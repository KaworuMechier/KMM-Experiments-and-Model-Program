"""
Unified data provider for all KMM-v3 sub-projects.
Supports: standard CSV datasets, Lorenz-63/96 generation, noise-perturbed data.
"""
import os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset

# ── Dataset registry ─────────────────────────────────────────────
DATASET_REGISTRY = {
    'ETTh1':  'ETTh1.csv',
    'ETTh2':  'ETTh2.csv',
    'ETTm1':  'ETTm1.csv',
    'ETTm2':  'ETTm2.csv',
    'ECL':    'ECL.csv',
    'Weather':'Weather.csv',
    'Traffic':'traffic.csv',
    'METR-LA':'metr-la_traffic.csv',
    'PEMS-BAY':'pems-bay_traffic.csv',
    'PJM':    'pjm_combined.csv',
    'ERA5_EC':'era5_east_china_990ch.csv',
}


def get_available_datasets():
    return list(DATASET_REGISTRY.keys())


# ── Standard CSV loading ─────────────────────────────────────────
def downsample_data(data, factor):
    """Take every N-th row. factor=3 means 5min->15min."""
    return data[::factor]


def load_csv_dataset(data_path, seq_len=96, pred_len=96, dataset='ECL', downsample=1):
    """Load a CSV time series, return (X_train, Y_train, X_val, Y_val, X_test, Y_test)."""
    df = pd.read_csv(data_path)

    # ETT datasets use a fixed date-based split
    ett_datasets = {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'}
    if dataset in ett_datasets:
        return _load_ett(df, seq_len, pred_len, dataset)

    # Generic: date column + numeric columns
    first_col = str(df.columns[0]).lower()
    # Detect datetime index column: named date/time/datetime, unnamed (pandas Unnamed: 0),
    # or the column values look like datetimes (contains ':' and '-')
    col0_sample = str(df.iloc[0, 0])
    is_datetime_col = (first_col in ('date', 'time', 'datetime')
                       or 'unnamed' in first_col
                       or first_col == ''
                       or (':' in col0_sample and '-' in col0_sample))
    if is_datetime_col:
        cols = df.columns[1:].tolist()
        df = df.rename(columns={df.columns[0]: 'date'})
    else:
        cols = df.columns.tolist()

    data = df[cols].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

    if downsample > 1:
        data = data[::downsample]
        print(f"  Downsampled {downsample}x: {len(data)} steps")

    total = len(data)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    def _make(s, e):
        X, Y = [], []
        for i in range(s, e - seq_len - pred_len + 1):
            X.append(data[i:i + seq_len])
            Y.append(data[i + seq_len:i + seq_len + pred_len])
        if len(X) == 0:
            return torch.empty(0, seq_len, data.shape[1]), torch.empty(0, pred_len, data.shape[1])
        return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

    X_train, Y_train = _make(0, train_end)
    X_val, Y_val = _make(train_end, val_end)
    X_test, Y_test = _make(val_end, total)

    print(f"Loaded {dataset}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def _load_ett(df, seq_len, pred_len, dataset):
    """ETT-specific split: first 12 months train, next 4 val, rest test."""
    cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    data = df[cols].values.astype(np.float32)
    total = len(df)
    multiplier = 4 if dataset in ('ETTm1', 'ETTm2') else 1
    train_end = 12 * 30 * 24 * multiplier
    val_end = train_end + 4 * 30 * 24 * multiplier

    def _make(s, e):
        X, Y = [], []
        for i in range(s, e - seq_len - pred_len + 1):
            X.append(data[i:i + seq_len])
            Y.append(data[i + seq_len:i + seq_len + pred_len])
        return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

    X_train, Y_train = _make(0, train_end)
    X_val, Y_val = _make(train_end, val_end)
    X_test, Y_test = _make(val_end, total)
    print(f"Loaded {dataset}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ── Lorenz-63 generator ──────────────────────────────────────────
def generate_lorenz63(n_steps=50000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0,
                      x0=None, discard=5000):
    """Generate Lorenz-63 trajectories."""
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0])

    xs = np.zeros((n_steps + discard, 3), dtype=np.float32)
    xs[0] = x0
    for i in range(1, n_steps + discard):
        x, y, z = xs[i - 1]
        xs[i, 0] = x + dt * sigma * (y - x)
        xs[i, 1] = y + dt * (x * (rho - z) - y)
        xs[i, 2] = z + dt * (x * y - beta * z)
    return xs[discard:]


def generate_lorenz63_control(n_steps=20000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Generate Lorenz-63 with random control input on x-component."""
    x0 = np.array([1.0, 1.0, 1.0])
    n_total = n_steps + 2000
    xs = np.zeros((n_total, 3), dtype=np.float32)
    us = np.random.randn(n_total).astype(np.float32) * 0.5
    xs[0] = x0
    for i in range(1, n_total):
        x, y, z = xs[i - 1]
        xs[i, 0] = x + dt * (sigma * (y - x) + us[i - 1])
        xs[i, 1] = y + dt * (x * (rho - z) - y)
        xs[i, 2] = z + dt * (x * y - beta * z)
    return xs[2000:], us[2000:]


# ── Noise-perturbed signal generator ─────────────────────────────
def generate_noisy_periodic(n_steps=10000, periods=(24, 168), noise_levels=(0, 0.5, 1.0, 2.0)):
    """Generate multi-periodic signals with controlled noise for robustness testing."""
    t = np.arange(n_steps, dtype=np.float32)
    clean = np.zeros(n_steps, dtype=np.float32)
    for p in periods:
        clean += np.sin(2 * np.pi * t / p)
    clean /= len(periods)

    datasets = {}
    for sigma in noise_levels:
        noisy = clean + np.random.randn(n_steps).astype(np.float32) * sigma
        datasets[f'sigma_{sigma}'] = noisy.reshape(-1, 1)
    datasets['clean'] = clean.reshape(-1, 1)
    return datasets


# ── DataLoader builder ───────────────────────────────────────────
def build_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                  batch_size=16, num_workers=0):
    """Build DataLoaders from tensors. Returns (train_loader, val_loader, test_loader)."""
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    test_ds = TensorDataset(X_test, Y_test)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


def get_normalization_stats(X_train):
    """Compute per-channel mean and std from training data."""
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True) + 1e-8
    return mean, std
