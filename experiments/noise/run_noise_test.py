#!/usr/bin/env python
"""
KMM as a System Diagnostic: "What kind of system generated this signal?"

Three system types:
  A: Chaotic  (Lorenz-63)     → |λ| → 1.0 (all modes at edge of stability)
  B: Periodic (sine + noise)  → |λ| distributed [0.7, 0.95] (structured)
  C: Noise    (pure Gaussian) → |λ| → 0 (no persistent dynamics)

Baselines (PatchTST, DLinear, TimesNet, FFT) can predict/analyze,
but CANNOT answer "Is this system chaotic?" — they have no eigenvalue spectrum.

Exp: Train KMM on each system type × multiple noise levels.
     Extract eigenvalue spectrum |λ_d| for each.
     Show: the spectrum CLASSIFIES the system type, regardless of noise level.
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
import torch.nn as nn


def generate_lorenz63(n_steps=10000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0, noise_std=0.0):
    """Lorenz-63 with optional additive noise."""
    x = np.zeros((n_steps + 2000, 3), dtype=np.float32)
    x[0] = [1.0, 1.0, 1.0]
    for i in range(1, len(x)):
        xc, yc, zc = x[i-1]
        x[i, 0] = xc + dt * sigma * (yc - xc)
        x[i, 1] = yc + dt * (xc * (rho - zc) - yc)
        x[i, 2] = zc + dt * (xc * yc - beta * zc)
    data = x[2000:]
    if noise_std > 0:
        data += np.random.randn(*data.shape).astype(np.float32) * noise_std
    return data


def generate_periodic(n_steps=10000, periods=(24, 168), noise_std=0.0):
    """Multi-periodic signal with noise."""
    t = np.arange(n_steps, dtype=np.float32)
    signal = np.zeros((n_steps, 3), dtype=np.float32)
    for ch in range(3):
        s = np.zeros(n_steps, dtype=np.float32)
        for p in periods:
            s += np.sin(2 * np.pi * t / p + ch * 0.5)
        s += np.random.randn(n_steps).astype(np.float32) * noise_std
        signal[:, ch] = s
    return signal


def generate_white_noise(n_steps=10000, noise_std=1.0):
    """Pure white noise."""
    return np.random.randn(n_steps, 3).astype(np.float32) * noise_std


def make_sequences(data, seq_len, pred_len):
    """Convert (T, C) time series to (N, L, C) input and (N, H, C) target."""
    data_t = torch.tensor(data, dtype=torch.float32)
    n = len(data) - seq_len - pred_len
    X = torch.stack([data_t[i:i+seq_len] for i in range(n)])
    Y = torch.stack([data_t[i+seq_len:i+seq_len+pred_len] for i in range(n)])
    return X, Y


def _zero_track_b(pred_len, d_koopman, device):
    class M(torch.nn.Module):
        def forward(self, x):
            return (torch.zeros(x.shape[0], pred_len, device=x.device),
                    torch.zeros(x.shape[0], d_koopman, device=x.device))
    return M().to(device)


def _zero_gate(device):
    class M(torch.nn.Module):
        def forward(self, ha, hl):
            return torch.zeros(ha.shape[0], 1, device=ha.device)
    return M().to(device)


def train_and_extract_spectrum(data, args, device):
    """Train KMM on data, return eigenvalue magnitudes |λ_d|."""
    X, Y = make_sequences(data, args.seq_len, args.pred_len)
    C = data.shape[1]

    # Normalize
    x_mean = X.mean(dim=(0,1), keepdim=True)
    x_std = X.std(dim=(0,1), keepdim=True) + 1e-8
    X_n, Y_n = (X - x_mean) / x_std, (Y - x_mean) / x_std

    n_train = int(len(X) * 0.8)
    X_tr, Y_tr = X_n[:n_train].to(device), Y_n[:n_train].to(device)
    X_te, Y_te = X_n[n_train:].to(device), Y_n[n_train:].to(device)

    # Build model
    configs = type('C', (), {})()
    configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
    configs.enc_in = C; configs.d_model = 32; configs.d_koopman = 16
    configs.n_blocks = 2; configs.dropout = 0.1

    model = Model(configs).to(device)
    model(torch.zeros(2, args.seq_len, C, device=device))
    model.track_b = _zero_track_b(args.pred_len, model.d_koopman, device)
    model.fusion_gate = _zero_gate(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), args.batch_size):
            idx = perm[i:i+args.batch_size]
            optimizer.zero_grad()
            out = model(X_tr[idx])
            pred = out[:, -args.pred_len:, :]
            loss = criterion(pred, Y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(X_te[:200])
        pred = out[:, -args.pred_len:, :]
        mse = criterion(pred, Y_te[:200]).item()

    # Extract spectrum
    r = torch.exp(-torch.exp(model.spectrum.nu)).detach().cpu().numpy()
    return np.sort(r)[::-1], mse


def main():
    parser = argparse.ArgumentParser('KMM-v3 System Diagnosis Experiment')
    parser.add_argument('--output_dir', type=str, default='results/chaos_robustness/noise')
    parser.add_argument('--seeds', type=str, default='2021')
    parser.add_argument('--n_steps', type=int, default=12000)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)

    # ── Experiment design ────────────────────────────────────────
    systems = {
        'Chaotic (Lorenz)': {
            'gen': lambda ns: generate_lorenz63(n_steps=args.n_steps, noise_std=ns),
            'noise_levels': [0.0, 0.5, 1.0, 2.0, 5.0],
        },
        'Periodic (Sine+Noise)': {
            'gen': lambda ns: generate_periodic(n_steps=args.n_steps, noise_std=ns),
            'noise_levels': [0.0, 0.5, 1.0, 2.0, 5.0],
        },
        'White Noise': {
            'gen': lambda ns: generate_white_noise(n_steps=args.n_steps, noise_std=max(ns, 0.5)),
            'noise_levels': [0.5, 1.0, 2.0, 5.0],
        },
    }

    for seed in [int(s.strip()) for s in args.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*70}")
        print(f"KMM System Diagnosis | Seed={seed}")
        print(f"{'='*70}")

        all_spectra = {}

        for sys_name, sys_config in systems.items():
            print(f"\n  {sys_name}:")
            for ns in sys_config['noise_levels']:
                label = f"{sys_name} σ={ns}"
                print(f"    Training {label}...", end=' ', flush=True)
                data = sys_config['gen'](ns)
                spectrum, mse = train_and_extract_spectrum(data, args, device)
                all_spectra[label] = (spectrum, mse)
                print(f"MSE={mse:.4f} |λ|=[{', '.join(f'{x:.3f}' for x in spectrum[:5])}]", flush=True)

        # ── Diagnostic table ──────────────────────────────────
        print(f"\n  {'='*60}")
        print(f"  System Diagnosis via Koopman Spectrum")
        print(f"  {'='*60}")
        print(f"  {'System':<30s} {'|λ|_max':>8s} {'|λ|_mean':>8s} {'Diagnosis':>15s}")
        print(f"  {'─'*65}")

        for label, (spec, mse) in sorted(all_spectra.items()):
            lam_max = spec[0]
            lam_mean = spec.mean()
            if lam_max > 0.97:
                diag = 'CHAOTIC'
            elif lam_max > 0.6:
                diag = 'PERIODIC'
            else:
                diag = 'NOISE'
            print(f"  {label:<30s} {lam_max:>8.4f} {lam_mean:>8.4f} {diag:>15s}")

        print(f"\n  KEY: |λ|→1.0 = CHAOS (edge of stability)")
        print(f"       |λ|∈[0.6,0.95] = PERIODIC (structured dynamics)")
        print(f"       |λ|→0 = NOISE (no persistent structure)")

        # ── Save results ──────────────────────────────────────
        for label, (spec, mse) in all_spectra.items():
            saver.save_csv('system_diagnosis.csv', [[
                seed, label, spec[0], spec.mean(), mse,
                ','.join(f'{x:.4f}' for x in spec[:10]),
                time.strftime('%Y-%m-%d %H:%M:%S')
            ]], header=['seed', 'system', 'lambda_max', 'lambda_mean', 'mse',
                         'top10_spectrum', 'timestamp'])

        # Save all spectra for plotting
        np.savez_compressed(os.path.join(args.output_dir, f'diagnosis_spectra_s{seed}.npz'),
                            labels=list(all_spectra.keys()),
                            spectra={k: v[0] for k, v in all_spectra.items()})

    print(f"\n  Results saved to {args.output_dir}/system_diagnosis.csv")


if __name__ == '__main__':
    main()
