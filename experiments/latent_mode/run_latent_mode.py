#!/usr/bin/env python
"""
Latent Mode Recovery: KMM's C_obs Advantage.

Generate: C=50 observable channels from D_true latent modes.
  x(t) = C_true @ g(t)   where g_d(t) = A_d·sin(ω_d·t + φ_d)
  C_true: random mixing matrix (D_true × C)

Three systems with different intrinsic dimensionality:
  D=3:   50 channels, driven by only 3 modes  (strong compression)
  D=10:  50 channels, driven by 10 modes      (moderate compression)
  D=30:  50 channels, driven by 30 modes      (weak compression)

Test: KMM (K=D_true) vs PatchTST (CI) vs iTransformer (CD) vs DLinear.
       KMM knows K via config. PatchTST treats 50 independent channels.

Hypothesis: KMM wins when D ≪ C (strong latent structure).
            Gap = KMM's C_obs captures the true mode structure.
            Gap widens: more noise, longer horizon, fewer training samples.
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
import torch.nn as nn


def generate_latent_mode_data(n_steps, C, D_true, noise_std=0.0, seed=42):
    """Generate C-channel data from D_true latent modes with random mixing.

    Returns: data (n_steps, C), C_true (D_true, C), true_modes (n_steps, D_true)
    """
    rng = np.random.RandomState(seed)

    # Generate D_true latent modes with distinct frequencies
    true_modes = np.zeros((n_steps, D_true), dtype=np.float32)
    t = np.arange(n_steps, dtype=np.float32)
    for d in range(D_true):
        omega = 2 * np.pi / (20 + d * 15)  # periods: 20, 35, 50, 65, ...
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 2.0)
        true_modes[:, d] = amp * np.sin(omega * t + phase)

    # Random mixing matrix
    C_true = rng.randn(D_true, C).astype(np.float32) * 0.5
    # Each channel gets contributions from 1-3 random modes (sparse mixing)
    mask = np.zeros((D_true, C), dtype=np.float32)
    for c in range(C):
        n_modes = rng.randint(1, min(4, D_true + 1))
        active = rng.choice(D_true, size=n_modes, replace=False)
        mask[active, c] = 1.0
    C_true = C_true * mask

    # Generate observations
    data = true_modes @ C_true  # (n_steps, C)
    data += rng.randn(*data.shape).astype(np.float32) * noise_std

    return data.astype(np.float32), C_true, true_modes


def _zero_track_b(pred_len, d_koopman, device):
    class M(torch.nn.Module):
        def forward(self, x):
            return (torch.zeros(x.shape[0], pred_len, device=x.device),
                    torch.zeros(x.shape[0], d_koopman, device=x.device))
    return M().to(device)

def _zero_gate(device, C_in):
    class M(torch.nn.Module):
        def forward(self, ha, hl):
            # ha: (B*K_proj, ...), need alpha shape (B*C_in, 1)
            B = ha.shape[0] // ha.shape[0]  # B*K / K, not quite right
            # Actually: ha.shape[0] = B * K_proj, we need B * C_in
            # The batch B is unknowable here, so just match the expected size
            # Fusion expects alpha.reshape(B, C_in, 1) → alpha has B*C_in elements
            n_total = hl.shape[0]  # h_local has B*C_in rows
            return torch.zeros(n_total, 1, device=ha.device)
    return M().to(device)


def train_kmm(X_tr, Y_tr, X_te, Y_te, C, K_proj, args, device):
    """Train KMM with specified K_proj."""
    configs = type('C', (), {})()
    configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
    configs.enc_in = C; configs.d_model = 64; configs.d_koopman = 32
    configs.n_blocks = 2; configs.dropout = 0.1

    # Override K_proj via model's C-adaptive logic
    model = Model(configs).to(device)
    model(torch.zeros(2, args.seq_len, C, device=device))
    model.track_b = _zero_track_b(args.pred_len, model.d_koopman, device)
    model.fusion_gate = _zero_gate(device, C)

    # Force K_proj to the desired value (monkey-patch C_obs on correct device)
    if K_proj != model.K_proj:
        new_c_obs = nn.Parameter(torch.empty(K_proj, C, device=device))
        nn.init.orthogonal_(new_c_obs)
        model.C_obs = new_c_obs
        model.K_proj = K_proj

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    n_train = len(X_tr)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i+args.batch_size]
            optimizer.zero_grad()
            out = model(X_tr[idx].to(device))
            pred = out[:, -args.pred_len:, :]
            loss = criterion(pred, Y_tr[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(X_te[:500].to(device))
        pred = out[:, -args.pred_len:, :]
        mse = criterion(pred, Y_te[:500].to(device)).item()
        mae = (pred - Y_te[:500].to(device)).abs().mean().item()

    return mse, mae, model


def train_dlinear(X_tr, Y_tr, X_te, Y_te, C, args, device):
    """DLinear baseline."""
    model = nn.Linear(args.seq_len, args.pred_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    n_train = len(X_tr)

    X_tr_t = X_tr.permute(0, 2, 1).to(device)
    Y_tr_t = Y_tr.permute(0, 2, 1).to(device)
    X_te_t = X_te[:500].permute(0, 2, 1).to(device)
    Y_te_t = Y_te[:500].permute(0, 2, 1).to(device)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i+args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), Y_tr_t[idx])
            loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_te_t).permute(0, 2, 1)
        mse = criterion(pred, Y_te[:500].to(device)).item()
    return mse, (pred - Y_te[:500].to(device)).abs().mean().item()


def main():
    parser = argparse.ArgumentParser('KMM-v3 Latent Mode Recovery')
    parser.add_argument('--output_dir', type=str, default='results/chaos_robustness/latent_mode')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)

    C = 50  # Fixed: 50 observable channels
    D_VALUES = [3, 10, 30]  # True latent dimensionality
    NOISE_LEVELS = [0.0, 0.2, 0.5, 1.0]
    KMM_K = [3, 8, 16]  # K_proj for KMM (≈ D_true or slightly less)

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*65}")
        print(f"Latent Mode Recovery | Seed={seed} | C={C}")
        print(f"{'='*65}")

        for di, D_true in enumerate(D_VALUES):
            for noise_std in NOISE_LEVELS:
                print(f"\n  D_true={D_true}, σ={noise_std}:")

                # Generate data
                data, C_true, true_modes = generate_latent_mode_data(
                    args.n_steps, C, D_true, noise_std, seed)

                data_t = torch.tensor(data, dtype=torch.float32)
                n = len(data) - args.seq_len - args.pred_len
                X = torch.stack([data_t[i:i+args.seq_len] for i in range(n)])
                Y = torch.stack([data_t[i+args.seq_len:i+args.seq_len+args.pred_len] for i in range(n)])

                # Normalize
                x_mean = X.mean(dim=(0,1), keepdim=True)
                x_std = X.std(dim=(0,1), keepdim=True) + 1e-8
                X_n, Y_n = (X - x_mean) / x_std, (Y - x_mean) / x_std

                n_train = int(len(X) * 0.7)
                X_tr, Y_tr = X_n[:n_train], Y_n[:n_train]
                X_te, Y_te = X_n[n_train:], Y_n[n_train:]

                # KMM
                kmm_mse, kmm_mae, model = train_kmm(
                    X_tr, Y_tr, X_te, Y_te, C, KMM_K[di], args, device)
                print(f"    KMM  (K={KMM_K[di]}):      MSE={kmm_mse:.4f}  MAE={kmm_mae:.4f}")

                # DLinear
                dl_mse, dl_mae = train_dlinear(X_tr, Y_tr, X_te, Y_te, C, args, device)
                print(f"    DLinear (shared): MSE={dl_mse:.4f}  MAE={dl_mae:.4f}")

                # Save
                saver.save_csv('latent_mode_results.csv', [[
                    seed, D_true, C, noise_std, KMM_K[di],
                    kmm_mse, kmm_mae, dl_mse, dl_mae,
                    time.strftime('%Y-%m-%d %H:%M:%S')
                ]], header=['seed', 'D_true', 'C', 'noise', 'K_proj',
                             'kmm_mse', 'kmm_mae', 'dlinear_mse', 'dlinear_mae', 'timestamp'])

    print(f"\nResults: {args.output_dir}/latent_mode_results.csv")


if __name__ == '__main__':
    main()
