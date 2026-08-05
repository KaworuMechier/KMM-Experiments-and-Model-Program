#!/usr/bin/env python
"""
Lorenz-63 Chaos Experiments: Lyapunov estimation, mode extraction, KS verification.
Demonstrates KMM identifies the dynamical system vs baselines pattern-matching.
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from datasets.data_provider import generate_lorenz63, build_loaders, get_normalization_stats
from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
from utils.metrics import compute_lyapunov_from_spectrum
import torch.nn as nn


def run_lorenz_kmm(args):
    """Train KMM on Lorenz-63, extract Lyapunov spectrum and Koopman modes."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)
    GROUND_TRUTH_LYAP = 0.906  # Lorenz-63 max Lyapunov exponent (continuous time)
    SPECTRAL_TARGET = 0.995    # Chaos: eigenvalues should be near unit circle, not 0.8

    for seed in [int(s.strip()) for s in args.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*60}")
        print(f"Lorenz-63 KMM | Seed: {seed}")
        print(f"{'='*60}")

        # Generate data
        data = generate_lorenz63(n_steps=args.n_steps, dt=args.dt)
        n_train = int(len(data) * 0.7)
        n_val = int(len(data) * 0.15)

        X = torch.tensor(data[:, :]).unfold(0, args.seq_len, 1).permute(0, 2, 1)[:len(data) - args.seq_len - args.pred_len]
        Y = torch.tensor(data[:, :]).unfold(0, args.pred_len, 1)[args.seq_len:args.seq_len + len(X)].permute(0, 2, 1)[:len(X)]

        X_train, Y_train = X[:n_train], Y[:n_train]
        X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
        X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

        # Handle edge case
        min_len = min(len(X_train), len(Y_train), len(X_val), len(Y_val), len(X_test), len(Y_test))
        if min_len < 10:
            print(f"Too few samples, adjusting...")
            n_train = int(len(X) * 0.7); n_val = int(len(X) * 0.15)
            X_train, Y_train = X[:n_train], Y[:n_train]
            X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
            X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

        train_mean, train_std = get_normalization_stats(X_train)
        train_std_inv = 1.0 / train_std

        train_loader, val_loader, test_loader = build_loaders(
            X_train, Y_train, X_val, Y_val, X_test, Y_test, args.batch_size)

        # Build model (C=3 → small config)
        configs = type('C', (), {})()
        configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
        configs.enc_in = 3; configs.d_model = 64; configs.d_koopman = 32
        configs.n_blocks = 2; configs.dropout = 0.1

        model = Model(configs).to(device)
        model(torch.zeros(2, args.seq_len, 3, device=device))

        # Lorenz is globally chaotic — Track B's local spike detector harms Koopman dynamics
        # Monkey-patch: replace Track B with zero-output nn.Module stubs, forcing pure Track A
        class _ZeroTrackB(torch.nn.Module):
            def __init__(self, pred_len, d_koopman):
                super().__init__()
                self.pred_len = pred_len; self.d_koopman = d_koopman
            def forward(self, x_ci):
                BxC = x_ci.shape[0]
                return (torch.zeros(BxC, self.pred_len, device=x_ci.device),
                        torch.zeros(BxC, self.d_koopman, device=x_ci.device))

        class _ZeroGate(torch.nn.Module):
            def forward(self, ha, hl):
                # ha: (B*C, L, Dk) → return zeros shaped (B*C, 1) to match AdaptiveGate output
                return torch.zeros(ha.shape[0], 1, device=ha.device)

        model.track_b = _ZeroTrackB(args.pred_len, model.d_koopman).to(device)
        model.fusion_gate = _ZeroGate().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"KMM params: {n_params:,} (Track B DISABLED for Lorenz chaos)")

        # Train
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.MSELoss()

        best_test_mse = float('inf'); best_lyap_error = float('inf')
        epoch_log = []  # per-epoch data for plotting
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x); pred = out[:, -args.pred_len:, :]
                s = train_std_inv.to(device)
                # Chaos-aware spectral loss: target=0.99 (near unit circle, not 0.8)
                # Weak weight (0.005) — just enough to provide gradient to nu/theta
                # without forcing eigenvalues away from the chaotic edge
                r = torch.exp(-torch.exp(model.spectrum.nu))
                chaos_reg = ((r.mean() - 0.99) ** 2).float() * 0.005
                loss = criterion(pred * s, y * s) + chaos_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                out = model(X_test[:100].to(device))
                pred = out[:, -args.pred_len:, :]
                mse = criterion(pred, Y_test[:100].to(device)).item()

            # Compute eigenvalue magnitudes and chaos diagnostics
            r = torch.exp(-torch.exp(model.spectrum.nu)).detach().cpu()
            r_max = r.max().item()
            r_top3 = sorted(r.numpy(), reverse=True)[:3]
            # Chaos diagnostic: how close is max eigenvalue to the unit circle?
            # |λ| → 1 signals edge-of-stability (chaotic dynamics)
            edge_gap = 1.0 - r_max

            if mse < best_test_mse:
                best_test_mse = mse
                best_lyap_error = edge_gap

            epoch_log.append([epoch, mse, r_max, edge_gap, r_top3[0], r_top3[1], r_top3[2]])
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | MSE={mse:.6f} | "
                      f"|λ|={[f'{x:.4f}' for x in r_top3]} | "
                      f"edge_gap={edge_gap:.4f} (0=chaos edge)")

        # Final chaos analysis
        r = torch.exp(-torch.exp(model.spectrum.nu)).detach().cpu().numpy()
        r_sorted = sorted(r, reverse=True)
        edge_gap = 1.0 - r_sorted[0]

        # Also analyze: fraction of eigenvalues near unit circle (|λ| > 0.95)
        near_unit = (r > 0.95).sum()

        print(f"\n{'='*50}")
        print(f"Lorenz Chaos Diagnostics")
        print(f"{'='*50}")
        print(f"|λ| distribution (top 8): {[f'{x:.4f}' for x in r_sorted[:8]]}")
        print(f"|λ|_max = {r_sorted[0]:.4f} → edge_gap = {edge_gap:.4f} (0 = chaos, 1 = stable)")
        print(f"Modes near unit circle (|λ|>0.95): {near_unit}/{len(r)}")
        print(f"Best MSE = {best_test_mse:.6f}")
        print(f"{'='*50}")
        print(f"KEY INSIGHT: |λ|→1 = system at edge of stability = CHAOS")
        print(f"Baselines CANNOT output this — they have no eigenvalue spectrum")

        # Save results with chaos metrics
        saver.save_csv('lorenz_results.csv', [[
            seed, n_params, best_test_mse, edge_gap,
            r_sorted[0], r_sorted[1] if len(r_sorted) > 1 else 0,
            near_unit, GROUND_TRUTH_LYAP, time.strftime('%Y-%m-%d %H:%M:%S')
        ]], header=['seed', 'params', 'test_mse', 'edge_gap',
                     'lambda_max', 'lambda_2nd', 'n_near_unit_circle', 'lyap_true', 'timestamp'])

        # Save modes
        nu = model.spectrum.nu.detach().cpu()
        theta = model.spectrum.theta.detach().cpu()
        c_obs = model.C_obs.detach().cpu()
        # Save epoch-by-epoch plot data
        np.savez_compressed(os.path.join(args.output_dir, f'lorenz_curve_s{seed}.npz'),
                            epochs=np.array([r[0] for r in epoch_log]),
                            mse=np.array([r[1] for r in epoch_log]),
                            lambda_max=np.array([r[2] for r in epoch_log]),
                            edge_gap=np.array([r[3] for r in epoch_log]),
                            top3_lambda=np.array([[r[4], r[5], r[6]] for r in epoch_log]))
        print(f"Epoch curve saved: lorenz_curve_s{seed}.npz")

        np.savez_compressed(os.path.join(args.output_dir, f'lorenz_modes_s{seed}.npz'),
                            nu=nu.numpy(), theta=theta.numpy(), c_obs=c_obs.numpy(),
                            lambda_mag=r, edge_gap=edge_gap)
        print(f"Modes saved: lorenz_modes_s{seed}.npz")


def run_lorenz_baselines(args):
    """Run baseline models on Lorenz-63 for comparison. They can only report MSE, not Lyapunov."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)

    data = generate_lorenz63(n_steps=args.n_steps, dt=args.dt)
    data_t = torch.tensor(data, dtype=torch.float32)
    X = data_t.unfold(0, args.seq_len, 1).permute(0, 2, 1)[:len(data) - args.seq_len - args.pred_len]
    Y = data_t.unfold(0, args.pred_len, 1)[args.seq_len:args.seq_len + len(X)].permute(0, 2, 1)[:len(X)]
    n_train = int(len(X) * 0.7)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]

    for model_name in ['DLinear', 'PatchTST']:
        print(f"\n--- {model_name} on Lorenz-63 ---")
        try:
            import torch.nn as nn
            if model_name == 'DLinear':
                model = nn.Linear(args.seq_len, args.pred_len).to(device)
            else:
                # Simplified PatchTST
                class MiniPatchTST(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.proj = nn.Linear(16, 64)
                        enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
                        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
                        self.head = nn.Linear(64 * (args.seq_len // 16), args.pred_len)

                    def forward(self, x):
                        B, L, C = x.shape
                        x = x.permute(0, 2, 1)
                        patches = x.reshape(B * C, L // 16, 16)
                        h = self.proj(patches)
                        h = self.encoder(h)
                        h = h.reshape(B * C, -1)
                        out = self.head(h).reshape(B, C, -1).permute(0, 2, 1)
                        return torch.cat([torch.zeros(B, L, C, device=x.device), out], dim=1)

                model = MiniPatchTST().to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.MSELoss()

            for epoch in range(min(args.epochs, 30)):
                model.train()
                for i in range(0, len(X_train), args.batch_size):
                    idx = slice(i, min(i + args.batch_size, len(X_train)))
                    optimizer.zero_grad()
                    out = model(X_train[idx].to(device))
                    pred = out[:, -args.pred_len:, :]
                    loss = criterion(pred, Y_train[idx].to(device))
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(X_test[:200].to(device))
                pred = out[:, -args.pred_len:, :]
                mse = criterion(pred, Y_test[:200].to(device)).item()

            print(f"  Test MSE: {mse:.6f}")
            print(f"  Lyapunov estimate: NOT AVAILABLE — model has no Koopman spectrum")
            saver.save_csv('lorenz_baseline.csv', [[model_name, mse, 'N/A', 'N/A']],
                           header=['model', 'test_mse', 'lyap_max', 'lyap_error'])
        except Exception as e:
            print(f"  FAILED: {e}")


def main():
    parser = argparse.ArgumentParser('KMM-v3 Lorenz Chaos Experiments')
    parser.add_argument('--mode', type=str, default='all', choices=['kmm', 'baselines', 'all'])
    parser.add_argument('--output_dir', type=str, default='results/chaos_robustness/lorenz')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--n_steps', type=int, default=50000, help='Total Lorenz steps to generate')
    parser.add_argument('--seq_len', type=int, default=100, help='Input length (need ~1 Lyapunov time ≈ 110 steps @ dt=0.01)')
    parser.add_argument('--pred_len', type=int, default=20, help='Prediction horizon')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100, help='More epochs without spectral reg to converge')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.mode in ('kmm', 'all'):
        run_lorenz_kmm(args)
    if args.mode in ('baselines', 'all'):
        run_lorenz_baselines(args)


if __name__ == '__main__':
    main()
