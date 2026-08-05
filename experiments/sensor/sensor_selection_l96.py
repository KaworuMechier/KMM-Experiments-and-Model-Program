#!/usr/bin/env python
"""
Sensor Selection on Lorenz-96: PCA vs KMM (C_obs).

Lorenz-96: dx_i/dt = (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F
  N=40 variables, F=8 (chaotic regime).
  Each variable is nonlinearly coupled to 4 neighbors.

Key: PCA selects variables with highest VARIANCE.
     KMM selects variables with highest DYNAMICAL INFLUENCE.

Test: Long-horizon prediction (H=10, 20, 50, 100).
      Short H: PCA works (momentum). Long H: KMM wins (dynamics).
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
import torch.nn as nn


def generate_lorenz96(N=40, F=8, dt=0.05, n_steps=15000, seed=42):
    """Generate Lorenz-96 data. N=40, F=8 -> chaotic."""
    rng = np.random.RandomState(seed)
    x = rng.randn(N).astype(np.float64) * 0.1
    x[0] += F  # slight perturbation
    data = np.zeros((n_steps, N), dtype=np.float32)
    for t in range(n_steps):
        data[t] = x.astype(np.float32)
        for i in range(N):
            xp1 = x[(i+1) % N]; xm1 = x[(i-1) % N]; xm2 = x[(i-2) % N]
            dx = (xp1 - xm2) * xm1 - x[i] + F
            x[i] += dt * dx
    return data


def _zero_mods(pred_len, dk, dev):
    class A(torch.nn.Module):
        def forward(self, x):
            return (torch.zeros(x.shape[0], pred_len, device=x.device),
                    torch.zeros(x.shape[0], dk, device=x.device))
    class B(torch.nn.Module):
        def forward(self, ha, hl):
            return torch.zeros(hl.shape[0], 1, device=ha.device)
    return A().to(dev), B().to(dev)


def greedy_selection(signatures, budget):
    C, D = signatures.shape
    sel = [int(np.argmax(np.linalg.norm(signatures, axis=1)))]
    rem = set(range(C)) - {sel[0]}
    while len(sel) < budget:
        span = signatures[sel]
        best, best_i = -1, -1
        for i in rem:
            si = signatures[i]
            proj = si @ span.T @ np.linalg.pinv(span @ span.T) @ span
            r = np.linalg.norm(si - proj)
            if r > best_i: best, best_i = r, i
        sel.append(best_i); rem.remove(best_i)
    return sel


def long_horizon_prediction_error(data, sel_idx, seq_len, pred_horizons):
    """Train linear predictor: selected[t-L:t] -> all[t+H], measure MSE."""
    data_t = torch.tensor(data, dtype=torch.float32)
    C = data_t.shape[1]; K = len(sel_idx)
    results = {}
    for H in pred_horizons:
        n = len(data) - seq_len - H
        if n < 100: continue
        X_m = torch.stack([data_t[i:i+seq_len][:, sel_idx].flatten() for i in range(n)])
        Y_full = data_t[seq_len+H-1 : seq_len+H-1+n]
        X_m = torch.cat([X_m, torch.ones(n,1)], dim=1)
        W = torch.linalg.lstsq(X_m, Y_full).solution
        mse = ((X_m @ W - Y_full)**2).mean().item()
        results[H] = mse
    return results


def main():
    parser = argparse.ArgumentParser('Lorenz-96 Sensor Selection')
    parser.add_argument('--output_dir', type=str, default='results/optimization')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--N', type=int, default=40)
    parser.add_argument('--F', type=float, default=8.0)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--n_steps', type=int, default=12000)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--pred_len', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)
    N = args.N
    K_BUDGETS = [3, 5, 10, 20]
    PRED_HORIZONS = [5, 10, 20, 50, 100]

    for seed in [int(s.strip()) for s in args.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*65}")
        print(f"Lorenz-96 Sensor Selection | Seed={seed} | N={N}")
        print(f"{'='*65}")

        data = generate_lorenz96(N=N, F=args.F, dt=args.dt, n_steps=args.n_steps, seed=seed)
        data_t = torch.tensor(data, dtype=torch.float32)
        print(f"  Data: {data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")

        n = len(data) - args.seq_len - args.pred_len
        X = torch.stack([data_t[i:i+args.seq_len] for i in range(n)])
        Y = torch.stack([data_t[i+args.seq_len:i+args.seq_len+args.pred_len] for i in range(n)])
        xm = X.mean(dim=(0,1), keepdim=True); xs = X.std(dim=(0,1), keepdim=True) + 1e-8
        Xn, Yn = (X - xm) / xs, (Y - xm) / xs
        nt = int(len(X) * 0.7)
        Xtr, Ytr = Xn[:nt].to(device), Yn[:nt].to(device)

        # Train KMM
        configs = type('C', (), {})()
        configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
        configs.enc_in = N; configs.d_model = 64; configs.d_koopman = 32
        configs.n_blocks = 2; configs.dropout = 0.1

        model = Model(configs).to(device)
        model(torch.zeros(1, args.seq_len, N, device=device))
        a,b = _zero_mods(args.pred_len, model.d_koopman, device)
        model.track_b = a; model.fusion_gate = b
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.MSELoss()

        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), args.batch_size):
                idx = perm[i:i+args.batch_size]; opt.zero_grad()
                loss = crit(model(Xtr[idx])[:, -args.pred_len:, :], Ytr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        print(f"  KMM trained ({args.epochs} epochs)")

        # Extract signatures
        c_obs = model.C_obs.detach().cpu().numpy()  # (K_proj, N)
        kmm_sig = c_obs.T  # (N, K_proj)

        # PCA signatures
        dc = data - data.mean(axis=0)
        _, _, Vt = np.linalg.svd(dc, full_matrices=False)
        pca_sig = Vt[:min(10,N), :].T

        # Compare sensor selection with LONG-HORIZON error
        for K_budget in K_BUDGETS:
            kmm_sel = greedy_selection(kmm_sig, K_budget)
            pca_sel = greedy_selection(pca_sig, K_budget)
            rng = np.random.RandomState(seed)
            rnd_sel = list(rng.choice(N, size=K_budget, replace=False))

            kmm_errs = long_horizon_prediction_error(data, kmm_sel, args.seq_len, PRED_HORIZONS)
            pca_errs = long_horizon_prediction_error(data, pca_sel, args.seq_len, PRED_HORIZONS)
            rnd_errs = long_horizon_prediction_error(data, rnd_sel, args.seq_len, PRED_HORIZONS)

            print(f"\n  K={K_budget}:")
            print(f"    {'H':>5s}  {'KMM':>10s} {'PCA':>10s} {'Random':>10s} {'Winner'}")
            for H in PRED_HORIZONS:
                if H not in kmm_errs: continue
                w = 'KMM' if kmm_errs[H] < pca_errs[H] else 'PCA'
                print(f"    {H:>5d}  {kmm_errs[H]:>10.6f} {pca_errs[H]:>10.6f} {rnd_errs[H]:>10.6f} {w:>6s}")
                saver.save_csv('l96_sensor_selection.csv', [[
                    seed, N, K_budget, H,
                    kmm_errs[H], pca_errs[H], rnd_errs[H],
                    'KMM' if kmm_errs[H] < pca_errs[H] else 'PCA',
                    time.strftime('%Y-%m-%d %H:%M:%S')
                ]], header=['seed','N','K','H','kmm','pca','random','winner','timestamp'])

    print(f"\n  Results: {args.output_dir}/l96_sensor_selection.csv")


if __name__ == '__main__':
    main()
