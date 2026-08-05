#!/usr/bin/env python
"""
Sensor Selection on KS Equation: PCA vs KMM (C_obs).

KS: u_t = -u*u_x - u_xx - u_xxxx, N=64 periodic grid.
  Energy cascades from large to small scales.
  High-variance points ≠ dynamically important points.
  PCA picks high-variance → wrong.
  KMM picks C_obs coverage → captures modal dynamics.

Test: long-horizon prediction (H=10, 30, 60, 120).
      Short H: PCA works (energy-rich points predict energy-rich).
      Long H: KMM wins (modal coverage > variance for dynamics).
"""
import os, sys, argparse, torch, numpy as np, time
from scipy.integrate import solve_ivp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
import torch.nn as nn


def generate_ks(N=64, L=22.0, dt=0.5, n_steps=10000, seed=42):
    """Stable KS integration via scipy Radau."""
    rng = np.random.RandomState(seed)

    def rhs(t, u):
        uh = np.fft.rfft(u); k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
        ux = np.fft.irfft(1j*k*uh, n=N); uxx = np.fft.irfft(-k**2*uh, n=N)
        uxxxx = np.fft.irfft(k**4*uh, n=N)
        return -u*ux - uxx - uxxxx

    u0 = rng.randn(N).astype(np.float64) * 0.1
    T = n_steps * dt
    t_eval = np.arange(0, T, dt)
    print(f"    Integrating KS (N={N}, {len(t_eval)} steps)...", flush=True)
    sol = solve_ivp(rhs, [0, T], u0, method='Radau', t_eval=t_eval,
                    max_step=dt, rtol=1e-6, atol=1e-8)
    d = sol.y.T.astype(np.float32)
    print(f"    Done. range=[{d.min():.3f}, {d.max():.3f}]", flush=True)
    return d


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


def long_horizon_error(data, sel_idx, seq_len, horizons):
    """Train linear predictor: selected[−L:] → all[+H], measure MSE."""
    data_t = torch.tensor(data, dtype=torch.float32); C = data_t.shape[1]
    results = {}
    for H in horizons:
        n = len(data) - seq_len - H
        if n < 100: continue
        X_m = torch.stack([data_t[i:i+seq_len][:, sel_idx].flatten() for i in range(n)])
        Y = data_t[seq_len+H-1 : seq_len+H-1+n]
        X_m = torch.cat([X_m, torch.ones(n,1)], dim=1)
        W = torch.linalg.lstsq(X_m, Y).solution
        results[H] = ((X_m @ W - Y)**2).mean().item()
    return results


def main():
    parser = argparse.ArgumentParser('KS Sensor Selection')
    parser.add_argument('--output_dir', type=str, default='results/optimization')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--L', type=float, default=22.0)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=80)
    parser.add_argument('--pred_len', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)
    N = args.N
    K_BUDGETS = [4, 8, 16, 32]
    HORIZONS = [10, 30, 60, 120]

    for seed in [int(s.strip()) for s in args.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*65}")
        print(f"KS Sensor Selection | Seed={seed} | N={N}")
        print(f"{'='*65}")

        data = generate_ks(N=N, L=args.L, dt=args.dt, n_steps=args.n_steps, seed=seed)
        data_t = torch.tensor(data, dtype=torch.float32)

        n = len(data) - args.seq_len - args.pred_len
        X = torch.stack([data_t[i:i+args.seq_len] for i in range(n)])
        Y = torch.stack([data_t[i+args.seq_len:i+args.seq_len+args.pred_len] for i in range(n)])
        xm = X.mean(dim=(0,1), keepdim=True); xs = X.std(dim=(0,1), keepdim=True) + 1e-8
        Xn, Yn = (X - xm) / xs, (Y - xm) / xs
        nt = int(len(X) * 0.7)
        Xtr, Ytr = Xn[:nt].to(device), Yn[:nt].to(device)

        configs = type('C', (), {})()
        configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
        configs.enc_in = N; configs.d_model = 128; configs.d_koopman = 32
        configs.n_blocks = 2; configs.dropout = 0.0

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

        c_obs = model.C_obs.detach().cpu().numpy()
        kmm_sig = c_obs.T  # (N, K_proj)
        dc = data - data.mean(axis=0)
        _, _, Vt = np.linalg.svd(dc, full_matrices=False)
        pca_sig = Vt[:min(16,N), :].T

        for K_budget in K_BUDGETS:
            kmm_sel = greedy_selection(kmm_sig, K_budget)
            pca_sel = greedy_selection(pca_sig, K_budget)
            rng = np.random.RandomState(seed)
            rnd_sel = list(rng.choice(N, size=K_budget, replace=False))

            kmm_e = long_horizon_error(data, kmm_sel, args.seq_len, HORIZONS)
            pca_e = long_horizon_error(data, pca_sel, args.seq_len, HORIZONS)
            rnd_e = long_horizon_error(data, rnd_sel, args.seq_len, HORIZONS)

            print(f"\n  K={K_budget}:")
            print(f"    {'H':>5s}  {'KMM':>10s} {'PCA':>10s} {'Random':>10s} {'Winner':>6s} {'Adv%':>6s}")
            n_win = 0
            for H in HORIZONS:
                if H not in kmm_e: continue
                w = 'KMM' if kmm_e[H] < pca_e[H] else 'PCA'
                adv = abs(kmm_e[H] - pca_e[H]) / max(kmm_e[H], pca_e[H]) * 100
                print(f"    {H:>5d}  {kmm_e[H]:>10.6f} {pca_e[H]:>10.6f} {rnd_e[H]:>10.6f} {w:>6s} {adv:>5.1f}%")
                if w == 'KMM': n_win += 1
                saver.save_csv('ks_sensor_selection.csv', [[
                    seed, N, K_budget, H,
                    kmm_e[H], pca_e[H], rnd_e[H], w, adv,
                    time.strftime('%Y-%m-%d %H:%M:%S')
                ]], header=['seed','N','K','H','kmm','pca','random','winner','adv_pct','timestamp'])

    print(f"\n  Results: {args.output_dir}/ks_sensor_selection.csv")


if __name__ == '__main__':
    main()
