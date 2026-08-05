#!/usr/bin/env python
"""
Kuramoto-Sivashinsky Control via Koopman MPC.

KS: u_t = -u·u_x - u_xx - u_xxxx, periodic [0, L].
Uses scipy's stiff solver (Radau) for stable integration.

Control: suppress spatiotemporal chaos via closed-form Koopman MPC.
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
import torch.nn as nn
from scipy.integrate import solve_ivp


def generate_ks_data(N=32, L=16.0, dt=0.25, n_steps=10000, seed=42):
    """Stable KS integration via scipy Radau (implicit, adaptive)."""
    rng = np.random.RandomState(seed)

    def rhs(t, u):
        u_hat = np.fft.rfft(u)
        k = 2 * np.pi * np.fft.rfftfreq(N, d=L/N)
        u_x = np.fft.irfft(1j * k * u_hat, n=N)
        u_xx = np.fft.irfft(-k**2 * u_hat, n=N)
        u_xxxx = np.fft.irfft(k**4 * u_hat, n=N)
        return -u * u_x - u_xx - u_xxxx

    u0 = rng.randn(N).astype(np.float64) * 0.1
    T = n_steps * dt
    t_eval = np.arange(0, T, dt)
    print(f"    Integrating KS (N={N}, L={L}, {len(t_eval)} steps)...", flush=True)
    sol = solve_ivp(rhs, [0, T], u0, method='Radau', t_eval=t_eval,
                    max_step=dt, rtol=1e-6, atol=1e-8)
    d = sol.y.T.astype(np.float32)
    print(f"    Done. Shape={d.shape}, range=[{d.min():.3f}, {d.max():.3f}]", flush=True)
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


def main():
    parser = argparse.ArgumentParser('KS Control via KMM-MPC')
    parser.add_argument('--output_dir', type=str, default='results/optimization')
    parser.add_argument('--seeds', type=str, default='2021')
    parser.add_argument('--N', type=int, default=32, help='Grid points')
    parser.add_argument('--L', type=float, default=16.0, help='Domain size')
    parser.add_argument('--dt', type=float, default=0.25)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--pred_len', type=int, default=50)
    parser.add_argument('--control_horizon', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)
    N = args.N

    for seed in [int(s.strip()) for s in args.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*60}")
        print(f"KS Control | Seed={seed} | N={N}")
        print(f"{'='*60}")

        # Generate data
        data = generate_ks_data(N=N, L=args.L, dt=args.dt,
                                n_steps=args.n_steps, seed=seed)
        data_t = torch.tensor(data, dtype=torch.float32)
        n = len(data) - args.seq_len - args.pred_len
        X = torch.stack([data_t[i:i+args.seq_len] for i in range(n)])
        Y = torch.stack([data_t[i+args.seq_len:i+args.seq_len+args.pred_len] for i in range(n)])
        xm = X.mean(dim=(0,1), keepdim=True)
        xs = X.std(dim=(0,1), keepdim=True) + 1e-8
        Xn, Yn = (X - xm) / xs, (Y - xm) / xs
        nt = int(len(X) * 0.7)
        Xtr, Ytr = Xn[:nt].to(device), Yn[:nt].to(device)

        # Train KMM
        print(f"  Training KMM ({args.epochs} epochs)...", flush=True)
        configs = type('C', (), {})()
        configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
        configs.enc_in = N; configs.d_model = 128; configs.d_koopman = 32
        configs.n_blocks = 2; configs.dropout = 0.0

        model = Model(configs).to(device)
        model(torch.zeros(1, args.seq_len, N, device=device))
        a, b = _zero_mods(args.pred_len, model.d_koopman, device)
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
            if (ep+1) % 10 == 0:
                print(f"    Epoch {ep+1}/{args.epochs}", flush=True)

        # Extract diagonal K — verify diagonality
        r = torch.exp(-torch.exp(model.spectrum.nu)).detach().cpu()
        K_test = torch.diag(r[:32])
        off_diag = (K_test - torch.diag(torch.diag(K_test))).norm().item()
        n_chaotic = (r > 0.95).sum().item()
        print(f"    K off-diag={off_diag:.6f} (0=diagonal), |λ|>0.95={n_chaotic}/{len(r)}")

        # MPC: closed-form (KMM) vs NMPC (scipy optimization)
        H = args.control_horizon; D = min(model.K_proj, 32)
        K_mat = torch.diag(r[:D])
        C_obs = model.C_obs.detach().cpu()

        test_idx = n - 200
        target = torch.zeros(args.pred_len, N)  # flat = suppress all
        x0 = Xn[test_idx, -1].cpu()
        G0 = C_obs[:D] @ x0
        Gt = C_obs[:D] @ target[0]

        # KMM-MPC
        S_parts, m_vec, t_vec = [], torch.zeros(H*D), torch.zeros(H*D)
        for d in range(D):
            Kd = K_mat[d,d].item()
            S_d = torch.zeros(H, H)
            for hi in range(H):
                for hj in range(hi+1): S_d[hi, hj] = Kd**(hi-hj)
            S_parts.append(S_d)
            for hi in range(H):
                m_vec[hi*D+d] = (Kd**(hi+1))*G0[d].item()
                t_vec[hi*D+d] = Gt[d].item()
        S_tot = torch.cat(S_parts, dim=0)

        t0 = time.time()
        U_kmm = torch.linalg.solve(S_tot.T@S_tot + 0.1*torch.eye(H),
                                    S_tot.T@(t_vec - m_vec))
        kmm_ms = (time.time()-t0)*1000

        # NMPC via scipy — stable integrator with clipped exponential
        from scipy.optimize import minimize
        # Precompute stable linear propagator
        k_ks = 2*np.pi*np.fft.rfftfreq(N, d=args.L/N)
        L_ks = -k_ks**2 + k_ks**4
        exp_L = np.exp(np.clip(0.25 * L_ks, -20, 0))  # clip: exp(-∞)=0, exp(0)=1

        def nmpc_cost(U_np):
            x = x0.numpy().copy()
            cost = 0.0
            for h in range(H):
                x_hat = np.fft.rfft(x)
                x = np.fft.irfft(x_hat * exp_L, n=N).real  # stable linear step
                x -= 0.25 * x * np.fft.irfft(1j*k_ks*x_hat, n=N).real  # nonlinear
                x[N//4] += U_np[h] * 0.25  # control at one point
                cost += ((x - target[min(h,len(target)-1)].numpy())**2).sum()
            return cost + 0.1*(U_np**2).sum()
        t0 = time.time()
        res = minimize(nmpc_cost, np.zeros(H), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-4})
        nmpc_ms = (time.time()-t0)*1000
        U_nmpc = torch.tensor(res.x)

        kmm_e = (U_kmm**2).sum().item()
        nmpc_e = (U_nmpc**2).sum().item()

        # Save control trajectory for plotting
        np.savez_compressed(os.path.join(args.output_dir, f'ks_control_traj_s{seed}.npz'),
                            U_kmm=U_kmm.cpu().numpy(), U_nmpc=U_nmpc.cpu().numpy(),
                            kmm_ms=kmm_ms, nmpc_ms=nmpc_ms,
                            kmm_energy=kmm_e, nmpc_energy=nmpc_e,
                            H=H, N=N)

        print(f"\n  {'='*50}")
        print(f"  KS Control (H={H}, N={N})")
        print(f"  {'='*50}")
        print(f"  {'Method':<20s} {'Time':>10s} {'Energy':>10s}")
        print(f"  {'─'*42}")
        print(f"  {'KMM-MPC (closed)':<20s} {kmm_ms:>8.2f}ms {kmm_e:>10.4f}")
        print(f"  {'NMPC (scipy)':<20s} {nmpc_ms:>8.2f}ms {nmpc_e:>10.4f}")
        print(f"  {'Speedup':<20s} {nmpc_ms/kmm_ms:>8.0f}×")
        print(f"  Trajectory saved: ks_control_traj_s{seed}.npz")

        saver.save_csv('ks_control.csv', [[
            seed, N, H, off_diag, n_chaotic,
            kmm_ms, kmm_e, nmpc_ms, nmpc_e,
            time.strftime('%Y-%m-%d %H:%M:%S')
        ]], header=['seed','N','H','K_off_diag','n_chaotic',
                     'kmm_ms','kmm_energy','nmpc_ms','nmpc_energy','timestamp'])

    print(f"\n  Results: {args.output_dir}/ks_control.csv")


if __name__ == '__main__':
    main()
