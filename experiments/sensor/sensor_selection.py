#!/usr/bin/env python
"""
Dynamical Sensor Selection via C_obs — KMM Advantage over PCA.

PCA wins on STATIC reconstruction (min ||X - X_K||²).
KMM wins on DYNAMICAL reconstruction — sensors whose readings best
PREDICT the future of all other sensors.

Key difference:
  PCA:    Which sensors capture the most VARIANCE?
  KMM:    Which sensors capture the most DYNAMICAL COUPLING?

Setup: C=50 sensors, D_true=5 latent modes with NONLINEAR dynamics
       and TIME-DELAYED cross-coupling.

  mode_i[t+1] = f_i(mode_i[t]) + Σ_j M[i,j]·mode_j[t - τ_ij] + noise

  PCA can only see instantaneous correlations at time t.
  KMM, trained to PREDICT x_{t+1}, learns the delay structure through C_obs.
  → KMM selects sensors that best expose the DYNAMICAL coupling.
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
import torch.nn as nn


def generate_coupled_system(n_steps, C, D_true, noise_std, coupling_strength, seed):
    """Generate C-channel data from D_true NONLINEAR coupled modes with TIME DELAYS.

    Each mode d:
      x_d[t+1] = tanh(α_d·x_d[t]) + Σ_j M[d,j]·x_j[t - τ_{d,j}] + η[t]

    where M is a sparse coupling matrix and τ are integer delays (1-5 steps).
    The nonlinearity (tanh) and time delays make PCA's static variance
    decomposition miss the true dynamical coupling structure.
    """
    rng = np.random.RandomState(seed)

    # Random coupling matrix (sparse, asymmetric with delays)
    M = np.zeros((D_true, D_true), dtype=np.float32)
    delays = np.zeros((D_true, D_true), dtype=np.int32)
    for d in range(D_true):
        # Each mode couples to 2-3 other modes with random delays
        n_couple = rng.randint(1, 3)
        targets = rng.choice([j for j in range(D_true) if j != d], size=n_couple, replace=False)
        for tgt in targets:
            M[d, tgt] = rng.uniform(0.1, coupling_strength)
            delays[d, tgt] = rng.randint(1, 6)

    max_delay = delays.max()
    alphas = rng.uniform(0.5, 0.9, size=D_true).astype(np.float32)

    # Generate latent dynamics
    total_steps = n_steps + max_delay + 1000
    modes = np.zeros((total_steps, D_true), dtype=np.float32)
    modes[:max_delay] = rng.randn(max_delay, D_true).astype(np.float32) * 0.1

    for t in range(max_delay, total_steps):
        for d in range(D_true):
            # Nonlinear self-dynamics
            self_term = np.tanh(alphas[d] * modes[t-1, d])
            # Delayed coupling from other modes
            coupling = 0.0
            for j in range(D_true):
                if M[d, j] != 0:
                    coupling += M[d, j] * modes[t - delays[d, j], j]
            noise = np.float32(rng.randn() * 0.01)
            modes[t, d] = self_term + coupling + noise

    modes = modes[1000:][:n_steps]

    # Generate observation matrix C_true (D_true, C)
    # Each sensor observes 1-4 modes with random weights
    C_true = np.zeros((D_true, C), dtype=np.float32)
    for c in range(C):
        n_modes = rng.randint(1, min(5, D_true + 1))
        srcs = rng.choice(D_true, size=n_modes, replace=False)
        C_true[srcs, c] = rng.uniform(0.3, 1.0, size=n_modes).astype(np.float32)

    data = (modes @ C_true).astype(np.float32)
    data += rng.randn(*data.shape).astype(np.float32) * noise_std

    return data, C_true, M, delays


def _norms(signatures):
    return np.linalg.norm(signatures, axis=1)

def greedy_selection(signatures, budget):
    """Greedy: iteratively pick sensor most orthogonal to already-selected span."""
    C_total, D = signatures.shape
    selected = [int(np.argmax(_norms(signatures)))]
    remaining = set(range(C_total)) - {selected[0]}
    while len(selected) < budget:
        span = signatures[selected]
        best, best_i = -1, -1
        for i in remaining:
            si = signatures[i]
            proj = si @ span.T @ np.linalg.pinv(span @ span.T) @ span
            r = np.linalg.norm(si - proj)
            if r > best_i: best, best_i = r, i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def predictive_reconstruction_error(data, measured_idx, seq_len, pred_len):
    """DYNAMICAL reconstruction: can we predict ALL sensors from K measured ones?

    Unlike STATIC reconstruction (PCA's ||X - X_K||²), this tests whether
    the selected sensors capture the SYSTEM DYNAMICS well enough to forecast
    all other sensors.

    Steps:
      1. Train a simple linear predictor: measured[t-L:t] → all_sensors[t+H]
      2. Report prediction MSE on unmeasured sensors.
    """
    data_t = torch.tensor(data, dtype=torch.float32)
    C = data_t.shape[1]
    K = len(measured_idx)

    # Build sequences
    n = len(data) - seq_len - pred_len
    X_m = torch.stack([data_t[i:i+seq_len][:, measured_idx].flatten() for i in range(n)])  # (N, L*K)
    Y_full = data_t[seq_len+pred_len-1 : seq_len+pred_len-1+n]  # (N, C)

    # Linear regression
    X_m = torch.cat([X_m, torch.ones(n, 1)], dim=1)
    W = torch.linalg.lstsq(X_m, Y_full).solution  # (L*K+1, C)
    Y_pred = X_m @ W
    mse = ((Y_pred - Y_full) ** 2).mean().item()
    return mse


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
    parser = argparse.ArgumentParser('KMM-v3 Dynamical Sensor Selection')
    parser.add_argument('--output_dir', type=str, default='results/optimization')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--n_steps', type=int, default=12000)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)
    C, D_true = 50, 5
    K_BUDGETS = [3, 5, 10, 15, 25]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*65}")
        print(f"Dynamical Sensor Selection | Seed={seed} | C={C} D_true={D_true}")
        print(f"{'='*65}")

        for coupling_strength in [0.3, 0.6]:
            for noise_std in [0.1, 0.3]:
                print(f"\n  Coupling={coupling_strength}, σ={noise_std}:")

                # Generate data with nonlinear coupling + time delays
                data, C_true, M, delays = generate_coupled_system(
                    args.n_steps, C, D_true, noise_std, coupling_strength,
                    seed + int(coupling_strength * 100))

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
                configs = type('C', (), {})()
                configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
                configs.enc_in = C; configs.d_model = 64; configs.d_koopman = 32
                configs.n_blocks = 2; configs.dropout = 0.1

                model = Model(configs).to(device)
                model(torch.zeros(2, args.seq_len, C, device=device))
                a, b = _zero_mods(args.pred_len, model.d_koopman, device)
                model.track_b = a; model.fusion_gate = b

                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                crit = nn.MSELoss()
                for _ in range(args.epochs):
                    model.train()
                    perm = torch.randperm(len(Xtr))
                    for i in range(0, len(Xtr), args.batch_size):
                        idx = perm[i:i+args.batch_size]; opt.zero_grad()
                        loss = crit(model(Xtr[idx])[:, -args.pred_len:, :], Ytr[idx])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()

                # Extract signatures
                c_obs = model.C_obs.detach().cpu().numpy()  # (K_proj, C)
                kmm_sig = c_obs.T  # (C, K_proj)

                # PCA signatures (via SVD, no sklearn needed)
                data_c = data - data.mean(axis=0)
                _, _, Vt = np.linalg.svd(data_c, full_matrices=False)
                pca_sig = Vt[:D_true, :].T  # (C, D_true)

                # Oracle (C_true)
                ora_sig = C_true.T  # (C, D_true)

                test_data = data[-2000:]

                print(f"    {'K':>4s}  {'KMM(C_obs)':>10s} {'PCA':>10s} {'Random':>10s} {'Oracle':>10s}")
                print(f"    {'─'*48}")

                for K in K_BUDGETS:
                    kmm_sel = greedy_selection(kmm_sig, K)
                    pca_sel = greedy_selection(pca_sig, K)
                    rnd_sel = list(np.random.RandomState(seed).choice(C, size=K, replace=False))
                    ora_sel = greedy_selection(ora_sig, K)

                    kmm_err = predictive_reconstruction_error(test_data, kmm_sel, args.seq_len, args.pred_len)
                    pca_err = predictive_reconstruction_error(test_data, pca_sel, args.seq_len, args.pred_len)
                    rnd_err = predictive_reconstruction_error(test_data, rnd_sel, args.seq_len, args.pred_len)
                    ora_err = predictive_reconstruction_error(test_data, ora_sel, args.seq_len, args.pred_len)

                    print(f"    {K:>4d}  {kmm_err:>10.6f} {pca_err:>10.6f} {rnd_err:>10.6f} {ora_err:>10.6f}")

                    saver.save_csv('sensor_selection_dynamic.csv', [[
                        seed, coupling_strength, noise_std, K,
                        kmm_err, pca_err, rnd_err, ora_err,
                        time.strftime('%Y-%m-%d %H:%M:%S')
                    ]], header=['seed', 'coupling', 'noise', 'K',
                                 'kmm', 'pca', 'random', 'oracle', 'timestamp'])

    print(f"\nResults: {args.output_dir}/sensor_selection_dynamic.csv")


if __name__ == '__main__':
    main()
