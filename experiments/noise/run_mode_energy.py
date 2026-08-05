#!/usr/bin/env python
"""
Mode Energy: Dominant vs Secondary Mode Separation.

Fix: Generate C=30 GENUINELY different channels from the same source.
Each channel = true_signal × random_gain + independent_noise.
C_obs must find the shared structure across diverse observations.

Key metric: Dominant Ratio = E_top3 / E_median
  Structured: top 3 modes carry >> median energy (ratio >> 10)
  Noise:      all modes carry similar energy (ratio ≈ 1)
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
import torch.nn as nn

C_TARGET = 30  # Genuinely different channels
D_K = 32


def make_multichannel(source_3d, C, noise_std, seed):
    """Create C diverse channels from 3D source with different gains and noise seeds."""
    rng = np.random.RandomState(seed)
    T = len(source_3d)
    data = np.zeros((T, C), dtype=np.float32)
    for c in range(C):
        # Each channel mixes 3 source dims with random weights + independent noise
        w = rng.uniform(0.2, 1.5, size=3).astype(np.float32)
        data[:, c] = (source_3d * w).sum(axis=1)
        data[:, c] += rng.randn(T).astype(np.float32) * noise_std
    return data


def generate_lorenz(n_steps, noise_std, dt=0.01):
    x = np.zeros((n_steps + 2000, 3), dtype=np.float32); x[0] = [1, 1, 1]
    for i in range(1, len(x)):
        xc, yc, zc = x[i-1]
        x[i,0] = xc + dt * 10 * (yc - xc)
        x[i,1] = yc + dt * (xc * (28 - zc) - yc)
        x[i,2] = zc + dt * (xc * yc - 8/3 * zc)
    d = x[2000:]
    if noise_std > 0: d += np.random.randn(*d.shape).astype(np.float32) * noise_std
    return d


def generate_periodic(n_steps, noise_std):
    t = np.arange(n_steps, dtype=np.float32)
    d = np.zeros((n_steps, 3), dtype=np.float32)
    for c in range(3):
        d[:, c] = np.sin(2*np.pi*t/24 + c) + 0.5*np.sin(2*np.pi*t/168 + c*3)
    if noise_std > 0: d += np.random.randn(*d.shape).astype(np.float32) * noise_std
    return d


def generate_noise(n_steps, std):
    return np.random.randn(n_steps, 3).astype(np.float32) * std


def _zero_mods(pred_len, dk, dev):
    class A(torch.nn.Module):
        def forward(self, x):
            return (torch.zeros(x.shape[0], pred_len, device=x.device),
                    torch.zeros(x.shape[0], dk, device=x.device))
    class B(torch.nn.Module):
        def forward(self, ha, hl):
            return torch.zeros(hl.shape[0], 1, device=ha.device)
    return A().to(dev), B().to(dev)


def train_and_analyze(data, args, device):
    """Train KMM, return: energies_sorted, dominant_ratio, top3_energy, median_energy"""
    data_t = torch.tensor(data, dtype=torch.float32)
    n = len(data) - args.seq_len - args.pred_len
    X = torch.stack([data_t[i:i+args.seq_len] for i in range(n)])
    Y = torch.stack([data_t[i+args.seq_len:i+args.seq_len+args.pred_len] for i in range(n)])
    xm = X.mean(dim=(0,1), keepdim=True)
    xs = X.std(dim=(0,1), keepdim=True) + 1e-8
    Xn, Yn = (X - xm) / xs, (Y - xm) / xs
    nt = int(len(X) * 0.8)
    Xtr, Ytr = Xn[:nt].to(device), Yn[:nt].to(device)

    configs = type('C', (), {})()
    configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
    configs.enc_in = C_TARGET; configs.d_model = 64; configs.d_koopman = D_K
    configs.n_blocks = 2; configs.dropout = 0.1

    model = Model(configs).to(device)
    model(torch.zeros(1, args.seq_len, C_TARGET, device=device))
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    # Energy = |λ_d| × ||C_obs[d,:]||²
    r = torch.exp(-torch.exp(model.spectrum.nu)).detach().cpu().numpy()
    cobs = model.C_obs.detach().cpu().numpy()
    Kp = model.K_proj
    D_eff = min(Kp, len(r))
    energies = r[:D_eff] * (cobs[:D_eff] ** 2).sum(axis=1)
    energies /= (energies.sum() + 1e-8)
    energies_sorted = np.sort(energies)[::-1]

    # Dominant vs secondary
    top3 = energies_sorted[:3].sum()
    median = np.median(energies_sorted)
    dom_ratio = top3 / (median + 1e-8)

    return energies_sorted, dom_ratio, top3, median


def main():
    parser = argparse.ArgumentParser('Mode Energy: Dominant vs Secondary')
    parser.add_argument('--output_dir', type=str, default='results/chaos_robustness/noise')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    saver = ResultSaver(args.output_dir)

    configs = [
        ('Lorenz     ', lambda: generate_lorenz(args.n_steps, 0.0), 0.0),
        ('Lorenz+noise', lambda: generate_lorenz(args.n_steps, 0.5), 0.2),
        ('Lorenz+noise', lambda: generate_lorenz(args.n_steps, 1.0), 0.3),
        ('Periodic   ', lambda: generate_periodic(args.n_steps, 0.0), 0.0),
        ('Periodic+n ', lambda: generate_periodic(args.n_steps, 0.5), 0.2),
        ('Periodic+n ', lambda: generate_periodic(args.n_steps, 1.0), 0.3),
        ('White Noise', lambda: generate_noise(args.n_steps, 0.5), 0.0),
        ('White Noise', lambda: generate_noise(args.n_steps, 1.0), 0.0),
        ('White Noise', lambda: generate_noise(args.n_steps, 2.0), 0.0),
    ]

    for seed in [int(s.strip()) for s in args.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*75}")
        print(f"Mode Energy: Dominant vs Secondary | Seed={seed} | C={C_TARGET} | D_k={D_K}")
        print(f"{'='*75}")
        print(f"  {'System':<15s} {'Top3%':>7s} {'Median%':>8s} {'DomRatio':>9s} {'Diagnosis'}")
        print(f"  {'─'*60}")

        for label, gen_fn, ch_noise in configs:
            src = gen_fn()
            data = make_multichannel(src, C_TARGET, ch_noise, seed + 1000)
            energies, dom_ratio, top3, median = train_and_analyze(data, args, device)

            if dom_ratio > 20:
                diag = 'STRONG STRUCTURE'
            elif dom_ratio > 5:
                diag = 'WEAK STRUCTURE'
            else:
                diag = 'NOISE (no dominant modes)'

            print(f"  {label:<15s} {top3:>6.3f}  {median:>7.4f}  {dom_ratio:>8.1f}  {diag}", flush=True)

            saver.save_csv('mode_energy.csv', [[
                seed, label.strip(), top3, median, dom_ratio,
                time.strftime('%Y-%m-%d %H:%M:%S')
            ]], header=['seed', 'system', 'top3_energy', 'median_energy', 'dominant_ratio', 'timestamp'])

    print(f"\n  KEY: DomRatio>>10 = STRUCTURE (few modes dominate)")
    print(f"       DomRatio≈1  = NOISE (all modes equal)")
    print(f"  Results: {args.output_dir}/mode_energy.csv")


if __name__ == '__main__':
    main()
