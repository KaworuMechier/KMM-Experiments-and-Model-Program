#!/usr/bin/env python
"""
Generate all TKDE paper figures from saved experiment data.
Usage: python scripts/plot_paper_figures.py
"""
import numpy as np, matplotlib.pyplot as plt, os, glob
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

OUT = 'results/figures'
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Fig 2: Lorenz eigenvalue convergence
# ═══════════════════════════════════════════════════════════════
def fig_lorenz_convergence():
    lorenz_dir = 'results/chaos_robustness/lorenz'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for seed in [2021, 2022, 2023]:
        f = os.path.join(lorenz_dir, f'lorenz_curve_s{seed}.npz')
        if not os.path.exists(f): continue
        d = np.load(f)
        ax1.plot(d['epochs'], d['edge_gap'], lw=2, label=f'seed={seed}')
        ax2.plot(d['epochs'], d['mse'], lw=2, alpha=0.7)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Edge Gap (1 - max|λ|)')
    ax1.set_title('Koopman Spectrum Convergence')
    ax1.legend(); ax1.set_yscale('log'); ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.0018, color='red', ls='--', alpha=0.5, label='converged')

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE')
    ax2.set_title('Prediction Loss'); ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)

    fig.suptitle('Lorenz-63: Unsupervised Emergence of Chaos Detection', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_lorenz_convergence.pdf', bbox_inches='tight')
    plt.close()
    print('Fig 2: Lorenz convergence ✓')


# ═══════════════════════════════════════════════════════════════
# Fig 3: Lorenz vs ECL eigenvalue distributions
# ═══════════════════════════════════════════════════════════════
def fig_eigenvalue_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Lorenz
    for seed, color in [(2021, 'red'), (2022, 'orange'), (2023, 'darkred')]:
        f = os.path.join('results/chaos_robustness/lorenz', f'lorenz_modes_s{seed}.npz')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True)
        lam = d['lambda_mag'].item() if 'lambda_mag' in dict(d) else np.exp(-np.exp(d['nu']))
        ax1.scatter(range(len(lam)), sorted(lam, reverse=True), c=color, s=20, alpha=0.7)
    ax1.axhline(y=0.95, color='gray', ls='--', alpha=0.5)
    ax1.set_title('Lorenz-63 (Chaotic)'); ax1.set_ylabel('|λ|'); ax1.set_ylim(0.9, 1.01)

    # ECL
    ecl_file = 'results/forecasting/kmm_v3/kmm_outputs_ECL_s2021.npz'
    if os.path.exists(ecl_file):
        d = np.load(ecl_file)
        lam = np.exp(-np.exp(d['spectrum_nu']))
        ax2.scatter(range(len(lam)), sorted(lam, reverse=True), c='steelblue', s=15)
    ax2.axhline(y=0.95, color='gray', ls='--', alpha=0.5)
    ax2.set_title('ECL (Stable Multi-Periodic)'); ax2.set_ylabel('|λ|')
    ax2.set_ylim(0.5, 1.01)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Mode Rank'); ax.grid(True, alpha=0.3)

    fig.suptitle('Koopman Spectrum: System Fingerprint', fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig3_eigenvalue_dist.pdf', bbox_inches='tight')
    plt.close()
    print('Fig 3: Eigenvalue distribution ✓')


# ═══════════════════════════════════════════════════════════════
# Fig 4: Latent Mode Recovery
# ═══════════════════════════════════════════════════════════════
def fig_latent_mode():
    csv_path = 'results/chaos_robustness/latent_mode/latent_mode_results.csv'
    data = {'D_true=3': {}, 'D_true=10': {}, 'D_true=30': {}}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(',')
                d_true = parts[1]; sigma = parts[3]
                kmm = float(parts[4]); dl = float(parts[5])
                key = f'D_true={d_true}'
                if sigma not in data.get(key, {}): data[key][sigma] = {'kmm': [], 'dl': []}
                data[key][sigma]['kmm'].append(kmm)
                data[key][sigma]['dl'].append(dl)

    sigmas = sorted(set(s for v in data.values() for s in v.keys()))
    if not sigmas: return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (label, sigma_data) in zip(axes, data.items()):
        x = np.arange(len(sigmas)); w = 0.35
        kmm_vals = [np.mean(sigma_data.get(s, {}).get('kmm', [0])) for s in sigmas]
        dl_vals = [np.mean(sigma_data.get(s, {}).get('dl', [0])) for s in sigmas]
        ax.bar(x - w/2, kmm_vals, w, label='KMM', color='steelblue')
        ax.bar(x + w/2, dl_vals, w, label='DLinear', color='lightcoral')
        ax.set_xticks(x); ax.set_xticklabels([f'σ={s}' for s in sigmas])
        ax.set_title(label); ax.set_ylabel('MSE'); ax.legend()

    fig.suptitle('Latent Mode Recovery — Information Sharing via C_obs', fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig4_latent_mode.pdf', bbox_inches='tight')
    plt.close()
    print('Fig 4: Latent mode recovery ✓')


# ═══════════════════════════════════════════════════════════════
# Fig 5: KS Sensor Selection
# ═══════════════════════════════════════════════════════════════
def fig_ks_sensor():
    csv_path = 'results/optimization/ks_sensor_selection.csv'
    if not os.path.exists(csv_path): return
    import pandas as pd
    df = pd.read_csv(csv_path)
    budgets = sorted(df['K'].unique())
    horizons = sorted(df['H'].unique())

    fig, axes = plt.subplots(1, len(budgets), figsize=(4*len(budgets), 4))
    for ax, K in zip(axes, budgets):
        sub = df[df['K'] == K]
        kmm_vals = [sub[sub['H']==h]['kmm'].mean() for h in horizons]
        pca_vals = [sub[sub['H']==h]['pca'].mean() for h in horizons]
        ax.plot(horizons, kmm_vals, 'o-', lw=2, label='KMM (C_obs)', color='steelblue')
        ax.plot(horizons, pca_vals, 's--', lw=2, label='PCA', color='lightcoral')
        ax.set_title(f'K={K}'); ax.set_xlabel('Horizon H'); ax.set_ylabel('MSE')
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('KS Sensor Selection: KMM vs PCA', fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig5_ks_sensor.pdf', bbox_inches='tight')
    plt.close()
    print('Fig 5: KS sensor selection ✓')


# ═══════════════════════════════════════════════════════════════
# Fig 6: Channel Capacity
# ═══════════════════════════════════════════════════════════════
def fig_channel_capacity():
    csv_path = 'results/analysis/channel_capacity.csv'
    if not os.path.exists(csv_path): return
    C, vram = [], []
    with open(csv_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if parts[1] == 'OK':
                C.append(int(parts[0])); vram.append(float(parts[2]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(C, vram, 'o-', lw=2, color='steelblue', markersize=8)
    ax.axhline(y=24, color='red', ls='--', alpha=0.7, label='RTX 3090 limit (24GB)')
    ax.axvline(x=325, color='orange', ls=':', alpha=0.7, label='Transformer OOM (C=325)')
    ax.set_xlabel('Channels C'); ax.set_ylabel('Peak VRAM (GB)')
    ax.set_title('KMM Channel Scaling on RTX 3090 24GB'); ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig6_channel_capacity.pdf', bbox_inches='tight')
    plt.close()
    print('Fig 6: Channel capacity ✓')


# ═══════════════════════════════════════════════════════════════
# Fig 7: KS MPC Control Trajectory
# ═══════════════════════════════════════════════════════════════
def fig_ks_mpc():
    traj_file = 'results/optimization/ks_control_traj_s2021.npz'
    if not os.path.exists(traj_file): return
    d = np.load(traj_file)
    U_kmm = d['U_kmm']; U_nmpc = d['U_nmpc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(len(U_kmm)), U_kmm, color='steelblue', alpha=0.8, label='KMM-MPC')
    ax1.set_title('KMM-MPC Control Sequence'); ax1.set_xlabel('Step'); ax1.set_ylabel('Control u')
    ax2.bar(range(len(U_nmpc)), U_nmpc, color='lightcoral', alpha=0.8, label='NMPC')
    ax2.set_title('NMPC Control Sequence'); ax2.set_xlabel('Step')
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f'KS Control (KMM {d[\"kmm_ms\"]:.0f}ms vs NMPC {d[\"nmpc_ms\"]:.0f}ms)', fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig7_ks_mpc.pdf', bbox_inches='tight')
    plt.close()
    print('Fig 7: KS MPC trajectory ✓')


if __name__ == '__main__':
    fig_lorenz_convergence()
    fig_eigenvalue_distribution()
    fig_latent_mode()
    fig_ks_sensor()
    fig_channel_capacity()
    fig_ks_mpc()
    print(f'\nAll figures saved to {OUT}/')
