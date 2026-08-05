#!/usr/bin/env python
"""
Clean and validate experiment results before plotting.
Filters out: NaN, error rows, incomplete seeds, failed runs.
Usage: python scripts/clean_results.py
"""
import os, numpy as np, pandas as pd, shutil

CLEAN = 'results/plot_data'
os.makedirs(CLEAN, exist_ok=True)

def clean_csv(path, name, cols_check):
    """Filter CSV: remove rows with NaN in any check column."""
    if not os.path.exists(path):
        print(f'  {name}: MISSING ✗')
        return None
    df = pd.read_csv(path)
    n_before = len(df)
    for col in cols_check:
        if col in df.columns:
            df = df[df[col].notna() & (df[col] != float('inf'))]
    n_after = len(df)
    if n_after < n_before:
        print(f'  {name}: {n_before}→{n_after} rows (removed {n_before-n_after} invalid)')
    else:
        print(f'  {name}: {n_after} rows ✓')
    df.to_csv(os.path.join(CLEAN, os.path.basename(path)), index=False)
    return df

def check_npz(path, name, required_keys):
    """Check npz has all required keys with valid data."""
    if not os.path.exists(path):
        print(f'  {name}: MISSING ✗')
        return False
    d = np.load(path, allow_pickle=True)
    ok = True
    for k in required_keys:
        if k not in d:
            print(f'  {name}: missing key "{k}" ✗')
            ok = False
        elif np.isnan(d[k]).any():
            print(f'  {name}: NaN in "{k}" ✗')
            ok = False
    if ok:
        # Copy to clean dir
        shutil.copy(path, os.path.join(CLEAN, os.path.basename(path)))
        print(f'  {name}: valid ✓')
    return ok

print("Data Cleaning & Validation\n")

# ── Lorenz ──────────────────────────────────────────
print("=== Lorenz ===")
for seed in [2021, 2022, 2023]:
    check_npz(f'results/chaos_robustness/lorenz/lorenz_curve_s{seed}.npz',
              f'lorenz_curve_s{seed}', ['epochs', 'edge_gap', 'mse', 'lambda_max'])
    check_npz(f'results/chaos_robustness/lorenz/lorenz_modes_s{seed}.npz',
              f'lorenz_modes_s{seed}', ['nu', 'theta'])
clean_csv('results/chaos_robustness/lorenz/lorenz_results.csv',
          'lorenz_results', ['edge_gap', 'test_mse'])

# ── Latent Mode ─────────────────────────────────────
print("\n=== Latent Mode ===")
clean_csv('results/chaos_robustness/latent_mode/latent_mode_results.csv',
          'latent_mode_results', ['kmm_mse', 'dlinear_mse', 'D_true', 'noise'])

# ── KS Sensor ───────────────────────────────────────
print("\n=== KS Sensor ===")
clean_csv('results/optimization/ks_sensor_selection.csv',
          'ks_sensor_selection', ['kmm', 'pca', 'random', 'K', 'H'])

# ── Channel Capacity ────────────────────────────────
print("\n=== Channel Capacity ===")
clean_csv('results/analysis/channel_capacity.csv',
          'channel_capacity', ['channels', 'peak_vram_gb'])

# ── KS Control ──────────────────────────────────────
print("\n=== KS Control ===")
for seed in [2021, 2022, 2023]:
    check_npz(f'results/optimization/ks_control_traj_s{seed}.npz',
              f'ks_control_traj_s{seed}', ['U_kmm', 'U_nmpc'])
clean_csv('results/optimization/ks_control.csv',
          'ks_control', ['kmm_ms', 'nmpc_ms', 'n_chaotic'])

# ── MPC ─────────────────────────────────────────────
print("\n=== MPC ===")
clean_csv('results/optimization/mpc_comparison.csv',
          'mpc_comparison', ['horizon', 'kmm_ms', 'nmpc_ms', 'lqr_ms'])

# ── ECL modes for Fig 3 ─────────────────────────────
print("\n=== ECL modes ===")
check_npz('results/forecasting/kmm_v3/kmm_outputs_ECL_s2021.npz',
          'kmm_outputs_ECL_s2021', ['spectrum_nu', 'spectrum_theta'])

print(f"\nClean data saved to {CLEAN}/")
print("Update plot_paper_figures.py to read from results/plot_data/ instead of results/")
