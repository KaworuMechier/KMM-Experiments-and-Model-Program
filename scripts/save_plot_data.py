#!/usr/bin/env python
"""
Check and fix data saving across all non-forecasting experiments.
Generates per-experiment numpy files for paper plotting.
"""
import os, sys, argparse, torch, numpy as np, time, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
OUT = 'results/plot_data'
os.makedirs(OUT, exist_ok=True)


def check_lorenz():
    """Lorenz: save per-epoch edge_gap for convergence plot."""
    print("=== Lorenz ===")
    # Check existing final results
    csv_path = 'results/chaos_robustness/lorenz/lorenz_results.csv'
    if os.path.exists(csv_path):
        print(f"  Final results: {csv_path} ✓")
    # Check mode files
    for seed in [2021, 2022, 2023]:
        npz = f'results/chaos_robustness/lorenz/lorenz_modes_s{seed}.npz'
        if os.path.exists(npz):
            print(f"  Modes s{seed}: {npz} ✓ (can plot final |λ| distribution)")
        else:
            print(f"  Modes s{seed}: MISSING ✗ — need to re-run with --mode kmm")
    # Per-epoch data
    print(f"  Per-epoch edge_gap: MISSING ✗ — run_lorenz.py needs epoch-by-epoch saving")
    print(f"  Fix: add CSV logging inside training loop in run_lorenz.py")
    print()


def check_ks_control():
    """KS Control: save per-method control trajectory for comparison plot."""
    print("=== KS Control ===")
    csv_path = 'results/optimization/ks_control.csv'
    if os.path.exists(csv_path):
        print(f"  Summary: {csv_path} ✓")
    else:
        print(f"  Summary: MISSING ✗")
    # Need: KMM-MPC control sequence, NMPC control sequence, cost per step
    print(f"  Control trajectory: MISSING ✗ — ks_control.py needs to save U_kmm, U_nmpc, rollout costs")
    print()


def check_mpc():
    """MPC comparison: save per-horizon detailed results."""
    print("=== MPC Comparison ===")
    csv_path = 'results/optimization/mpc_comparison.csv'
    if os.path.exists(csv_path):
        print(f"  Summary: {csv_path} ✓")
    else:
        print(f"  Summary: MISSING ✗")
    print(f"  Per-horizon NMPC trajectory: MISSING ✗ — compare_mpc.py needs per-step rollout data")
    print()


def check_ks_sensor():
    """KS Sensor Selection: check saved data."""
    print("=== KS Sensor Selection ===")
    csv_path = 'results/optimization/ks_sensor_selection.csv'
    if os.path.exists(csv_path):
        print(f"  Results: {csv_path} ✓ (per-seed, per-K, per-H)")
    else:
        print(f"  Results: MISSING ✗")
    print()


def check_latent_mode():
    """Latent Mode Recovery: check saved data."""
    print("=== Latent Mode Recovery ===")
    csv_path = 'results/chaos_robustness/latent_mode/latent_mode_results.csv'
    if os.path.exists(csv_path):
        print(f"  Results: {csv_path} ✓")
    else:
        print(f"  Results: MISSING ✗ — check run_latent_mode.py output")
    print()


def check_channel_capacity():
    """Channel Capacity: check CSV."""
    print("=== Channel Capacity ===")
    csv_path = 'results/analysis/channel_capacity.csv'
    if os.path.exists(csv_path):
        print(f"  Results: {csv_path} ✓ (C, VRAM, params per step)")
    else:
        print(f"  Results: MISSING ✗")
    print()


def main():
    print("KMM-v3 Plot Data Availability Check\n")
    check_lorenz()
    check_ks_control()
    check_mpc()
    check_ks_sensor()
    check_latent_mode()
    check_channel_capacity()


if __name__ == '__main__':
    main()
