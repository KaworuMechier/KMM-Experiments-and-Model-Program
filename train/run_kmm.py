#!/usr/bin/env python
"""
KMM-v3 Forecasting Runner — 3 seeds, modal output saving.
Usage:
  python run.py --dataset ECL --data_path data/ECL.csv --seeds 2021,2022,2023
  python run.py --dataset all --data_dir data/ --seeds 2021,2022,2023
"""
import os, sys, argparse, time, json, torch, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from datasets.data_provider import load_csv_dataset, build_loaders, get_normalization_stats, DATASET_REGISTRY
from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
from utils.memory_guard import preflight_check, smart_batch_size
from train_kmm import train_and_evaluate


def run_single(dataset_name, data_path, seed, args, saver):
    """Run KMM v3 on a single dataset with a single seed."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Seed: {seed} | Device: {device}")
    print(f"{'='*60}")

    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_csv_dataset(
        data_path, args.seq_len, args.pred_len, dataset_name, downsample=args.downsample)
    C = X_train.shape[-1]

    # Pre-flight memory check
    ok, rec_bs, est_gb = preflight_check(C, args.seq_len, args.pred_len, device)
    if not ok:
        print(f"  [SKIP] Dataset C={C} too large: needs {est_gb:.1f} GB at B=1, "
              f"free={torch.cuda.mem_get_info(device.index)[0]/1024**3:.1f} GB")
        return {'best_val_tsl': float('nan'), 'best_test_tsl': float('nan')}

    actual_bs = smart_batch_size(args.batch_size, C, args.seq_len, args.pred_len, device)
    # Long-horizon: decoder reverse projection scales with H, clamp batch
    if args.pred_len >= 336: actual_bs = min(actual_bs, 8)
    if args.pred_len >= 720: actual_bs = min(actual_bs, 4)
    H_val = args.pred_len  # avoid shadowing
    print(f"  Batch size: {actual_bs} (requested={args.batch_size}, est_peak={est_gb:.1f} GB)")

    train_mean, train_std = get_normalization_stats(X_train)
    train_std_inv = 1.0 / train_std
    train_loader, val_loader, test_loader = build_loaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test, actual_bs)

    # Build model
    configs = type('C', (), {})()
    configs.seq_len = args.seq_len
    configs.pred_len = args.pred_len
    configs.enc_in = C
    configs.d_model = args.d_model
    configs.d_koopman = args.d_koopman
    configs.n_blocks = args.n_blocks
    configs.dropout = args.dropout

    model = Model(configs).to(device)
    model(torch.zeros(2, args.seq_len, C, device=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Train
    results = train_and_evaluate(
        model, train_loader, val_loader, test_loader,
        train_mean, train_std, train_std_inv, device, args)

    # Save results
    row = [dataset_name, args.pred_len, seed, n_params,
           results['best_val_tsl'], results['best_test_tsl'],
           results.get('best_val_mse', 0), results.get('best_test_mse', 0),
           time.strftime('%Y-%m-%d %H:%M:%S')]
    saver.save_csv('kmm_v3_results.csv', [row],
                   header=['dataset', 'pred_len', 'seed', 'params',
                           'best_val_tsl', 'best_test_tsl', 'best_val_mse', 'best_test_mse', 'timestamp'])

    # Save Koopman modes
    if results.get('final_model') is not None:
        m = results['final_model']
        nu = m.spectrum.nu.detach()
        theta = m.spectrum.theta.detach()
        c_obs = m.C_obs.detach()

        # Get a sample prediction for mode extraction
        m.eval()
        with torch.no_grad():
            sample_x = X_test[:32].to(device)
            out = m(sample_x)
            pred = out[:, -args.pred_len:, :]
            h_before = m._h_before_latent.detach() if hasattr(m, '_h_before_latent') else torch.zeros(1)
            h_track_a = m._h_track_a.detach() if hasattr(m, '_h_track_a') else torch.zeros(1)

        modes_data = {
            'pred': pred.cpu().numpy(),
            'target': Y_test[:32].numpy(),
            'h_before': h_before.cpu().numpy(),
            'h_track_a': h_track_a.cpu().numpy(),
            'nu': nu.cpu().numpy(),
            'theta': theta.cpu().numpy(),
            'c_obs': c_obs.cpu().numpy(),
        }
        saver.save_json(f'modes_{dataset_name}_s{seed}.json', {
            'lyapunov_estimates': np.log(np.exp(-np.exp(nu.cpu().numpy())).clip(min=1e-8)).tolist(),
            'eigenvalue_magnitudes': np.exp(-np.exp(nu.cpu().numpy())).tolist(),
            'modal_frequencies': theta.cpu().numpy().tolist(),
            'n_params': n_params,
            'best_test_tsl': results['best_test_tsl'],
        })
        saver.save_model_outputs(
            f'kmm_outputs_{dataset_name}_s{seed}.npz',
            pred.cpu(), Y_test[:32],
            h_track_a.cpu(),
            (nu.cpu(), theta.cpu()))

    return results


def main():
    parser = argparse.ArgumentParser(description='KMM-v3 Forecasting')
    parser.add_argument('--dataset', type=str, default='ECL', help='Dataset name or "all"')
    parser.add_argument('--data_path', type=str, default='', help='CSV path (overrides registry)')
    parser.add_argument('--data_dir', type=str, default='datasets/data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results/forecasting/kmm_v3')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=str, default='96,192,336,720', help='Comma-separated, e.g. "96,192,336,720"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_koopman', type=int, default=128)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=-1, help='-1=model auto, >0=override (use 0.15 for long-horizon)')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample factor (3 = 5min->15min)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    pred_lens = [int(p.strip()) for p in args.pred_len.split(',')]
    saver = ResultSaver(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine dataset list
    if args.dataset == 'all':
        datasets_to_run = list(DATASET_REGISTRY.keys())
    else:
        datasets_to_run = [args.dataset]

    for ds_name in datasets_to_run:
        data_path = args.data_path or os.path.join(args.data_dir,
                                                   DATASET_REGISTRY.get(ds_name, f'{ds_name}.csv'))
        if not os.path.exists(data_path):
            print(f"SKIP {ds_name}: data not found at {data_path}")
            continue

        for pred_len in pred_lens:
            args.pred_len = pred_len
            for seed in seeds:
                try:
                    run_single(ds_name, data_path, seed, args, saver)
                except Exception as e:
                    print(f"ERROR {ds_name} p{pred_len} s{seed}: {e}")
                    saver.save_csv('errors.csv', [[ds_name, pred_len, seed, str(e), time.strftime('%Y-%m-%d %H:%M:%S')]],
                                   header=['dataset', 'pred_len', 'seed', 'error', 'timestamp'])


if __name__ == '__main__':
    main()
