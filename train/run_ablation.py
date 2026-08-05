#!/usr/bin/env python
"""
KMM-v3 Ablation Study: incremental component contributions.
Runs KMM v2.1 → +FNO → +LinearRNN → +DynamicCobs → +Wavelet → +FreDF → +Variance
"""
import os, sys, argparse, torch, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from datasets.data_provider import load_csv_dataset, build_loaders, get_normalization_stats, DATASET_REGISTRY
from models.koopman_mixer import Model
from utils.result_saver import ResultSaver
from forecasting.kmm_v3.train_kmm import train_and_evaluate


ABLATION_STEPS = [
    ('v2.1_baseline', {}),
    ('+FNO', {'use_fno': True}),
    ('+LinearRNN', {'use_fno': True, 'use_linear_rnn': True}),
    ('+DynamicCobs', {'use_fno': True, 'use_linear_rnn': True, 'use_dynamic_cobs': True}),
    ('+Wavelet', {'use_fno': True, 'use_linear_rnn': True, 'use_dynamic_cobs': True, 'use_wavelet': True}),
    ('+FreDF', {'use_fno': True, 'use_linear_rnn': True, 'use_dynamic_cobs': True, 'use_wavelet': True, 'use_freq_loss': True}),
    ('v3_full', {'use_fno': True, 'use_linear_rnn': True, 'use_dynamic_cobs': True, 'use_wavelet': True, 'use_freq_loss': True, 'use_variance': True}),
]

# Reverse ablation (remove one at a time)
REVERSE_ABLATION = [
    ('v3_minus_FNO', {'use_fno': False}),
    ('v3_minus_LinearRNN', {'use_linear_rnn': False}),
    ('v3_minus_DynamicCobs', {'use_dynamic_cobs': False}),
    ('v3_minus_Wavelet', {'use_wavelet': False}),
    ('v3_minus_FreDF', {'use_freq_loss': False}),
    ('v3_minus_Variance', {'use_variance': False}),
]


def run_ablation(dataset_name, data_path, seed, args, saver):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed); np.random.seed(seed)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_csv_dataset(
        data_path, args.seq_len, args.pred_len, dataset_name)
    C = X_train.shape[-1]
    train_mean, train_std = get_normalization_stats(X_train)
    train_std_inv = 1.0 / train_std
    train_loader, val_loader, test_loader = build_loaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test, args.batch_size)

    # Determine config based on C
    if C <= 10:
        d_model, d_koopman, nblk, drop = 64, 16, 1, 0.45
    elif C <= 50:
        d_model, d_koopman, nblk, drop = 64, 48, 2, 0.1
    else:
        d_model, d_koopman, nblk, drop = 128, 64, 2, 0.0

    base_config = type('C', (), {})()
    base_config.seq_len = args.seq_len; base_config.pred_len = args.pred_len
    base_config.enc_in = C
    base_config.d_model = d_model; base_config.d_koopman = d_koopman
    base_config.n_blocks = nblk; base_config.dropout = drop

    results = []
    for step_name, ab_flags in ABLATION_STEPS:
        print(f"\n  --- {step_name} ---")
        configs = type('C', (), {})()
        for k, v in base_config.__dict__.items():
            setattr(configs, k, v)
        for k, v in ab_flags.items():
            setattr(configs, k, v)

        model = Model(configs).to(device)
        model(torch.zeros(2, args.seq_len, C, device=device))
        n_params = sum(p.numel() for p in model.parameters())

        try:
            r = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                   train_mean, train_std, train_std_inv, device,
                                   type('A', (), {'epochs': args.epochs, 'lr': args.lr,
                                                  'batch_size': args.batch_size,
                                                  'no_manifold_loss': False})())
            results.append([dataset_name, seed, step_name, n_params,
                            r['best_val_tsl'], r['best_test_tsl'], r['time']])
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append([dataset_name, seed, step_name, 0, float('nan'), float('nan'), 0])

    # Save all ablation rows
    saver.save_csv('ablation_results.csv', results,
                   header=['dataset', 'seed', 'step', 'params', 'val_tsl', 'test_tsl', 'time'])
    return results


def main():
    parser = argparse.ArgumentParser('KMM-v3 Ablation Study')
    parser.add_argument('--dataset', type=str, default='ECL')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='datasets/data')
    parser.add_argument('--output_dir', type=str, default='results/forecasting/ablation')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=str, default='96,192,336,720', help='Comma-separated')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    pred_lens = [int(p.strip()) for p in args.pred_len.split(',')]
    saver = ResultSaver(args.output_dir)

    datasets_to_run = [args.dataset] if args.dataset != 'all' else list(DATASET_REGISTRY.keys())
    for ds_name in datasets_to_run:
        data_path = args.data_path or os.path.join(args.data_dir,
                                                   DATASET_REGISTRY.get(ds_name, f'{ds_name}.csv'))
        if not os.path.exists(data_path):
            print(f"SKIP {ds_name}")
            continue
        for pred_len in pred_lens:
            args.pred_len = pred_len
            for seed in seeds:
                run_ablation(ds_name, data_path, seed, args, saver)


if __name__ == '__main__':
    main()
