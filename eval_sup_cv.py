# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import glob
import os
import re

import numpy as np

from eval_sup import build_eval_loader, evaluate_checkpoint, load_model, resolve_device, str2bool


METRIC_NAMES = ['auc', 'accuracy', 'sen', 'spe', 'f1']
METRIC_LABELS = {
    'auc': 'AUC',
    'accuracy': 'ACC',
    'sen': 'Sen',
    'spe': 'Spe',
    'f1': 'F1',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-validation summary for supervised checkpoints')

    parser.add_argument('--load_glob_template', type=str, required=True,
                        help='Glob template for checkpoints, use {fold} placeholder, e.g. saved_models/.../fold{fold}_*/model_best.pth')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)

    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--model_key', type=str, default='model', choices=['model', 'ema_model'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='../uda_data')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--crop_ratio', type=float, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--label_ratio', type=float, default=None)
    parser.add_argument('--ulb_num_labels', type=int, default=None)
    parser.add_argument('--lb_imb_ratio', type=int, default=1)
    parser.add_argument('--ulb_imb_ratio', type=int, default=1)
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=None)

    parser.add_argument('--lpath', type=str, default='')
    parser.add_argument('--ulpath', type=str, default='')
    parser.add_argument('--valdata', type=str, default='')

    parser.add_argument('--eval_dest', type=str, default='auto', choices=['auto', 'eval', 'test'])
    parser.add_argument('--summary_csv', type=str, default='')
    parser.add_argument('--method_suffix', type=str, default='',
                        help='Optional suffix appended to inferred method name, e.g. best or latest')

    return parser.parse_args()


def metric_summary(results):
    summary = {}
    for metric_name in METRIC_NAMES:
        values = [item[metric_name] for item in results if item[metric_name] is not None]
        if not values:
            continue
        values = np.asarray(values, dtype=float)
        summary[metric_name] = {
            'mean': values.mean(),
            'std': values.std(ddof=0),
            'count': len(values),
        }
    return summary


def fold_mean_results(fold_results_map):
    results = []
    for fold in sorted(fold_results_map):
        summary = metric_summary(fold_results_map[fold])
        fold_result = {'fold': fold}
        for metric_name in METRIC_NAMES:
            fold_result[metric_name] = summary[metric_name]['mean'] if metric_name in summary else None
        results.append(fold_result)
    return results


def print_summary(title, results):
    print(title)
    summary = metric_summary(results)
    for metric_name in METRIC_NAMES:
        if metric_name not in summary:
            continue
        metric = summary[metric_name]
        print(f'{METRIC_LABELS[metric_name]} mean: {metric["mean"]:.6f}')
        print(f'{METRIC_LABELS[metric_name]} std: {metric["std"]:.6f}')
    print()


def infer_method_name(load_glob_template, checkpoint_paths, method_suffix=''):
    candidate = ''

    if checkpoint_paths:
        candidate = os.path.basename(os.path.dirname(checkpoint_paths[0]))
    if not candidate:
        candidate = os.path.basename(os.path.dirname(load_glob_template))

    candidate = re.sub(r'_fold\{fold\}.*$', '', candidate)
    candidate = re.sub(r'_fold\d+.*$', '', candidate)
    candidate = re.sub(r'[\*\?\[\]]', '', candidate)
    candidate = candidate.rstrip('_- ')

    if not candidate:
        raise ValueError(f'Unable to infer method name from template: {load_glob_template}')

    suffix = str(method_suffix).strip()
    if suffix:
        candidate = f'{candidate}_{suffix}'

    return candidate


def append_summary_csv(csv_path, method_name, pooled_summary, num_checkpoints, num_folds):
    if not csv_path:
        return

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    header = ['method', 'AUC', 'ACC', 'Sen', 'Spe', 'F1']

    row = [method_name]
    for metric_name in METRIC_NAMES:
        metric = pooled_summary.get(metric_name)
        if metric is None:
            row.append('')
        else:
            row.append(f'{metric["mean"]:.6f}±{metric["std"]:.6f}')

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if not write_header:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            existing_header = next(reader, [])
        if existing_header != header:
            raise ValueError(
                f'Existing CSV header does not match the current format in {csv_path}. '
                f'Please remove the old file or choose a new --summary_csv path.'
            )

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f'Saved pooled summary row for {method_name} to {csv_path}')


def main():
    args = parse_args()
    device = resolve_device()

    all_results = []
    per_fold_results = {}
    method_name = ''

    for fold in args.folds:
        checkpoint_paths = sorted(glob.glob(args.load_glob_template.format(fold=fold)))
        if not checkpoint_paths:
            print(f'Fold {fold}: no checkpoints matched {args.load_glob_template.format(fold=fold)}')
            print()
            continue

        args.fold = fold
        args.load_path = checkpoint_paths[0]
        if not method_name:
            method_name = infer_method_name(args.load_glob_template, checkpoint_paths, args.method_suffix)
        eval_name, eval_dset, eval_loader = build_eval_loader(args)

        fold_results = []
        print(f'========== Fold {fold} ({eval_name}) ==========')
        for checkpoint_path in checkpoint_paths:
            model = load_model(args, checkpoint_path, device)
            metrics = evaluate_checkpoint(model, eval_dset, eval_loader, device, csv_path='')
            fold_results.append(metrics)
            all_results.append(metrics)

            summary = ', '.join(
                f'{name}={value:.6f}'
                for name, value in metrics.items()
                if value is not None and isinstance(value, (int, float, np.floating))
            )
            print(f'{checkpoint_path}: {summary}')
        print()

        per_fold_results[fold] = fold_results
        print_summary(f'Fold {fold} summary ({len(fold_results)} checkpoint(s))', fold_results)

    if all_results:
        pooled_summary = metric_summary(all_results)
        print_summary(
            f'Overall pooled summary ({len(all_results)} checkpoint(s), direct aggregation, recommended for paper main result)',
            all_results,
        )
        append_summary_csv(args.summary_csv, method_name, pooled_summary, len(all_results), len(per_fold_results))

        fold_mean_stats = fold_mean_results(per_fold_results)
        if fold_mean_stats:
            print_summary(
                f'Overall fold-mean summary ({len(fold_mean_stats)} fold mean(s), recommended for fold-level analysis)',
                fold_mean_stats,
            )
    else:
        print('No checkpoints were evaluated.')


if __name__ == '__main__':
    main()
