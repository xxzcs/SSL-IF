# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import glob
import os
import re

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from semilearn.core.utils import get_dataset, get_net_builder


METRIC_KEYS = ['auc', 'accuracy', 'sen', 'spe', 'f1']
METRIC_LABELS = {
    'auc': 'AUC',
    'accuracy': 'ACC',
    'sen': 'Sen',
    'spe': 'Spe',
    'f1': 'F1',
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate supervised checkpoints')

    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_paths', nargs='*', default=None)
    parser.add_argument('--load_glob', type=str, default='')

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)

    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--model_key', type=str, default='model', choices=['model', 'ema_model'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
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
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=None)

    parser.add_argument('--lpath', type=str, default='')
    parser.add_argument('--ulpath', type=str, default='')
    parser.add_argument('--valdata', type=str, default='')

    parser.add_argument('--eval_dest', type=str, default='auto', choices=['auto', 'eval', 'test'])
    parser.add_argument('--csvname', type=str, default='')
    parser.add_argument('--summary_csv', type=str, default='')
    parser.add_argument('--method_suffix', type=str, default='',
                        help='Optional suffix appended to inferred method name, e.g. best or latest')

    return parser.parse_args()


def resolve_checkpoint_paths(args):
    checkpoint_paths = []

    if args.load_path:
        checkpoint_paths.append(args.load_path)
    if args.load_paths:
        checkpoint_paths.extend(args.load_paths)
    if args.load_glob:
        checkpoint_paths.extend(sorted(glob.glob(args.load_glob)))

    checkpoint_paths = [os.path.normpath(path) for path in checkpoint_paths if path]
    checkpoint_paths = list(dict.fromkeys(checkpoint_paths))

    if not checkpoint_paths:
        raise ValueError('Please provide at least one checkpoint via --load_path, --load_paths, or --load_glob.')

    return checkpoint_paths


def strip_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def resolve_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pick_eval_dataset(dataset_dict, eval_dest):
    if eval_dest == 'eval':
        return 'eval', dataset_dict['eval']
    if eval_dest == 'test':
        if dataset_dict['test'] is None:
            raise ValueError('Requested test split, but dataset does not provide test data.')
        return 'test', dataset_dict['test']

    if dataset_dict['test'] is not None:
        return 'test', dataset_dict['test']
    return 'eval', dataset_dict['eval']


def unpack_batch(batch, device):
    if isinstance(batch, dict):
        x = batch['x_lb']
        y = batch['y_lb']
    else:
        x, y = batch

    if isinstance(x, dict):
        x = {key: value.to(device) for key, value in x.items()}
    else:
        x = x.to(device)
    y = y.to(device)
    return x, y


def get_sample_path(dataset, index):
    if hasattr(dataset, 'imgs') and index < len(dataset.imgs):
        return dataset.imgs[index][0]
    if hasattr(dataset, 'samples') and index < len(dataset.samples):
        sample = dataset.samples[index]
        if isinstance(sample, (tuple, list)) and len(sample) > 0:
            return sample[0]
    if hasattr(dataset, 'data') and index < len(dataset.data):
        sample = dataset.data[index]
        if isinstance(sample, str):
            return sample
    return str(index)


def save_predictions_csv(csv_path, sample_paths, labels, probs):
    if not csv_path:
        return

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    header = ['sample', 'label'] + [f'prob_{class_idx}' for class_idx in range(probs.shape[1])]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for index, sample_path in enumerate(sample_paths):
            writer.writerow([sample_path, int(labels[index]), *[round(prob, 9) for prob in probs[index].tolist()]])


def build_eval_loader(args):
    args.save_dir = os.path.dirname(args.load_path) if args.load_path else ''
    args.save_name = ''
    args.epoch = 1
    args.num_train_iter = 1

    dataset_dict = get_dataset(
        args,
        'supervised',
        args.dataset,
        args.num_labels,
        args.num_classes,
        args.data_dir,
        args.lpath,
        args.ulpath,
        args.include_lb_to_ulb,
    )
    if dataset_dict is None:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    eval_name, eval_dset = pick_eval_dataset(dataset_dict, args.eval_dest)
    eval_loader = DataLoader(
        eval_dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return eval_name, eval_dset, eval_loader


def load_model(args, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if args.model_key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain key '{args.model_key}'. Available keys: {list(checkpoint.keys())}")

    model_state = strip_module_prefix(checkpoint[args.model_key])
    # 兼容被 SimMatch_Net / CoMatch_Net 等包装的权重:
    # 取 backbone.* 子集并去掉前缀, 丢弃投影头 (mlp_proj.* 等), 再加载进普通 backbone
    if any(k.startswith('backbone.') for k in model_state):
        model_state = {k[len('backbone.'):]: v
                       for k, v in model_state.items()
                       if k.startswith('backbone.')}
    net_builder = get_net_builder(args.net, args.net_from_name)
    model = net_builder(num_classes=args.num_classes, pretrained=False, pretrained_path='')
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(model, eval_dset, eval_loader, device, csv_path=''):
    sample_paths = []
    all_labels = []
    all_preds = []
    all_probs = []
    running_index = 0

    with torch.no_grad():
        for batch in eval_loader:
            x, y = unpack_batch(batch, device)
            logits = model(x)['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            batch_size = y.shape[0]
            if csv_path:
                for offset in range(batch_size):
                    sample_paths.append(get_sample_path(eval_dset, running_index + offset))
            running_index += batch_size

            all_labels.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    metrics = {
        'auc': None,
        'accuracy': accuracy_score(labels, preds),
        'sen': None,
        'spe': None,
        'f1': f1_score(labels, preds, average='binary' if probs.shape[1] == 2 else 'macro', zero_division=0),
        'confusion_matrix': confusion_matrix(labels, preds, labels=list(range(probs.shape[1]))),
    }

    if probs.shape[1] == 2:
        metrics['sen'] = recall_score(labels, preds, pos_label=1, zero_division=0)
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['spe'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['sen'] = recall_score(labels, preds, average='macro', zero_division=0)

    try:
        if probs.shape[1] == 2:
            metrics['auc'] = roc_auc_score(labels, probs[:, 1])
        else:
            metrics['auc'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception as error:
        print(f'AUC calculation failed: {error}')

    if csv_path:
        save_predictions_csv(csv_path, sample_paths, labels, probs)

    return metrics


def print_single_result(eval_name, checkpoint_path, model_key, metrics):
    print(f'Eval split: {eval_name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Model key: {model_key}')
    for metric_name in METRIC_KEYS:
        if metrics[metric_name] is not None:
            print(f'{METRIC_LABELS[metric_name]}: {metrics[metric_name]}')
    if metrics.get('confusion_matrix') is not None:
        print('Confusion Matrix:')
        print(metrics['confusion_matrix'])


def print_summary(results):
    print(f'Evaluated {len(results)} checkpoint(s).')
    for metric_name in METRIC_KEYS:
        values = [item[metric_name] for item in results if item[metric_name] is not None]
        if not values:
            continue
        values = np.asarray(values, dtype=float)
        print(f'{METRIC_LABELS[metric_name]} mean: {values.mean():.6f}')
        print(f'{METRIC_LABELS[metric_name]} std: {values.std(ddof=0):.6f}')

    confusion_matrices = [item['confusion_matrix'] for item in results if item.get('confusion_matrix') is not None]
    if confusion_matrices:
        stacked_confusion = np.stack(confusion_matrices, axis=0).astype(float)
        mean_confusion = stacked_confusion.mean(axis=0)
        std_confusion = stacked_confusion.std(axis=0, ddof=0)
        print('Confusion Matrix mean:')
        print(mean_confusion)
        print('Confusion Matrix std:')
        print(std_confusion)


def metric_summary(results):
    summary = {}
    for metric_name in METRIC_KEYS:
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


def infer_method_name(checkpoint_paths, load_glob, method_suffix=''):
    candidate = ''
    if checkpoint_paths:
        candidate = os.path.basename(os.path.dirname(checkpoint_paths[0]))
    if not candidate and load_glob:
        candidate = os.path.basename(os.path.dirname(load_glob))

    candidate = re.sub(r'_[0-9]+$', '', candidate)
    candidate = re.sub(r'[\*\?\[\]]', '', candidate)
    candidate = candidate.rstrip('_- ')

    if not candidate:
        raise ValueError('Unable to infer method name for summary CSV output.')

    suffix = str(method_suffix).strip()
    if suffix:
        candidate = f'{candidate}_{suffix}'

    return candidate


def append_summary_csv(csv_path, method_name, summary):
    if not csv_path:
        return

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    header = ['method', 'AUC', 'ACC', 'Sen', 'Spe', 'F1']
    row = [method_name]
    for metric_name in METRIC_KEYS:
        metric = summary.get(metric_name)
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

    print(f'Saved summary row for {method_name} to {csv_path}')


def main():
    args = parse_args()
    device = resolve_device()

    checkpoint_paths = resolve_checkpoint_paths(args)
    if len(checkpoint_paths) > 1 and args.csvname:
        raise ValueError('Batch summary mode does not support --csvname. Omit it to compute mean/std across multiple checkpoints.')

    args.load_path = checkpoint_paths[0]
    eval_name, eval_dset, eval_loader = build_eval_loader(args)

    results = []
    for checkpoint_path in checkpoint_paths:
        model = load_model(args, checkpoint_path, device)
        metrics = evaluate_checkpoint(
            model,
            eval_dset,
            eval_loader,
            device,
            csv_path=args.csvname if len(checkpoint_paths) == 1 else '',
        )
        results.append(metrics)

        if len(checkpoint_paths) == 1:
            print_single_result(eval_name, checkpoint_path, args.model_key, metrics)
        else:
            summary = ', '.join(
                f'{name}={value:.6f}' for name, value in metrics.items()
                if value is not None and name != 'confusion_matrix'
            )
            print(f'{checkpoint_path}: {summary}')
            if metrics.get('confusion_matrix') is not None:
                print('Confusion Matrix:')
                print(metrics['confusion_matrix'])

    if len(checkpoint_paths) > 1:
        print_summary(results)
        summary = metric_summary(results)
        method_name = infer_method_name(checkpoint_paths, args.load_glob, args.method_suffix)
        append_summary_csv(args.summary_csv, method_name, summary)


if __name__ == '__main__':
    main()
