#!/usr/bin/env python3
"""Paired BUS evaluation with a validation-selected shared decision threshold.

The threshold is selected independently for each method by maximizing mean
validation F1 across seeds. Ties are resolved by choosing the value closest to
0.5. The selected threshold is then applied unchanged to every test seed.
"""

import argparse
import csv
import glob
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval_sup import build_eval_loader, load_model, unpack_batch


METRICS = ("auc", "acc", "sen", "spe", "f1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_glob", required=True)
    parser.add_argument("--if_glob", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--threshold_step", type=float, default=0.001)
    return parser.parse_args()


def seed_from_path(path):
    match = re.search(r"_s(\d+)(?:/|$)", path)
    if not match:
        raise ValueError(f"Cannot infer seed from {path}")
    return int(match.group(1))


def resolve_pairs(base_glob, if_glob):
    base = {seed_from_path(path): path for path in sorted(glob.glob(base_glob))}
    with_if = {seed_from_path(path): path for path in sorted(glob.glob(if_glob))}
    if not base or base.keys() != with_if.keys():
        raise ValueError(f"Seed mismatch: base={sorted(base)}, if={sorted(with_if)}")
    return [(seed, base[seed], with_if[seed]) for seed in sorted(base)]


def eval_args(load_path, destination):
    return SimpleNamespace(
        load_path=load_path, dataset="bus", num_classes=2, net="resnet18",
        net_from_name=False, model_key="ema_model", batch_size=16,
        num_workers=4, data_dir="../uda_data", img_size=224,
        crop_ratio=0.875, max_length=512, max_length_seconds=4.0,
        sample_rate=16000, num_labels=878, label_ratio=None,
        ulb_num_labels=None, lb_imb_ratio=1, ulb_imb_ratio=1,
        include_lb_to_ulb=False, fold=1, seed=0, split_seed=None,
        lpath="../data_split/28/labeled_images_20_9.pth",
        ulpath="../data_split/28/unlabeled_images_80_9.pth",
        valdata="", eval_dest=destination,
    )


def predict(checkpoint_path, destination, device):
    args = eval_args(checkpoint_path, destination)
    _, _, loader = build_eval_loader(args)
    model = load_model(args, checkpoint_path, device)
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch(batch, device)
            logits = model(x)["logits"]
            labels.append(y.cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "auc": roc_auc_score(labels, probs),
        "acc": accuracy_score(labels, preds),
        "sen": recall_score(labels, preds, pos_label=1, zero_division=0),
        "spe": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(labels, preds, zero_division=0),
    }


def choose_threshold(validation_outputs, step):
    candidates = np.arange(step, 1.0, step)
    scores = np.asarray([
        np.mean([f1_score(y, p >= threshold, zero_division=0) for y, p in validation_outputs])
        for threshold in candidates
    ])
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    chosen_index = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return float(candidates[chosen_index]), float(scores[chosen_index])


def paired_stats(base_rows, if_rows, metric):
    base = np.asarray([row[metric] for row in base_rows])
    with_if = np.asarray([row[metric] for row in if_rows])
    differences = with_if - base
    t_p = float(ttest_rel(with_if, base).pvalue) if len(base) > 1 else np.nan
    try:
        w_p = float(wilcoxon(differences).pvalue)
    except ValueError:
        w_p = np.nan
    return base, with_if, differences, t_p, w_p


def main():
    args = parse_args()
    pairs = resolve_pairs(args.base_glob, args.if_glob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs = {"base": {"eval": [], "test": []}, "if": {"eval": [], "test": []}}
    for _, base_path, if_path in pairs:
        for name, path in (("base", base_path), ("if", if_path)):
            for destination in ("eval", "test"):
                outputs[name][destination].append(predict(path, destination, device))

    thresholds = {}
    validation_f1 = {}
    for name in ("base", "if"):
        thresholds[name], validation_f1[name] = choose_threshold(
            outputs[name]["eval"], args.threshold_step
        )

    rows = {"base": [], "if": []}
    default_rows = {"base": [], "if": []}
    for name in ("base", "if"):
        for labels, probs in outputs[name]["test"]:
            rows[name].append(metrics(labels, probs, thresholds[name]))
            default_rows[name].append(metrics(labels, probs, 0.5))

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "threshold_mode", "method", "threshold", "val_mean_f1", "metric",
            "mean", "std", "paired_delta_mean", "paired_t_p", "wilcoxon_p",
        ])
        for mode, selected_rows, selected_thresholds in (
            ("default", default_rows, {"base": 0.5, "if": 0.5}),
            ("val_max_mean_f1", rows, thresholds),
        ):
            for name in ("base", "if"):
                for metric in METRICS:
                    base, with_if, delta, t_p, w_p = paired_stats(
                        selected_rows["base"], selected_rows["if"], metric
                    )
                    values = base if name == "base" else with_if
                    writer.writerow([
                        mode, name, selected_thresholds[name],
                        validation_f1.get(name, "") if mode != "default" else "",
                        metric, values.mean(), values.std(ddof=0),
                        delta.mean(), t_p, w_p,
                    ])

    print(f"base threshold={thresholds['base']:.3f}, val mean F1={validation_f1['base']:.6f}")
    print(f"if threshold={thresholds['if']:.3f}, val mean F1={validation_f1['if']:.6f}")
    for metric in METRICS:
        base, with_if, delta, t_p, w_p = paired_stats(rows["base"], rows["if"], metric)
        print(
            f"{metric}: base={base.mean():.6f}±{base.std(ddof=0):.6f}, "
            f"if={with_if.mean():.6f}±{with_if.std(ddof=0):.6f}, "
            f"delta={delta.mean():+.6f}, paired_t_p={t_p:.6g}, wilcoxon_p={w_p:.6g}"
        )
    print(f"saved {args.output_csv}")


if __name__ == "__main__":
    main()
