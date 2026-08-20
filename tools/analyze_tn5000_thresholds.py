#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval_sup import build_eval_loader, load_model, resolve_device, unpack_batch


METRIC_KEYS = ["auc", "acc", "bacc", "sen", "spe", "f1"]


METHODS = [
    ("supervised", "supervised_tn5000_{ratio}_{seed}", "model"),
    ("fixmatch", "fixmatch_tn5000_{ratio}_{seed}", "ema_model"),
    ("fixmatch_if", "fixmatch_ifcf_hard_tn5000_{ratio}_{seed}", "ema_model"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio_tag", default="28", help="TN5000 ratio tag: 19/15/28/37/all")
    parser.add_argument("--kind", default="latest", choices=["latest", "best"])
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--data_dir", default="../uda_data/TN5000")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold_step", type=float, default=0.001)
    parser.add_argument("--summary_csv", default="results/tn5000_threshold_summary.csv")
    parser.add_argument("--detail_csv", default="results/tn5000_threshold_detail.csv")
    return parser.parse_args()


def ratio_to_num_labels(ratio_tag):
    mapping = {
        "19": 350,
        "15": 525,
        "28": 700,
        "37": 1050,
        "all": 3500,
    }
    if ratio_tag not in mapping:
        raise ValueError(f"Unsupported ratio_tag={ratio_tag}")
    return mapping[ratio_tag]


def eval_args(load_path, model_key, data_dir, batch_size, num_workers, num_labels, seed):
    return SimpleNamespace(
        load_path=load_path,
        dataset="tn5000",
        num_classes=2,
        net="resnet18",
        net_from_name=False,
        model_key=model_key,
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        img_size=224,
        crop_ratio=0.875,
        max_length=512,
        max_length_seconds=4.0,
        sample_rate=16000,
        num_labels=num_labels,
        label_ratio=None,
        ulb_num_labels=None,
        lb_imb_ratio=1,
        ulb_imb_ratio=1,
        include_lb_to_ulb=False,
        fold=1,
        seed=seed,
        split_seed=seed,
        lpath="",
        ulpath="",
        valdata="",
        eval_dest="eval",
    )


def predict(args, device, destination):
    local_args = SimpleNamespace(**vars(args))
    local_args.eval_dest = destination
    _, _, loader = build_eval_loader(local_args)
    model = load_model(local_args, local_args.load_path, device)
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch(batch, device)
            logits = model(x)["logits"]
            labels.append(y.cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def compute_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sen = recall_score(labels, preds, pos_label=1, zero_division=0)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": roc_auc_score(labels, probs),
        "acc": accuracy_score(labels, preds),
        "bacc": 0.5 * (sen + spe),
        "sen": sen,
        "spe": spe,
        "f1": f1_score(labels, preds, zero_division=0),
    }


def choose_threshold(validation_outputs, step, objective):
    candidates = np.arange(step, 1.0, step)
    if objective == "f1":
        scores = np.asarray([
            np.mean([f1_score(y, p >= threshold, zero_division=0) for y, p in validation_outputs])
            for threshold in candidates
        ])
    elif objective == "bacc":
        scores = np.asarray([
            np.mean([
                compute_metrics(y, p, threshold)["bacc"]
                for y, p in validation_outputs
            ])
            for threshold in candidates
        ])
    else:
        raise ValueError(f"Unsupported objective={objective}")

    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    chosen = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return float(candidates[chosen]), float(scores[chosen])


def summarize(rows):
    out = {}
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = (float(values.mean()), float(values.std(ddof=0)))
    return out


def fmt(mean, std):
    return f"{mean:.6f}±{std:.6f}"


def main():
    args = parse_args()
    device = resolve_device()
    num_labels = ratio_to_num_labels(args.ratio_tag)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    ckpt_name = "latest_model.pth" if args.kind == "latest" else "model_best.pth"

    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.detail_csv) or ".", exist_ok=True)

    detail_rows = []
    summary_rows = []

    for method, pattern, model_key in METHODS:
        eval_outputs = []
        test_outputs = []
        active_seeds = []

        for seed in seeds:
            run_dir = os.path.join("saved_models", "usb_cv", pattern.format(ratio=args.ratio_tag, seed=seed))
            ckpt_path = os.path.join(run_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f"missing {ckpt_path}")
                continue

            method_args = eval_args(
                load_path=ckpt_path,
                model_key=model_key,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_labels=num_labels,
                seed=seed,
            )
            eval_outputs.append(predict(method_args, device, "eval"))
            test_outputs.append(predict(method_args, device, "test"))
            active_seeds.append(seed)

        if not active_seeds:
            continue

        threshold_modes = [
            ("default_0.5", 0.5, None),
            ("val_max_f1", None, "f1"),
            ("val_max_bacc", None, "bacc"),
        ]

        selected_thresholds = {}
        selected_scores = {}
        for mode, fixed_threshold, objective in threshold_modes:
            if fixed_threshold is not None:
                selected_thresholds[mode] = fixed_threshold
                selected_scores[mode] = None
            else:
                threshold, score = choose_threshold(eval_outputs, args.threshold_step, objective)
                selected_thresholds[mode] = threshold
                selected_scores[mode] = score

        for mode, threshold in selected_thresholds.items():
            rows = []
            for seed, (labels, probs) in zip(active_seeds, test_outputs):
                metrics = compute_metrics(labels, probs, threshold)
                rows.append(metrics)
                detail_rows.append({
                    "method": method,
                    "kind": args.kind,
                    "ratio_tag": args.ratio_tag,
                    "seed": seed,
                    "threshold_mode": mode,
                    "threshold": threshold,
                    **metrics,
                })

            stats = summarize(rows)
            summary_rows.append({
                "method": method,
                "kind": args.kind,
                "ratio_tag": args.ratio_tag,
                "n": len(rows),
                "threshold_mode": mode,
                "threshold": f"{threshold:.3f}",
                "val_objective": "" if selected_scores[mode] is None else f"{selected_scores[mode]:.6f}",
                "AUC": fmt(*stats["auc"]),
                "ACC": fmt(*stats["acc"]),
                "BACC": fmt(*stats["bacc"]),
                "Sen": fmt(*stats["sen"]),
                "Spe": fmt(*stats["spe"]),
                "F1": fmt(*stats["f1"]),
            })
            print(
                f"{method} {mode} thr={threshold:.3f} n={len(rows)} "
                f"AUC={fmt(*stats['auc'])} ACC={fmt(*stats['acc'])} "
                f"BACC={fmt(*stats['bacc'])} Sen={fmt(*stats['sen'])} "
                f"Spe={fmt(*stats['spe'])} F1={fmt(*stats['f1'])}"
            )

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method", "kind", "ratio_tag", "n", "threshold_mode", "threshold", "val_objective",
                "AUC", "ACC", "BACC", "Sen", "Spe", "F1",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(args.detail_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "kind", "ratio_tag", "seed", "threshold_mode", "threshold"] + METRIC_KEYS,
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"saved {args.summary_csv}")
    print(f"saved {args.detail_csv}")


if __name__ == "__main__":
    main()
