#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_sup import build_eval_loader, evaluate_checkpoint, load_model, resolve_device


METRICS = ["auc", "accuracy", "sen", "spe", "f1"]
LABELS = {"auc": "AUC", "accuracy": "ACC", "sen": "Sen", "spe": "Spe", "f1": "F1"}


METHODS = [
    ("supervised", "supervised_gdph_28", False),
    ("fixmatch_if", "fixmatch_ifcf_gdph_28", True),
    ("refixmatch", "refixmatch_gdph_28", True),
    ("softmatch", "softmatch_gdph_28", True),
    ("simmatch", "simmatch_gdph_28_da1", False),
    ("simmatchv2", "simmatchv2_gdph_28_da1", False),
]


def parse_fold_seed(path):
    match = re.search(r"_fold(\d+)_(\d+)", path)
    if not match:
        raise ValueError(f"Cannot parse fold/seed from {path}")
    return int(match.group(1)), int(match.group(2))


def method_dirs(root, prefix, require_rundone):
    dirs = []
    for name in sorted(os.listdir(root)):
        if not name.startswith(prefix + "_fold"):
            continue
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if require_rundone and not os.path.exists(os.path.join(full, "RUN_DONE")):
            continue
        dirs.append(full)
    return dirs


def summarize(rows):
    out = {}
    for key in METRICS:
        values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=float)
        out[key] = (float(values.mean()), float(values.std(ddof=0)), len(values))
    return out


def format_metric(summary, key):
    mean, std, _ = summary[key]
    return f"{mean:.6f}+/-{std:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="saved_models/usb_cv")
    parser.add_argument("--summary_csv", default="results/gdph28_best_latest_recomputed_summary.csv")
    parser.add_argument("--detail_csv", default="results/gdph28_best_latest_recomputed_detail.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device()
    print(f"device {device}")

    base_args = SimpleNamespace(
        load_path="",
        dataset="gdph",
        num_classes=2,
        net="resnet18",
        net_from_name=False,
        model_key="ema_model",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir="../uda_data/GDPH",
        img_size=224,
        crop_ratio=0.875,
        max_length=512,
        max_length_seconds=4.0,
        sample_rate=16000,
        num_labels=384,
        label_ratio=0.2,
        ulb_num_labels=None,
        lb_imb_ratio=1,
        ulb_imb_ratio=1,
        include_lb_to_ulb=False,
        fold=0,
        seed=0,
        split_seed=0,
        lpath="",
        ulpath="",
        valdata="",
        eval_dest="eval",
    )

    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.detail_csv) or ".", exist_ok=True)

    detail_rows = []
    summary_rows = []
    loader_cache = {}

    for method, prefix, require_rundone in METHODS:
        dirs = method_dirs(args.root, prefix, require_rundone)
        print(f"{method}: {len(dirs)} dirs, require_rundone={require_rundone}")
        if not dirs:
            continue

        for kind, ckpt_name in [("best", "model_best.pth"), ("latest", "latest_model.pth")]:
            rows = []
            for directory in dirs:
                ckpt = os.path.join(directory, ckpt_name)
                if not os.path.exists(ckpt):
                    print(f"missing {ckpt}")
                    continue
                fold, seed = parse_fold_seed(directory)
                eval_args = SimpleNamespace(**vars(base_args))
                eval_args.fold = fold
                eval_args.seed = seed
                eval_args.load_path = ckpt
                if fold not in loader_cache:
                    loader_cache[fold] = build_eval_loader(eval_args)
                _, eval_dset, eval_loader = loader_cache[fold]
                model = load_model(eval_args, ckpt, device)
                metrics = evaluate_checkpoint(model, eval_dset, eval_loader, device, csv_path="")
                row = {"method": method, "prefix": prefix, "kind": kind, "fold": fold, "seed": seed, **metrics}
                rows.append(row)
                detail_rows.append(row)
                print(
                    f"{method} {kind} fold{fold} seed{seed}: "
                    + ", ".join(f"{LABELS[k]}={metrics[k]:.6f}" for k in METRICS)
                )
            if rows:
                summary = summarize(rows)
                summary_rows.append(
                    {
                        "method": method,
                        "kind": kind,
                        "n": len(rows),
                        "AUC": format_metric(summary, "auc"),
                        "ACC": format_metric(summary, "accuracy"),
                        "Sen": format_metric(summary, "sen"),
                        "Spe": format_metric(summary, "spe"),
                        "F1": format_metric(summary, "f1"),
                    }
                )
                print(
                    f"SUMMARY {method} {kind} n={len(rows)} "
                    + " ".join(f"{LABELS[k]}={format_metric(summary, k)}" for k in METRICS)
                )

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "kind", "n", "AUC", "ACC", "Sen", "Spe", "F1"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(args.detail_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "prefix", "kind", "fold", "seed", "auc", "accuracy", "sen", "spe", "f1"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"saved {args.summary_csv}")
    print(f"saved {args.detail_csv}")


if __name__ == "__main__":
    main()
