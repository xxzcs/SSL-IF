#!/usr/bin/env python3
"""Build a readable inventory of all BUS experiment summary rows."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "saved_models" / "usb_cv"
OUT_MD = RESULTS / "BUS_ALL_EXPERIMENTS.md"
OUT_CSV = RESULTS / "BUS_ALL_EXPERIMENTS.csv"

EXCLUDE = {
    "classic_cv.csv", "classic_cv_imb.csv", "gdph_cv_summary.csv",
    "simmatch_gdph_cv.csv", "simmatch_gdph_cv_foldstd_SUPPLEMENTARY.csv",
    "simmatch_gdph_gtprior_cv.csv", "simmatchv2_gdph_cv.csv",
    "tn5000_summary.csv", "usb_audio.csv", "usb_cv.csv", "usb_nlp.csv",
}

KEYS = [
    "algorithm", "net", "seed", "num_labels", "uratio", "epoch",
    "num_train_iter", "batch_size", "lr", "optim", "sched", "p_cutoff",
    "T", "ulb_loss_ratio", "ifrank_combine", "if_target",
    "ifrank_loss_weight", "if_lambda", "csim_lambda", "corrT",
    "num_references", "ref_select", "ref_cand_k", "use_strong_if",
    "ifrank_warmup_epochs", "ifrank_warmup_mode",
]


def parse_args(model_dir: Path) -> dict[str, str]:
    log = model_dir / "log.txt"
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    match = re.search(r"Arguments: Namespace\((.*?)\)\n", text)
    if not match:
        return {}
    body = match.group(1)
    values: dict[str, str] = {}
    for key in KEYS:
        m = re.search(rf"(?:^|, ){re.escape(key)}=(.*?)(?=, [A-Za-z_]\w*=|$)", body)
        if m:
            values[key] = m.group(1).strip("'\"")
    return values


def checkpoint(method: str) -> str:
    if re.search(r"(?:^|_)latest(?:_|$)", method):
        return "latest"
    if re.search(r"(?:^|_)best(?:_|$)", method):
        return "best"
    return "未标明"


def family(algorithm: str, method: str) -> tuple[str, str]:
    low = (algorithm + " " + method).lower()
    if "simmatchv2" in low:
        base = "SimMatchV2"
    elif "simmatch" in low:
        base = "SimMatch"
    elif "refixmatch" in low:
        base = "ReFixMatch"
    elif "freematch" in low or re.search(r"(^|_)fm_", low):
        base = "FreeMatch"
    elif "softmatch" in low:
        base = "SoftMatch"
    elif "adamatch" in low:
        base = "AdaMatch"
    elif "flexmatch" in low or re.search(r"(^|_)fx_", low):
        base = "FlexMatch"
    elif "fixmatch" in low or any(x in low for x in ("ifcf", "ifrank", "capt", "p90", "warm", "wu_")):
        base = "FixMatch"
    elif "supervised" in low:
        base = "Supervised"
    else:
        base = algorithm or "未识别"
    enhanced = any(x in low for x in ("ifcf", "ifrank", "_if_", "infuse", "capt", "corr", "refselect"))
    kind = "+IF/IFCF" if enhanced else "Base/原方法"
    return base, kind


def compact_params(args: dict[str, str], model: str) -> str:
    if not args:
        return "配置日志未找到；见实验名/来源脚本"
    core = []
    for key in KEYS:
        if key not in args or key == "seed":
            continue
        value = args[key]
        if value in ("None", "False") and key.startswith("if"):
            continue
        core.append(f"{key}={value}")
    return "; ".join(core) + (f"; 参数样本={model}" if model else "")


def main() -> None:
    model_names = sorted((p.name for p in MODELS.iterdir() if p.is_dir()), key=len, reverse=True)
    args_cache: dict[str, dict[str, str]] = {}
    rows = []
    for path in sorted(RESULTS.glob("*.csv")):
        if path.name in EXCLUDE:
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"method", "AUC", "ACC", "Sen", "Spe", "F1"}.issubset(reader.fieldnames):
                continue
            for raw in reader:
                method = raw["method"].strip()
                model = next((name for name in model_names if method == name or method.startswith(name + "_")), "")
                if model not in args_cache:
                    args_cache[model] = parse_args(MODELS / model) if model else {}
                args = args_cache[model]
                base, kind = family(args.get("algorithm", ""), method)
                rows.append({
                    "Base": base, "类别": kind, "实验/方法名": method,
                    "checkpoint": checkpoint(method), "AUC": raw["AUC"],
                    "ACC": raw["ACC"], "Sen": raw["Sen"], "Spe": raw["Spe"],
                    "F1": raw["F1"], "关键参数配置": compact_params(args, model),
                    "结果来源": path.name,
                })

    fields = list(rows[0])
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

    sources = sorted({r["结果来源"] for r in rows})
    lines = [
        "# BUS 数据集全部实验汇总", "",
        f"共收录 **{len(rows)}** 条评估记录，来自 **{len(sources)}** 个 BUS 结果文件。",
        "指标均直接保留为 `均值 ± 标准差`；best/latest 分行保留。参数来自对应模型 `log.txt` 的 Arguments；找不到日志的历史记录已明确标注。",
        "", "共同数据设置：BUS 二分类；标注 878、未标注 3512；数据划分为 `data_split/28/*_20_9.pth`；测试目的地为 test。", "",
        "| Base | 类别 | 实验/方法名 | ckpt | AUC | ACC | Sen | Spe | F1 | 关键参数配置 | 来源 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        vals = [r["Base"], r["类别"], r["实验/方法名"], r["checkpoint"], r["AUC"], r["ACC"], r["Sen"], r["Spe"], r["F1"], r["关键参数配置"], r["结果来源"]]
        lines.append("| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in vals) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    main()
