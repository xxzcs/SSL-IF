#!/usr/bin/env python3
# 一键对比: in_fuse strength 扫描 vs SimMatch 基线。
# 读 results/simmatch_if_bus_infuse_summary.csv (本实验) 与 results/simmatch_bus_summary.csv (基线),
# 输出按 strength 排序的对比表 (AUC/ACC/F1 + 相对基线 ΔAUC), 并给 s0 sanity 判定。
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
INFUSE_CSV = os.path.join(ROOT, 'results', 'simmatch_if_bus_infuse_summary.csv')
BASE_CSV = os.path.join(ROOT, 'results', 'simmatch_bus_summary.csv')
BASE_KEY = 'simmatch_bus_878_da1_da1_best'      # SimMatch da1 基线 (best)
SANITY_TOL = 0.012                              # s0 与基线 AUC 容差


def _mean(cell):
    """'0.8531±0.0064' -> 0.8531 ; 解析失败返回 None"""
    if cell is None:
        return None
    try:
        return float(str(cell).split('±')[0])
    except ValueError:
        return None


def load_summary(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows[r['method']] = r
    return rows


def seed_count(tag):
    return len(glob.glob(os.path.join(
        ROOT, 'saved_models', 'usb_cv', f'simmatch_if_bus_{tag}_*', 'model_best.pth')))


def main():
    base = load_summary(BASE_CSV).get(BASE_KEY, {})
    base_auc = _mean(base.get('AUC'))
    base_acc = _mean(base.get('ACC'))
    base_f1 = _mean(base.get('F1'))

    infuse = load_summary(INFUSE_CSV)
    # 收集 (strength, kind) -> row
    recs = []
    for method, r in infuse.items():
        m = re.search(r'infuse_s(\d+).*_(best|latest)$', method)
        if not m:
            continue
        recs.append((int(m.group(1)), m.group(2), r))
    recs.sort(key=lambda x: (x[1] != 'best', x[0]))   # best 在前, 按 strength 升序

    print('=' * 92)
    print('  SimMatch-IF  in_fuse 模式  strength 扫描  vs  SimMatch 基线(da1)')
    print('=' * 92)
    if base_auc is not None:
        print(f'  基线 da1 (best): AUC={base_auc:.4f}  ACC={base_acc:.4f}  F1={base_f1:.4f}')
    else:
        print('  [warn] 未找到基线行', BASE_KEY)
    if not recs:
        print('\n  (尚无 in_fuse 结果, summary 为空或未评测)')
        print('=' * 92)
        return 0

    print('-' * 92)
    print(f'  {"strength":>8} {"ckpt":>6} {"n":>3} {"AUC":>16} {"ACC":>16} {"F1":>16} {"ΔAUC":>9}')
    print('-' * 92)
    for s, kind, r in recs:
        auc = _mean(r.get('AUC'))
        d = '' if (auc is None or base_auc is None) else f'{auc - base_auc:+.4f}'
        n = seed_count(f'infuse_s{s}')
        print(f'  {s:>8} {kind:>6} {n:>3} {str(r.get("AUC","")):>16} '
              f'{str(r.get("ACC","")):>16} {str(r.get("F1","")):>16} {d:>9}')
    print('-' * 92)

    # ---- s0 sanity ----
    s0 = next((r for s, k, r in recs if s == 0 and k == 'best'), None)
    if s0 is not None and base_auc is not None:
        a0 = _mean(s0.get('AUC'))
        if a0 is None:
            print('  [sanity] s0 的 AUC 解析失败, 需人工检查')
        elif abs(a0 - base_auc) <= SANITY_TOL:
            print(f'  [sanity] PASS: s0(AUC={a0:.4f}) 与基线({base_auc:.4f}) 差 {a0-base_auc:+.4f}'
                  f' (≤{SANITY_TOL}), strength=0 退化为基线成立, 融合管线干净。')
        else:
            print(f'  [sanity] WARN: s0(AUC={a0:.4f}) 与基线({base_auc:.4f}) 差 {a0-base_auc:+.4f}'
                  f' (>{SANITY_TOL}), strength=0 本应≈基线, 需排查 (但仍可能在 seed 方差内)。')
    else:
        print('  [sanity] 暂无 s0_best 结果, 跳过判定。')

    # ---- 最佳 ----
    best_recs = [(s, _mean(r.get('AUC'))) for s, k, r in recs if k == 'best' and _mean(r.get('AUC'))]
    if best_recs:
        bs, ba = max(best_recs, key=lambda x: x[1])
        tag = 'best>基线 ✅' if (base_auc and ba > base_auc) else '未超基线'
        print(f'  [最佳] strength={bs}: AUC={ba:.4f}  ({tag})')
    print('=' * 92)
    return 0


if __name__ == '__main__':
    sys.exit(main())
