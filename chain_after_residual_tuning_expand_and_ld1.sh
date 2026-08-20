#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate wssl 2>/dev/null

LOG=logs/chain_after_residual_tuning_expand_and_ld1.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] waiting for BUS ld0.5 residual tuning 3-seed sweep..."
while [ ! -f results/ALL_DONE_BUS_FIXMATCH_IFCF_RESIDUAL_TUNING_LD05 ]; do
  sleep 120
done

DECISION_FILE=results/residual_tuning_selection.txt
python3 - <<'PY' > "$DECISION_FILE"
import csv, re
from pathlib import Path

path = Path("results/fixmatch_ifcf_bus_ld05_residual_tuning_latest.csv")
rows = list(csv.DictReader(path.open()))

def parse_metric(s):
    return float(s.split('±')[0])

name_map = {
    "lam05": "lambda05",
    "lam075": "lambda075",
    "ramp": "ramp",
}

parsed = []
for row in rows:
    m = re.search(r"residual_product_balanced_(lam05|lam075|ramp)", row["method"])
    if not m:
        continue
    key = name_map[m.group(1)]
    auc = parse_metric(row["AUC"])
    acc = parse_metric(row["ACC"])
    f1 = parse_metric(row["F1"])
    parsed.append((key, auc, acc, f1))

parsed.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
best = parsed[0]

close = []
for item in parsed:
    if (best[1] - item[1] <= 0.0015 and
        best[2] - item[2] <= 0.0015 and
        best[3] - item[3] <= 0.0015):
        close.append(item[0])

print("BEST=" + best[0])
print("BEST_AUC=%.6f" % best[1])
print("BEST_ACC=%.6f" % best[2])
print("BEST_F1=%.6f" % best[3])
print("SELECTED=" + " ".join(close if len(close) > 1 else [best[0]]))
PY

cat "$DECISION_FILE"
. "$DECISION_FILE"

echo "[$(date)] expanding selected variants to 5 seeds: $SELECTED"
VARIANTS="$SELECTED" RESET_SUMMARY=1 SEEDS="4 5" bash run_bus_fixmatch_ifcf_residual_expand_ld05.sh

python3 - <<'PY' > results/residual_expand_best_variant.txt
import csv, re
from pathlib import Path

path = Path("results/fixmatch_ifcf_bus_ld05_residual_expand_latest.csv")
rows = list(csv.DictReader(path.open()))

def parse_metric(s):
    return float(s.split('±')[0])

name_map = {
    "lam05": "lambda05",
    "lam075": "lambda075",
    "ramp": "ramp",
}

parsed = []
for row in rows:
    m = re.search(r"residual_product_balanced_(lam05|lam075|ramp)", row["method"])
    if not m:
        continue
    key = name_map[m.group(1)]
    auc = parse_metric(row["AUC"])
    acc = parse_metric(row["ACC"])
    f1 = parse_metric(row["F1"])
    parsed.append((key, auc, acc, f1))

parsed.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
best = parsed[0][0]
print("VARIANT=" + best)
PY

. results/residual_expand_best_variant.txt
echo "[$(date)] best 5-seed variant: $VARIANT; starting ld1.0 5-seed replication"
VARIANT="$VARIANT" RESET_SUMMARY=1 SEEDS="1 2 3 4 5" bash run_bus_fixmatch_ifcf_residual_best_ld1.sh
