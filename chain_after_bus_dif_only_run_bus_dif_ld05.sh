#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] waiting for BUS FixMatch+DIF-only ld0.5 completion..."
while [ ! -f results/ALL_DONE_BUS_FIXMATCH_DIF_ONLY_LD05 ]; do
  sleep 120
done

echo "[$(date)] BUS FixMatch+DIF-only ld0.5 done, starting BUS FixMatch+DIF ld0.5"
bash run_bus_fixmatch_dif_ld05.sh
