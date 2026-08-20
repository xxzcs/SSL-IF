#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] waiting for GDPH IF corrT=0.5 warmup=5 completion..."
while [ ! -f results/ALL_DONE_GDPH_IF_T05_WU5 ]; do
  sleep 120
done

echo "[$(date)] GDPH IF corrT=0.5 warmup=5 done, starting BUS FixMatch+DIF-only ld0.5"
bash run_bus_fixmatch_dif_only_ld05.sh
