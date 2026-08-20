#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] waiting for BUS FixMatch+IFCF ld0.5 ablation completion..."
while [ ! -f results/ALL_DONE_FIXMATCH_IFCF_BUS_LD05_ABLATION ]; do
  sleep 120
done

echo "[$(date)] BUS ld0.5 ablation done, starting GDPH FixMatch+IF corrT=0.5 warmup=5"
METHODS=if_hard_t05_warmup5 \
SUMMARY=results/gdph_if_t05_wu5_latest.csv \
FOLDMEAN=results/gdph_if_t05_wu5_foldmean.csv \
bash run_gdph_requested_experiments.sh

touch results/ALL_DONE_GDPH_IF_T05_WU5
echo "[$(date)] GDPH FixMatch+IF corrT=0.5 warmup=5 done"
