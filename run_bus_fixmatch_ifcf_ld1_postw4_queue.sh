#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/bus_fixmatch_ifcf_ld1_postw4_queue_${STAMP}.log}
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== post-w4 queue waiting ====="
while [ ! -f results/ALL_DONE_BUS_FIXMATCH_IFCF_LD1_WU1_WEIGHT24 ]; do
  sleep 60
done

echo "[$(date)] ===== start if_lambda2/5 ====="
bash run_bus_fixmatch_ifcf_ld1_wu1_iflambda25.sh

echo "[$(date)] ===== start corrT0.5 ====="
bash run_bus_fixmatch_ifcf_ld1_wu1_corrt05.sh

echo "[$(date)] ===== post-w4 queue done ====="
