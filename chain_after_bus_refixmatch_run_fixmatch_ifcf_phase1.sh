#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/chain_after_bus_refixmatch_run_fixmatch_ifcf_phase1_${STAMP}.log}
WAIT_FLAG=${WAIT_FLAG:-results/ALL_DONE_BUS_REFIXMATCH_LD1_RECHECK}

mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== Waiting BUS ReFixMatch ld=1.0 queue ====="
echo "wait_flag=${WAIT_FLAG}"
echo "log=${LOG}"

while [ ! -f "$WAIT_FLAG" ]; do
  sleep 60
done

echo "[$(date)] ===== ReFixMatch done; start FixMatch+IFCF phase1 ====="
bash run_bus_fixmatch_ifcf_ld1_phase1.sh
echo "[$(date)] ===== Chain done ====="
