#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/chain_after_bus_ld1_recheck_run_refixmatch_${STAMP}.log}
WAIT_FLAG=${WAIT_FLAG:-results/ALL_DONE_BUS_LD1_RECHECK_4METHODS}

mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== Waiting BUS ld=1.0 4-method queue ====="
echo "wait_flag=${WAIT_FLAG}"
echo "log=${LOG}"

while [ ! -f "$WAIT_FLAG" ]; do
  sleep 60
done

echo "[$(date)] ===== BUS ld=1.0 4-method queue done; start ReFixMatch ====="
bash run_bus_refixmatch_ld1_recheck.sh
echo "[$(date)] ===== Chain done ====="
