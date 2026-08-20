#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/simmatch_if_bus_ld05_wu510_after_srcmt_${STAMP}.log}

mkdir -p logs results
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== Wait SRC-MT then run SimMatch+IF BUS ld0.5 warmup5/10 ====="
echo "log=${LOG}"

while pgrep -f "train_SRC_MT.py" >/dev/null 2>&1; do
  echo "[$(date)] SRC-MT still running; sleep 60s"
  sleep 60
done

echo "[$(date)] SRC-MT finished; start SimMatch+IF BUS ld0.5 warmup5/10"
bash run_simmatch_if_bus_ld05_wu510.sh
echo "[$(date)] ===== queued task finished ====="
