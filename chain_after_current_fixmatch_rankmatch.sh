#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs

STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=${CHAIN_LOG:-logs/chain_fixmatch_rankmatch_${STAMP}.log}
BUS_LOG=${BUS_LOG:-logs/fixmatch_rankmatch_bus_after_current_${STAMP}.log}
GDPH_LOG=${GDPH_LOG:-logs/fixmatch_rankmatch_gdph_after_current_${STAMP}.log}

exec > >(tee -a "$CHAIN_LOG") 2>&1

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] wait for current GDPH requested queue to finish"
echo "chain_log=${CHAIN_LOG}"
echo "bus_log=${BUS_LOG}"
echo "gdph_log=${GDPH_LOG}"

while pgrep -f "run_gdph_requested_experiments.sh" >/dev/null 2>&1; do
  echo "[$(date)] current queue still running..."
  sleep 300
done

while pgrep -f "train(_ifcf)?\\.py" >/dev/null 2>&1; do
  echo "[$(date)] residual training process still running..."
  sleep 120
done

echo "[$(date)] current queue finished; start BUS"
LOG="$BUS_LOG" bash run_fixmatch_rankmatch_bus.sh

echo "[$(date)] BUS finished; start GDPH"
LOG="$GDPH_LOG" bash run_fixmatch_rankmatch_gdph.sh

touch results/ALL_DONE_FIXMATCH_RANKMATCH_CHAIN
echo "[$(date)] chain finished"
