#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-10}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-2000}"
BUS_SEEDS="${BUS_SEEDS:-1 2 3 4 5}"
GDPH_FOLDS="${GDPH_FOLDS:-0 1 2 3 4}"
GDPH_SEEDS="${GDPH_SEEDS:-1 2 3 4 5}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-logs/src_mt_faithful_queue_${STAMP}.log}"

mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

gpu_idle() {
  local line util mem
  line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU" | head -n 1 || true)"
  util="$(echo "$line" | cut -d',' -f1 | tr -d ' ')"
  mem="$(echo "$line" | cut -d',' -f2 | tr -d ' ')"
  [[ -n "$util" && -n "$mem" ]] || return 1
  [[ "$util" -le "$UTIL_THRESHOLD" && "$mem" -le "$MEM_THRESHOLD_MB" ]]
}

echo "[$(date)] ===== faithful SRC-MT queue start ====="
echo "log=${LOG}"
echo "gpu=${GPU} util_threshold=${UTIL_THRESHOLD} mem_threshold_mb=${MEM_THRESHOLD_MB} check_interval=${CHECK_INTERVAL}"
echo "bus_seeds=${BUS_SEEDS}"
echo "gdph_folds=${GDPH_FOLDS} gdph_seeds=${GDPH_SEEDS}"

until gpu_idle; do
  echo "[$(date)] GPU ${GPU} busy; sleep ${CHECK_INTERVAL}s"
  sleep "$CHECK_INTERVAL"
done

echo "[$(date)] GPU ${GPU} is idle; start BUS faithful 5-seed"
GPU="$GPU" \
SEEDS="$BUS_SEEDS" \
ARCH="resnet18" \
EPOCHS="50" \
BATCH_SIZE="32" \
LABELED_BS="16" \
BASE_LR="0.009375" \
INITIAL_LR="0.0" \
LR_RAMPUP="2" \
CONSISTENCY_RAMPUP="30" \
OPTIMIZER="sgd" \
WDECAY="0.0005" \
bash third_party/SRC-MT/run_bus_src_mt_5seed.sh

echo "[$(date)] BUS faithful run done; start GDPH faithful 5-fold x 5-seed"
GPU="$GPU" \
FOLDS="$GDPH_FOLDS" \
SEEDS="$GDPH_SEEDS" \
ARCH="resnet18" \
EPOCHS="50" \
BATCH_SIZE="32" \
LABELED_BS="16" \
BASE_LR="0.009375" \
INITIAL_LR="0.0" \
LR_RAMPUP="2" \
CONSISTENCY_RAMPUP="30" \
OPTIMIZER="sgd" \
WDECAY="0.0005" \
NUM_LABELS="384" \
bash third_party/SRC-MT/run_gdph_src_mt_5fold5seed.sh

echo "[$(date)] ===== faithful SRC-MT queue done ====="
