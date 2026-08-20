#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs

STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=${CHAIN_LOG:-logs/chain_after_srcmt_rankmatch_ld1_${STAMP}.log}
BUS_LOG=${BUS_LOG:-logs/fixmatch_rankmatch_bus_ld1_after_srcmt_${STAMP}.log}
SUMMARY=${SUMMARY:-results/fixmatch_rankmatch_bus_ld1_pcut09_latest.csv}

exec > >(tee -a "$CHAIN_LOG") 2>&1

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] wait for current SRC-MT queue to finish"
echo "chain_log=${CHAIN_LOG}"
echo "bus_log=${BUS_LOG}"
echo "summary=${SUMMARY}"

while pgrep -f "third_party/SRC-MT.*run_bus_src_mt_5seed|third_party/SRC-MT.*run_gdph_src_mt_5fold5seed|third_party/SRC-MT/code/train_SRC_MT.py" >/dev/null 2>&1; do
  echo "[$(date)] SRC-MT still running..."
  sleep 300
done

echo "[$(date)] SRC-MT finished; start BUS rankmatch ld=1.0"
LOG="$BUS_LOG" SUMMARY="$SUMMARY" LAYER_DECAY=1.0 bash run_fixmatch_rankmatch_bus.sh

touch results/ALL_DONE_FIXMATCH_RANKMATCH_BUS_LD1
echo "[$(date)] BUS rankmatch ld=1.0 done"
