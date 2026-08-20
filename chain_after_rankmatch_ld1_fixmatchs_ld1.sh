#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs

STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=${CHAIN_LOG:-logs/chain_after_rankmatch_ld1_fixmatchs_ld1_${STAMP}.log}

exec > >(tee -a "$CHAIN_LOG") 2>&1

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] wait for BUS rankmatch ld=1.0 to finish"
echo "chain_log=${CHAIN_LOG}"

while pgrep -f "fixmatch_rankmatch_bus_ld1|config/_rankmatch_bus_fixmatch_rankmatch_bus_ld1_|train_ifcf.py --c config/_rankmatch_bus_fixmatch_rankmatch_bus_ld1_" >/dev/null 2>&1; do
  echo "[$(date)] rankmatch ld1 still running..."
  sleep 300
done

echo "[$(date)] rankmatch ld1 finished; start BUS fixmatch ld1"
bash run_fixmatch_bus_ld1.sh

echo "[$(date)] BUS fixmatch ld1 finished; start BUS fixmatch+ifcf ld1"
bash run_fixmatch_ifcf_bus_ld1.sh

touch results/ALL_DONE_FIXMATCH_CHAIN_LD1
echo "[$(date)] chain finished"
