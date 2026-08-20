#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/chain_after_freematch_gdph_simmatch_ld05_${STAMP}.log}

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== Wait FreeMatch GDPH then run SimMatch BUS ld0.5 ====="
echo "log=${LOG}"

while [ ! -f results/ALL_DONE_FREEMATCH_GDPH_LD1 ]; do
  echo "[$(date)] FreeMatch GDPH ld1 not done yet; sleep 60s"
  sleep 60
done

echo "[$(date)] FreeMatch GDPH ld1 done; starting SimMatch BUS ld0.5"
bash run_simmatch_bus_ld05.sh
