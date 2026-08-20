#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs results

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/chain_after_simmatch_ld05_srcmt_${STAMP}.log}

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== Wait SimMatch BUS ld0.5 then run SRC-MT BUS 5seed -> GDPH 5foldx5seed ====="
echo "log=${LOG}"

while [ ! -f results/ALL_DONE_SIMMATCH_BUS_LD05 ]; do
  echo "[$(date)] SimMatch BUS ld0.5 not done yet; sleep 60s"
  sleep 60
done

echo "[$(date)] SimMatch BUS ld0.5 done; starting SRC-MT BUS 5seed"
bash third_party/SRC-MT/run_bus_src_mt_5seed.sh

echo "[$(date)] SRC-MT BUS 5seed done; starting SRC-MT GDPH 5foldx5seed"
bash third_party/SRC-MT/run_gdph_src_mt_5fold5seed.sh

touch results/ALL_DONE_SRCMT_CHAIN
echo "[$(date)] ===== SRC-MT chain done ====="
