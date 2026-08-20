#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/chain_after_fixmatch_ifcf_phase1_run_soft_phase_${STAMP}.log}
WAIT_FLAG=${WAIT_FLAG:-results/ALL_DONE_BUS_FIXMATCH_IFCF_LD1_HARDSTRONG_WU510}

mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== Waiting FixMatch+IFCF hard-strong warmup5/10 ====="
echo "wait_flag=${WAIT_FLAG}"
echo "log=${LOG}"

while [ ! -f "$WAIT_FLAG" ]; do
  sleep 60
done

echo "[$(date)] ===== Hard-strong warmup5/10 done; start soft phase ====="
bash run_bus_fixmatch_ifcf_ld1_soft_phase.sh
echo "[$(date)] ===== Chain done ====="
