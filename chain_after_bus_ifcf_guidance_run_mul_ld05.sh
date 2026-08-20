#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate wssl 2>/dev/null

LOG=logs/chain_after_bus_ifcf_guidance_run_mul_ld05.nohup.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] waiting for BUS ld0.5 guidance sweep completion..."
while [ ! -f results/ALL_DONE_BUS_FIXMATCH_IFCF_GUIDANCE_LD05 ]; do
  sleep 120
done

echo "[$(date)] BUS ld0.5 guidance sweep finished; starting multiplicative guidance sweep."
bash run_bus_fixmatch_ifcf_guidance_mul_ld05.sh
