#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] waiting for GDPH MeanTeacher-AP completion..."
while [ ! -f results/ALL_DONE_MEANTEACHER_AP_GDPH ]; do
  sleep 120
done

echo "[$(date)] GDPH done, starting FixMatch+IFCF BUS ld0.5 ablation queue"
bash run_fixmatch_ifcf_bus_ld05_ablation_queue.sh
