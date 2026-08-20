#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] waiting for BUS MeanTeacher-AP completion..."
while [ ! -f results/ALL_DONE_MEANTEACHER_AP_BUS ]; do
  sleep 120
done

echo "[$(date)] BUS done, starting GDPH MeanTeacher-AP"
bash run_meanteacher_ap_gdph.sh
