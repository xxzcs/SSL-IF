#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

DATA_DIR=${DATA_DIR:-../uda_data/TN5000}
CE_CLASS_WEIGHTS=${CE_CLASS_WEIGHTS:-1.6957,0.7091}
P_CUTOFF=${P_CUTOFF:-0.9}
FORCE_RERUN=${FORCE_RERUN:-0}

wait_for_no_training() {
  while pgrep -f "python3 train.py --c /tmp/tmp" >/dev/null; do
    echo "[$(date)] Waiting for current train.py job to finish..."
    sleep 60
  done
}

echo "[$(date)] TN5000 weighted pipeline started"
wait_for_no_training

echo "[$(date)] Start supervised weighted sweep"
DATA_DIR="${DATA_DIR}" \
CE_CLASS_WEIGHTS="${CE_CLASS_WEIGHTS}" \
FORCE_RERUN="${FORCE_RERUN}" \
bash run_tn5000_supervised_weighted_sweep.sh

echo "[$(date)] Start fixmatch weighted sweep"
DATA_DIR="${DATA_DIR}" \
CE_CLASS_WEIGHTS="${CE_CLASS_WEIGHTS}" \
P_CUTOFF="${P_CUTOFF}" \
FORCE_RERUN="${FORCE_RERUN}" \
bash run_tn5000_fixmatch_weighted_sweep.sh

echo "[$(date)] TN5000 weighted pipeline finished"
