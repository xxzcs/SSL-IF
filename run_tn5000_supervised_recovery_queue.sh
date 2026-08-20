#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

echo "[$(date)] TN5000 supervised recovery queue started"

echo "[$(date)] Start focal-loss line"
bash run_tn5000_supervised_focal20.sh

echo "[$(date)] Start short-training 10 epoch line"
SHORT_EPOCHS=10 SAVE_SUFFIX=short10e bash run_tn5000_supervised_short20.sh

echo "[$(date)] Start short-training 20 epoch line"
SHORT_EPOCHS=20 SAVE_SUFFIX=short20e bash run_tn5000_supervised_short20.sh

echo "[$(date)] Start short-training 30 epoch line"
SHORT_EPOCHS=30 SAVE_SUFFIX=short30e bash run_tn5000_supervised_short20.sh

echo "[$(date)] TN5000 supervised recovery queue finished"
