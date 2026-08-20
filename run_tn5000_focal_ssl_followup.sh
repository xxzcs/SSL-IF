#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

DATA_DIR=${DATA_DIR:-../uda_data/TN5000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
CE_CLASS_WEIGHTS=${CE_CLASS_WEIGHTS:-1.6957,0.7091}
FOCAL_GAMMA=${FOCAL_GAMMA:-1.5}
FORCE_RERUN=${FORCE_RERUN:-0}

run_case() {
  local method=$1
  local suffix=$2
  echo "[$(date)] Start ${method} focal followup suffix=${suffix}"
  METHOD="${method}" \
  RATIO=20 \
  SEEDS="${SEEDS}" \
  DATA_DIR="${DATA_DIR}" \
  FORCE_RERUN="${FORCE_RERUN}" \
  CE_CLASS_WEIGHTS="${CE_CLASS_WEIGHTS}" \
  SAVE_SUFFIX="${suffix}" \
  AUTO_EVAL=1 \
  EVAL_KINDS=latest \
  EVAL_THRESHOLD_MODES=default_0.5,val_max_bacc \
  EXTRA_CONFIG_APPEND="focal_gamma: ${FOCAL_GAMMA}" \
  bash run_tn5000_ratio.sh
}

run_case fixmatch "focal_g15_wce"
run_case fixmatch_if "focal_g15_wce"
