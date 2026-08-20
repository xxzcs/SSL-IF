#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

RATIO=${RATIO:-20}
SEEDS=${SEEDS:-"1 2 3 4 5"}
DATA_DIR=${DATA_DIR:-../uda_data/TN5000}
LR=${LR:-0.002}
P_CUTOFF=${P_CUTOFF:-0.9}
FORCE_RERUN=${FORCE_RERUN:-0}
EVAL_KINDS=${EVAL_KINDS:-latest}
SAVE_SUFFIX=${SAVE_SUFFIX:-b_full_aug_bal_split}
BALANCE_MODE=${BALANCE_MODE:-full_train_aug_balanced_split}

run_case() {
  local method=$1
  echo "[$(date)] ===== ${method} mode=${BALANCE_MODE} suffix=${SAVE_SUFFIX} start ====="
  METHOD="${method}" \
  RATIO="${RATIO}" \
  SEEDS="${SEEDS}" \
  DATA_DIR="${DATA_DIR}" \
  LR="${LR}" \
  P_CUTOFF="${P_CUTOFF}" \
  CE_CLASS_WEIGHTS="" \
  TN5000_BALANCE_MODE="${BALANCE_MODE}" \
  SAVE_SUFFIX="${SAVE_SUFFIX}" \
  EVAL_KINDS="${EVAL_KINDS}" \
  EVAL_THRESHOLD_MODES="default_0.5" \
  FORCE_RERUN="${FORCE_RERUN}" \
  AUTO_EVAL=1 \
  bash run_tn5000_ratio.sh
  echo "[$(date)] ===== ${method} mode=${BALANCE_MODE} suffix=${SAVE_SUFFIX} done ====="
}

run_case supervised
run_case fixmatch
