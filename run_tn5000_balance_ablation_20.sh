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

run_case() {
  local method=$1
  local mode=$2
  local suffix=$3
  echo "[$(date)] ===== ${method} mode=${mode} suffix=${suffix} start ====="
  METHOD="${method}" \
  RATIO="${RATIO}" \
  SEEDS="${SEEDS}" \
  DATA_DIR="${DATA_DIR}" \
  LR="${LR}" \
  P_CUTOFF="${P_CUTOFF}" \
  CE_CLASS_WEIGHTS="" \
  TN5000_BALANCE_MODE="${mode}" \
  SAVE_SUFFIX="${suffix}" \
  EVAL_KINDS="${EVAL_KINDS}" \
  EVAL_THRESHOLD_MODES="default_0.5" \
  FORCE_RERUN="${FORCE_RERUN}" \
  AUTO_EVAL=1 \
  bash run_tn5000_ratio.sh
  echo "[$(date)] ===== ${method} mode=${mode} suffix=${suffix} done ====="
}

# A: labeled only balanced
run_case supervised label_balanced a_label_bal
run_case fixmatch label_balanced a_label_bal

# B: whole train balanced, then split labeled/unlabeled while keeping
# each original sample and its oversampled copies in the same bucket.
run_case supervised full_train_balanced_split b_full_bal_split
run_case fixmatch full_train_balanced_split b_full_bal_split
