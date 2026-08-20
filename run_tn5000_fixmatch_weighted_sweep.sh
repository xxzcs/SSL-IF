#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

DATA_DIR=${DATA_DIR:-../uda_data/TN5000}
CE_CLASS_WEIGHTS=${CE_CLASS_WEIGHTS:-1.6957,0.7091}
FORCE_RERUN=${FORCE_RERUN:-0}
P_CUTOFF=${P_CUTOFF:-0.9}

run_case() {
  local ratio=$1
  local batch_size=$2
  local eval_batch_size=$3
  local lr=$4
  local train_iter=$5
  local eval_iter=$6
  local warmup_iter=$7
  local lr_drop_iter=$8

  echo "[$(date)] Queue fixmatch ratio=${ratio} bs=${batch_size} lr=${lr}"
  METHOD=fixmatch \
  RATIO="${ratio}" \
  BATCH_SIZE="${batch_size}" \
  EVAL_BATCH_SIZE="${eval_batch_size}" \
  LR="${lr}" \
  TRAIN_ITER_OVERRIDE="${train_iter}" \
  EVAL_ITER_OVERRIDE="${eval_iter}" \
  LOG_ITER_OVERRIDE="${eval_iter}" \
  WARMUP_ITER_OVERRIDE="${warmup_iter}" \
  LR_DROP_ITER_OVERRIDE="${lr_drop_iter}" \
  CE_CLASS_WEIGHTS="${CE_CLASS_WEIGHTS}" \
  P_CUTOFF="${P_CUTOFF}" \
  FORCE_RERUN="${FORCE_RERUN}" \
  DATA_DIR="${DATA_DIR}" \
  bash run_tn5000_ratio.sh
}

# single setting: bs=8, uratio=30, lr=0.002
run_case 10 8 16 0.002 2200 44 88 "660 1320 1980"
run_case 15 8 16 0.002 3300 66 132 "990 1980 2970"
run_case 20 8 16 0.002 4400 88 176 "1320 2640 3960"
run_case 30 8 16 0.002 6600 132 264 "1980 3960 5940"
