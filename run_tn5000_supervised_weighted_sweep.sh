#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

DATA_DIR=${DATA_DIR:-../uda_data/TN5000}
CE_CLASS_WEIGHTS=${CE_CLASS_WEIGHTS:-1.6957,0.7091}
FORCE_RERUN=${FORCE_RERUN:-0}

run_case() {
  local ratio=$1
  local batch_size=$2
  local eval_batch_size=$3
  local lr=$4
  local train_iter=$5
  local eval_iter=$6
  local warmup_iter=$7
  local lr_drop_iter=$8

  echo "[$(date)] Queue supervised ratio=${ratio} bs=${batch_size} lr=${lr}"
  METHOD=supervised \
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
  FORCE_RERUN="${FORCE_RERUN}" \
  DATA_DIR="${DATA_DIR}" \
  bash run_tn5000_ratio.sh
}

# setting A: bs=8, lr=0.002
run_case 10 8 16 0.002 2200 44 88 "660 1320 1980"
run_case 15 8 16 0.002 3300 66 132 "990 1980 2970"
run_case 20 8 16 0.002 4400 88 176 "1320 2640 3960"
run_case 30 8 16 0.002 6600 132 264 "1980 3960 5940"
run_case all 8 16 0.002 22000 440 880 "6600 13200 19800"

# setting B: bs=16, lr=0.004
# Keep 50 epochs by halving the step-based schedule from setting A.
run_case 10 16 32 0.004 1100 22 44 "330 660 990"
run_case 15 16 32 0.004 1650 33 66 "495 990 1485"
run_case 20 16 32 0.004 2200 44 88 "660 1320 1980"
run_case 30 16 32 0.004 3300 66 132 "990 1980 2970"
run_case all 16 32 0.004 11000 220 440 "3300 6600 9900"
