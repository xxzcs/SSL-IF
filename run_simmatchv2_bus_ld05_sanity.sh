#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true

SUMMARY=${SUMMARY:-results/simmatchv2_bus_ld05_sanity.csv}
mkdir -p results logs

run_case() {
  local case_name=$1
  local save_prefix=$2
  local use_da=$3
  local lambda_ee=$4
  local lambda_ne=$5
  local topn=$6
  local queue_k=$7

  echo "[$(date)] ===== sanity case ${case_name} start ====="
  BASE=config/usb_cv/simmatchv2/simmatchv2_bus_878_0.yaml \
  SAVE_PREFIX="${save_prefix}" \
  DATA_TAG=da1 \
  SEEDS=1 \
  USE_DA="${use_da}" \
  LAYER_DECAY=0.5 \
  LAMBDA_EE="${lambda_ee}" \
  LAMBDA_NE="${lambda_ne}" \
  TOPN="${topn}" \
  QUEUE_K="${queue_k}" \
  bash simmatchv2_bus_878.sh

  for kind in best latest; do
    if [ "$kind" = "best" ]; then
      ck=model_best.pth
    else
      ck=latest_model.pth
    fi
    python3 eval_sup.py \
      --dataset bus --num_classes 2 \
      --summary_csv "$SUMMARY" \
      --net resnet18 --model_key ema_model \
      --data_dir ../uda_data --batch_size 16 --num_labels 878 \
      --eval_dest test \
      --lpath ../data_split/28/labeled_images_20_9.pth \
      --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/${save_prefix}_da1_1/${ck}" \
      --method_suffix "${case_name}_${kind}"
  done
  echo "[$(date)] ===== sanity case ${case_name} done ====="
}

run_case da_off simmatchv2_bus_ld05_sanity_daoff False 5.0 5.0 128 3360
run_case lambda1 simmatchv2_bus_ld05_sanity_l1 True 1.0 1.0 128 3360
run_case smallgraph simmatchv2_bus_ld05_sanity_smallg True 5.0 5.0 64 1680

