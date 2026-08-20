#!/bin/bash
# USB SimMatchV2 (BUS, 878 标注), 仅 DA开(da1) × 5 个 seed = 5 run
# 健壮版: 不用 set -e; 跳过已完成(model_best.pth 存在); 单个 run 失败重试一次, 仍失败则跳过继续
cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true

BASE=${BASE:-config/usb_cv/simmatchv2/simmatchv2_bus_878_0.yaml}
SAVE_PREFIX=${SAVE_PREFIX:-simmatchv2_bus_878}
USE_DA=${USE_DA:-True}
DATA_TAG=${DATA_TAG:-da1}
LAYER_DECAY=${LAYER_DECAY:-}
SEEDS=${SEEDS:-"1 2 3 4 5"}
TOPN=${TOPN:-}
QUEUE_K=${QUEUE_K:-}
LAMBDA_EE=${LAMBDA_EE:-}
LAMBDA_NE=${LAMBDA_NE:-}

run_one () {
  local save_name=$1 seed=$2 useda=$3
  local tmp; tmp=$(mktemp)
  sed "s|^save_name:.*|save_name: ${save_name}|" "$BASE" | \
  sed "s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|" | \
  sed "s/^seed:.*/seed: ${seed}/" | \
  sed "s/^use_da:.*/use_da: ${useda}/" > "$tmp"
  if [ -n "$LAYER_DECAY" ]; then
    sed -i "s/^layer_decay:.*/layer_decay: ${LAYER_DECAY}/" "$tmp"
  fi
  if [ -n "$TOPN" ]; then
    sed -i "s/^topn:.*/topn: ${TOPN}/" "$tmp"
  fi
  if [ -n "$QUEUE_K" ]; then
    sed -i "s/^K:.*/K: ${QUEUE_K}/" "$tmp"
  fi
  if [ -n "$LAMBDA_EE" ]; then
    sed -i "s/^lambda_ee:.*/lambda_ee: ${LAMBDA_EE}/" "$tmp"
  fi
  if [ -n "$LAMBDA_NE" ]; then
    sed -i "s/^lambda_ne:.*/lambda_ne: ${LAMBDA_NE}/" "$tmp"
  fi
  python3 train.py --c "$tmp"
  local rc=$?
  rm -f "$tmp"
  return $rc
}

for i in $SEEDS; do
  save_name="${SAVE_PREFIX}_${DATA_TAG}_${i}"
  if [ -f "saved_models/usb_cv/${save_name}/model_best.pth" ]; then
    echo "=== skip ${save_name} (已完成) ==="; continue
  fi
  echo "=== train ${save_name} ==="
  run_one "$save_name" "$i" "$USE_DA"
  if [ $? -ne 0 ]; then
    echo "[warn] ${save_name} 失败, 重试一次"; sleep 5
    rm -rf "saved_models/usb_cv/${save_name}"
    run_one "$save_name" "$i" "$USE_DA" || echo "[warn] ${save_name} 重试仍失败, 跳过"
  fi
done
echo "=== BUS SimMatchV2 训练循环结束 ==="
