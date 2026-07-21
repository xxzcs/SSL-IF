#!/bin/bash
# USB SimMatchV2 (BUS, 878 标注), 仅 DA开(da1) × 5 个 seed = 5 run
# 健壮版: 不用 set -e; 跳过已完成(model_best.pth 存在); 单个 run 失败重试一次, 仍失败则跳过继续
BASE=config/usb_cv/simmatchv2/simmatchv2_bus_878_0.yaml

run_one () {
  local save_name=$1 seed=$2 useda=$3
  local tmp; tmp=$(mktemp)
  sed "s|^save_name:.*|save_name: ${save_name}|" "$BASE" | \
  sed "s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|" | \
  sed "s/^seed:.*/seed: ${seed}/" | \
  sed "s/^use_da:.*/use_da: ${useda}/" > "$tmp"
  python3 train.py --c "$tmp"
  local rc=$?
  rm -f "$tmp"
  return $rc
}

useda=True; datag=da1
for i in 1 2 3 4 5; do
  save_name="simmatchv2_bus_878_${datag}_${i}"
  if [ -f "saved_models/usb_cv/${save_name}/model_best.pth" ]; then
    echo "=== skip ${save_name} (已完成) ==="; continue
  fi
  echo "=== train ${save_name} ==="
  run_one "$save_name" "$i" "$useda"
  if [ $? -ne 0 ]; then
    echo "[warn] ${save_name} 失败, 重试一次"; sleep 5
    rm -rf "saved_models/usb_cv/${save_name}"
    run_one "$save_name" "$i" "$useda" || echo "[warn] ${save_name} 重试仍失败, 跳过"
  fi
done
echo "=== BUS SimMatchV2 训练循环结束 ==="
