#!/bin/bash
# USB SimMatchV2 on GDPH: 0.1/0.2/0.3 (19/28/37) × DA开(da1) × 5折 × 5seed = 75 run
# 健壮版: 不用 set -e; 跳过已完成; 单个 run 失败(如偶发段错误)重试一次, 仍失败则跳过继续
declare -A CFG=(
  [19]=config/usb_cv/simmatchv2/simmatchv2_gdph_19_0.yaml
  [28]=config/usb_cv/simmatchv2/simmatchv2_gdph_28_0.yaml
  [37]=config/usb_cv/simmatchv2/simmatchv2_gdph_37_0.yaml
)

run_one () {
  local base=$1 save_name=$2 fold=$3 seed=$4 useda=$5
  local tmp; tmp=$(mktemp)
  sed "s|^save_name:.*|save_name: ${save_name}|" "$base" | \
  sed "s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|" | \
  sed "s/^fold:.*/fold: ${fold}/" | \
  sed "s/^seed:.*/seed: ${seed}/" | \
  sed "s/^split_seed:.*/split_seed: 0/" | \
  sed "s/^use_da:.*/use_da: ${useda}/" > "$tmp"
  python3 train.py --c "$tmp"
  local rc=$?
  rm -f "$tmp"
  return $rc
}

for rtag in 19 28 37; do
  base=${CFG[$rtag]}
  useda=True
  for fold in 0 1 2 3 4; do
    for seed in 1 2 3 4 5; do
      save_name="simmatchv2_gdph_${rtag}_da1_fold${fold}_${seed}"
      if [ -f "saved_models/usb_cv/${save_name}/model_best.pth" ]; then
        echo "=== skip ${save_name} (已完成) ==="; continue
      fi
      echo "=== train ${save_name} ==="
      run_one "$base" "$save_name" "$fold" "$seed" "$useda"
      if [ $? -ne 0 ]; then
        echo "[warn] ${save_name} 失败, 重试一次"; sleep 5
        rm -rf "saved_models/usb_cv/${save_name}"
        run_one "$base" "$save_name" "$fold" "$seed" "$useda" || echo "[warn] ${save_name} 重试仍失败, 跳过"
      fi
    done
  done
done
echo "=== 全部训练循环结束 ==="
