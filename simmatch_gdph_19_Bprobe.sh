#!/bin/bash
# B 方案探针: gt-prior + in_loss_ratio 5->2 + p_cutoff 0.9->0.95
# 4 个代表性 seed, 看能否同时压住两种崩溃方向并抬 AUC
# 健壮版: 不用 set -e; 跳过已完成; 失败重试一次。存到 *_Bprobe_* 不覆盖。
BASE=config/usb_cv/simmatch/simmatch_gdph_19_Bprobe.yaml

# fold seed
RUNS=( "1 3" "0 1" "1 2" "3 4" )

run_one () {
  local fold=$1 seed=$2 save_name=$3
  local tmp; tmp=$(mktemp)
  sed "s|^save_name:.*|save_name: ${save_name}|" "$BASE" | \
  sed "s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|" | \
  sed "s/^fold:.*/fold: ${fold}/" | \
  sed "s/^seed:.*/seed: ${seed}/" | \
  sed "s/^split_seed:.*/split_seed: 0/" > "$tmp"
  python3 train.py --c "$tmp"; local rc=$?
  rm -f "$tmp"; return $rc
}

for fs in "${RUNS[@]}"; do
  set -- $fs; fold=$1; seed=$2
  save_name="simmatch_gdph_19_Bprobe_fold${fold}_${seed}"
  if [ -f "saved_models/usb_cv/${save_name}/model_best.pth" ]; then
    echo "=== skip ${save_name} (已完成) ==="; continue
  fi
  echo "=== train ${save_name} (fold${fold} seed${seed}) ==="
  run_one "$fold" "$seed" "$save_name"
  if [ $? -ne 0 ]; then
    echo "[warn] ${save_name} 失败, 重试一次"; sleep 5
    rm -rf "saved_models/usb_cv/${save_name}"
    run_one "$fold" "$seed" "$save_name" || echo "[warn] ${save_name} 重试仍失败, 跳过"
  fi
done
echo "=== 训练结束, 开始评测 ==="

for fs in "${RUNS[@]}"; do
  set -- $fs; fold=$1; seed=$2
  save_name="simmatch_gdph_19_Bprobe_fold${fold}_${seed}"
  for ck in model_best.pth latest_model.pth; do
    [ -f "saved_models/usb_cv/${save_name}/${ck}" ] || continue
    echo "----- eval ${save_name} ${ck} -----"
    python3 eval_sup_cv.py \
      --load_glob_template "saved_models/usb_cv/${save_name}/${ck}" \
      --folds ${fold} \
      --dataset gdph --num_classes 2 --net resnet18 --model_key ema_model \
      --data_dir ../uda_data/GDPH \
      --label_ratio 0.1 --num_labels 192 \
      --batch_size 16 --eval_dest eval || echo "[warn] eval ${save_name} ${ck} 失败"
  done
done
echo "=== ALL_DONE_BPROBE ==="
