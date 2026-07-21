#!/bin/bash
# 全量验证: 10%(ratio19) gt-prior, 5折×5seed=25 run, 与 uniform 的 0.710 池化对比
# 健壮版: 不用 set -e; 跳过已完成(含之前 3-seed 探针); 失败重试一次。存到 *_gtprior_* 不覆盖原结果。
BASE=config/usb_cv/simmatch/simmatch_gdph_19_gtprior.yaml

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

for fold in 0 1 2 3 4; do
  for seed in 1 2 3 4 5; do
    save_name="simmatch_gdph_19_gtprior_fold${fold}_${seed}"
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
done
echo "=== 全部训练结束, 开始 25-ckpt 池化评测 ==="

mkdir -p results
for kind in best latest; do
  if [ "$kind" = "best" ]; then ck=model_best.pth; else ck=latest_model.pth; fi
  echo "===== eval gtprior ${kind} (25 ckpt 池化) ====="
  python3 eval_sup_cv.py \
    --load_glob_template "saved_models/usb_cv/simmatch_gdph_19_gtprior_fold{fold}_*/${ck}" \
    --folds 0 1 2 3 4 \
    --dataset gdph --num_classes 2 --net resnet18 --model_key ema_model \
    --data_dir ../uda_data/GDPH \
    --label_ratio 0.1 --num_labels 192 \
    --batch_size 16 --eval_dest eval \
    --summary_csv results/simmatch_gdph_gtprior_cv.csv \
    --method_suffix gtprior_${kind} || echo "[warn] eval gtprior ${kind} 失败"
done
echo "===== gtprior 汇总 ====="; cat results/simmatch_gdph_gtprior_cv.csv
echo "=== ALL_DONE_GTPRIOR_FULL ==="
