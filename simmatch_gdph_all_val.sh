#!/bin/bash
# SimMatch GDPH 评测: 每比例(DA开) 评 best 和 latest
# 每折匹配 5 个 seed, 5 折共 25 个 checkpoint 直接聚合 -> mean±std (5次5折)
set -e

declare -A NL=( [19]=192 [28]=384 [37]=577 )
declare -A LR=( [19]=0.1 [28]=0.2 [37]=0.3 )

mkdir -p results
for rtag in 19 28 37; do
  for da in 1; do
    for kind in best latest; do
      if [ "$kind" = "best" ]; then ck=model_best.pth; else ck=latest_model.pth; fi
      echo "===== eval ${rtag} da${da} ${kind} ====="
      python3 eval_sup_cv.py \
        --load_glob_template "saved_models/usb_cv/simmatch_gdph_${rtag}_da${da}_fold{fold}_*/${ck}" \
        --folds 0 1 2 3 4 \
        --dataset gdph --num_classes 2 --net resnet18 --model_key ema_model \
        --data_dir ../uda_data/GDPH \
        --label_ratio ${LR[$rtag]} --num_labels ${NL[$rtag]} \
        --batch_size 16 --eval_dest eval \
        --summary_csv results/simmatch_gdph_cv.csv \
        --method_suffix ${kind} || echo "[warn] eval ${rtag} da${da} ${kind} 失败"
    done
  done
done
echo "===== GDPH 最终汇总 ====="; cat results/simmatch_gdph_cv.csv
