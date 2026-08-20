#!/bin/bash
# SimMatch-ifrank BUS 测试: 按 ifrank_lambda 分组(各5seed)在 test 集评 best+latest
# 健壮版: 不用 set -e, 单个评测失败不影响其它
for lam in 0.0 1.0; do
  for kind in best latest; do
    if [ "$kind" = "best" ]; then ck=model_best.pth; else ck=latest_model.pth; fi
    echo "===== eval lambda=${lam} ${kind} ====="
    python3 eval_sup.py \
      --dataset bus --num_classes 2 \
      --summary_csv results/BUS_ALL_EXPERIMENTS.csv \
      --net resnet18 --model_key ema_model \
      --data_dir ../uda_data --batch_size 16 --num_labels 878 \
      --eval_dest test \
      --lpath ../data_split/28/labeled_images_20_9.pth \
      --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/simmatch_ifrank_bus_878_l${lam}_*/${ck}" \
      --method_suffix l${lam}_${kind} || echo "[warn] eval lambda=${lam} ${kind} 失败"
  done
done
echo "===== SimMatch-ifrank BUS 汇总 ====="; cat results/BUS_ALL_EXPERIMENTS.csv
