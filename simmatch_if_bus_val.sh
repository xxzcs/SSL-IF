#!/bin/bash
# SimMatch-IF BUS 测试: 按 tag 分组(各5seed)在 test 集评 best+latest
TAGS="rep_base rep_mulF rep_mulT rep_bal"
for tag in $TAGS; do
  for kind in best latest; do
    if [ "$kind" = "best" ]; then ck=model_best.pth; else ck=latest_model.pth; fi
    echo "===== eval ${tag} ${kind} ====="
    python3 eval_sup.py \
      --dataset bus --num_classes 2 \
      --summary_csv results/simmatch_if_bus_summary.csv \
      --net resnet18 --model_key ema_model \
      --data_dir ../uda_data --batch_size 16 --num_labels 878 \
      --eval_dest test \
      --lpath ../data_split/28/labeled_images_20_9.pth \
      --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/simmatch_if_bus_${tag}_*/${ck}" \
      --method_suffix ${tag}_${kind} || echo "[warn] eval ${tag} ${kind} 失败"
  done
done
echo "===== SimMatch-IF BUS 汇总 ====="; cat results/simmatch_if_bus_summary.csv
