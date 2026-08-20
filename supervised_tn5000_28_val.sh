#!/bin/bash

set -e

DATA_DIR="${DATA_DIR:-../uda_data/TN5000}"

python3 eval_sup.py \
    --dataset tn5000 \
    --num_classes 2 \
    --summary_csv results/tn5000_summary.csv \
    --net resnet18 \
    --model_key model \
    --data_dir "${DATA_DIR}" \
    --batch_size 16 \
    --num_labels 700 \
    --label_ratio 0.2 \
    --eval_dest eval \
    --method_suffix best \
    --load_glob "saved_models/usb_cv/supervised_tn5000_28_*/model_best.pth"

python3 eval_sup.py \
    --dataset tn5000 \
    --num_classes 2 \
    --summary_csv results/tn5000_summary.csv \
    --net resnet18 \
    --model_key model \
    --data_dir "${DATA_DIR}" \
    --batch_size 16 \
    --num_labels 700 \
    --label_ratio 0.2 \
    --eval_dest eval \
    --method_suffix latest \
    --load_glob "saved_models/usb_cv/supervised_tn5000_28_*/latest_model.pth"
