#!/bin/bash

set -e

python3 eval_sup.py \
    --dataset tn5000 \
    --num_classes 2 \
    --summary_csv results/tn5000_summary.csv \
    --net resnet18 \
    --model_key model \
    --data_dir ../uda_data/TN5000 \
    --batch_size 16 \
    --num_labels 1050 \
    --label_ratio 0.3 \
    --eval_dest eval \
    --method_suffix best \
    --load_glob "saved_models/usb_cv/supervised_tn5000_37_*/model_best.pth"

python3 eval_sup.py \
    --dataset tn5000 \
    --num_classes 2 \
    --summary_csv results/tn5000_summary.csv \
    --net resnet18 \
    --model_key model \
    --data_dir ../uda_data/TN5000 \
    --batch_size 16 \
    --num_labels 1050 \
    --label_ratio 0.3 \
    --eval_dest eval \
    --method_suffix latest \
    --load_glob "saved_models/usb_cv/supervised_tn5000_37_*/latest_model.pth"
