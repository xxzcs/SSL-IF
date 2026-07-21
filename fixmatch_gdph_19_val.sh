#!/bin/bash

set -e

python3 eval_sup_cv.py \
    --dataset gdph \
    --num_classes 2 \
    --summary_csv results/gdph_cv_summary.csv \
    --net resnet18 \
    --model_key model \
    --data_dir ../uda_data/GDPH \
    --batch_size 16 \
    --num_labels 192 \
    --label_ratio 0.1 \
    --eval_dest eval \
    --folds 0 1 2 3 4 \
    --load_glob_template "saved_models/usb_cv/fixmatch_gdph_fold{fold}_*/model_best.pth"
