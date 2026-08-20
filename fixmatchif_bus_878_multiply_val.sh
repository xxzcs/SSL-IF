#!/bin/bash

set -e

ifw=1.0
for corrT in 0.01 0.02 0.1 0.2; do
    # 验证 best
    python3 eval_sup.py \
        --dataset bus \
        --num_classes 2 \
        --summary_csv results/BUS_ALL_EXPERIMENTS.csv \
        --net resnet18 \
        --model_key ema_model \
        --data_dir ../uda_data \
        --batch_size 16 \
        --num_labels 878 \
        --eval_dest test \
        --lpath ../data_split/28/labeled_images_20_9.pth \
        --ulpath ../data_split/28/unlabeled_images_80_9.pth \
        --load_glob "saved_models/usb_cv/fixmatchif_bus_878_multiply_ifw${ifw}_corrT${corrT}_*/model_best.pth" \
        --method_suffix best

    # 验证 latest
        python3 eval_sup.py \
            --dataset bus \
            --num_classes 2 \
            --summary_csv results/BUS_ALL_EXPERIMENTS.csv \
            --net resnet18 \
            --model_key ema_model \
            --data_dir ../uda_data \
            --batch_size 16 \
            --num_labels 878 \
            --eval_dest test \
            --lpath ../data_split/28/labeled_images_20_9.pth \
            --ulpath ../data_split/28/unlabeled_images_80_9.pth \
            --load_glob "saved_models/usb_cv/fixmatchif_bus_878_multiply_ifw${ifw}_corrT${corrT}_*/latest_model.pth" \
            --method_suffix latest
done
