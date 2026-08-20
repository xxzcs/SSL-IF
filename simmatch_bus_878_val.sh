#!/bin/bash
# SimMatch BUS 测试: 在 test 集上评测 best 和 latest, DA 开/关 各一组, 聚合多 seed -> mean±std
set -e

for datag in da1 da0; do
    # best
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
        --load_glob "saved_models/usb_cv/simmatch_bus_878_${datag}_*/model_best.pth" \
        --method_suffix ${datag}_best

    # latest
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
        --load_glob "saved_models/usb_cv/simmatch_bus_878_${datag}_*/latest_model.pth" \
        --method_suffix ${datag}_latest
done
