#!/bin/bash
# SimMatchV2 BUS 测试: test 集上评 best+latest, 仅 DA开(da1), 聚合多seed
# 健壮版: 不用 set -e, 单个评测失败不影响其它
for datag in da1; do
    for kind in best latest; do
        if [ "$kind" = "best" ]; then ck=model_best.pth; else ck=latest_model.pth; fi
        python3 eval_sup.py \
            --dataset bus --num_classes 2 \
            --summary_csv results/simmatchv2_bus_summary.csv \
            --net resnet18 --model_key ema_model \
            --data_dir ../uda_data --batch_size 16 --num_labels 878 \
            --eval_dest test \
            --lpath ../data_split/28/labeled_images_20_9.pth \
            --ulpath ../data_split/28/unlabeled_images_80_9.pth \
            --load_glob "saved_models/usb_cv/simmatchv2_bus_878_${datag}_*/${ck}" \
            --method_suffix ${datag}_${kind}
    done
done
