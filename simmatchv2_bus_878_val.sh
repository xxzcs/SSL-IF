#!/bin/bash
# SimMatchV2 BUS 测试: test 集上评 best+latest, 聚合多seed
# 健壮版: 不用 set -e, 单个评测失败不影响其它
cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true

SUMMARY_CSV=${SUMMARY_CSV:-results/BUS_ALL_EXPERIMENTS.csv}
SAVE_PREFIX=${SAVE_PREFIX:-simmatchv2_bus_878}
DATA_TAGS=${DATA_TAGS:-da1}
METHOD_SUFFIX_PREFIX=${METHOD_SUFFIX_PREFIX:-}

for datag in $DATA_TAGS; do
    for kind in best latest; do
        if [ "$kind" = "best" ]; then ck=model_best.pth; else ck=latest_model.pth; fi
        suffix="${datag}_${kind}"
        if [ -n "$METHOD_SUFFIX_PREFIX" ]; then
            suffix="${METHOD_SUFFIX_PREFIX}_${suffix}"
        fi
        python3 eval_sup.py \
            --dataset bus --num_classes 2 \
            --summary_csv "$SUMMARY_CSV" \
            --net resnet18 --model_key ema_model \
            --data_dir ../uda_data --batch_size 16 --num_labels 878 \
            --eval_dest test \
            --lpath ../data_split/28/labeled_images_20_9.pth \
            --ulpath ../data_split/28/unlabeled_images_80_9.pth \
            --load_glob "saved_models/usb_cv/${SAVE_PREFIX}_${datag}_*/${ck}" \
            --method_suffix "$suffix"
    done
done
