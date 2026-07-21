#!/bin/bash
# USB 框架内跑 SimMatch (GDPH, 0.1 标注), 5 折 × 5 seed, 与 fixmatch_gdph_19.sh 同协议
set -e

BASE_CONFIG="config/usb_cv/simmatch/simmatch_gdph_19_0.yaml"

for fold in 0 1 2 3 4;
do
    for run in 1 2 3 4 5;
    do
        seed=${run}
        save_name="simmatch_gdph_19_fold${fold}_${run}"
        load_path="./saved_models/usb_cv/${save_name}/latest_model.pth"
        temp_config=$(mktemp)

        sed "s/^save_name:.*/save_name: ${save_name}/" "${BASE_CONFIG}" | \
        sed "s|^load_path:.*|load_path: ${load_path}|" | \
        sed "s/^fold:.*/fold: ${fold}/" | \
        sed "s/^seed:.*/seed: ${seed}/" > "${temp_config}"

        python3 train.py --c "${temp_config}"
        rm -f "${temp_config}"
    done
done
