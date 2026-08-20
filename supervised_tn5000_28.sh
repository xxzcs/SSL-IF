#!/bin/bash

set -e

BASE_CONFIG="config/usb_cv/supervised/supervised_tn5000_28_0.yaml"
DATA_DIR="${DATA_DIR:-../uda_data/TN5000}"

for run in 1 2 3 4 5;
do
    seed=${run}
    save_name="supervised_tn5000_28_${run}"
    load_path="./saved_models/usb_cv/${save_name}/latest_model.pth"
    temp_config=$(mktemp)

    sed "s/^save_name:.*/save_name: ${save_name}/" "${BASE_CONFIG}" | \
    sed "s|^load_path:.*|load_path: ${load_path}|" | \
    sed "s|^data_dir:.*|data_dir: ${DATA_DIR}|" | \
    sed "s/^seed:.*/seed: ${seed}/" > "${temp_config}"

    python3 train.py --c "${temp_config}"
    rm -f "${temp_config}"
done
