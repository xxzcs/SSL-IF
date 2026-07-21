#!/bin/bash
# USB 框架内跑 SimMatch (BUS, 878 标注), DA 开/关 × 5 个 seed
# 训练预算 (5500 iter / 220 warmup / lr 0.0046875 / 默认cosine) 与你之前的 BUS 方法一致
set -e

BASE=config/usb_cv/simmatch/simmatch_bus_878_0.yaml

for da in 1 0; do
    if [ "$da" = "1" ]; then useda=True; datag=da1; else useda=False; datag=da0; fi
    for i in 1 2 3 4 5; do
        seed=$i
        save_name="simmatch_bus_878_${datag}_${i}"
        tmp=$(mktemp)
        sed "s|^save_name:.*|save_name: ${save_name}|" "$BASE" | \
        sed "s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|" | \
        sed "s/^seed:.*/seed: ${seed}/" | \
        sed "s/^use_da:.*/use_da: ${useda}/" > "$tmp"

        python3 train.py --c "$tmp"
        rm -f "$tmp"
    done
done
