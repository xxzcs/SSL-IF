#!/bin/bash
# BUS SimMatchV2 (878 标注, 仅 da1, 5 seed) 训练 + 评测 一条龙, 可断点续跑
cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true
echo "[$(date)] === BUS SimMatchV2 训练开始 (da1 × 5 seed = 5 run) ==="
bash simmatchv2_bus_878.sh || echo "[warn] 训练有报错"
echo "[$(date)] === 训练结束, 评测 ==="
bash simmatchv2_bus_878_val.sh || echo "[warn] 评测有报错"
echo "[$(date)] ALL_DONE_SIMMATCHV2_BUS"
