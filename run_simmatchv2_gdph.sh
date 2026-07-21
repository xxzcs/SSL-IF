#!/bin/bash
# GDPH SimMatchV2 (19/28/37, 仅 da1, 5折×5seed=75 run) 训练 + 评测 一条龙, 可断点续跑
cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true
echo "[$(date)] === GDPH SimMatchV2 训练开始 (19/28/37 × da1 × 5折 × 5seed = 75 run) ==="
bash simmatchv2_gdph_all.sh || echo "[warn] 训练有报错"
echo "[$(date)] === 训练结束, 评测 ==="
bash simmatchv2_gdph_all_val.sh || echo "[warn] 评测有报错"
echo "[$(date)] ALL_DONE_SIMMATCHV2_GDPH"
