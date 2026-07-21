#!/bin/bash
cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true
echo "[$(date)] === GDPH 5次5折 训练开始 (3比例 × DA开关 × 5折 × 5seed = 150 run) ==="
bash simmatch_gdph_all.sh || echo "[warn] 训练有报错"
echo "[$(date)] === 训练结束, 评测 ==="
rm -f results/simmatch_gdph_cv.csv
bash simmatch_gdph_all_val.sh || echo "[warn] 评测有报错"
echo "[$(date)] ALL_DONE_SIMMATCH_GDPH"
