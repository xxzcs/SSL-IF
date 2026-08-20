#!/bin/bash
cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true
echo "[$(date)] === 开始 SimMatch BUS 训练 (da1×5 + da0×5) ==="
bash simmatch_bus_878.sh || echo "[warn] 训练有报错"
echo "[$(date)] === 训练结束, 开始测试 ==="
mkdir -p results; rm -f results/BUS_ALL_EXPERIMENTS.csv
bash simmatch_bus_878_val.sh || echo "[warn] 测试有报错"
echo "[$(date)] ALL_DONE_SIMMATCH_BUS"
cat results/BUS_ALL_EXPERIMENTS.csv
