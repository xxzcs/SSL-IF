#!/bin/bash
# SimMatch-IF BUS (mode{add,replace}×weight{0,100}×5seed=20run) 训练+评测 一条龙, 可断点续跑
cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true
echo "[$(date)] === SimMatch-IF BUS 训练开始 (add/replace × w0/100 × 5seed = 20run) ==="
bash simmatch_if_bus.sh || echo "[warn] 训练有报错"
echo "[$(date)] === 训练结束, 评测 ==="
bash simmatch_if_bus_val.sh || echo "[warn] 评测有报错"
echo "[$(date)] ALL_DONE_SIMMATCH_IF_BUS"
