#!/bin/bash
# SimMatch-ifrank BUS (878, da1, lambda∈{0,1}×5seed=10run) 训练+评测 一条龙, 可断点续跑
cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true
echo "[$(date)] === SimMatch-ifrank BUS 训练开始 (lambda 0.0/1.0 × 5 seed = 10 run) ==="
bash simmatch_ifrank_bus.sh || echo "[warn] 训练有报错"
echo "[$(date)] === 训练结束, 评测 ==="
bash simmatch_ifrank_bus_val.sh || echo "[warn] 评测有报错"
echo "[$(date)] ALL_DONE_SIMMATCH_IFRANK_BUS"
