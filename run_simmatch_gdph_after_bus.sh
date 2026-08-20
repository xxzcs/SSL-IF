#!/bin/bash
# 等 BUS 跑完(其 summary csv 在最后才生成) -> 再训练+评测 GDPH, 避免抢 GPU
cd /home/xiexiaozheng/Semi-supervised-learning
source activate wssl 2>/dev/null || conda activate wssl || true

echo "[$(date)] 等待 BUS 完成 (results/BUS_ALL_EXPERIMENTS.csv) ..."
for i in $(seq 1 900); do   # 最多等 15 小时
  if [ -f results/BUS_ALL_EXPERIMENTS.csv ] && ! pgrep -f "train.py" >/dev/null 2>&1; then
    echo "[$(date)] 检测到 BUS 已完成"; break
  fi
  sleep 60
done

echo "[$(date)] === 开始 GDPH 训练 (0.1/0.2/0.3 × DA开关 × 5折) ==="
bash simmatch_gdph_all.sh || echo "[warn] GDPH 训练有报错"
echo "[$(date)] === GDPH 训练结束, 开始评测 ==="
rm -f results/simmatch_gdph_cv.csv
bash simmatch_gdph_all_val.sh || echo "[warn] GDPH 评测有报错"
echo "[$(date)] ALL_DONE_SIMMATCH_GDPH"
