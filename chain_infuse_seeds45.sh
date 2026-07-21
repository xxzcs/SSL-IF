#!/bin/bash
# 链式补 seed: 等今晚 3-seed in_fuse sweep 完成后, 自动补 seed 4/5 -> 全 5-seed, 重评+重出报告。
# 不改动正在跑的过夜任务; 跳过已训好的 seed 1/2/3。
cd /home/xiexiaozheng/Semi-supervised-learning || exit 1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true
MARKER_LOG=logs/overnight_infuse_20260629_235417.log

echo "[$(date)] chain-seed45: 等待 3-seed sweep 完成 (ALL_OVERNIGHT_INFUSE_DONE @ $MARKER_LOG)..."
for i in $(seq 1 300); do
  grep -qE "ALL_OVERNIGHT_INFUSE_DONE|ABORTED_NO_S0_MODEL" "$MARKER_LOG" 2>/dev/null && break
  sleep 180
done

if grep -q "ABORTED_NO_S0_MODEL" "$MARKER_LOG" 2>/dev/null; then
  echo "[$(date)] chain-seed45: 3-seed sweep 已中止(s0 崩溃), 不补 seed。"; exit 0
fi
# 确保没有训练进程在跑再开始
while pgrep -f "train_simmatch_if" >/dev/null 2>&1; do sleep 30; done

echo "[$(date)] chain-seed45: 开始补 seed 4/5 (全部 strength)..."
export SEEDS="4 5"
./simmatch_if_bus_infuse.sh infuse_s0 infuse_s1 infuse_s2 infuse_s3 infuse_s4

REPORT="results/REPORT_infuse_5seed_$(date +%Y%m%d_%H%M%S).txt"
python3 compare_infuse.py > "$REPORT" 2>&1
echo "[$(date)] chain-seed45: 5-seed 报告 -> $REPORT"
echo "[$(date)] ALL_INFUSE_5SEED_DONE"
