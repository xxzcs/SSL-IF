#!/bin/bash
# 等 GDPH SimMatchV2 跑完(ALL_DONE_SIMMATCHV2_GDPH)后, 自动启动 BUS SimMatch-ifrank 消融实验。
# 带 GPU 空闲保护; 自身可被监视。
cd /home/xiexiaozheng/Semi-supervised-learning || exit 1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true

echo "[$(date)] chain2: 等待 GDPH V2 完成 (ALL_DONE_SIMMATCHV2_GDPH)..."
for i in $(seq 1 260); do
  L=$(ls -1t logs/gdphv2_*.log 2>/dev/null | head -1)
  if [ -n "$L" ] && grep -q "ALL_DONE_SIMMATCHV2_GDPH" "$L" 2>/dev/null; then
    echo "[$(date)] chain2: 检测到 GDPH V2 完成 ($L)"; break
  fi
  sleep 180
done

# 保护: 等任何残留 train.py 退出, 避免抢 GPU
while pgrep -f "train\.py" >/dev/null 2>&1; do echo "[$(date)] chain2: 仍有 train.py 在跑, 等待..."; sleep 60; done

ILOG="logs/ifrankbus_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] chain2: 启动 BUS SimMatch-ifrank -> $ILOG"
setsid bash run_simmatch_ifrank_bus.sh > "$ILOG" 2>&1 < /dev/null &
echo "[$(date)] chain2: 已后台启动 ifrank BUS (PID $!), 日志 $ILOG"
