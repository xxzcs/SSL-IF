#!/bin/bash
# 等 BUS SimMatchV2 跑完(ALL_DONE_SIMMATCHV2_BUS)后, 自动启动 GDPH SimMatchV2 (75 run) 训练+评测。
# 带 GPU 空闲保护; 自身可被监视。
cd /home/xiexiaozheng/Semi-supervised-learning || exit 1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true

echo "[$(date)] chain: 等待 BUS 完成 (ALL_DONE_SIMMATCHV2_BUS)..."
for i in $(seq 1 220); do
  L=$(ls -1t logs/resume_bus_*.log 2>/dev/null | head -1)
  if [ -n "$L" ] && grep -q "ALL_DONE_SIMMATCHV2_BUS" "$L" 2>/dev/null; then
    echo "[$(date)] chain: 检测到 BUS 完成 ($L)"; break
  fi
  sleep 180
done

# 保护: 等任何残留 train.py 退出, 避免抢 GPU
while pgrep -f "train\.py" >/dev/null 2>&1; do echo "[$(date)] chain: 仍有 train.py 在跑, 等待..."; sleep 60; done

GLOG="logs/gdphv2_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] chain: 启动 GDPH SimMatchV2 -> $GLOG"
setsid bash run_simmatchv2_gdph.sh > "$GLOG" 2>&1 < /dev/null &
echo "[$(date)] chain: 已后台启动 GDPH V2 (PID $!), 日志 $GLOG"
