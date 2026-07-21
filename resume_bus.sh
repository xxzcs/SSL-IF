#!/bin/bash
# 续跑 BUS SimMatchV2 (da1, 5 seed), 自动跳过已完成的 run。
# 用法: resume-bus          # 后台续跑(防重复启动) + 写日志
#       resume-bus status   # 只看当前进度, 不启动
#       resume-bus tail     # 实时跟踪日志
PROJ=/home/xiexiaozheng/Semi-supervised-learning
LOGDIR=$PROJ/logs

latest_log () { ls -1t "$LOGDIR"/resume_bus_*.log 2>/dev/null | head -1; }
is_running () { pgrep -af "run_simmatchv2_bus.sh|simmatchv2_bus_878.sh|train\.py" >/dev/null 2>&1; }

case "$1" in
  status)
    if is_running; then echo "✅ 训练正在运行:"; pgrep -af "train\.py|run_simmatchv2_bus" | head -3
    else echo "⏹️  当前没有 BUS 训练在跑"; fi
    L=$(latest_log); [ -n "$L" ] && { echo "--- 最近进度 ($L) ---"; grep "=== train" "$L" | tail -2; grep -E "iteration|ALL_DONE" "$L" | tail -1; }
    exit 0 ;;
  tail)
    L=$(latest_log); [ -n "$L" ] && tail -f "$L" || echo "还没有日志文件"; exit 0 ;;
esac

if is_running; then
  echo "⚠️  已有训练在跑, 不重复启动。看进度: resume-bus status"
  pgrep -af "train\.py|run_simmatchv2_bus" | head -3
  exit 0
fi

mkdir -p "$LOGDIR"
LOG=$LOGDIR/resume_bus_$(date +%Y%m%d_%H%M%S).log
cd "$PROJ" || exit 1
setsid bash run_simmatchv2_bus.sh > "$LOG" 2>&1 < /dev/null &
echo "🚀 已后台启动 BUS SimMatchV2 (PID $!), 自动跳过已完成的 run"
echo "   日志: $LOG"
echo "   看进度: resume-bus status   |   实时日志: resume-bus tail"
