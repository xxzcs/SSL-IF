#!/bin/bash
# 过夜编排: SimMatch-IF in_fuse strength 扫描 (0/1/2/3/4, 各 3 seed)。
#   Phase A: 先跑 s0 (sanity). 若 s0 训练崩溃(无模型产出)=> in_fuse 代码有问题, 终止 sweep。
#   Phase B: s0 正常 => 跑 s1 s2 s3 s4。
#   最后用 compare_infuse.py 生成对比报告。
# 不做质量硬门槛(s0 数值是否=基线留待人工判读), 只在"崩溃"时止损, 以最大化过夜产出。
cd /home/xiexiaozheng/Semi-supervised-learning || exit 1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true
export SEEDS="1 2 3"

TS=$(date +%Y%m%d_%H%M%S)
RUNLOG="logs/overnight_infuse_${TS}.log"
REPORT="results/REPORT_infuse_${TS}.txt"

{
  echo "[$(date)] ===== 过夜 in_fuse 开始 (SEEDS=$SEEDS) ====="

  echo "[$(date)] --- Phase A: s0 (sanity) ---"
  ./simmatch_if_bus_infuse.sh infuse_s0

  if ! ls saved_models/usb_cv/simmatch_if_bus_infuse_s0_*/model_best.pth >/dev/null 2>&1; then
    echo "[$(date)] [FATAL] s0 未产出任何 model_best.pth => in_fuse 训练崩溃, 终止 sweep。"
    echo "ABORTED_NO_S0_MODEL"
    python3 compare_infuse.py > "$REPORT" 2>&1 || true
    echo "[$(date)] 报告(失败) -> $REPORT"
    exit 1
  fi
  echo "[$(date)] s0 已产出模型, 继续 sweep。"

  echo "[$(date)] --- Phase B: s1 s2 s3 s4 ---"
  ./simmatch_if_bus_infuse.sh infuse_s1 infuse_s2 infuse_s3 infuse_s4

  echo "[$(date)] --- 生成对比报告 ---"
  python3 compare_infuse.py > "$REPORT" 2>&1
  echo "[$(date)] 报告 -> $REPORT"
  echo "[$(date)] ===== ALL_OVERNIGHT_INFUSE_DONE ====="
} 2>&1 | tee -a "$RUNLOG"
echo "RUNLOG=$RUNLOG"
echo "REPORT=$REPORT"
