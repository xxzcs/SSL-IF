#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/chain_after_current_freematch_gdph_${STAMP}.log}

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== Wait current task then run FreeMatch GDPH ld1 ====="
echo "log=${LOG}"

while pgrep -f "run_fixmatch_rankmatch_bus.sh|config/_rankmatch_bus_fixmatch_rankmatch_bus_|fixmatch_rankmatch_bus_s[1-5]" >/dev/null 2>&1; do
  echo "[$(date)] current BUS RankMatch task still running; sleep 60s"
  sleep 60
done

echo "[$(date)] current task cleared; starting FreeMatch GDPH ld1"
bash run_freematch_gdph_ld1.sh
