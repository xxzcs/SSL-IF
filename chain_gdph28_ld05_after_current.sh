#!/usr/bin/env bash
# Wait for the currently running GDPH 20% queue to finish, then start the ld05 queue.
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs

STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=${CHAIN_LOG:-logs/chain_gdph28_ld05_after_current_${STAMP}.log}
LD05_LOG=${LD05_LOG:-logs/gdph28_ld05_full_${STAMP}.log}
LD05_SUMMARY=${LD05_SUMMARY:-results/gdph28_ld05_latest.csv}
LD05_METHODS=${LD05_METHODS:-"fixmatch fixmatch_if flexmatch freematch refixmatch softmatch adamatch"}

exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[$(date)] wait for current non-ld05 GDPH 20% training to finish"
echo "chain_log=${CHAIN_LOG}"
echo "ld05_log=${LD05_LOG}"
echo "ld05_summary=${LD05_SUMMARY}"
echo "ld05_methods=${LD05_METHODS}"

while true; do
  running=$(pgrep -af 'python3 .*train(_ifcf)?\.py --c config/_gdph28_.*_gdph_28_fold' | grep -v '_ld05_' || true)
  if [ -z "$running" ]; then
    break
  fi
  echo "[$(date)] current GDPH queue still running:"
  echo "$running" | head -20
  sleep 300
done

echo "[$(date)] current queue clear; start layer_decay=0.5 queue"
LOG="$LD05_LOG" SUMMARY="$LD05_SUMMARY" METHODS="$LD05_METHODS" bash run_gdph28_ld05_queue.sh
echo "[$(date)] ld05 queue finished"
