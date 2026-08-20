#!/usr/bin/env bash
# Wait until the current non-ld05 SoftMatch GDPH 20% runs finish, skip AdaMatch,
# then start only FixMatch and FixMatch+IF with layer_decay=0.5.
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs results

STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=${CHAIN_LOG:-logs/chain_ld05_fixmatch_after_softmatch_${STAMP}.log}
LD05_LOG=${LD05_LOG:-logs/gdph28_ld05_fixmatch_first_${STAMP}.log}
LD05_SUMMARY=${LD05_SUMMARY:-results/gdph28_ld05_fixmatch_first_latest.csv}

exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[$(date)] wait for softmatch_gdph_28 RUN_DONE count to reach 25"
echo "chain_log=${CHAIN_LOG}"
echo "ld05_log=${LD05_LOG}"
echo "ld05_summary=${LD05_SUMMARY}"

while true; do
  done_count=$(find saved_models/usb_cv -maxdepth 2 -name RUN_DONE | grep -c 'softmatch_gdph_28_fold' || true)
  echo "[$(date)] softmatch RUN_DONE: ${done_count}/25"
  if [ "$done_count" -ge 25 ]; then
    break
  fi
  sleep 120
done

echo "[$(date)] SoftMatch complete; stop old priority queue before AdaMatch"
pkill -f 'bash run_gdph28_priority_queue.sh' || true
pkill -f 'python3 train.py --c config/_gdph28_adamatch_gdph_28_fold' || true
sleep 10

echo "[$(date)] start ld05 FixMatch, hard FixMatch+IF, and soft FixMatch+IF"
LOG="$LD05_LOG" \
SUMMARY="$LD05_SUMMARY" \
METHODS="fixmatch fixmatch_if fixmatch_if_soft" \
bash run_gdph28_ld05_queue.sh

echo "[$(date)] ld05 FixMatch-first queue finished"
