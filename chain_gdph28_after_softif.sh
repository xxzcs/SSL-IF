#!/usr/bin/env bash
# Wait for the current GDPH softIF queue to finish, then continue with the
# remaining ld1 methods in order.
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
mkdir -p logs

STAMP=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG=${CHAIN_LOG:-logs/chain_gdph28_after_softif_${STAMP}.log}
NEXT_LOG=${NEXT_LOG:-logs/gdph28_ld1_after_softif_${STAMP}.log}
SUMMARY=${SUMMARY:-results/gdph28_ld1_screen_w1_full.csv}
FOLDMEAN=${FOLDMEAN:-$SUMMARY}
NEXT_METHODS=${NEXT_METHODS:-"softmatch freematch adamatch"}

exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[$(date)] wait for current softIF queue to finish"
echo "chain_log=${CHAIN_LOG}"
echo "next_log=${NEXT_LOG}"
echo "summary=${SUMMARY}"
echo "next_methods=${NEXT_METHODS}"

while [ ! -f results/ALL_DONE_GDPH28_LD1 ]; do
  echo "[$(date)] softIF queue still running..."
  sleep 300
done

echo "[$(date)] softIF queue finished; start remaining methods"
LOG="$NEXT_LOG" SUMMARY="$SUMMARY" FOLDMEAN="$FOLDMEAN" METHODS="$NEXT_METHODS" bash run_gdph28_ld1_queue.sh
echo "[$(date)] remaining ld1 queue finished"
