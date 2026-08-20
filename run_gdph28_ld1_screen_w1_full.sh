#!/usr/bin/env bash
# GDPH 20% layer_decay=1.0 follow-up queue:
# finish fixmatch_ifcf_soft_gdph_28_screen_w1 as 5 folds x 5 seeds first.
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/gdph28_ld1_screen_w1_full_${STAMP}.log}
SUMMARY=${SUMMARY:-results/gdph28_ld1_screen_w1_full.csv}
FOLDMEAN=${FOLDMEAN:-$SUMMARY}
METHODS=${METHODS:-"if_soft_w1_screen"}
SCREEN_SEEDS=${SCREEN_SEEDS:-"1 2 3 4 5"}

LOG="$LOG" SUMMARY="$SUMMARY" FOLDMEAN="$FOLDMEAN" METHODS="$METHODS" SCREEN_SEEDS="$SCREEN_SEEDS" \
  exec bash run_gdph28_ld1_queue.sh
