#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

SEEDS=${SEEDS:-"1 2 3"}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/bus_ld1_ifcf_param_probe_${STAMP}.log}
SUMMARY=${SUMMARY:-results/bus_ld1_ifcf_param_probe_latest.csv}

mkdir -p logs results
exec > >(tee -a "$LOG") 2>&1

wait_until_idle() {
  while pgrep -f "python .*train_ifcf.py|python .*train.py|python3 .*train_ifcf.py|python3 .*train.py" >/dev/null 2>&1; do
    echo "[$(date)] another training job is still running; sleep 60s"
    sleep 60
  done
}

run_one() {
  local method=$1
  local corrt=$2
  local warmup=$3
  local weight=$4

  echo "[$(date)] ===== queue ${method} corrT=${corrt} warmup=${warmup} weight=${weight} ====="
  wait_until_idle
  SEEDS="$SEEDS" \
  METHOD="$method" \
  CORRT="$corrt" \
  WARMUP="$warmup" \
  WEIGHT="$weight" \
  IF_TARGET="hard" \
  STRONG_IF="True" \
  SUMMARY="$SUMMARY" \
  bash run_bus_ifcf_ld1_method_targeted.sh
}

echo "[$(date)] ===== BUS ld1 IFCF parameter probe queue start ====="
echo "seeds=${SEEDS}"
echo "summary=${SUMMARY}"
echo "log=${LOG}"

run_one freematch_ifcf 0.5 10 1.0
run_one freematch_ifcf 0.9 5 1.0
run_one refixmatch_ifcf 0.5 10 1.0
run_one refixmatch_ifcf 0.9 5 1.0

touch results/ALL_DONE_BUS_LD1_IFCF_PARAM_PROBE
echo "[$(date)] ===== BUS ld1 IFCF parameter probe queue done ====="
