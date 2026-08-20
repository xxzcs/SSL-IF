#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

METHOD=${METHOD:-supervised}
RATIO=${RATIO:-20}
SEEDS=${SEEDS:-"1 2 3 4 5"}
DATA_DIR=${DATA_DIR:-../uda_data/TN5000}
FORCE_RERUN=${FORCE_RERUN:-0}
BEST_METRIC=${BEST_METRIC:-balanced_acc}
CE_CLASS_WEIGHTS=${CE_CLASS_WEIGHTS:-}
AUTO_EVAL=${AUTO_EVAL:-1}
EVAL_ONLY=${EVAL_ONLY:-0}
EVAL_KINDS=${EVAL_KINDS:-latest}
TN5000_BALANCE_MODE=${TN5000_BALANCE_MODE:-none}
SAVE_SUFFIX=${SAVE_SUFFIX:-}
EVAL_THRESHOLD_MODES=${EVAL_THRESHOLD_MODES:-default_0.5,val_max_bacc}
EXTRA_CONFIG_APPEND=${EXTRA_CONFIG_APPEND:-}

case "$RATIO" in
  10)
    NUM_LABELS=350
    LABEL_RATIO=0.1
    TRAIN_ITER=2200
    EVAL_ITER=44
    LOG_ITER=44
    WARMUP_ITER=88
    LR_DROP_ITER="660 1320 1980"
    RATIO_TAG=19
    ;;
  15)
    NUM_LABELS=525
    LABEL_RATIO=0.15
    TRAIN_ITER=3300
    EVAL_ITER=66
    LOG_ITER=66
    WARMUP_ITER=132
    LR_DROP_ITER="990 1980 2970"
    RATIO_TAG=15
    ;;
  20)
    NUM_LABELS=700
    LABEL_RATIO=0.2
    TRAIN_ITER=4400
    EVAL_ITER=88
    LOG_ITER=88
    WARMUP_ITER=176
    LR_DROP_ITER="1320 2640 3960"
    RATIO_TAG=28
    ;;
  30)
    NUM_LABELS=1050
    LABEL_RATIO=0.3
    TRAIN_ITER=6600
    EVAL_ITER=132
    LOG_ITER=132
    WARMUP_ITER=264
    LR_DROP_ITER="1980 3960 5940"
    RATIO_TAG=37
    ;;
  all)
    NUM_LABELS=3500
    LABEL_RATIO=1.0
    TRAIN_ITER=22000
    EVAL_ITER=440
    LOG_ITER=440
    WARMUP_ITER=880
    LR_DROP_ITER="6600 13200 19800"
    RATIO_TAG=all
    ;;
  *)
    echo "Unknown RATIO=$RATIO"
    exit 1
    ;;
esac

NUM_LABELS=${NUM_LABELS_OVERRIDE:-$NUM_LABELS}
LABEL_RATIO=${LABEL_RATIO_OVERRIDE:-$LABEL_RATIO}
EPOCH=${EPOCH_OVERRIDE:-50}
TRAIN_ITER=${TRAIN_ITER_OVERRIDE:-$TRAIN_ITER}
EVAL_ITER=${EVAL_ITER_OVERRIDE:-$EVAL_ITER}
LOG_ITER=${LOG_ITER_OVERRIDE:-$LOG_ITER}
WARMUP_ITER=${WARMUP_ITER_OVERRIDE:-$WARMUP_ITER}
LR_DROP_ITER=${LR_DROP_ITER_OVERRIDE:-$LR_DROP_ITER}

case "$METHOD" in
  supervised)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/supervised/supervised_tn5000_28_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="supervised_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=1
    EMA_M=0.0
    LR=${LR:-0.0046875}
    EXTRA_CONFIG=""
    ;;
  fixmatch)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/fixmatch/fixmatch_tn5000_19_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="fixmatch_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    P_CUTOFF=${P_CUTOFF:-0.9}
    EXTRA_CONFIG=$(cat <<EOF
p_cutoff: ${P_CUTOFF}
EOF
)
    ;;
  adamatch)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/adamatch/adamatch_bus_878_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="adamatch_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    P_CUTOFF=${P_CUTOFF:-0.9}
    EXTRA_CONFIG=$(cat <<EOF
p_cutoff: ${P_CUTOFF}
EOF
)
    ;;
  flexmatch)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/flexmatch/flexmatch_bus_878_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="flexmatch_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    P_CUTOFF=${P_CUTOFF:-0.9}
    EXTRA_CONFIG=$(cat <<EOF
p_cutoff: ${P_CUTOFF}
thresh_warmup: True
EOF
)
    ;;
  freematch)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/freematch/freematch_tn5000_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="freematch_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.0
    LR=${LR:-0.0046875}
    EXTRA_CONFIG=$(cat <<'EOF'
ema_p: 0.999
ent_loss_ratio: 0.001
EOF
)
    ;;
  refixmatch)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="refixmatch_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    P_CUTOFF=${P_CUTOFF:-0.9}
    EXTRA_CONFIG=$(cat <<EOF
p_cutoff: ${P_CUTOFF}
EOF
)
    ;;
  softmatch)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/softmatch/softmatch_bus_878_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="softmatch_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    EXTRA_CONFIG=$(cat <<'EOF'
dist_align: True
dist_uniform: True
n_sigma: 2
per_class: False
ulb_loss_ratio: 1.0
EOF
)
    ;;
  simmatch)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/simmatch/simmatch_bus_878_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="simmatch_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    P_CUTOFF=${P_CUTOFF:-0.9}
    EXTRA_CONFIG=$(cat <<EOF
T: 0.1
p_cutoff: ${P_CUTOFF}
proj_size: 128
K: ${NUM_LABELS}
in_loss_ratio: 5.0
ulb_loss_ratio: 10.0
smoothing_alpha: 0.9
da_len: 256
use_epass: False
use_da: True
EOF
)
    ;;
  simmatchv2)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/simmatchv2/simmatchv2_bus_878_0.yaml}
    LAUNCHER=${LAUNCHER:-train.py}
    SAVE_PREFIX="simmatchv2_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    P_CUTOFF=${P_CUTOFF:-0.9}
    EXTRA_CONFIG=$(cat <<EOF
t: 0.1
alpha: 0.1
topn: 128
K: 3360
proj_size: 128
p_cutoff: ${P_CUTOFF}
ulb_loss_ratio: 10.0
lambda_ee: 5.0
lambda_ne: 5.0
da_len: 256
use_da: True
EOF
)
    ;;
  fixmatch_if)
    BASE_CONFIG=${BASE_CONFIG:-config/usb_cv/fixmatch/fixmatch_tn5000_19_0.yaml}
    LAUNCHER=${LAUNCHER:-train_ifcf.py}
    SAVE_PREFIX="fixmatch_ifcf_hard_tn5000_${RATIO_TAG}"
    BATCH_SIZE=${BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    URATIO=${URATIO:-30}
    EMA_M=0.999
    LR=${LR:-0.0046875}
    P_CUTOFF=${P_CUTOFF:-0.9}
    EXTRA_CONFIG=$(cat <<'EOF'
algorithm: fixmatch_ifcf
ifrank_combine: multiply_balanced
if_target: hard
if_tracin_scale: 1.0
ifrank_loss_weight: 1.0
if_lambda: 1
csim_lambda: 1
corrT: 0.9
num_references: 4
ref_select: by_instance
ref_cand_k: 8
use_strong_if: True
ifrank_warmup_epochs: 5
ifrank_warmup_mode: zero
EOF
)
    EXTRA_CONFIG="${EXTRA_CONFIG}
p_cutoff: ${P_CUTOFF}"
    ;;
  *)
    echo "Unknown METHOD=$METHOD"
    exit 1
    ;;
esac

if [ -n "${EXTRA_CONFIG_APPEND}" ]; then
  if [ -n "${EXTRA_CONFIG}" ]; then
    EXTRA_CONFIG="${EXTRA_CONFIG}
${EXTRA_CONFIG_APPEND}"
  else
    EXTRA_CONFIG="${EXTRA_CONFIG_APPEND}"
  fi
fi

run_eval() {
  local kind=$1
  local result_prefix="${SAVE_PREFIX}"
  if [ -n "${SAVE_SUFFIX}" ]; then
    result_prefix="${result_prefix}_${SAVE_SUFFIX}"
  fi
  local summary="results/${result_prefix}_${kind}_threshold_summary.csv"
  local detail="results/${result_prefix}_${kind}_threshold_detail.csv"
  echo "[$(date)] Eval ${METHOD} ratio=${RATIO} kind=${kind}"
  python3 tools/eval_tn5000_method_thresholds.py \
    --method "${METHOD}" \
    --ratio_tag "${RATIO_TAG}" \
    --kind "${kind}" \
    --save_suffix "${SAVE_SUFFIX}" \
    --threshold_modes "${EVAL_THRESHOLD_MODES}" \
    --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
    --data_dir "${DATA_DIR}" \
    --summary_csv "${summary}" \
    --detail_csv "${detail}" \
    --num_workers 0
}

if [ "${EVAL_ONLY}" = "1" ]; then
  for kind in ${EVAL_KINDS}; do
    run_eval "${kind}"
  done
  exit 0
fi

for seed in $SEEDS; do
  SAVE_NAME="${SAVE_PREFIX}_${seed}"
  if [ -n "${SAVE_SUFFIX}" ]; then
    SAVE_NAME="${SAVE_NAME}_${SAVE_SUFFIX}"
  fi
  LOAD_PATH="./saved_models/usb_cv/${SAVE_NAME}/latest_model.pth"
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$LOAD_PATH" ]; then
    echo "[$(date)] Skip existing run ${SAVE_NAME}"
    continue
  fi

  TMP_CONFIG=$(mktemp)
  sed "s/^save_name:.*/save_name: ${SAVE_NAME}/" "${BASE_CONFIG}" | \
    sed "s|^load_path:.*|load_path: ${LOAD_PATH}|" | \
    sed "s|^data_dir:.*|data_dir: ${DATA_DIR}|" | \
    sed "s/^dataset:.*/dataset: tn5000/" | \
    sed "s/^seed:.*/seed: ${seed}/" | \
    sed "s/^split_seed:.*/split_seed: ${seed}/" | \
    sed "s/^epoch:.*/epoch: ${EPOCH}/" | \
    sed "s/^num_labels:.*/num_labels: ${NUM_LABELS}/" | \
    sed "s/^label_ratio:.*/label_ratio: ${LABEL_RATIO}/" | \
    sed "s/^num_train_iter:.*/num_train_iter: ${TRAIN_ITER}/" | \
    sed "s/^num_eval_iter:.*/num_eval_iter: ${EVAL_ITER}/" | \
    sed "s/^num_log_iter:.*/num_log_iter: ${LOG_ITER}/" | \
    sed "s/^num_warmup_iter:.*/num_warmup_iter: ${WARMUP_ITER}/" | \
    sed "s/^batch_size:.*/batch_size: ${BATCH_SIZE}/" | \
    sed "s/^eval_batch_size:.*/eval_batch_size: ${EVAL_BATCH_SIZE}/" | \
    sed "s/^uratio:.*/uratio: ${URATIO}/" | \
    sed "s/^ema_m:.*/ema_m: ${EMA_M}/" | \
    sed "s/^lr:.*/lr: ${LR}/" | \
    sed "s|^lpath:.*|lpath: ''|" | \
    sed "s|^ulpath:.*|ulpath: ''|" | \
    sed "s/^lr_drop_iter:.*/lr_drop_iter: ${LR_DROP_ITER}/" > "${TMP_CONFIG}"

  # Some base YAMLs end without a trailing newline. Add one before any append.
  printf "\n" >> "${TMP_CONFIG}"

  if grep -q "^best_metric:" "${TMP_CONFIG}"; then
    sed -i "s/^best_metric:.*/best_metric: ${BEST_METRIC}/" "${TMP_CONFIG}"
  else
    printf "%s\n" "best_metric: ${BEST_METRIC}" >> "${TMP_CONFIG}"
  fi

  if [ -n "${EXTRA_CONFIG}" ]; then
    while IFS= read -r extra_line; do
      [ -z "${extra_line}" ] && continue
      extra_key=${extra_line%%:*}
      if grep -q "^${extra_key}:" "${TMP_CONFIG}"; then
        extra_value=${extra_line#*: }
        sed -i "s|^${extra_key}:.*|${extra_key}: ${extra_value}|" "${TMP_CONFIG}"
      else
        printf "%s\n" "${extra_line}" >> "${TMP_CONFIG}"
      fi
    done <<< "${EXTRA_CONFIG}"
  fi

  if grep -q "^tn5000_balance_mode:" "${TMP_CONFIG}"; then
    sed -i "s/^tn5000_balance_mode:.*/tn5000_balance_mode: ${TN5000_BALANCE_MODE}/" "${TMP_CONFIG}"
  else
    printf "%s\n" "tn5000_balance_mode: ${TN5000_BALANCE_MODE}" >> "${TMP_CONFIG}"
  fi

  if [ -n "${CE_CLASS_WEIGHTS}" ]; then
    if grep -q "^ce_class_weights:" "${TMP_CONFIG}"; then
      sed -i "s/^ce_class_weights:.*/ce_class_weights: ${CE_CLASS_WEIGHTS}/" "${TMP_CONFIG}"
    else
      printf "%s\n" "ce_class_weights: ${CE_CLASS_WEIGHTS}" >> "${TMP_CONFIG}"
    fi
  fi

  echo "[$(date)] Start ${METHOD} ratio=${RATIO} seed=${seed} data_dir=${DATA_DIR}"
  python3 "${LAUNCHER}" --c "${TMP_CONFIG}"
  rm -f "${TMP_CONFIG}"
done

if [ "${AUTO_EVAL}" = "1" ]; then
  for kind in ${EVAL_KINDS}; do
    run_eval "${kind}"
  done
fi
