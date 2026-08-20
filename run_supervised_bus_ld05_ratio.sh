#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

BASE=${BASE:-config/usb_cv/supervised/supervised_gdph_28_0.yaml}
RATIO=${RATIO:-5}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
RESET_SUMMARY=${RESET_SUMMARY:-1}

case "$RATIO" in
  5)
    NUM_LABELS=220
    LABEL_RATIO=0.05
    NUM_TRAIN_ITER=700
    NUM_EVAL_ITER=14
    NUM_LOG_ITER=14
    NUM_WARMUP_ITER=28
    LR_DROP_ITER="210 420 630"
    LPATH="../data_split/5/labeled_images_5_1.pth"
    ULPATH="../data_split/5/unlabeled_images_95_1.pth"
    NAME_PREFIX="supervised_bus_ld05_ratio5"
    METHOD_SUFFIX="supervised_bus_ld05_ratio5"
    SUMMARY="results/supervised_bus_ld05_ratio5_latest.csv"
    ;;
  10)
    NUM_LABELS=439
    LABEL_RATIO=0.10
    NUM_TRAIN_ITER=1400
    NUM_EVAL_ITER=28
    NUM_LOG_ITER=28
    NUM_WARMUP_ITER=56
    LR_DROP_ITER="420 840 1260"
    LPATH="../data_split/19/labeled_images_10_new.pth"
    ULPATH="../data_split/19/unlabeled_images_90_new.pth"
    NAME_PREFIX="supervised_bus_ld05_ratio10"
    METHOD_SUFFIX="supervised_bus_ld05_ratio10"
    SUMMARY="results/supervised_bus_ld05_ratio10_latest.csv"
    ;;
  15)
    NUM_LABELS=659
    LABEL_RATIO=0.15
    NUM_TRAIN_ITER=4150
    NUM_EVAL_ITER=83
    NUM_LOG_ITER=83
    NUM_WARMUP_ITER=166
    LR_DROP_ITER="1162 2407 3652"
    LPATH="../data_split/15_85/labeled_images_15_1.pth"
    ULPATH="../data_split/15_85/unlabeled_images_85_1.pth"
    NAME_PREFIX="supervised_bus_ld05_ratio15"
    METHOD_SUFFIX="supervised_bus_ld05_ratio15"
    SUMMARY="results/supervised_bus_ld05_ratio15_latest.csv"
    ;;
  30)
    NUM_LABELS=1317
    LABEL_RATIO=0.30
    NUM_TRAIN_ITER=4150
    NUM_EVAL_ITER=83
    NUM_LOG_ITER=83
    NUM_WARMUP_ITER=166
    LR_DROP_ITER="1245 2490 3735"
    LPATH="../data_split/37/labeled_images_30_1.pth"
    ULPATH="../data_split/37/unlabeled_images_70_1.pth"
    NAME_PREFIX="supervised_bus_ld05_ratio30"
    METHOD_SUFFIX="supervised_bus_ld05_ratio30"
    SUMMARY="results/supervised_bus_ld05_ratio30_latest.csv"
    ;;
  *)
    echo "Unknown RATIO=$RATIO"
    exit 1
    ;;
esac

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
LR=${LR:-0.009375}
LAYER_DECAY=${LAYER_DECAY:-0.5}

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/supervised_bus_ld05_ratio${RATIO}_${STAMP}.log}

mkdir -p logs results config
exec > >(tee -a "$LOG") 2>&1

if [ "$RESET_SUMMARY" = "1" ]; then
  rm -f "$SUMMARY"
fi

setkv() {
  local f=$1 k=$2 v=$3
  python3 - "$f" "$k" "$v" <<'PY'
import re
import sys

path, key, value = sys.argv[1:]
with open(path, "r", encoding="utf-8") as fh:
    text = fh.read()
if text and not text.endswith("\n"):
    text += "\n"
pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
line = f"{key}: {value}"
if pattern.search(text):
    text = pattern.sub(line, text)
else:
    text += line + "\n"
with open(path, "w", encoding="utf-8") as fh:
    fh.write(text)
PY
}

echo "[$(date)] ===== Supervised BUS ld=0.5 ratio=${RATIO}% start ====="
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
echo "log=${LOG}"

for s in $SEEDS; do
  sn="${NAME_PREFIX}_s${s}"
  if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${sn}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${sn}"
    continue
  fi
  if [ -d "saved_models/usb_cv/${sn}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${sn}/latest_model.pth" ]; }; then
    rm -rf "saved_models/usb_cv/${sn}"
  fi
  tmp="config/_${sn}.yaml"
  cp "$BASE" "$tmp"
  setkv "$tmp" algorithm supervised
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" dataset bus
  setkv "$tmp" data_dir ../uda_data
  setkv "$tmp" num_labels "$NUM_LABELS"
  setkv "$tmp" label_ratio "$LABEL_RATIO"
  setkv "$tmp" uratio 1
  setkv "$tmp" batch_size "$TRAIN_BATCH_SIZE"
  setkv "$tmp" eval_batch_size "$EVAL_BATCH_SIZE"
  setkv "$tmp" num_train_iter "$NUM_TRAIN_ITER"
  setkv "$tmp" num_eval_iter "$NUM_EVAL_ITER"
  setkv "$tmp" num_log_iter "$NUM_LOG_ITER"
  setkv "$tmp" num_warmup_iter "$NUM_WARMUP_ITER"
  setkv "$tmp" lr_drop_iter "$LR_DROP_ITER"
  setkv "$tmp" lr "$LR"
  setkv "$tmp" layer_decay "$LAYER_DECAY"
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" lpath "$LPATH"
  setkv "$tmp" ulpath "$ULPATH"
  echo "[$(date)] === train ${sn} ==="
  if ! python train.py --c "$tmp"; then
    echo "[$(date)] [retry] ${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
    python train.py --c "$tmp"
  fi
  if [ ! -f "saved_models/usb_cv/${sn}/latest_model.pth" ]; then
    echo "[$(date)] [error] missing latest_model.pth for ${sn}"
    exit 1
  fi
  rm -f "$tmp"
  touch "saved_models/usb_cv/${sn}/RUN_DONE"
done

for kind in best latest; do
  [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py \
    --dataset bus \
    --num_classes 2 \
    --summary_csv "$SUMMARY" \
    --net resnet18 \
    --model_key ema_model \
    --data_dir ../uda_data \
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_labels "$NUM_LABELS" \
    --eval_dest test \
    --lpath "$LPATH" \
    --ulpath "$ULPATH" \
    --load_glob "saved_models/usb_cv/${NAME_PREFIX}_s*/${ck}" \
    --method_suffix "${METHOD_SUFFIX}_${kind}"
done

touch "results/ALL_DONE_SUPERVISED_BUS_LD05_RATIO${RATIO}"
echo "[$(date)] ===== Supervised BUS ld=0.5 ratio=${RATIO}% done ====="
