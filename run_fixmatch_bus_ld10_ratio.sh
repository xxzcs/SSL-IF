#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml}
RATIO=${RATIO:-5}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
RESET_SUMMARY=${RESET_SUMMARY:-1}

case "$RATIO" in
  5)
    NUM_LABELS=220
    NUM_TRAIN_ITER=1400
    NUM_EVAL_ITER=28
    NUM_WARMUP_ITER=56
    LR_DROP_ITER="392 812 1234"
    LPATH="../data_split/5/labeled_images_5_1.pth"
    ULPATH="../data_split/5/unlabeled_images_95_1.pth"
    NAME_PREFIX="fixmatch_bus_ld10_ratio5"
    METHOD_SUFFIX="fixmatch_bus_ld10_ratio5"
    SUMMARY="results/fixmatch_bus_ld10_ratio5_latest.csv"
    ;;
  *)
    echo "Unknown RATIO=$RATIO"
    exit 1
    ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/fixmatch_bus_ld10_ratio${RATIO}_${STAMP}.log}

mkdir -p logs results config
exec > >(tee -a "$LOG") 2>&1

if [ "$RESET_SUMMARY" = "1" ]; then
  rm -f "$SUMMARY"
fi

setkv() {
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}:" "$f"; then
    sed -i "s|^${k}:.*|${k}: ${v}|" "$f"
  else
    printf "%s: %s\n" "$k" "$v" >> "$f"
  fi
}

echo "[$(date)] ===== FixMatch BUS ld=1.0 ratio=${RATIO}% start ====="
echo "base=${BASE}"
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
  setkv "$tmp" algorithm fixmatch
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" layer_decay 1.0
  setkv "$tmp" num_labels "$NUM_LABELS"
  setkv "$tmp" num_train_iter "$NUM_TRAIN_ITER"
  setkv "$tmp" num_eval_iter "$NUM_EVAL_ITER"
  setkv "$tmp" num_log_iter "$NUM_EVAL_ITER"
  setkv "$tmp" num_warmup_iter "$NUM_WARMUP_ITER"
  setkv "$tmp" lr_drop_iter "$LR_DROP_ITER"
  setkv "$tmp" lpath "$LPATH"
  setkv "$tmp" ulpath "$ULPATH"
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  echo "[$(date)] === train ${sn} ==="
  python train.py --c "$tmp" || {
    echo "[$(date)] [retry] ${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
    python train.py --c "$tmp"
  }
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
    --batch_size 16 \
    --num_labels "$NUM_LABELS" \
    --eval_dest test \
    --lpath "$LPATH" \
    --ulpath "$ULPATH" \
    --load_glob "saved_models/usb_cv/${NAME_PREFIX}_s*/${ck}" \
    --method_suffix "${METHOD_SUFFIX}_${kind}"
done

touch "results/ALL_DONE_FIXMATCH_BUS_LD10_RATIO${RATIO}"
echo "[$(date)] ===== FixMatch BUS ld=1.0 ratio=${RATIO}% done ====="
