#!/usr/bin/env bash
set -euo pipefail

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
    NAME_PREFIX="fixmatch_ifcf_bus_ld05_ratio5"
    METHOD_SUFFIX="fixmatch_ifcf_bus_ld05_ratio5"
    SUMMARY="results/fixmatch_ifcf_bus_ld05_ratio5_latest.csv"
    ;;
  10)
    NUM_LABELS=439
    NUM_TRAIN_ITER=5500
    NUM_EVAL_ITER=55
    NUM_WARMUP_ITER=220
    LR_DROP_ITER="1595 3245 4895"
    LPATH="../data_split/19/labeled_images_10_new.pth"
    ULPATH="../data_split/19/unlabeled_images_90_new.pth"
    NAME_PREFIX="fixmatch_ifcf_bus_ld05_ratio10"
    METHOD_SUFFIX="fixmatch_ifcf_bus_ld05_ratio10"
    SUMMARY="results/fixmatch_ifcf_bus_ld05_ratio10_latest.csv"
    ;;
  15)
    NUM_LABELS=659
    NUM_TRAIN_ITER=4150
    NUM_EVAL_ITER=83
    NUM_WARMUP_ITER=166
    LR_DROP_ITER="1162 2407 3652"
    LPATH="../data_split/15_85/labeled_images_15_1.pth"
    ULPATH="../data_split/15_85/unlabeled_images_85_1.pth"
    NAME_PREFIX="fixmatch_ifcf_bus_ld05_ratio15"
    METHOD_SUFFIX="fixmatch_ifcf_bus_ld05_ratio15"
    SUMMARY="results/fixmatch_ifcf_bus_ld05_ratio15_latest.csv"
    ;;
  30)
    NUM_LABELS=1317
    NUM_TRAIN_ITER=8250
    NUM_EVAL_ITER=165
    NUM_WARMUP_ITER=330
    LR_DROP_ITER="2310 4785 7260"
    LPATH="../data_split/37/labeled_images_30_1.pth"
    ULPATH="../data_split/37/unlabeled_images_70_1.pth"
    NAME_PREFIX="fixmatch_ifcf_bus_ld05_ratio30"
    METHOD_SUFFIX="fixmatch_ifcf_bus_ld05_ratio30"
    SUMMARY="results/fixmatch_ifcf_bus_ld05_ratio30_latest.csv"
    ;;
  *)
    echo "Unknown RATIO=$RATIO"
    exit 1
    ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/fixmatch_ifcf_bus_ld05_ratio${RATIO}_${STAMP}.log}

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

echo "[$(date)] ===== FixMatch+IFCF BUS ld=0.5 ratio=${RATIO}% start ====="
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
  setkv "$tmp" algorithm fixmatch_ifcf
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" layer_decay 0.5
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
  setkv "$tmp" ifrank_combine multiply_balanced
  setkv "$tmp" corrT 0.9
  setkv "$tmp" use_strong_if True
  setkv "$tmp" if_lambda 1
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" if_tracin_scale 1.0
  setkv "$tmp" num_references 4
  setkv "$tmp" ref_select by_instance
  setkv "$tmp" ref_cand_k 8
  setkv "$tmp" if_target hard
  setkv "$tmp" ifrank_loss_weight 1.0
  setkv "$tmp" ifrank_warmup_mode zero
  setkv "$tmp" ifrank_warmup_epochs 5
  setkv "$tmp" ifrank_if_ramp_epochs 5
  setkv "$tmp" if_mean_reduce True
  echo "[$(date)] === train ${sn} ==="
  if ! python train_ifcf.py --c "$tmp"; then
    echo "[$(date)] [retry] ${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
    python train_ifcf.py --c "$tmp"
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
    --batch_size 16 \
    --num_labels "$NUM_LABELS" \
    --eval_dest test \
    --lpath "$LPATH" \
    --ulpath "$ULPATH" \
    --load_glob "saved_models/usb_cv/${NAME_PREFIX}_s*/${ck}" \
    --method_suffix "${METHOD_SUFFIX}_${kind}"
done

touch "results/ALL_DONE_FIXMATCH_IFCF_BUS_LD05_RATIO${RATIO}"
echo "[$(date)] ===== FixMatch+IFCF BUS ld=0.5 ratio=${RATIO}% done ====="
