#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate wssl 2>/dev/null

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/bus_ifcf_residual_best_ld1_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1

FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
SUMMARY=${SUMMARY:-results/fixmatch_ifcf_bus_ld1_residual_best_latest.csv}
SEEDS=${SEEDS:-"1 2 3 4 5"}
VARIANT=${VARIANT:-lambda05}
RESET_SUMMARY=${RESET_SUMMARY:-1}

if [ "$RESET_SUMMARY" = "1" ]; then
  rm -f "$SUMMARY"
fi

setkv() {
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}:" "$f"; then
    sed -i "s|^${k}:.*|${k}: ${v}|" "$f"
  else
    echo "${k}: ${v}" >> "$f"
  fi
}

case "$VARIANT" in
  lambda05)
    PREFIX="ifcf_bus_ld1_residual_product_balanced_lam05"
    COMBINE="residual_product_balanced"
    IFLAMBDA="0.5"
    MODE="zero"
    WE="5"
    RE="5"
    ;;
  lambda075)
    PREFIX="ifcf_bus_ld1_residual_product_balanced_lam075"
    COMBINE="residual_product_balanced"
    IFLAMBDA="0.75"
    MODE="zero"
    WE="5"
    RE="5"
    ;;
  ramp)
    PREFIX="ifcf_bus_ld1_residual_product_balanced_ramp"
    COMBINE="residual_product_balanced"
    IFLAMBDA="1.0"
    MODE="cosine_then_if_ramp"
    WE="5"
    RE="5"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT"
    exit 1
    ;;
esac

COMMON="corrT=0.9 use_strong_if=True csim_lambda=1 if_tracin_scale=1.0 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard"

train_one() {
  local sn=$1 seed=$2
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml
  cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" layer_decay 1.0
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110
  setkv "$tmp" lr 0.0046875
  for kv in $COMMON; do
    setkv "$tmp" "${kv%%=*}" "${kv#*=}"
  done
  setkv "$tmp" ifrank_combine "$COMBINE"
  setkv "$tmp" if_lambda "$IFLAMBDA"
  setkv "$tmp" ifrank_warmup_mode "$MODE"
  setkv "$tmp" ifrank_warmup_epochs "$WE"
  setkv "$tmp" ifrank_if_ramp_epochs "$RE"
  echo "[$(date)] === train $sn (ld1 variant=$VARIANT) ==="
  python train_ifcf.py --c "$tmp" || {
    echo "[retry] $sn"
    rm -rf "saved_models/usb_cv/$sn"
    python train_ifcf.py --c "$tmp" || echo "[fail] $sn"
  }
  rm -f "$tmp"
}

for s in $SEEDS; do
  train_one "${PREFIX}_s${s}" "$s"
done

python3 eval_sup.py \
  --dataset bus \
  --num_classes 2 \
  --summary_csv "$SUMMARY" \
  --net resnet18 \
  --model_key ema_model \
  --data_dir ../uda_data \
  --batch_size 16 \
  --num_labels 878 \
  --eval_dest test \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob "saved_models/usb_cv/${PREFIX}_s*/latest_model.pth" \
  --method_suffix "${PREFIX}_latest"

touch results/ALL_DONE_BUS_FIXMATCH_IFCF_RESIDUAL_BEST_LD1
echo "[$(date)] ===== BUS ld1 residual best replication done ====="
