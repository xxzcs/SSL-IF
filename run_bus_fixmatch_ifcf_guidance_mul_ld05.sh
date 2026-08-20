#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate wssl 2>/dev/null

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/bus_ifcf_guidance_mul_ld05_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1

FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
SUMMARY=${SUMMARY:-results/fixmatch_ifcf_bus_ld05_guidance_mul_latest.csv}
SEEDS=${SEEDS:-"1 2 3 4 5"}
VARIANTS=${VARIANTS:-"residual_product_balanced gated_cosine_balanced"}
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

COMMON="corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 if_tracin_scale=1.0 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard"

train_one() {
  local sn=$1 seed=$2 combine=$3
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml
  cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110
  setkv "$tmp" lr 0.0046875
  for kv in $COMMON; do
    setkv "$tmp" "${kv%%=*}" "${kv#*=}"
  done
  setkv "$tmp" ifrank_warmup_mode "zero"
  setkv "$tmp" ifrank_warmup_epochs 5
  setkv "$tmp" ifrank_if_ramp_epochs 5
  setkv "$tmp" ifrank_combine "$combine"
  echo "[$(date)] === train $sn (combine=$combine zero5) ==="
  python train_ifcf.py --c "$tmp" || {
    echo "[retry] $sn"
    rm -rf "saved_models/usb_cv/$sn"
    python train_ifcf.py --c "$tmp" || echo "[fail] $sn"
  }
  rm -f "$tmp"
}

eval_one() {
  local glob=$1 suffix=$2
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
    --load_glob "saved_models/usb_cv/${glob}/latest_model.pth" \
    --method_suffix "$suffix"
}

run_variant() {
  local combine=$1 prefix=$2
  for s in $SEEDS; do
    train_one "${prefix}_s${s}" "$s" "$combine"
  done
  eval_one "${prefix}_s*" "${prefix}_latest"
}

for variant in $VARIANTS; do
  case "$variant" in
    residual_product_balanced)
      run_variant "residual_product_balanced" "ifcf_bus_ld05_residual_product_balanced"
      ;;
    gated_cosine_balanced)
      run_variant "gated_cosine_balanced" "ifcf_bus_ld05_gated_cosine_balanced"
      ;;
    *)
      echo "[skip] unknown variant: $variant"
      ;;
  esac
done

touch results/ALL_DONE_BUS_FIXMATCH_IFCF_GUIDANCE_MUL_LD05
echo "[$(date)] ===== BUS ld0.5 guidance multiplicative sweep done ====="
