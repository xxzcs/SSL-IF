#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate wssl 2>/dev/null

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/bus_ifcf_residual_expand_ld05_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1

FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
SUMMARY=${SUMMARY:-results/fixmatch_ifcf_bus_ld05_residual_expand_latest.csv}
SEEDS=${SEEDS:-"4 5"}
VARIANTS=${VARIANTS:-"lambda05"}
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

COMMON="corrT=0.9 use_strong_if=True csim_lambda=1 if_tracin_scale=1.0 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard"

train_one() {
  local sn=$1 seed=$2 combine=$3 if_lambda=$4 mode=$5 we=$6 re=$7
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
  setkv "$tmp" ifrank_combine "$combine"
  setkv "$tmp" if_lambda "$if_lambda"
  setkv "$tmp" ifrank_warmup_mode "$mode"
  setkv "$tmp" ifrank_warmup_epochs "$we"
  setkv "$tmp" ifrank_if_ramp_epochs "$re"
  echo "[$(date)] === train $sn (combine=$combine if_lambda=$if_lambda mode=$mode warmup=$we ramp=$re) ==="
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
  local prefix=$1 combine=$2 if_lambda=$3 mode=$4 we=$5 re=$6
  for s in $SEEDS; do
    train_one "${prefix}_s${s}" "$s" "$combine" "$if_lambda" "$mode" "$we" "$re"
  done
  eval_one "${prefix}_s*" "${prefix}_latest"
}

for variant in $VARIANTS; do
  case "$variant" in
    lambda05)
      run_variant "ifcf_bus_ld05_residual_product_balanced_lam05" "residual_product_balanced" "0.5" "zero" 5 5
      ;;
    lambda075)
      run_variant "ifcf_bus_ld05_residual_product_balanced_lam075" "residual_product_balanced" "0.75" "zero" 5 5
      ;;
    lambda1)
      run_variant "ifcf_bus_ld05_residual_product_balanced_lam1" "residual_product_balanced" "1.0" "zero" 5 5
      ;;
    ramp)
      run_variant "ifcf_bus_ld05_residual_product_balanced_ramp" "residual_product_balanced" "1.0" "cosine_then_if_ramp" 5 5
      ;;
    *)
      echo "[skip] unknown variant: $variant"
      ;;
  esac
done

touch results/ALL_DONE_BUS_FIXMATCH_IFCF_RESIDUAL_EXPAND_LD05
echo "[$(date)] ===== BUS ld0.5 residual expand sweep done ====="
