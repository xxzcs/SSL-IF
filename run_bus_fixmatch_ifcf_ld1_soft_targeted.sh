#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

SEEDS=${SEEDS:-"1 2 3"}
FORCE_RERUN=${FORCE_RERUN:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/bus_fixmatch_ifcf_ld1_soft_targeted_${STAMP}.log}
BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml}
SUMMARY=${SUMMARY:-results/fixmatch_ifcf_bus_ld1_soft_targeted_latest.csv}
CORRT=${CORRT:-0.5}
WARMUP=${WARMUP:-10}
WEIGHT=${WEIGHT:-1.0}
STRONG_IF=${STRONG_IF:-True}
IF_TARGET=${IF_TARGET:-soft}

mkdir -p logs results config
exec > >(tee -a "$LOG") 2>&1

setkv() {
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}:" "$f"; then
    sed -i "s|^${k}:.*|${k}: ${v}|" "$f"
  else
    printf "%s: %s\n" "$k" "$v" >> "$f"
  fi
}

tag_key() {
  echo "$1" | tr '.' 'p'
}

CORRT_KEY=$(tag_key "$CORRT")
WEIGHT_KEY=$(tag_key "$WEIGHT")
TAG="fixmatch_ifcf_bus_ld1_soft_t${CORRT_KEY}_w${WEIGHT_KEY}_wu${WARMUP}"
if [ "$IF_TARGET" = "hard" ]; then
  TAG="fixmatch_ifcf_bus_ld1_hard_t${CORRT_KEY}_w${WEIGHT_KEY}_wu${WARMUP}"
fi

echo "[$(date)] ===== BUS FixMatch+IFCF ld=1.0 targeted start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
echo "corrT=${CORRT}"
echo "weight=${WEIGHT}"
echo "warmup=${WARMUP}"
echo "use_strong_if=${STRONG_IF}"
echo "if_target=${IF_TARGET}"
echo "force_rerun=${FORCE_RERUN}"
echo "log=${LOG}"

python3 - <<'PY'
import sys
import torch
n = torch.cuda.device_count()
print(f"[gpu-check] torch.cuda.device_count()={n}", flush=True)
sys.exit(0 if n > 0 else 1)
PY

for s in $SEEDS; do
  sn="${TAG}_s${s}"
  tmp="config/_${sn}.yaml"

  if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${sn}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${sn}"
    continue
  fi

  if [ -d "saved_models/usb_cv/${sn}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${sn}/latest_model.pth" ]; }; then
    rm -rf "saved_models/usb_cv/${sn}"
  fi

  cp "$BASE" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" layer_decay 1.0
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" dataset bus
  setkv "$tmp" data_dir ../uda_data
  setkv "$tmp" lpath ../data_split/28/labeled_images_20_9.pth
  setkv "$tmp" ulpath ../data_split/28/unlabeled_images_80_9.pth
  setkv "$tmp" ifrank_combine multiply_balanced
  setkv "$tmp" corrT "$CORRT"
  setkv "$tmp" use_strong_if "$STRONG_IF"
  setkv "$tmp" if_lambda 1
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" num_references 4
  setkv "$tmp" ref_select by_instance
  setkv "$tmp" ref_cand_k 8
  setkv "$tmp" if_target "$IF_TARGET"
  setkv "$tmp" ifrank_loss_weight "$WEIGHT"
  setkv "$tmp" ifrank_warmup_mode zero
  setkv "$tmp" ifrank_warmup_epochs "$WARMUP"
  setkv "$tmp" if_mean_reduce True

  echo "[$(date)] === train ${sn} (corrT=${CORRT}, weight=${WEIGHT}, warmup=${WARMUP}, if_target=${IF_TARGET}) ==="
  python train_ifcf.py --c "$tmp" || {
    echo "[$(date)] [retry] ${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
    python train_ifcf.py --c "$tmp"
  }
  rc=$?
  rm -f "$tmp"
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [fail] ${sn}"
    exit $rc
  fi

  touch "saved_models/usb_cv/${sn}/RUN_DONE"
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
  --load_glob "saved_models/usb_cv/${TAG}_s*/latest_model.pth" \
  --method_suffix "${TAG}_latest"

touch "results/${TAG}_DONE"
echo "[$(date)] ===== BUS FixMatch+IFCF ld=1.0 targeted done ====="
