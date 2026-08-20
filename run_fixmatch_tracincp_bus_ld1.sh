#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/fixmatch_tracincp_bus_ld1_${STAMP}.log}
BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml}
SUMMARY=${SUMMARY:-results/fixmatch_tracincp_bus_ld1_layerdecay_latest.csv}
NAME_PREFIX=${NAME_PREFIX:-fixmatch_tracincp_bus_ld1}
METHOD_SUFFIX=${METHOD_SUFFIX:-fixmatch_tracincp_bus_ld1}

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

echo "[$(date)] ===== FixMatch TracInCP BUS ld=1.0 start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
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
  setkv "$tmp" algorithm fixmatch_if
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" layer_decay 1.0
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110
  setkv "$tmp" gpu None
  setkv "$tmp" combine multiply_balanced
  setkv "$tmp" corrT 0.9
  setkv "$tmp" use_strong_if True
  setkv "$tmp" if_lambda 1
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" num_references 4
  setkv "$tmp" ifrank_loss_weight 1.0
  setkv "$tmp" if_target hard

  echo "[$(date)] === train ${sn} (Captum TracInCP) ==="
  python train.py --c "$tmp" || {
    echo "[$(date)] [retry] ${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
    python train.py --c "$tmp"
  }
  local_rc=$?
  rm -f "$tmp"
  if [ $local_rc -ne 0 ]; then
    echo "[$(date)] [fail] ${sn}"
    exit $local_rc
  fi
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
    --num_labels 878 \
    --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth \
    --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/${NAME_PREFIX}_s*/${ck}" \
    --method_suffix "${METHOD_SUFFIX}_${kind}"
done

touch results/ALL_DONE_FIXMATCH_TRACINCP_BUS_LD1
echo "[$(date)] ===== FixMatch TracInCP BUS ld=1.0 done ====="
