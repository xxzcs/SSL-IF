#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/meanteacher_ap/meanteacher_ap_bus_878_0.yaml}
SUMMARY=${SUMMARY:-results/meanteacher_ap_bus_summary.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/meanteacher_ap_bus_${STAMP}.log}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}

mkdir -p logs results config

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== MeanTeacher-AP BUS start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
echo "log=${LOG}"

ensure_trailing_newline() {
  local f=$1
  [ -s "$f" ] || return 0
  tail -c 1 "$f" | od -An -t x1 | tr -d ' \n' | grep -qi '^0a$' && return 0
  printf "\n" >> "$f"
}

setkv() {
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}:" "$f"; then
    sed -i "s|^${k}:.*|${k}: ${v}|" "$f"
  else
    ensure_trailing_newline "$f"
    printf "%s: %s\n" "$k" "$v" >> "$f"
  fi
}

check_gpu() {
  python3 - <<'PY'
import sys
import torch
n = torch.cuda.device_count()
print(f"[gpu-check] torch.cuda.device_count()={n}", flush=True)
sys.exit(0 if n > 0 else 1)
PY
}

run_one() {
  local seed=$1
  local save_name="meanteacher_ap_bus_s${seed}"
  local tmp="config/_${save_name}.yaml"

  if [ "$FORCE_RERUN" != "1" ] && grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${save_name}/log.txt" 2>/dev/null; then
    echo "[$(date)] [skip] ${save_name} already complete"
    return 0
  fi

  if [ -d "saved_models/usb_cv/${save_name}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${save_name}/latest_model.pth" ]; }; then
    echo "[$(date)] [clean] removing saved_models/usb_cv/${save_name}"
    rm -rf "saved_models/usb_cv/${save_name}"
  fi

  cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$save_name"
  setkv "$tmp" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True

  echo "[$(date)] === train ${save_name} ==="
  python3 train.py --c "$tmp"
  rm -f "$tmp"
}

eval_group() {
  local pattern="saved_models/usb_cv/meanteacher_ap_bus_s*/latest_model.pth"
  if ! compgen -G "$pattern" > /dev/null; then
    echo "[$(date)] [warn] no checkpoints found; skip eval"
    return 0
  fi

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
    --load_glob "$pattern" \
    --method_suffix "meanteacher_ap_bus_latest" \
    || echo "[$(date)] [warn] eval latest failed"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort"; exit 1; }

for seed in $SEEDS; do
  run_one "$seed"
done

eval_group

touch results/ALL_DONE_MEANTEACHER_AP_BUS
echo "[$(date)] ===== MeanTeacher-AP BUS done. ${SUMMARY} ====="
