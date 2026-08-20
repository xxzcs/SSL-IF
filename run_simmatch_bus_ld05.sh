#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/simmatch/simmatch_bus_878_0.yaml}
SUMMARY=${SUMMARY:-results/simmatch_bus_ld05_summary.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/simmatch_bus_ld05_${STAMP}.log}
SEEDS=${SEEDS:-"1 2 3 4 5"}
USE_DA=${USE_DA:-1}
FORCE_RERUN=${FORCE_RERUN:-0}
mkdir -p logs results config

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== SimMatch BUS ld0.5 start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
echo "use_da=${USE_DA}"
echo "log=${LOG}"

check_gpu() {
  python3 - <<'PY'
import sys
import torch
n = torch.cuda.device_count()
print(f"[gpu-check] torch.cuda.device_count()={n}", flush=True)
sys.exit(0 if n > 0 else 1)
PY
}

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

run_one() {
  local seed=$1
  local datag useda save_name tmp rc
  if [ "$USE_DA" = "1" ]; then
    datag="da1ld05"
    useda=True
  else
    datag="da0ld05"
    useda=False
  fi

  save_name="simmatch_bus_878_${datag}_${seed}"
  tmp="config/_simmatch_bus_ld05_${save_name}.yaml"

  if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${save_name}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${save_name} RUN_DONE exists"
    return 0
  fi

  if [ -d "saved_models/usb_cv/${save_name}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${save_name}/latest_model.pth" ]; }; then
    echo "[$(date)] [clean] removing existing directory saved_models/usb_cv/${save_name}"
    rm -rf "saved_models/usb_cv/${save_name}"
  fi

  cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$save_name"
  setkv "$tmp" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" use_da "$useda"
  setkv "$tmp" layer_decay 0.5
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None

  echo "[$(date)] === train ${save_name} ==="
  python3 train.py --c "$tmp"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} failed with rc=${rc}; retry once after cleanup"
    rm -rf "saved_models/usb_cv/${save_name}"
    python3 train.py --c "$tmp"
    rc=$?
  fi

  rm -f "$tmp"
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} still failed; continue"
    return 0
  fi

  touch "saved_models/usb_cv/${save_name}/RUN_DONE"
}

eval_all() {
  local datag
  if [ "$USE_DA" = "1" ]; then
    datag="da1ld05"
  else
    datag="da0ld05"
  fi

  if ! compgen -G "saved_models/usb_cv/simmatch_bus_878_${datag}_*/latest_model.pth" > /dev/null; then
    echo "[$(date)] [warn] no checkpoints found for ${datag}; skip eval"
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
    --load_glob "saved_models/usb_cv/simmatch_bus_878_${datag}_*/model_best.pth" \
    --method_suffix "${datag}_best" \
    || echo "[$(date)] [warn] eval ${datag} best failed"

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
    --load_glob "saved_models/usb_cv/simmatch_bus_878_${datag}_*/latest_model.pth" \
    --method_suffix "${datag}_latest" \
    || echo "[$(date)] [warn] eval ${datag} latest failed"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort SimMatch BUS ld0.5"; exit 1; }

for seed in $SEEDS; do
  run_one "$seed"
done

eval_all
touch results/ALL_DONE_SIMMATCH_BUS_LD05
echo "[$(date)] ===== SimMatch BUS ld0.5 done. ${SUMMARY} ====="
