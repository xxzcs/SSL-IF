#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/meanteacher_ap/meanteacher_ap_gdph_28_0.yaml}
SUMMARY=${SUMMARY:-results/meanteacher_ap_gdph_28_latest.csv}
FOLDMEAN=${FOLDMEAN:-results/meanteacher_ap_gdph_28_foldmean.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/meanteacher_ap_gdph_${STAMP}.log}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
mkdir -p logs results config

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== MeanTeacher-AP GDPH start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "foldmean=${FOLDMEAN}"
echo "folds=${FOLDS}"
echo "seeds=${SEEDS}"
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

make_cfg() {
  local tmp=$1 save_name=$2 fold=$3 seed=$4
  cp "$BASE" "$tmp"

  setkv "$tmp" save_name "$save_name"
  setkv "$tmp" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None

  setkv "$tmp" fold "$fold"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" split_seed 0
}

run_one() {
  local fold=$1 seed=$2
  local sn="meanteacher_ap_gdph_28_fold${fold}_${seed}"
  local tmp="config/_meanteacher_ap_gdph_${sn}.yaml"

  if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${sn}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${sn} RUN_DONE exists"
    return 0
  fi

  if [ -d "saved_models/usb_cv/${sn}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${sn}/latest_model.pth" ]; }; then
    echo "[$(date)] [clean] removing existing directory saved_models/usb_cv/${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
  fi

  make_cfg "$tmp" "$sn" "$fold" "$seed"
  echo "[$(date)] === train ${sn} ==="
  python3 train.py --c "$tmp"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${sn} failed with rc=${rc}; retry once after cleanup"
    rm -rf "saved_models/usb_cv/${sn}"
    python3 train.py --c "$tmp"
    rc=$?
  fi

  rm -f "$tmp"
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${sn} still failed; continue"
    return 0
  fi

  touch "saved_models/usb_cv/${sn}/RUN_DONE"
}

eval_latest() {
  if ! compgen -G "saved_models/usb_cv/meanteacher_ap_gdph_28_fold0_*/latest_model.pth" > /dev/null; then
    echo "[$(date)] [warn] no fold0 checkpoints found; skip eval"
    return 0
  fi

  python3 eval_sup_cv.py \
    --load_glob_template "saved_models/usb_cv/meanteacher_ap_gdph_28_fold{fold}_*/latest_model.pth" \
    --folds 0 1 2 3 4 \
    --dataset gdph --num_classes 2 --net resnet18 --model_key ema_model \
    --data_dir ../uda_data/GDPH \
    --label_ratio 0.2 --num_labels 384 \
    --batch_size 16 --num_workers 0 --eval_dest eval \
    --summary_csv "$SUMMARY" \
    --foldmean_csv "$FOLDMEAN" \
    --method_suffix latest || echo "[$(date)] [warn] eval latest failed"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort MeanTeacher-AP GDPH"; exit 1; }

for fold in $FOLDS; do
  for seed in $SEEDS; do
    run_one "$fold" "$seed"
  done
done

eval_latest
touch results/ALL_DONE_MEANTEACHER_AP_GDPH
echo "[$(date)] ===== MeanTeacher-AP GDPH done. ${FOLDMEAN} ====="

