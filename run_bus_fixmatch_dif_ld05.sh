#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml}
SUMMARY=${SUMMARY:-results/fixmatch_dif_bus_ld05_latest.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/fixmatch_dif_bus_ld05_${STAMP}.log}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}

mkdir -p logs results config

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== BUS FixMatch+DIF ld0.5 start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
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

run_one() {
  local seed=$1
  local save_name="fixmatch_dif_bus_ld05_wu5_s${seed}"
  local tmp="config/_${save_name}.yaml"

  if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${save_name}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${save_name}"
    return 0
  fi

  if [ -d "saved_models/usb_cv/${save_name}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${save_name}/latest_model.pth" ]; }; then
    echo "[$(date)] [clean] removing saved_models/usb_cv/${save_name}"
    rm -rf "saved_models/usb_cv/${save_name}"
  fi

  cp "$BASE" "$tmp"
  setkv "$tmp" algorithm fixmatch_dif
  setkv "$tmp" save_name "$save_name"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" batch_size 8
  setkv "$tmp" layer_decay 0.5
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" ifrank_combine multiply_balanced
  setkv "$tmp" ifrank_score_mode fused
  setkv "$tmp" corrT 0.9
  setkv "$tmp" use_strong_if True
  setkv "$tmp" if_lambda 1
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" if_target hard
  setkv "$tmp" ifrank_loss_weight 1.0
  setkv "$tmp" if_tracin_scale 1.0
  setkv "$tmp" num_references 4
  setkv "$tmp" ref_select by_instance
  setkv "$tmp" ref_cand_k 8
  setkv "$tmp" ifrank_warmup_mode zero
  setkv "$tmp" ifrank_warmup_epochs 5
  setkv "$tmp" if_mean_reduce True

  echo "[$(date)] === train ${save_name} ==="
  python3 train_ifcf.py --c "$tmp"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} failed with rc=${rc}; retry once after cleanup"
    rm -rf "saved_models/usb_cv/${save_name}"
    python3 train_ifcf.py --c "$tmp"
    rc=$?
  fi

  rm -f "$tmp"
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} still failed; continue"
    return 0
  fi

  touch "saved_models/usb_cv/${save_name}/RUN_DONE"
}

eval_latest() {
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
    --load_glob "saved_models/usb_cv/fixmatch_dif_bus_ld05_wu5_s*/latest_model.pth" \
    --method_suffix latest || echo "[$(date)] [warn] eval latest failed"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort BUS FixMatch+DIF"; exit 1; }

for seed in $SEEDS; do
  run_one "$seed"
done

eval_latest

touch results/ALL_DONE_BUS_FIXMATCH_DIF_LD05
echo "[$(date)] ===== BUS FixMatch+DIF ld0.5 done ====="
