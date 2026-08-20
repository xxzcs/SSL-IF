#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml}
SUMMARY=${SUMMARY:-results/fixmatch_rankmatch_bus_pcut09_latest.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/fixmatch_rankmatch_bus_${STAMP}.log}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
LAYER_DECAY=${LAYER_DECAY:-0.5}
NAME_PREFIX=${NAME_PREFIX:-fixmatch_rankmatch_bus}
METHOD_SUFFIX=${METHOD_SUFFIX:-fixmatch_rankmatch_bus_latest}
mkdir -p logs results config

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== FixMatch RankMatch-style BUS start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
echo "layer_decay=${LAYER_DECAY}"
echo "name_prefix=${NAME_PREFIX}"
echo "method_suffix=${METHOD_SUFFIX}"
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
  local sn="${NAME_PREFIX}_s${seed}"
  local tmp="config/_rankmatch_bus_${sn}.yaml"

  if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${sn}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${sn} RUN_DONE exists"
    return 0
  fi

  if [ -d "saved_models/usb_cv/${sn}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${sn}/latest_model.pth" ]; }; then
    echo "[$(date)] [clean] removing existing directory saved_models/usb_cv/${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
  fi

  cp "$BASE" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" layer_decay "$LAYER_DECAY"

  # RankMatch-style path: orthogonal references + cosine-only affinity.
  setkv "$tmp" ifrank_combine multiply_balanced
  setkv "$tmp" ifrank_score_mode cosine
  setkv "$tmp" if_lambda 0
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" num_references 4
  setkv "$tmp" ref_select orthogonal
  setkv "$tmp" corrT 0.9
  setkv "$tmp" if_target soft
  setkv "$tmp" use_strong_if True
  setkv "$tmp" if_mean_reduce True
  setkv "$tmp" ifrank_loss_weight 1.0
  setkv "$tmp" ifrank_warmup_mode zero
  setkv "$tmp" ifrank_warmup_epochs 1

  echo "[$(date)] === train ${sn} ==="
  python3 train_ifcf.py --c "$tmp"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${sn} failed with rc=${rc}; retry once after cleanup"
    rm -rf "saved_models/usb_cv/${sn}"
    python3 train_ifcf.py --c "$tmp"
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
  if ! compgen -G "saved_models/usb_cv/${NAME_PREFIX}_s*/latest_model.pth" > /dev/null; then
    echo "[$(date)] [warn] no BUS checkpoints found; skip eval"
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
    --load_glob "saved_models/usb_cv/${NAME_PREFIX}_s*/latest_model.pth" \
    --method_suffix "${METHOD_SUFFIX}" \
    || echo "[$(date)] [warn] eval latest failed"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort BUS queue"; exit 1; }

for seed in $SEEDS; do
  run_one "$seed"
done

eval_latest
touch results/ALL_DONE_FIXMATCH_RANKMATCH_BUS
echo "[$(date)] ===== FixMatch RankMatch-style BUS done. ${SUMMARY} ====="
