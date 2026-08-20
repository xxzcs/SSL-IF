#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_gdph_0.yaml}
SUMMARY=${SUMMARY:-results/fixmatch_rankmatch_gdph_latest.csv}
FOLDMEAN=${FOLDMEAN:-results/fixmatch_rankmatch_gdph_foldmean_latest.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/fixmatch_rankmatch_gdph_${STAMP}.log}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
mkdir -p logs results config

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

GDPH_NUM_LABELS=${GDPH_NUM_LABELS:-384}
GDPH_LABEL_RATIO=${GDPH_LABEL_RATIO:-0.2}
GDPH_NUM_TRAIN_ITER=${GDPH_NUM_TRAIN_ITER:-2400}
GDPH_NUM_LOG_ITER=${GDPH_NUM_LOG_ITER:-48}
GDPH_NUM_EVAL_ITER=${GDPH_NUM_EVAL_ITER:-48}
GDPH_NUM_WARMUP_ITER=${GDPH_NUM_WARMUP_ITER:-96}
GDPH_LR_DROP_ITER=${GDPH_LR_DROP_ITER:-"720 1440 2160"}
GDPH_BATCH_SIZE=${GDPH_BATCH_SIZE:-8}

echo "[$(date)] ===== FixMatch RankMatch-style GDPH start ====="
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

prepare_common() {
  local cfg=$1 save_name=$2 fold=$3 seed=$4

  setkv "$cfg" algorithm fixmatch_ifcf
  setkv "$cfg" save_name "$save_name"
  setkv "$cfg" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$cfg" resume False
  setkv "$cfg" overwrite True
  setkv "$cfg" multiprocessing_distributed False
  setkv "$cfg" gpu None

  setkv "$cfg" dataset gdph
  setkv "$cfg" data_dir ../uda_data/GDPH
  setkv "$cfg" num_classes 2
  setkv "$cfg" net resnet18
  setkv "$cfg" net_from_name False
  setkv "$cfg" img_size 224
  setkv "$cfg" crop_ratio 0.875
  setkv "$cfg" use_pretrain False

  setkv "$cfg" fold "$fold"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" split_seed 0
  setkv "$cfg" num_labels "$GDPH_NUM_LABELS"
  setkv "$cfg" label_ratio "$GDPH_LABEL_RATIO"
  setkv "$cfg" uratio 30
  setkv "$cfg" batch_size "$GDPH_BATCH_SIZE"
  setkv "$cfg" eval_batch_size 16
  setkv "$cfg" num_train_iter "$GDPH_NUM_TRAIN_ITER"
  setkv "$cfg" num_log_iter "$GDPH_NUM_LOG_ITER"
  setkv "$cfg" num_eval_iter "$GDPH_NUM_EVAL_ITER"
  setkv "$cfg" num_warmup_iter "$GDPH_NUM_WARMUP_ITER"
  setkv "$cfg" lr 0.0046875
  setkv "$cfg" sched linear
  setkv "$cfg" lr_drop_iter "$GDPH_LR_DROP_ITER"
  setkv "$cfg" layer_decay 1.0
  setkv "$cfg" train_sampler RandomSampler
  setkv "$cfg" num_workers 4
  setkv "$cfg" lpath "''"
  setkv "$cfg" ulpath "''"

  setkv "$cfg" ifrank_combine multiply_balanced
  setkv "$cfg" ifrank_score_mode cosine
  setkv "$cfg" if_lambda 0
  setkv "$cfg" csim_lambda 1
  setkv "$cfg" num_references 4
  setkv "$cfg" ref_select orthogonal
  setkv "$cfg" corrT 0.9
  setkv "$cfg" if_target soft
  setkv "$cfg" use_strong_if True
  setkv "$cfg" if_mean_reduce True
  setkv "$cfg" ifrank_loss_weight 1.0
  setkv "$cfg" ifrank_warmup_mode zero
  setkv "$cfg" ifrank_warmup_epochs 1
}

run_one() {
  local fold=$1 seed=$2
  local sn="fixmatch_rankmatch_gdph_28_fold${fold}_${seed}"
  local tmp="config/_rankmatch_${sn}.yaml"

  if [ -f "saved_models/usb_cv/${sn}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${sn} RUN_DONE exists"
    return 0
  fi

  if [ -d "saved_models/usb_cv/${sn}" ] && [ ! -f "saved_models/usb_cv/${sn}/latest_model.pth" ]; then
    echo "[$(date)] [clean] removing stale directory saved_models/usb_cv/${sn}"
    rm -rf "saved_models/usb_cv/${sn}"
  fi

  cp "$BASE" "$tmp"
  prepare_common "$tmp" "$sn" "$fold" "$seed"

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
  if ! compgen -G "saved_models/usb_cv/fixmatch_rankmatch_gdph_28_fold*/latest_model.pth" > /dev/null; then
    echo "[$(date)] [warn] no GDPH checkpoints found; skip eval"
    return 0
  fi
  python3 eval_sup_cv.py \
    --load_glob_template "saved_models/usb_cv/fixmatch_rankmatch_gdph_28_fold{fold}_*/latest_model.pth" \
    --folds 0 1 2 3 4 \
    --dataset gdph \
    --num_classes 2 \
    --net resnet18 \
    --model_key ema_model \
    --data_dir ../uda_data/GDPH \
    --label_ratio "$GDPH_LABEL_RATIO" \
    --num_labels "$GDPH_NUM_LABELS" \
    --batch_size 16 \
    --num_workers 0 \
    --eval_dest eval \
    --summary_csv "$SUMMARY" \
    --foldmean_csv "$FOLDMEAN" \
    --method_suffix latest \
    || echo "[$(date)] [warn] eval latest failed"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort GDPH queue"; exit 1; }

for fold in $FOLDS; do
  for seed in $SEEDS; do
    run_one "$fold" "$seed"
  done
done

eval_latest
touch results/ALL_DONE_FIXMATCH_RANKMATCH_GDPH
echo "[$(date)] ===== FixMatch RankMatch-style GDPH done. ${FOLDMEAN} ====="
