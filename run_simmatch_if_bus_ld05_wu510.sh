#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml}
SUMMARY=${SUMMARY:-results/simmatch_if_bus_ld05_wu510_summary.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/simmatch_if_bus_ld05_wu510_${STAMP}.log}
SEEDS=${SEEDS:-"1 2 3 4 5"}
WARMUPS=${WARMUPS:-"5 10"}
FORCE_RERUN=${FORCE_RERUN:-0}

mkdir -p logs results config

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== SimMatch+IF BUS ld0.5 warmup5/10 start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
echo "warmups=${WARMUPS}"
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
  local warmup=$2
  local save_name="simmatch_if_bus_ld05_hard_ct09_wu${warmup}_s${seed}"
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
  setkv "$tmp" algorithm simmatch_if
  setkv "$tmp" save_name "$save_name"
  setkv "$tmp" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" batch_size 8
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" T 0.1
  setkv "$tmp" ema_m 0.999
  setkv "$tmp" use_da True
  setkv "$tmp" use_epass False
  setkv "$tmp" layer_decay 0.5
  setkv "$tmp" ifrank_mode add
  setkv "$tmp" ifrank_combine multiply_balanced
  setkv "$tmp" ifrank_loss_weight 1.0
  setkv "$tmp" if_lambda 1
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" corrT 0.9
  setkv "$tmp" num_references 4
  setkv "$tmp" ref_select by_instance
  setkv "$tmp" ref_cand_k 8
  setkv "$tmp" use_strong_if True
  setkv "$tmp" if_target hard
  setkv "$tmp" if_mean_reduce True
  setkv "$tmp" ifrank_warmup_epochs "$warmup"
  setkv "$tmp" ifrank_warmup_mode zero
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" num_log_iter 110

  echo "[$(date)] === train ${save_name} ==="
  python3 train_simmatch_if.py --c "$tmp"
  rm -f "$tmp"
}

eval_group() {
  local warmup=$1
  local pattern="saved_models/usb_cv/simmatch_if_bus_ld05_hard_ct09_wu${warmup}_s*/latest_model.pth"
  if ! compgen -G "$pattern" > /dev/null; then
    echo "[$(date)] [warn] no checkpoints found for warmup=${warmup}; skip eval"
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
    --method_suffix "simmatch_if_ld05_hard_ct09_wu${warmup}_latest" \
    || echo "[$(date)] [warn] eval warmup=${warmup} latest failed"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort"; exit 1; }

for warmup in $WARMUPS; do
  for seed in $SEEDS; do
    run_one "$seed" "$warmup"
  done
done

for warmup in $WARMUPS; do
  eval_group "$warmup"
done

touch results/ALL_DONE_SIMMATCH_IF_BUS_LD05_WU510
echo "[$(date)] ===== SimMatch+IF BUS ld0.5 warmup5/10 done. ${SUMMARY} ====="
