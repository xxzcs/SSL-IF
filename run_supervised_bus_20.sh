#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

BASE=${BASE:-config/usb_cv/supervised/supervised_gdph_28_0.yaml}
SUMMARY=${SUMMARY:-results/supervised_bus_20_ld1_latest.csv}
NAME_PREFIX=${NAME_PREFIX:-supervised_bus_20_ld1}
METHOD_SUFFIX=${METHOD_SUFFIX:-supervised_bus_20_ld1}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
LAYER_DECAY=${LAYER_DECAY:-1.0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
LR=${LR:-0.009375}
NUM_TRAIN_ITER=${NUM_TRAIN_ITER:-2750}
NUM_EVAL_ITER=${NUM_EVAL_ITER:-55}
NUM_LOG_ITER=${NUM_LOG_ITER:-55}
NUM_WARMUP_ITER=${NUM_WARMUP_ITER:-110}
LR_DROP_ITER=${LR_DROP_ITER:-"825 1650 2475"}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/supervised_bus_20_${STAMP}.log}

mkdir -p logs results config
exec > >(tee -a "$LOG") 2>&1

setkv() {
  local f=$1 k=$2 v=$3
  python3 - "$f" "$k" "$v" <<'PY'
import re
import sys

path, key, value = sys.argv[1:]
with open(path, "r", encoding="utf-8") as fh:
    text = fh.read()
if text and not text.endswith("\n"):
    text += "\n"
pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
line = f"{key}: {value}"
if pattern.search(text):
    text = pattern.sub(line, text)
else:
    text += line + "\n"
with open(path, "w", encoding="utf-8") as fh:
    fh.write(text)
PY
}

echo "[$(date)] ===== Supervised BUS 20% start ====="
echo "base=${BASE}"
echo "summary=${SUMMARY}"
echo "name_prefix=${NAME_PREFIX}"
echo "layer_decay=${LAYER_DECAY}"
echo "train_batch_size=${TRAIN_BATCH_SIZE}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "lr=${LR}"
echo "num_train_iter=${NUM_TRAIN_ITER}"
echo "num_eval_iter=${NUM_EVAL_ITER}"
echo "num_log_iter=${NUM_LOG_ITER}"
echo "num_warmup_iter=${NUM_WARMUP_ITER}"
echo "lr_drop_iter=${LR_DROP_ITER}"
echo "seeds=${SEEDS}"
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
  setkv "$tmp" algorithm supervised
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" dataset bus
  setkv "$tmp" data_dir ../uda_data
  setkv "$tmp" num_labels 878
  setkv "$tmp" label_ratio 0.2
  setkv "$tmp" uratio 1
  setkv "$tmp" batch_size "$TRAIN_BATCH_SIZE"
  setkv "$tmp" eval_batch_size "$EVAL_BATCH_SIZE"
  setkv "$tmp" num_train_iter "$NUM_TRAIN_ITER"
  setkv "$tmp" num_eval_iter "$NUM_EVAL_ITER"
  setkv "$tmp" num_log_iter "$NUM_LOG_ITER"
  setkv "$tmp" num_warmup_iter "$NUM_WARMUP_ITER"
  setkv "$tmp" lr_drop_iter "$LR_DROP_ITER"
  setkv "$tmp" lr "$LR"
  setkv "$tmp" layer_decay "$LAYER_DECAY"
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" lpath ../data_split/28/labeled_images_20_9.pth
  setkv "$tmp" ulpath ../data_split/28/unlabeled_images_80_9.pth
  echo "[$(date)] === train ${sn} ==="
  python train.py --c "$tmp" || { echo "[$(date)] [retry] ${sn}"; rm -rf "saved_models/usb_cv/${sn}"; python train.py --c "$tmp"; }
  rm -f "$tmp"
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
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_labels 878 \
    --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth \
    --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/${NAME_PREFIX}_s*/${ck}" \
    --method_suffix "${METHOD_SUFFIX}_${kind}"
done

touch "results/${METHOD_SUFFIX}_DONE"
echo "[$(date)] ===== Supervised BUS 20% done ====="
