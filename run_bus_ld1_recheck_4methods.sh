#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

METHODS=${METHODS:-"adamatch flexmatch freematch softmatch"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/bus_ld1_recheck_4methods_${STAMP}.log}

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

base_cfg() {
  case "$1" in
    adamatch) echo "config/usb_cv/adamatch/adamatch_bus_878_0.yaml" ;;
    flexmatch) echo "config/usb_cv/flexmatch/flexmatch_bus_878_0.yaml" ;;
    freematch) echo "config/usb_cv/freematch/freematch_bus_878_0.yaml" ;;
    softmatch) echo "config/usb_cv/softmatch/softmatch_bus_878_0.yaml" ;;
    *) return 1 ;;
  esac
}

summary_csv() {
  case "$1" in
    adamatch) echo "results/adamatch_bus_ld1_recheck_latest.csv" ;;
    flexmatch) echo "results/flexmatch_bus_ld1_recheck_latest.csv" ;;
    freematch) echo "results/freematch_bus_ld1_recheck_latest.csv" ;;
    softmatch) echo "results/softmatch_bus_ld1_recheck_latest.csv" ;;
    *) return 1 ;;
  esac
}

train_method() {
  local method=$1
  local base
  local summary
  local train_bin="train.py"

  base=$(base_cfg "$method") || return 1
  summary=$(summary_csv "$method") || return 1

  echo "[$(date)] ===== ${method} BUS ld=1.0 recheck start ====="
  echo "base=${base}"
  echo "summary=${summary}"

  for s in $SEEDS; do
    local sn="${method}_bus_ld1_recheck_s${s}"
    local tmp="config/_${sn}.yaml"

    if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${sn}/RUN_DONE" ]; then
      echo "[$(date)] [skip] ${sn}"
      continue
    fi

    if [ -d "saved_models/usb_cv/${sn}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${sn}/latest_model.pth" ]; }; then
      rm -rf "saved_models/usb_cv/${sn}"
    fi

    cp "$base" "$tmp"
    setkv "$tmp" algorithm "$method"
    setkv "$tmp" save_name "$sn"
    setkv "$tmp" seed "$s"
    setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
    setkv "$tmp" lr 0.0046875
    setkv "$tmp" layer_decay 1.0
    setkv "$tmp" multiprocessing_distributed False
    setkv "$tmp" gpu None
    setkv "$tmp" resume False
    setkv "$tmp" overwrite True
    setkv "$tmp" lpath ../data_split/28/labeled_images_20_9.pth
    setkv "$tmp" ulpath ../data_split/28/unlabeled_images_80_9.pth
    setkv "$tmp" dataset bus
    setkv "$tmp" data_dir ../uda_data

    case "$method" in
      adamatch|flexmatch)
        setkv "$tmp" p_cutoff 0.9
        ;;
    esac

    echo "[$(date)] === train ${sn} ==="
    python "$train_bin" --c "$tmp" || {
      echo "[$(date)] [retry] ${sn}"
      rm -rf "saved_models/usb_cv/${sn}"
      python "$train_bin" --c "$tmp"
    }
    local rc=$?
    rm -f "$tmp"
    if [ $rc -ne 0 ]; then
      echo "[$(date)] [fail] ${sn}"
      return $rc
    fi
    touch "saved_models/usb_cv/${sn}/RUN_DONE"
  done

  python3 eval_sup.py \
    --dataset bus \
    --num_classes 2 \
    --summary_csv "$summary" \
    --net resnet18 \
    --model_key ema_model \
    --data_dir ../uda_data \
    --batch_size 16 \
    --num_labels 878 \
    --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth \
    --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/${method}_bus_ld1_recheck_s*/latest_model.pth" \
    --method_suffix "${method}_bus_ld1_recheck_latest"

  echo "[$(date)] ===== ${method} BUS ld=1.0 recheck done ====="
}

echo "[$(date)] ===== BUS ld=1.0 recheck queue start ====="
echo "methods=${METHODS}"
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

for method in $METHODS; do
  train_method "$method"
done

touch results/ALL_DONE_BUS_LD1_RECHECK_4METHODS
echo "[$(date)] ===== BUS ld=1.0 recheck queue done ====="
