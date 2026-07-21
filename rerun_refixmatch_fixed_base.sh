#!/usr/bin/env bash
# Corrected, paper-faithful ReFixMatch BUS baseline: 5 seeds, latest checkpoint.
set -euo pipefail
cd /home/xiexiaozheng/Semi-supervised-learning
source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh
conda activate wssl

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/refixmatch_fixed_base_${STAMP}.log"
SUMMARY="results/refixmatch_fixed_base_summary.csv"
BASE="config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml"
exec > >(tee -a "$LOG") 2>&1

setkv() {
  local file=$1 key=$2 value=$3
  if grep -qE "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    echo "${key}: ${value}" >> "$file"
  fi
}

run_one() {
  local seed=$1 name="refixmatch_fixed_base_s${seed}"
  if [[ -f "saved_models/usb_cv/${name}/latest_model.pth" ]]; then
    echo "[$(date)] [skip] ${name} already complete"
    return
  fi
  local cfg="config/_${name}.yaml"
  cp "$BASE" "$cfg"
  setkv "$cfg" save_name "$name"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" resume False
  setkv "$cfg" overwrite True
  setkv "$cfg" multiprocessing_distributed False
  setkv "$cfg" num_log_iter 110
  echo "[$(date)] === ${name} ==="
  python train.py --c "$cfg"
  rm -f "$cfg"
}

# GPU/data/backprop smoke test under the corrected loss.
SMOKE="config/_smoke_refixmatch_fixed.yaml"
cp "$BASE" "$SMOKE"
setkv "$SMOKE" save_name smoke_refixmatch_fixed
setkv "$SMOKE" epoch 1
setkv "$SMOKE" num_train_iter 3
setkv "$SMOKE" num_eval_iter 3
setkv "$SMOKE" num_log_iter 1
setkv "$SMOKE" resume False
setkv "$SMOKE" overwrite True
setkv "$SMOKE" multiprocessing_distributed False
echo "[$(date)] === corrected ReFixMatch smoke test ==="
python train.py --c "$SMOKE"
rm -f "$SMOKE"

for seed in 1 2 3 4 5; do
  run_one "$seed"
done

python eval_sup.py \
  --dataset bus --num_classes 2 --summary_csv "$SUMMARY" \
  --net resnet18 --model_key ema_model --data_dir ../uda_data \
  --batch_size 16 --num_labels 878 --eval_dest test \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob 'saved_models/usb_cv/refixmatch_fixed_base_s[12345]/latest_model.pth' \
  --method_suffix refixmatch_fixed_base_latest

touch REFIXMATCH_FIXED_BASE_DONE
echo "[$(date)] ===== corrected ReFixMatch Base 5-seed complete ====="
