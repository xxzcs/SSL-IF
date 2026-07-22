#!/usr/bin/env bash
# FreeMatch EMA screening: paired Base vs soft IF, validation only, 3 seeds.
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh
conda activate wssl

BASE=config/usb_cv/freematch/freematch_bus_878_0.yaml
SUMMARY=results/freematch_ema999_val_screen.csv

setkv() {
  local file=$1 key=$2 value=$3
  if grep -qE "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    echo "${key}: ${value}" >> "$file"
  fi
}

run_base() {
  local seed=$1 name="fm_ema999_base_s${seed}"
  if grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${name}/log.txt" 2>/dev/null; then
    echo "[$(date)] [skip] ${name} complete"
    return
  fi
  local cfg="config/_${name}.yaml"
  cp "$BASE" "$cfg"
  setkv "$cfg" algorithm freematch
  setkv "$cfg" save_name "$name"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" ema_m 0.999
  setkv "$cfg" resume False
  setkv "$cfg" overwrite True
  setkv "$cfg" multiprocessing_distributed False
  setkv "$cfg" num_log_iter 110
  python train.py --c "$cfg"
  rm -f "$cfg"
}

run_soft_if() {
  local seed=$1 name="fm_ema999_soft_w1_s${seed}"
  if grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${name}/log.txt" 2>/dev/null; then
    echo "[$(date)] [skip] ${name} complete"
    return
  fi
  local cfg="config/_${name}.yaml"
  cp "$BASE" "$cfg"
  setkv "$cfg" algorithm freematch_ifcf
  setkv "$cfg" save_name "$name"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" ema_m 0.999
  setkv "$cfg" resume False
  setkv "$cfg" overwrite True
  setkv "$cfg" multiprocessing_distributed False
  setkv "$cfg" num_log_iter 110
  setkv "$cfg" ifrank_combine multiply_balanced
  setkv "$cfg" if_lambda 1
  setkv "$cfg" csim_lambda 1
  setkv "$cfg" num_references 4
  setkv "$cfg" ref_select by_instance
  setkv "$cfg" ref_cand_k 8
  setkv "$cfg" ifrank_loss_weight 1.0
  setkv "$cfg" if_target soft
  setkv "$cfg" use_strong_if True
  setkv "$cfg" corrT 0.9
  setkv "$cfg" ifrank_warmup_epochs 5
  setkv "$cfg" ifrank_warmup_mode zero
  python train_ifcf.py --c "$cfg"
  rm -f "$cfg"
}

for seed in 1 2 3; do
  run_base "$seed"
  run_soft_if "$seed"
done

rm -f "$SUMMARY"
python eval_sup.py \
  --dataset bus --num_classes 2 --summary_csv "$SUMMARY" \
  --net resnet18 --model_key ema_model --data_dir ../uda_data \
  --batch_size 16 --num_labels 878 --eval_dest eval \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob 'saved_models/usb_cv/fm_ema999_base_s[123]/latest_model.pth' \
  --method_suffix fm_ema999_base_latest_val
python eval_sup.py \
  --dataset bus --num_classes 2 --summary_csv "$SUMMARY" \
  --net resnet18 --model_key ema_model --data_dir ../uda_data \
  --batch_size 16 --num_labels 878 --eval_dest eval \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob 'saved_models/usb_cv/fm_ema999_soft_w1_s[123]/latest_model.pth' \
  --method_suffix fm_ema999_soft_w1_latest_val

touch FREEMATCH_EMA999_VAL_SCREEN_DONE
echo "[$(date)] FreeMatch EMA=0.999 validation screening complete"
