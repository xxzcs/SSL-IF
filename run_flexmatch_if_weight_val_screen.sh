#!/usr/bin/env bash
# FlexMatch IF weight screening: p=0.9, hard IF, zero5, 3 seeds, validation only.
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh
conda activate wssl

BASE=config/usb_cv/flexmatch/flexmatch_bus_878_0.yaml
SUMMARY=results/flexmatch_if_weight_val_screen.csv

setkv() {
  local file=$1 key=$2 value=$3
  if grep -qE "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    echo "${key}: ${value}" >> "$file"
  fi
}

run_one() {
  local tag=$1 weight=$2 seed=$3 name="flex_if_${tag}_s${seed}"
  if grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${name}/log.txt" 2>/dev/null; then
    echo "[$(date)] [skip] ${name} complete"
    return
  fi

  local cfg="config/_${name}.yaml"
  cp "$BASE" "$cfg"
  setkv "$cfg" algorithm flexmatch_ifcf
  setkv "$cfg" save_name "$name"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" p_cutoff 0.9
  setkv "$cfg" lr 0.0046875
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
  setkv "$cfg" ifrank_loss_weight "$weight"
  setkv "$cfg" if_target hard
  setkv "$cfg" use_strong_if True
  setkv "$cfg" corrT 0.9
  setkv "$cfg" ifrank_warmup_epochs 5
  setkv "$cfg" ifrank_warmup_mode zero

  echo "[$(date)] === ${name}, IF weight=${weight} ==="
  python train_ifcf.py --c "$cfg"
  rm -f "$cfg"
}

for spec in "w010 0.1" "w025 0.25" "w050 0.5"; do
  read -r tag weight <<< "$spec"
  for seed in 1 2 3; do
    run_one "$tag" "$weight" "$seed"
  done
done

rm -f "$SUMMARY"
python eval_sup.py \
  --dataset bus --num_classes 2 --summary_csv "$SUMMARY" \
  --net resnet18 --model_key ema_model --data_dir ../uda_data \
  --batch_size 16 --num_labels 878 --eval_dest eval \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob 'saved_models/usb_cv/flex_if_w010_s[123]/latest_model.pth' \
  --method_suffix flex_if_w010_latest_val
python eval_sup.py \
  --dataset bus --num_classes 2 --summary_csv "$SUMMARY" \
  --net resnet18 --model_key ema_model --data_dir ../uda_data \
  --batch_size 16 --num_labels 878 --eval_dest eval \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob 'saved_models/usb_cv/flex_if_w025_s[123]/latest_model.pth' \
  --method_suffix flex_if_w025_latest_val
python eval_sup.py \
  --dataset bus --num_classes 2 --summary_csv "$SUMMARY" \
  --net resnet18 --model_key ema_model --data_dir ../uda_data \
  --batch_size 16 --num_labels 878 --eval_dest eval \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob 'saved_models/usb_cv/flex_if_w050_s[123]/latest_model.pth' \
  --method_suffix flex_if_w050_latest_val

touch FLEXMATCH_IF_WEIGHT_VAL_SCREEN_DONE
echo "[$(date)] FlexMatch IF weight validation screening complete"
