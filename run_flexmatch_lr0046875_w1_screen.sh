#!/usr/bin/env bash
# Establish the fair FlexMatch p=0.9, lr=0.0046875 Base vs IF-weight=1 baseline.
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh
conda activate wssl

BASE=config/usb_cv/flexmatch/flexmatch_bus_878_0.yaml
SUMMARY=results/flexmatch_lr0046875_w1_val_screen.csv

setkv() {
  local file=$1 key=$2 value=$3
  if grep -qE "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    echo "${key}: ${value}" >> "$file"
  fi
}

run_one() {
  local kind=$1 seed=$2
  local name="flex_lr0046875_${kind}_s${seed}"
  if grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${name}/log.txt" 2>/dev/null; then
    echo "[$(date)] [skip] ${name} complete"
    return
  fi

  local cfg="config/_${name}.yaml"
  cp "$BASE" "$cfg"
  setkv "$cfg" save_name "$name"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" batch_size 8
  setkv "$cfg" lr 0.0046875
  setkv "$cfg" p_cutoff 0.9
  setkv "$cfg" resume False
  setkv "$cfg" overwrite True
  setkv "$cfg" multiprocessing_distributed False
  setkv "$cfg" num_log_iter 110

  local launcher=train.py
  if [[ "$kind" == ifw1 ]]; then
    launcher=train_ifcf.py
    setkv "$cfg" algorithm flexmatch_ifcf
    setkv "$cfg" ifrank_combine multiply_balanced
    setkv "$cfg" if_lambda 1
    setkv "$cfg" csim_lambda 1
    setkv "$cfg" num_references 4
    setkv "$cfg" ref_select by_instance
    setkv "$cfg" ref_cand_k 8
    setkv "$cfg" ifrank_loss_weight 1.0
    setkv "$cfg" if_target hard
    setkv "$cfg" use_strong_if True
    setkv "$cfg" corrT 0.9
    setkv "$cfg" ifrank_warmup_epochs 5
    setkv "$cfg" ifrank_warmup_mode zero
  else
    setkv "$cfg" algorithm flexmatch
  fi

  echo "[$(date)] === ${name} ==="
  python "$launcher" --c "$cfg"
  rm -f "$cfg"
}

for seed in 1 2 3; do run_one base "$seed"; done
for seed in 1 2 3; do run_one ifw1 "$seed"; done

rm -f "$SUMMARY"
COMMON=(
  --dataset bus --num_classes 2 --net resnet18 --model_key ema_model
  --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest eval
  --lpath ../data_split/28/labeled_images_20_9.pth
  --ulpath ../data_split/28/unlabeled_images_80_9.pth
)
python eval_sup.py "${COMMON[@]}" --summary_csv "$SUMMARY" \
  --load_glob 'saved_models/usb_cv/flex_lr0046875_base_s[123]/latest_model.pth' \
  --method_suffix flex_lr0046875_p90_base_latest_val
python eval_sup.py "${COMMON[@]}" --summary_csv "$SUMMARY" \
  --load_glob 'saved_models/usb_cv/flex_lr0046875_ifw1_s[123]/latest_model.pth' \
  --method_suffix flex_lr0046875_p90_hard_if_w1_warmup5_latest_val

touch FLEXMATCH_LR0046875_W1_SCREEN_DONE
echo "[$(date)] FlexMatch lr=0.0046875 Base/IF-w1 validation screen complete"
