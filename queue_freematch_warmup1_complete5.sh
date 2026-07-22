#!/usr/bin/env bash
# Complete the early FreeMatch soft-IF warmup1 experiment from 3 to 5 seeds.
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh
conda activate wssl

while [[ ! -f FLEXMATCH_LR0046875_W1_SCREEN_DONE ]]; do
  sleep 60
done

BASE=config/usb_cv/freematch/freematch_bus_878_0.yaml
SUMMARY_TEST=results/freematch_warmup1_5seed_summary.csv
SUMMARY_VAL=results/freematch_warmup1_5seed_val.csv

setkv() {
  local file=$1 key=$2 value=$3
  if grep -qE "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    echo "${key}: ${value}" >> "$file"
  fi
}

run_one() {
  local seed=$1 name="gen_fm_ifw1_s${seed}"
  if grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${name}/log.txt" 2>/dev/null; then
    echo "[$(date)] [skip] ${name} complete"
    return
  fi

  local cfg="config/_${name}.yaml"
  cp "$BASE" "$cfg"
  setkv "$cfg" algorithm freematch_ifcf
  setkv "$cfg" save_name "$name"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" batch_size 8
  setkv "$cfg" lr 0.0046875
  setkv "$cfg" ema_m 0.0
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
  setkv "$cfg" ifrank_warmup_epochs 1
  setkv "$cfg" ifrank_warmup_mode zero

  echo "[$(date)] === ${name}: FreeMatch soft IF w1, warmup1 ==="
  python train_ifcf.py --c "$cfg"
  rm -f "$cfg"
}

run_one 4
run_one 5

COMMON=(
  --dataset bus --num_classes 2 --net resnet18 --model_key ema_model
  --data_dir ../uda_data --batch_size 16 --num_labels 878
  --lpath ../data_split/28/labeled_images_20_9.pth
  --ulpath ../data_split/28/unlabeled_images_80_9.pth
)

rm -f "$SUMMARY_TEST" "$SUMMARY_VAL"
for destination in test eval; do
  if [[ "$destination" == test ]]; then summary=$SUMMARY_TEST; else summary=$SUMMARY_VAL; fi
  python eval_sup.py "${COMMON[@]}" --eval_dest "$destination" \
    --summary_csv "$summary" \
    --load_glob 'saved_models/usb_cv/gen_fm_base_s[12345]/latest_model.pth' \
    --method_suffix "freematch_lr0046875_base_latest_${destination}"
  python eval_sup.py "${COMMON[@]}" --eval_dest "$destination" \
    --summary_csv "$summary" \
    --load_glob 'saved_models/usb_cv/gen_fm_ifw1_s[12345]/latest_model.pth' \
    --method_suffix "freematch_lr0046875_soft_if_w1_warmup1_latest_${destination}"
done

touch FREEMATCH_WARMUP1_5SEED_DONE
echo "[$(date)] FreeMatch warmup1 five-seed completion and summaries done"
