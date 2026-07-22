#!/usr/bin/env bash
# SimMatch + hard IF screen aligned with the Fix/Flex/ReFix IF recipe.
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh
conda activate wssl

BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SUMMARY_VAL=results/simmatch_hard_if_unified_3seed_val.csv
SUMMARY_TEST=results/simmatch_hard_if_unified_3seed_test.csv

setkv() {
  local file=$1 key=$2 value=$3
  if grep -qE "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    echo "${key}: ${value}" >> "$file"
  fi
}

run_one() {
  local seed=$1
  local name="simmatch_if_hard_w1_unified_s${seed}"
  if grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${name}/log.txt" 2>/dev/null; then
    echo "[$(date)] skip ${name}, already complete"
    return
  fi

  local cfg="config/_${name}.yaml"
  cp "$BASE" "$cfg"
  setkv "$cfg" algorithm simmatch_if
  setkv "$cfg" save_name "$name"
  setkv "$cfg" load_path "./saved_models/usb_cv/${name}/latest_model.pth"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" batch_size 8
  setkv "$cfg" lr 0.0046875
  setkv "$cfg" p_cutoff 0.9
  setkv "$cfg" T 0.1
  setkv "$cfg" ema_m 0.999
  setkv "$cfg" use_da True
  setkv "$cfg" use_epass False
  setkv "$cfg" ifrank_mode add
  setkv "$cfg" ifrank_combine multiply_balanced
  setkv "$cfg" ifrank_loss_weight 1.0
  setkv "$cfg" if_lambda 1
  setkv "$cfg" csim_lambda 1
  setkv "$cfg" corrT 0.9
  setkv "$cfg" num_references 4
  setkv "$cfg" ref_select by_instance
  setkv "$cfg" ref_cand_k 8
  setkv "$cfg" use_strong_if True
  setkv "$cfg" if_target hard
  setkv "$cfg" if_mean_reduce True
  setkv "$cfg" ifrank_warmup_epochs 5
  setkv "$cfg" ifrank_warmup_mode zero
  setkv "$cfg" resume False
  setkv "$cfg" overwrite True
  setkv "$cfg" multiprocessing_distributed False
  setkv "$cfg" num_log_iter 110

  echo "[$(date)] train ${name}"
  python train_simmatch_if.py --c "$cfg"
  rm -f "$cfg"
}

for seed in 1 2 3; do
  run_one "$seed"
done

COMMON=(
  --dataset bus --num_classes 2 --net resnet18 --model_key ema_model
  --data_dir ../uda_data --batch_size 16 --num_labels 878
  --lpath ../data_split/28/labeled_images_20_9.pth
  --ulpath ../data_split/28/unlabeled_images_80_9.pth
)

rm -f "$SUMMARY_VAL" "$SUMMARY_TEST"
for dest in eval test; do
  if [[ "$dest" == eval ]]; then summary=$SUMMARY_VAL; else summary=$SUMMARY_TEST; fi
  python eval_sup.py "${COMMON[@]}" --eval_dest "$dest" \
    --summary_csv "$summary" \
    --load_glob 'saved_models/usb_cv/simmatch_bus_878_da1_[123]/latest_model.pth' \
    --method_suffix "simmatch_da1_base_latest_${dest}_s123"
  python eval_sup.py "${COMMON[@]}" --eval_dest "$dest" \
    --summary_csv "$summary" \
    --load_glob 'saved_models/usb_cv/simmatch_if_hard_w1_unified_s[123]/latest_model.pth' \
    --method_suffix "simmatch_hard_if_w1_unified_latest_${dest}_s123"
done

touch SIMMATCH_HARD_IF_UNIFIED_3SEED_DONE
echo "[$(date)] SimMatch hard IF unified 3seed screen complete"
