#!/usr/bin/env bash
# Overnight queue for BUS IF experiments, 2026-07-21.
# Runs one GPU task at a time and avoids choosing checkpoints by test results.
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh
conda activate wssl

LOG_PREFIX="[overnight $(date +%Y%m%d_%H%M%S)]"
echo "${LOG_PREFIX} start"

setkv() {
  local file=$1 key=$2 value=$3
  if grep -qE "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    echo "${key}: ${value}" >> "$file"
  fi
}

is_complete() {
  local name=$1
  grep -q "GPU 0 training is FINISHED" "saved_models/usb_cv/${name}/log.txt" 2>/dev/null
}

eval_group() {
  local summary=$1 destination=$2 glob=$3 suffix=$4
  python eval_sup.py \
    --dataset bus --num_classes 2 --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 \
    --eval_dest "$destination" \
    --lpath ../data_split/28/labeled_images_20_9.pth \
    --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --summary_csv "$summary" \
    --load_glob "$glob" \
    --method_suffix "$suffix"
}

run_flex_one() {
  local kind=$1 seed=$2
  local base=config/usb_cv/flexmatch/flexmatch_bus_878_0.yaml
  local name="flex_lr0046875_${kind}_s${seed}"
  if is_complete "$name"; then
    echo "[$(date)] skip ${name}, already complete"
    return
  fi

  local cfg="config/_${name}.yaml"
  cp "$base" "$cfg"
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

  echo "[$(date)] train ${name}"
  python "$launcher" --c "$cfg"
  rm -f "$cfg"
}

run_freematch_warmup1_one() {
  local seed=$1
  local base=config/usb_cv/freematch/freematch_bus_878_0.yaml
  local name="gen_fm_ifw1_s${seed}"
  if is_complete "$name"; then
    echo "[$(date)] skip ${name}, already complete"
    return
  fi

  local cfg="config/_${name}.yaml"
  cp "$base" "$cfg"
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

  echo "[$(date)] train ${name}"
  python train_ifcf.py --c "$cfg"
  rm -f "$cfg"
}

run_refix_if_one() {
  local seed=$1
  local base=config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml
  local name="refixmatch_fixed_ifw1_s${seed}"
  if is_complete "$name"; then
    echo "[$(date)] skip ${name}, already complete"
    return
  fi

  local cfg="config/_${name}.yaml"
  cp "$base" "$cfg"
  setkv "$cfg" algorithm refixmatch_ifcf
  setkv "$cfg" save_name "$name"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" batch_size 8
  setkv "$cfg" lr 0.0046875
  setkv "$cfg" p_cutoff 0.9
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
  setkv "$cfg" if_target hard
  setkv "$cfg" use_strong_if True
  setkv "$cfg" corrT 0.9
  setkv "$cfg" ifrank_warmup_epochs 5
  setkv "$cfg" ifrank_warmup_mode zero

  echo "[$(date)] train ${name}"
  python train_ifcf.py --c "$cfg"
  rm -f "$cfg"
}

echo "[$(date)] waiting for current FlexMatch 3-seed screen"
while [[ ! -f FLEXMATCH_LR0046875_W1_SCREEN_DONE ]]; do
  sleep 60
done

echo "[$(date)] phase 1: complete FlexMatch lr=0.0046875 p=0.9 to five seeds"
for seed in 4 5; do run_flex_one base "$seed"; done
for seed in 4 5; do run_flex_one ifw1 "$seed"; done

rm -f results/flexmatch_lr0046875_w1_5seed_val.csv results/flexmatch_lr0046875_w1_5seed_test.csv
for dest in eval test; do
  if [[ "$dest" == eval ]]; then summary=results/flexmatch_lr0046875_w1_5seed_val.csv; else summary=results/flexmatch_lr0046875_w1_5seed_test.csv; fi
  eval_group "$summary" "$dest" 'saved_models/usb_cv/flex_lr0046875_base_s[12345]/latest_model.pth' "flex_lr0046875_p90_base_latest_${dest}"
  eval_group "$summary" "$dest" 'saved_models/usb_cv/flex_lr0046875_ifw1_s[12345]/latest_model.pth' "flex_lr0046875_p90_hard_if_w1_warmup5_latest_${dest}"
done
touch FLEXMATCH_LR0046875_W1_5SEED_DONE

echo "[$(date)] phase 2: complete FreeMatch warmup1 soft IF to five seeds"
for seed in 4 5; do run_freematch_warmup1_one "$seed"; done

rm -f results/freematch_warmup1_5seed_summary.csv results/freematch_warmup1_5seed_val.csv
for dest in test eval; do
  if [[ "$dest" == test ]]; then summary=results/freematch_warmup1_5seed_summary.csv; else summary=results/freematch_warmup1_5seed_val.csv; fi
  eval_group "$summary" "$dest" 'saved_models/usb_cv/gen_fm_base_s[12345]/latest_model.pth' "freematch_lr0046875_base_latest_${dest}"
  eval_group "$summary" "$dest" 'saved_models/usb_cv/gen_fm_ifw1_s[12345]/latest_model.pth' "freematch_lr0046875_soft_if_w1_warmup1_latest_${dest}"
done
touch FREEMATCH_WARMUP1_5SEED_DONE

echo "[$(date)] phase 3: run corrected ReFixMatch p=0.9 hard IF, five seeds"
for seed in 1 2 3 4 5; do run_refix_if_one "$seed"; done

rm -f results/refixmatch_fixed_ifw1_summary.csv results/refixmatch_fixed_ifw1_val.csv
for dest in test eval; do
  if [[ "$dest" == test ]]; then summary=results/refixmatch_fixed_ifw1_summary.csv; else summary=results/refixmatch_fixed_ifw1_val.csv; fi
  eval_group "$summary" "$dest" 'saved_models/usb_cv/refixmatch_fixed_base_s[12345]/latest_model.pth' "refixmatch_fixed_p90_base_latest_${dest}"
  eval_group "$summary" "$dest" 'saved_models/usb_cv/refixmatch_fixed_ifw1_s[12345]/latest_model.pth' "refixmatch_fixed_p90_hard_if_w1_warmup5_latest_${dest}"
done
touch REFIXMATCH_FIXED_IFW1_5SEED_DONE

echo "[$(date)] phase 4: summarize existing SimMatch latest checkpoints"
rm -f results/simmatch_existing_latest_test.csv results/simmatch_existing_latest_val.csv
for dest in test eval; do
  if [[ "$dest" == test ]]; then summary=results/simmatch_existing_latest_test.csv; else summary=results/simmatch_existing_latest_val.csv; fi
  eval_group "$summary" "$dest" 'saved_models/usb_cv/simmatch_bus_878_da1_*/latest_model.pth' "simmatch_da1_base_latest_${dest}" || true
  eval_group "$summary" "$dest" 'saved_models/usb_cv/simmatch_if_bus_f5_c4_s[12345]/latest_model.pth' "simmatch_if_f5_c4_latest_${dest}" || true
  eval_group "$summary" "$dest" 'saved_models/usb_cv/simmatch_ifrank_bus_878_l1.0_[12345]/latest_model.pth' "simmatch_ifrank_l1_latest_${dest}" || true
done
touch SIMMATCH_EXISTING_LATEST_SUMMARY_DONE

echo "[$(date)] overnight queue complete"
touch OVERNIGHT_AUTO_IF_QUEUE_20260721_DONE
