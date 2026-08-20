#!/usr/bin/env bash
set -u
cd /home/xiexiaozheng/Semi-supervised-learning

BASE=config/usb_cv/softmatch/softmatch_bus_878_0.yaml
SUMMARY=results/softmatch_soft_if_w1_5seed_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/softmatch_soft_if_w1_complete5_${STAMP}.log
mkdir -p logs results config

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate wssl 2>/dev/null

exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== SoftMatch soft IF w1 seed4/5 completion start, log=$LOG ====="

setkv() {
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}:" "$f"; then
    sed -i "s|^${k}:.*|${k}: ${v}|" "$f"
  else
    echo "${k}: ${v}" >> "$f"
  fi
}

run_one() {
  local seed=$1
  local sn="sm_ifsw1_s${seed}"
  if [ -f "saved_models/usb_cv/${sn}/latest_model.pth" ] && [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then
    echo "[skip] ${sn} already complete"
    return 0
  fi

  local tmp="config/_sm_${sn}.yaml"
  cp "$BASE" "$tmp"
  setkv "$tmp" algorithm softmatch_ifcf
  setkv "$tmp" save_name "$sn"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" ifrank_combine multiply_balanced
  setkv "$tmp" corrT 0.9
  setkv "$tmp" use_strong_if True
  setkv "$tmp" if_lambda 1
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" num_references 4
  setkv "$tmp" ref_select by_instance
  setkv "$tmp" ref_cand_k 8
  setkv "$tmp" if_target soft
  setkv "$tmp" if_mean_reduce True
  setkv "$tmp" ifrank_loss_weight 1.0

  echo "[$(date)] === train ${sn}: SoftMatch soft IF w1 ==="
  python train_ifcf.py --c "$tmp" || {
    echo "[warn] ${sn} failed once, retrying"
    python train_ifcf.py --c "$tmp" || {
      echo "[error] ${sn} failed twice"
      rm -f "$tmp"
      return 1
    }
  }
  rm -f "$tmp"
}

eval_group() {
  local dest=$1
  for kind in best latest; do
    local ck
    if [ "$kind" = best ]; then ck=model_best.pth; else ck=latest_model.pth; fi
    python3 eval_sup.py \
      --dataset bus \
      --num_classes 2 \
      --summary_csv "$SUMMARY" \
      --net resnet18 \
      --model_key ema_model \
      --data_dir ../uda_data \
      --batch_size 16 \
      --num_labels 878 \
      --eval_dest "$dest" \
      --lpath ../data_split/28/labeled_images_20_9.pth \
      --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/sm_ifsw1_s[12345]/${ck}" \
      --method_suffix "softmatch_soft_if_w1_5seed_${kind}_${dest}"
  done
}

run_one 4
run_one 5

eval_group test
eval_group eval

touch SOFTMATCH_SOFT_IF_W1_5SEED_DONE
echo "[$(date)] ===== SoftMatch soft IF w1 seed4/5 completion done. Summary: $SUMMARY ====="
