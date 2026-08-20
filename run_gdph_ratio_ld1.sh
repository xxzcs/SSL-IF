#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

METHOD=${METHOD:-supervised}
RATIO=${RATIO:-10}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
RESET_SUMMARY=${RESET_SUMMARY:-1}

case "$RATIO" in
  10)
    GDPH_NUM_LABELS=192
    GDPH_LABEL_RATIO=0.1
    SUP_TRAIN_ITER=600
    SUP_EVAL_ITER=12
    SUP_LOG_ITER=12
    SUP_WARMUP_ITER=24
    SUP_LR_DROP_ITER="180 360 540"
    SSL_TRAIN_ITER=1200
    SSL_EVAL_ITER=24
    SSL_LOG_ITER=24
    SSL_WARMUP_ITER=48
    SSL_LR_DROP_ITER="360 720 1080"
    RATIO_TAG=19
    ;;
  15)
    GDPH_NUM_LABELS=288
    GDPH_LABEL_RATIO=0.15
    SUP_TRAIN_ITER=900
    SUP_EVAL_ITER=18
    SUP_LOG_ITER=18
    SUP_WARMUP_ITER=36
    SUP_LR_DROP_ITER="270 540 810"
    SSL_TRAIN_ITER=1800
    SSL_EVAL_ITER=36
    SSL_LOG_ITER=36
    SSL_WARMUP_ITER=72
    SSL_LR_DROP_ITER="540 1080 1620"
    RATIO_TAG=15
    ;;
  30)
    GDPH_NUM_LABELS=577
    GDPH_LABEL_RATIO=0.3
    SUP_TRAIN_ITER=1850
    SUP_EVAL_ITER=37
    SUP_LOG_ITER=37
    SUP_WARMUP_ITER=74
    SUP_LR_DROP_ITER="555 1110 1665"
    SSL_TRAIN_ITER=3650
    SSL_EVAL_ITER=73
    SSL_LOG_ITER=73
    SSL_WARMUP_ITER=146
    SSL_LR_DROP_ITER="1095 2190 3285"
    RATIO_TAG=37
    ;;
  *)
    echo "Unknown RATIO=$RATIO"
    exit 1
    ;;
esac

case "$METHOD" in
  supervised)
    BASE=${BASE:-config/usb_cv/supervised/supervised_gdph_28_0.yaml}
    ALGORITHM=supervised
    LAUNCHER=train.py
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    LR=${LR:-0.009375}
    URATIO=1
    NUM_TRAIN_ITER=$SUP_TRAIN_ITER
    NUM_EVAL_ITER=$SUP_EVAL_ITER
    NUM_LOG_ITER=$SUP_LOG_ITER
    NUM_WARMUP_ITER=$SUP_WARMUP_ITER
    LR_DROP_ITER=$SUP_LR_DROP_ITER
    NAME_PREFIX="supervised_gdph_${RATIO_TAG}_ld1"
    SUMMARY="results/supervised_gdph_${RATIO_TAG}_ld1_latest.csv"
    FOLDMEAN="results/supervised_gdph_${RATIO_TAG}_ld1_foldmean.csv"
    ;;
  fixmatch)
    BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_gdph_0.yaml}
    ALGORITHM=fixmatch
    LAUNCHER=train.py
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    LR=${LR:-0.0046875}
    URATIO=30
    NUM_TRAIN_ITER=$SSL_TRAIN_ITER
    NUM_EVAL_ITER=$SSL_EVAL_ITER
    NUM_LOG_ITER=$SSL_LOG_ITER
    NUM_WARMUP_ITER=$SSL_WARMUP_ITER
    LR_DROP_ITER=$SSL_LR_DROP_ITER
    NAME_PREFIX="fixmatch_gdph_${RATIO_TAG}_ld1"
    SUMMARY="results/fixmatch_gdph_${RATIO_TAG}_ld1_latest.csv"
    FOLDMEAN="results/fixmatch_gdph_${RATIO_TAG}_ld1_foldmean.csv"
    ;;
  fixmatch_if)
    BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_gdph_0.yaml}
    ALGORITHM=fixmatch_ifcf
    LAUNCHER=train_ifcf.py
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
    EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    LR=${LR:-0.0046875}
    URATIO=30
    NUM_TRAIN_ITER=$SSL_TRAIN_ITER
    NUM_EVAL_ITER=$SSL_EVAL_ITER
    NUM_LOG_ITER=$SSL_LOG_ITER
    NUM_WARMUP_ITER=$SSL_WARMUP_ITER
    LR_DROP_ITER=$SSL_LR_DROP_ITER
    NAME_PREFIX="fixmatch_ifcf_hard_gdph_${RATIO_TAG}_ld1"
    SUMMARY="results/fixmatch_ifcf_hard_gdph_${RATIO_TAG}_ld1_latest.csv"
    FOLDMEAN="results/fixmatch_ifcf_hard_gdph_${RATIO_TAG}_ld1_foldmean.csv"
    ;;
  *)
    echo "Unknown METHOD=$METHOD"
    exit 1
    ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/${NAME_PREFIX}_${STAMP}.log}

mkdir -p logs results config
exec > >(tee -a "$LOG") 2>&1

if [ "$RESET_SUMMARY" = "1" ]; then
  rm -f "$SUMMARY" "$FOLDMEAN"
fi

ensure_trailing_newline() {
  local f=$1
  [ -s "$f" ] || return 0
  tail -c 1 "$f" | od -An -t x1 | tr -d ' \n' | grep -qi '^0a$' && return 0
  printf "\n" >> "$f"
}

setkv() {
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}:" "$f"; then
    sed -i "s|^${k}:.*|${k}: ${v}|" "$f"
  else
    ensure_trailing_newline "$f"
    printf "%s: %s\n" "$k" "$v" >> "$f"
  fi
}

prepare_common_cfg() {
  local cfg=$1 save_name=$2 fold=$3 seed=$4

  setkv "$cfg" algorithm "$ALGORITHM"
  setkv "$cfg" save_name "$save_name"
  setkv "$cfg" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$cfg" resume False
  setkv "$cfg" overwrite True
  setkv "$cfg" multiprocessing_distributed False
  setkv "$cfg" gpu None

  setkv "$cfg" dataset gdph
  setkv "$cfg" data_dir ../uda_data/GDPH
  setkv "$cfg" num_classes 2
  setkv "$cfg" net resnet18
  setkv "$cfg" net_from_name False
  setkv "$cfg" img_size 224
  setkv "$cfg" crop_ratio 0.875
  setkv "$cfg" use_pretrain False

  setkv "$cfg" fold "$fold"
  setkv "$cfg" seed "$seed"
  setkv "$cfg" split_seed 0
  setkv "$cfg" num_labels "$GDPH_NUM_LABELS"
  setkv "$cfg" label_ratio "$GDPH_LABEL_RATIO"
  setkv "$cfg" uratio "$URATIO"
  setkv "$cfg" batch_size "$TRAIN_BATCH_SIZE"
  setkv "$cfg" eval_batch_size "$EVAL_BATCH_SIZE"
  setkv "$cfg" num_train_iter "$NUM_TRAIN_ITER"
  setkv "$cfg" num_log_iter "$NUM_LOG_ITER"
  setkv "$cfg" num_eval_iter "$NUM_EVAL_ITER"
  setkv "$cfg" num_warmup_iter "$NUM_WARMUP_ITER"
  setkv "$cfg" lr "$LR"
  setkv "$cfg" sched linear
  setkv "$cfg" lr_drop_iter "$LR_DROP_ITER"
  setkv "$cfg" layer_decay 1.0
  setkv "$cfg" train_sampler RandomSampler
  setkv "$cfg" num_workers 4
  setkv "$cfg" lpath "''"
  setkv "$cfg" ulpath "''"
}

apply_if_overrides() {
  local cfg=$1
  setkv "$cfg" ifrank_combine multiply_balanced
  setkv "$cfg" if_lambda 1
  setkv "$cfg" csim_lambda 1
  setkv "$cfg" if_tracin_scale 1.0
  setkv "$cfg" num_references 4
  setkv "$cfg" ref_select by_instance
  setkv "$cfg" ref_cand_k 8
  setkv "$cfg" use_strong_if True
  setkv "$cfg" corrT 0.9
  setkv "$cfg" ifrank_warmup_epochs 5
  setkv "$cfg" ifrank_warmup_mode zero
  setkv "$cfg" if_target hard
  setkv "$cfg" ifrank_loss_weight 1.0
}

echo "[$(date)] ===== GDPH ratio=${RATIO}% method=${METHOD} ld=1.0 start ====="
echo "summary=${SUMMARY}"
echo "foldmean=${FOLDMEAN}"
echo "folds=${FOLDS}"
echo "seeds=${SEEDS}"
echo "log=${LOG}"

for fold in $FOLDS; do
  for seed in $SEEDS; do
    save_name="${NAME_PREFIX}_fold${fold}_${seed}"
    if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${save_name}/RUN_DONE" ]; then
      echo "[$(date)] [skip] ${save_name}"
      continue
    fi
    if [ -d "saved_models/usb_cv/${save_name}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${save_name}/latest_model.pth" ]; }; then
      rm -rf "saved_models/usb_cv/${save_name}"
    fi
    tmp="config/_${save_name}.yaml"
    cp "$BASE" "$tmp"
    prepare_common_cfg "$tmp" "$save_name" "$fold" "$seed"
    if [ "$METHOD" = "fixmatch_if" ]; then
      apply_if_overrides "$tmp"
    fi
    echo "[$(date)] === train ${save_name} ==="
    if ! python3 "$LAUNCHER" --c "$tmp"; then
      echo "[$(date)] [retry] ${save_name}"
      rm -rf "saved_models/usb_cv/${save_name}"
      python3 "$LAUNCHER" --c "$tmp"
    fi
    if [ ! -f "saved_models/usb_cv/${save_name}/latest_model.pth" ]; then
      echo "[$(date)] [error] missing latest_model.pth for ${save_name}"
      exit 1
    fi
    rm -f "$tmp"
    touch "saved_models/usb_cv/${save_name}/RUN_DONE"
  done
done

python3 eval_sup_cv.py \
  --load_glob_template "saved_models/usb_cv/${NAME_PREFIX}_fold{fold}_*/latest_model.pth" \
  --folds 0 1 2 3 4 \
  --dataset gdph --num_classes 2 --net resnet18 --model_key ema_model \
  --data_dir ../uda_data/GDPH \
  --label_ratio "$GDPH_LABEL_RATIO" --num_labels "$GDPH_NUM_LABELS" \
  --batch_size "$EVAL_BATCH_SIZE" --num_workers 0 --eval_dest eval \
  --summary_csv "$SUMMARY" \
  --foldmean_csv "$FOLDMEAN" \
  --method_suffix latest

touch "results/ALL_DONE_${NAME_PREFIX^^}"
echo "[$(date)] ===== GDPH ratio=${RATIO}% method=${METHOD} ld=1.0 done ====="
