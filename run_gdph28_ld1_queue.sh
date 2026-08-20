#!/usr/bin/env bash
# GDPH 20% queue for final AAAI results with layer_decay=1.0.
# Order: FixMatch anchor -> small IF screens -> remaining baselines.
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/gdph28_ld1_${STAMP}.log}
SUMMARY=${SUMMARY:-results/gdph28_ld1_latest.csv}
FOLDMEAN=${FOLDMEAN:-$SUMMARY}
mkdir -p logs results config

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

exec > >(tee -a "$LOG") 2>&1

FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SCREEN_SEEDS=${SCREEN_SEEDS:-"1"}
METHODS=${METHODS:-"fixmatch if_soft_w1_screen if_hard_w2_screen flexmatch freematch refixmatch softmatch adamatch simmatch"}

GDPH_NUM_LABELS=384
GDPH_LABEL_RATIO=0.2
GDPH_NUM_TRAIN_ITER=2400
GDPH_NUM_LOG_ITER=48
GDPH_NUM_EVAL_ITER=48
GDPH_NUM_WARMUP_ITER=96
GDPH_LR_DROP_ITER="720 1440 2160"
GDPH_LAYER_DECAY=1.0
GDPH_BATCH_SIZE=${GDPH_BATCH_SIZE:-8}

IF_COMMON="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 use_strong_if=True corrT=0.9 ifrank_warmup_epochs=5 ifrank_warmup_mode=zero"

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

prepare_common_gdph28() {
  local cfg=$1 save_name=$2 fold=$3 seed=$4 algorithm=$5

  setkv "$cfg" algorithm "$algorithm"
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
  setkv "$cfg" uratio 30
  setkv "$cfg" batch_size "$GDPH_BATCH_SIZE"
  setkv "$cfg" eval_batch_size 16
  setkv "$cfg" num_train_iter "$GDPH_NUM_TRAIN_ITER"
  setkv "$cfg" num_log_iter "$GDPH_NUM_LOG_ITER"
  setkv "$cfg" num_eval_iter "$GDPH_NUM_EVAL_ITER"
  setkv "$cfg" num_warmup_iter "$GDPH_NUM_WARMUP_ITER"
  setkv "$cfg" lr 0.0046875
  setkv "$cfg" sched linear
  setkv "$cfg" lr_drop_iter "$GDPH_LR_DROP_ITER"
  setkv "$cfg" layer_decay "$GDPH_LAYER_DECAY"
  setkv "$cfg" train_sampler RandomSampler
  setkv "$cfg" num_workers 4

  # GDPH uses label.csv fold + label_ratio, not BUS split files.
  setkv "$cfg" lpath "''"
  setkv "$cfg" ulpath "''"
}

base_config_for_method() {
  case "$1" in
    fixmatch|if_soft_w1_screen|if_hard_w2_screen|if_soft_w2_screen) echo "config/usb_cv/fixmatch/fixmatch_gdph_0.yaml" ;;
    flexmatch) echo "config/usb_cv/flexmatch/flexmatch_bus_878_fixed.yaml" ;;
    freematch) echo "config/usb_cv/freematch/freematch_gdph_0.yaml" ;;
    refixmatch) echo "config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml" ;;
    softmatch) echo "config/usb_cv/softmatch/softmatch_bus_878_0.yaml" ;;
    adamatch) echo "config/usb_cv/adamatch/adamatch_bus_878_0.yaml" ;;
    simmatch) echo "config/usb_cv/simmatch/simmatch_gdph_28_0.yaml" ;;
    *) return 1 ;;
  esac
}

algorithm_for_method() {
  case "$1" in
    fixmatch) echo "fixmatch" ;;
    if_soft_w1_screen|if_hard_w2_screen|if_soft_w2_screen) echo "fixmatch_ifcf" ;;
    flexmatch) echo "flexmatch" ;;
    freematch) echo "freematch" ;;
    refixmatch) echo "refixmatch" ;;
    softmatch) echo "softmatch" ;;
    adamatch) echo "adamatch" ;;
    simmatch) echo "simmatch" ;;
    *) return 1 ;;
  esac
}

launcher_for_method() {
  case "$1" in
    if_soft_w1_screen|if_hard_w2_screen|if_soft_w2_screen) echo "train_ifcf.py" ;;
    fixmatch|flexmatch|freematch|refixmatch|softmatch|adamatch|simmatch) echo "train.py" ;;
    *) return 1 ;;
  esac
}

save_prefix_for_method() {
  case "$1" in
    fixmatch) echo "fixmatch_gdph_28" ;;
    if_soft_w1_screen) echo "fixmatch_ifcf_soft_gdph_28_screen_w1" ;;
    if_hard_w2_screen) echo "fixmatch_ifcf_gdph_28_screen_w2" ;;
    if_soft_w2_screen) echo "fixmatch_ifcf_soft_gdph_28_screen_w2" ;;
    flexmatch) echo "flexmatch_gdph_28" ;;
    freematch) echo "freematch_gdph_28" ;;
    refixmatch) echo "refixmatch_gdph_28_ld1" ;;
    softmatch) echo "softmatch_gdph_28_ld1" ;;
    adamatch) echo "adamatch_gdph_28_ld1" ;;
    simmatch) echo "simmatch_gdph_28_ld1" ;;
    *) return 1 ;;
  esac
}

seeds_for_method() {
  case "$1" in
    if_soft_w1_screen|if_hard_w2_screen|if_soft_w2_screen) echo "$SCREEN_SEEDS" ;;
    *) echo "$SEEDS" ;;
  esac
}

apply_method_overrides() {
  local cfg=$1 method=$2

  if [ "$method" = "if_soft_w1_screen" ]; then
    for kv in $IF_COMMON; do setkv "$cfg" "${kv%%=*}" "${kv#*=}"; done
    setkv "$cfg" if_target soft
    setkv "$cfg" ifrank_loss_weight 1.0
  elif [ "$method" = "if_hard_w2_screen" ]; then
    for kv in $IF_COMMON; do setkv "$cfg" "${kv%%=*}" "${kv#*=}"; done
    setkv "$cfg" if_target hard
    setkv "$cfg" ifrank_loss_weight 2.0
  elif [ "$method" = "if_soft_w2_screen" ]; then
    for kv in $IF_COMMON; do setkv "$cfg" "${kv%%=*}" "${kv#*=}"; done
    setkv "$cfg" if_target soft
    setkv "$cfg" ifrank_loss_weight 2.0
  fi
}

run_one() {
  local method=$1 fold=$2 seed=$3
  local base algorithm launcher prefix save_name tmp rc
  base=$(base_config_for_method "$method") || { echo "[error] unknown method: $method"; return 1; }
  algorithm=$(algorithm_for_method "$method")
  launcher=$(launcher_for_method "$method")
  prefix=$(save_prefix_for_method "$method")
  save_name="${prefix}_fold${fold}_${seed}"

  if [ -f "saved_models/usb_cv/${save_name}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${save_name} RUN_DONE exists"
    return 0
  fi

  tmp="config/_gdph28_ld1_${save_name}.yaml"
  cp "$base" "$tmp"
  prepare_common_gdph28 "$tmp" "$save_name" "$fold" "$seed" "$algorithm"
  apply_method_overrides "$tmp" "$method"

  echo "[$(date)] === train ${save_name} (${algorithm}, layer_decay=${GDPH_LAYER_DECAY}) ==="
  python3 "$launcher" --c "$tmp"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} failed with rc=${rc}; retry once after cleanup"
    rm -rf "saved_models/usb_cv/${save_name}"
    python3 "$launcher" --c "$tmp"
    rc=$?
  fi
  rm -f "$tmp"
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} still failed; continue queue"
  else
    touch "saved_models/usb_cv/${save_name}/RUN_DONE"
  fi
}

eval_method_latest() {
  local method=$1 prefix
  prefix=$(save_prefix_for_method "$method")
  echo "[$(date)] === eval latest ${prefix} ==="
  python3 eval_sup_cv.py \
    --load_glob_template "saved_models/usb_cv/${prefix}_fold{fold}_*/latest_model.pth" \
    --folds 0 1 2 3 4 \
    --dataset gdph --num_classes 2 --net resnet18 --model_key ema_model \
    --data_dir ../uda_data/GDPH \
    --label_ratio "$GDPH_LABEL_RATIO" --num_labels "$GDPH_NUM_LABELS" \
    --batch_size 16 --num_workers 0 --eval_dest eval \
    --summary_csv "$SUMMARY" \
    --foldmean_csv "$FOLDMEAN" \
    --method_suffix latest || echo "[$(date)] [warn] eval ${prefix} latest failed"
}

echo "[$(date)] ===== GDPH 20% layer_decay=1.0 queue start ====="
echo "methods=${METHODS}"
echo "folds=${FOLDS}"
echo "seeds=${SEEDS}"
echo "screen_seeds=${SCREEN_SEEDS}"
echo "summary=${SUMMARY}"
echo "foldmean=${FOLDMEAN}"
echo "log=${LOG}"

for method in $METHODS; do
  echo "[$(date)] ----- method ${method} start -----"
  for fold in $FOLDS; do
    for seed in $(seeds_for_method "$method"); do
      run_one "$method" "$fold" "$seed"
    done
  done
  eval_method_latest "$method"
  echo "[$(date)] ----- method ${method} done -----"
done

touch results/ALL_DONE_GDPH28_LD1
echo "[$(date)] ===== GDPH 20% layer_decay=1.0 queue done: ${SUMMARY} ====="
