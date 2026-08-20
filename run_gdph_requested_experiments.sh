#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/gdph_requested_${STAMP}.log}
SUMMARY=${SUMMARY:-results/gdph_requested_latest.csv}
FOLDMEAN=${FOLDMEAN:-results/gdph_requested_foldmean.csv}
mkdir -p logs results config

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

exec > >(tee -a "$LOG") 2>&1

FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
METHODS=${METHODS:-"if_hard_warmup1 if_hard_warmup10 softmatch adamatch"}

GDPH_NUM_LABELS=384
GDPH_LABEL_RATIO=0.2
GDPH_NUM_TRAIN_ITER=2400
GDPH_NUM_LOG_ITER=48
GDPH_NUM_EVAL_ITER=48
GDPH_NUM_WARMUP_ITER=96
GDPH_LR_DROP_ITER="720 1440 2160"
GDPH_LAYER_DECAY=1.0
GDPH_BATCH_SIZE=${GDPH_BATCH_SIZE:-8}

IF_COMMON="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 use_strong_if=True corrT=0.9 ifrank_warmup_mode=zero"

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
  setkv "$cfg" lpath "''"
  setkv "$cfg" ulpath "''"
}

base_config_for_method() {
  case "$1" in
    if_hard_warmup1|if_hard_warmup10|if_hard_t05_warmup5) echo "config/usb_cv/fixmatch/fixmatch_gdph_0.yaml" ;;
    softmatch) echo "config/usb_cv/softmatch/softmatch_bus_878_0.yaml" ;;
    adamatch) echo "config/usb_cv/adamatch/adamatch_bus_878_0.yaml" ;;
    *) return 1 ;;
  esac
}

algorithm_for_method() {
  case "$1" in
    if_hard_warmup1|if_hard_warmup10|if_hard_t05_warmup5) echo "fixmatch_ifcf" ;;
    softmatch) echo "softmatch" ;;
    adamatch) echo "adamatch" ;;
    *) return 1 ;;
  esac
}

launcher_for_method() {
  case "$1" in
    if_hard_warmup1|if_hard_warmup10|if_hard_t05_warmup5) echo "train_ifcf.py" ;;
    softmatch|adamatch) echo "train.py" ;;
    *) return 1 ;;
  esac
}

save_prefix_for_method() {
  case "$1" in
    if_hard_warmup1) echo "fixmatch_ifcf_hard_gdph_28_warmup1_ld1" ;;
    if_hard_warmup10) echo "fixmatch_ifcf_hard_gdph_28_warmup10_ld1" ;;
    if_hard_t05_warmup5) echo "fixmatch_ifcf_hard_gdph_28_t05_warmup5_ld1" ;;
    softmatch) echo "softmatch_gdph_28_ld1" ;;
    adamatch) echo "adamatch_gdph_28_ld1" ;;
    *) return 1 ;;
  esac
}

apply_method_overrides() {
  local cfg=$1 method=$2

  case "$method" in
    if_hard_warmup1)
      for kv in $IF_COMMON; do setkv "$cfg" "${kv%%=*}" "${kv#*=}"; done
      setkv "$cfg" if_target hard
      setkv "$cfg" ifrank_loss_weight 1.0
      setkv "$cfg" ifrank_warmup_epochs 1
      ;;
    if_hard_warmup10)
      for kv in $IF_COMMON; do setkv "$cfg" "${kv%%=*}" "${kv#*=}"; done
      setkv "$cfg" if_target hard
      setkv "$cfg" ifrank_loss_weight 1.0
      setkv "$cfg" ifrank_warmup_epochs 10
      ;;
    if_hard_t05_warmup5)
      for kv in $IF_COMMON; do setkv "$cfg" "${kv%%=*}" "${kv#*=}"; done
      setkv "$cfg" corrT 0.5
      setkv "$cfg" if_target hard
      setkv "$cfg" ifrank_loss_weight 1.0
      setkv "$cfg" ifrank_warmup_epochs 5
      ;;
    adamatch)
      setkv "$cfg" layer_decay 1.0
      ;;
  esac
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

  if [ -d "saved_models/usb_cv/${save_name}" ]; then
    echo "[$(date)] [clean] removing stale directory saved_models/usb_cv/${save_name}"
    rm -rf "saved_models/usb_cv/${save_name}"
  fi

  tmp="config/_gdph_requested_${save_name}.yaml"
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

echo "[$(date)] ===== GDPH requested experiments start ====="
echo "methods=${METHODS}"
echo "folds=${FOLDS}"
echo "seeds=${SEEDS}"
echo "summary=${SUMMARY}"
echo "foldmean=${FOLDMEAN}"
echo "log=${LOG}"

for method in $METHODS; do
  echo "[$(date)] ----- method ${method} start -----"
  for fold in $FOLDS; do
    for seed in $SEEDS; do
      run_one "$method" "$fold" "$seed"
    done
  done
  eval_method_latest "$method"
  echo "[$(date)] ----- method ${method} done -----"
done

echo "[$(date)] ===== GDPH requested experiments done ====="
