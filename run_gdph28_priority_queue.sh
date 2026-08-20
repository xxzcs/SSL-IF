#!/usr/bin/env bash
# GDPH 20% priority queue for AAAI main-table completion.
# Default: 5 folds x 5 seeds, latest-checkpoint evaluation.
# Override examples:
#   METHODS="fixmatch_if" FOLDS="0" SEEDS="1" bash run_gdph28_priority_queue.sh
#   METHODS="fixmatch_if refixmatch softmatch adamatch" bash run_gdph28_priority_queue.sh
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/gdph28_priority_${STAMP}.log}
SUMMARY=${SUMMARY:-results/gdph28_priority_latest.csv}
FOLDMEAN=${FOLDMEAN:-$SUMMARY}
mkdir -p logs results config

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

exec > >(tee -a "$LOG") 2>&1

FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
METHODS=${METHODS:-"fixmatch_if refixmatch softmatch adamatch"}

GDPH_NUM_LABELS=384
GDPH_LABEL_RATIO=0.2
GDPH_NUM_TRAIN_ITER=2400
GDPH_NUM_LOG_ITER=48
GDPH_NUM_EVAL_ITER=48
GDPH_NUM_WARMUP_ITER=96
GDPH_LR_DROP_ITER="720 1440 2160"

# Locked BUS IF recipe from phase2.sh:
# hard / strong / corrT0.9 / multiply_balanced / w1 / warmup5-zero /
# num_ref4 / by_instance / ref_cand_k8 / lambda=1.
IFPLUG="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard use_strong_if=True corrT=0.9 ifrank_warmup_epochs=5 ifrank_warmup_mode=zero"

setkv() {
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}:" "$f"; then
    sed -i "s|^${k}:.*|${k}: ${v}|" "$f"
  else
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
  setkv "$cfg" batch_size 8
  setkv "$cfg" eval_batch_size 16
  setkv "$cfg" num_train_iter "$GDPH_NUM_TRAIN_ITER"
  setkv "$cfg" num_log_iter "$GDPH_NUM_LOG_ITER"
  setkv "$cfg" num_eval_iter "$GDPH_NUM_EVAL_ITER"
  setkv "$cfg" num_warmup_iter "$GDPH_NUM_WARMUP_ITER"
  setkv "$cfg" lr 0.0046875
  setkv "$cfg" sched linear
  setkv "$cfg" lr_drop_iter "$GDPH_LR_DROP_ITER"
  setkv "$cfg" train_sampler RandomSampler
  setkv "$cfg" num_workers 4

  # BUS configs use explicit file splits. GDPH uses fold + label_ratio splits.
  setkv "$cfg" lpath "''"
  setkv "$cfg" ulpath "''"
}

base_config_for_method() {
  case "$1" in
    fixmatch_if) echo "config/usb_cv/fixmatch/fixmatch_gdph_0.yaml" ;;
    refixmatch) echo "config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml" ;;
    softmatch) echo "config/usb_cv/softmatch/softmatch_bus_878_0.yaml" ;;
    adamatch) echo "config/usb_cv/adamatch/adamatch_bus_878_0.yaml" ;;
    *) return 1 ;;
  esac
}

algorithm_for_method() {
  case "$1" in
    fixmatch_if) echo "fixmatch_ifcf" ;;
    refixmatch) echo "refixmatch" ;;
    softmatch) echo "softmatch" ;;
    adamatch) echo "adamatch" ;;
    *) return 1 ;;
  esac
}

launcher_for_method() {
  case "$1" in
    fixmatch_if) echo "train_ifcf.py" ;;
    refixmatch|softmatch|adamatch) echo "train.py" ;;
    *) return 1 ;;
  esac
}

save_prefix_for_method() {
  case "$1" in
    fixmatch_if) echo "fixmatch_ifcf_gdph_28" ;;
    refixmatch) echo "refixmatch_gdph_28" ;;
    softmatch) echo "softmatch_gdph_28" ;;
    adamatch) echo "adamatch_gdph_28" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local method=$1 fold=$2 seed=$3
  local base algorithm launcher prefix save_name tmp
  base=$(base_config_for_method "$method") || { echo "[error] unknown method: $method"; return 1; }
  algorithm=$(algorithm_for_method "$method")
  launcher=$(launcher_for_method "$method")
  prefix=$(save_prefix_for_method "$method")
  save_name="${prefix}_fold${fold}_${seed}"

  if [ -f "saved_models/usb_cv/${save_name}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${save_name} RUN_DONE exists"
    return 0
  fi

  tmp="config/_gdph28_${save_name}.yaml"
  cp "$base" "$tmp"
  prepare_common_gdph28 "$tmp" "$save_name" "$fold" "$seed" "$algorithm"

  if [ "$method" = "fixmatch_if" ]; then
    for kv in $IFPLUG; do
      setkv "$tmp" "${kv%%=*}" "${kv#*=}"
    done
  fi

  echo "[$(date)] === train ${save_name} (${algorithm}) ==="
  python3 "$launcher" --c "$tmp"
  local rc=$?
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
  return 0
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

echo "[$(date)] ===== GDPH 20% priority queue start ====="
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

touch results/ALL_DONE_GDPH28_PRIORITY
echo "[$(date)] ===== GDPH 20% priority queue done: ${SUMMARY} ====="
