#!/usr/bin/env bash
set -u

cd /home/xiexiaozheng/Semi-supervised-learning

BASE=${BASE:-config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml}
STAGE1_SUMMARY=${STAGE1_SUMMARY:-results/fixmatch_ifcf_bus_ld05_ablation_stage1_latest.csv}
SUMMARY=${SUMMARY:-results/fixmatch_ifcf_bus_ld05_ablation_latest.csv}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/fixmatch_ifcf_bus_ld05_ablation_${STAMP}.log}
SEEDS=${SEEDS:-"1 2 3 4 5"}
FORCE_RERUN=${FORCE_RERUN:-0}
LAYER_DECAY=${LAYER_DECAY:-0.5}

mkdir -p logs results config

source /home/xiexiaozheng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== FixMatch+IFCF BUS ld0.5 ablation queue start ====="
echo "base=${BASE}"
echo "stage1_summary=${STAGE1_SUMMARY}"
echo "summary=${SUMMARY}"
echo "seeds=${SEEDS}"
echo "layer_decay=${LAYER_DECAY}"
echo "log=${LOG}"

check_gpu() {
  python3 - <<'PY'
import sys
import torch
n = torch.cuda.device_count()
print(f"[gpu-check] torch.cuda.device_count()={n}", flush=True)
sys.exit(0 if n > 0 else 1)
PY
}

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

run_one() {
  local seed=$1
  local tag=$2
  local ref_select=$3
  local score_mode=$4
  local warmup=$5
  local save_name="${tag}_s${seed}"
  local tmp="config/_${save_name}.yaml"

  if [ "$FORCE_RERUN" != "1" ] && [ -f "saved_models/usb_cv/${save_name}/RUN_DONE" ]; then
    echo "[$(date)] [skip] ${save_name}"
    return 0
  fi

  if [ -d "saved_models/usb_cv/${save_name}" ] && { [ "$FORCE_RERUN" = "1" ] || [ ! -f "saved_models/usb_cv/${save_name}/latest_model.pth" ]; }; then
    echo "[$(date)] [clean] removing saved_models/usb_cv/${save_name}"
    rm -rf "saved_models/usb_cv/${save_name}"
  fi

  cp "$BASE" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf
  setkv "$tmp" save_name "$save_name"
  setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${save_name}/latest_model.pth"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" lr 0.0046875
  setkv "$tmp" batch_size 8
  setkv "$tmp" layer_decay "$LAYER_DECAY"
  setkv "$tmp" multiprocessing_distributed False
  setkv "$tmp" gpu None
  setkv "$tmp" resume False
  setkv "$tmp" overwrite True
  setkv "$tmp" ifrank_combine multiply_balanced
  setkv "$tmp" ifrank_score_mode "$score_mode"
  setkv "$tmp" corrT 0.9
  setkv "$tmp" use_strong_if True
  setkv "$tmp" if_lambda 1
  setkv "$tmp" csim_lambda 1
  setkv "$tmp" num_references 4
  setkv "$tmp" ref_select "$ref_select"
  setkv "$tmp" ref_cand_k 8
  setkv "$tmp" if_target hard
  setkv "$tmp" ifrank_loss_weight 1.0
  setkv "$tmp" ifrank_warmup_mode zero
  setkv "$tmp" ifrank_warmup_epochs "$warmup"
  setkv "$tmp" if_mean_reduce True

  echo "[$(date)] === train ${save_name} (ref=${ref_select} score=${score_mode} warmup=${warmup}) ==="
  python3 train_ifcf.py --c "$tmp"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} failed with rc=${rc}; retry once after cleanup"
    rm -rf "saved_models/usb_cv/${save_name}"
    python3 train_ifcf.py --c "$tmp"
    rc=$?
  fi

  rm -f "$tmp"
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [warn] ${save_name} still failed; continue"
    return 0
  fi

  touch "saved_models/usb_cv/${save_name}/RUN_DONE"
}

eval_group() {
  local tag=$1
  local summary_csv=$2
  local method_suffix="${tag}_latest"
  local pattern="saved_models/usb_cv/${tag}_s*/latest_model.pth"

  if ! compgen -G "$pattern" > /dev/null; then
    echo "[$(date)] [warn] no checkpoints for ${tag}; skip eval"
    return 0
  fi

  python3 eval_sup.py \
    --dataset bus \
    --num_classes 2 \
    --summary_csv "$summary_csv" \
    --net resnet18 \
    --model_key ema_model \
    --data_dir ../uda_data \
    --batch_size 16 \
    --num_labels 878 \
    --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth \
    --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "$pattern" \
    --method_suffix "$method_suffix" \
    || echo "[$(date)] [warn] eval latest failed for ${tag}"
}

pick_best_warmup() {
  python3 - "$STAGE1_SUMMARY" <<'PY'
import csv
import sys
path = sys.argv[1]
best = None
best_auc = -1.0
with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row["method"]
        if "fixmatch_ifcf_bus_ld05_orthogonal_fused_wu" not in method:
            continue
        auc = float(row["AUC"].split("±", 1)[0])
        warmup = method.split("_wu", 1)[1].split("_", 1)[0]
        if auc > best_auc:
            best_auc = auc
            best = warmup
if best is None:
    raise SystemExit("failed to choose warmup from stage1 summary")
print(best)
PY
}

run_group() {
  local tag=$1
  local ref_select=$2
  local score_mode=$3
  local warmup=$4
  local summary_csv=$5
  local seed

  for seed in $SEEDS; do
    run_one "$seed" "$tag" "$ref_select" "$score_mode" "$warmup"
  done
  eval_group "$tag" "$summary_csv"
}

check_gpu || { echo "[$(date)] [error] no visible GPU; abort BUS ablation queue"; exit 1; }

run_group "fixmatch_ifcf_bus_ld05_orthogonal_fused_wu5" orthogonal fused 5 "$STAGE1_SUMMARY"
run_group "fixmatch_ifcf_bus_ld05_orthogonal_fused_wu10" orthogonal fused 10 "$STAGE1_SUMMARY"

BEST_WARMUP=$(pick_best_warmup)
echo "[$(date)] stage1 best warmup=${BEST_WARMUP}"

run_group "fixmatch_ifcf_bus_ld05_random_fused_wu${BEST_WARMUP}" random fused "$BEST_WARMUP" "$SUMMARY"
run_group "fixmatch_ifcf_bus_ld05_supportonly_fused_wu${BEST_WARMUP}" support_only fused "$BEST_WARMUP" "$SUMMARY"
run_group "fixmatch_ifcf_bus_ld05_byinstance_cosine_wu${BEST_WARMUP}" by_instance cosine "$BEST_WARMUP" "$SUMMARY"
run_group "fixmatch_ifcf_bus_ld05_byinstance_if_wu${BEST_WARMUP}" by_instance if "$BEST_WARMUP" "$SUMMARY"

touch results/ALL_DONE_FIXMATCH_IFCF_BUS_LD05_ABLATION
echo "[$(date)] ===== FixMatch+IFCF BUS ld0.5 ablation queue done ====="
