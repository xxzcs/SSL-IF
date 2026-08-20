#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate wssl 2>/dev/null || true

RATIO=${RATIO:-20}
SEEDS=${SEEDS:-"1 2 3 4 5"}
DATA_DIR=${DATA_DIR:-../uda_data/TN5000}
CE_CLASS_WEIGHTS=${CE_CLASS_WEIGHTS:-1.6957,0.7091}
LR=${LR:-0.002}
P_CUTOFF=${P_CUTOFF:-0.9}
FORCE_RERUN=${FORCE_RERUN:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-logs/tn5000_compare20_pipeline_${STAMP}.log}

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== TN5000 compare pipeline start ====="
echo "ratio=${RATIO} seeds=${SEEDS} data_dir=${DATA_DIR} lr=${LR} ce_class_weights=${CE_CLASS_WEIGHTS} p_cutoff=${P_CUTOFF}"

# USB main-table candidates. Each run auto-produces:
#   results/<method_prefix>_latest_threshold_summary.csv
#   results/<method_prefix>_best_threshold_summary.csv
USB_METHODS=(
  supervised
  fixmatch
  fixmatch_if
  adamatch
  flexmatch
  freematch
  softmatch
  refixmatch
  simmatch
  simmatchv2
)

for method in "${USB_METHODS[@]}"; do
  echo "[$(date)] ---- start ${method} ----"
  METHOD="${method}" \
  RATIO="${RATIO}" \
  SEEDS="${SEEDS}" \
  DATA_DIR="${DATA_DIR}" \
  CE_CLASS_WEIGHTS="${CE_CLASS_WEIGHTS}" \
  LR="${LR}" \
  P_CUTOFF="${P_CUTOFF}" \
  FORCE_RERUN="${FORCE_RERUN}" \
  AUTO_EVAL=1 \
  bash run_tn5000_ratio.sh
  echo "[$(date)] ---- done ${method} ----"
done

cat <<'EOF'
External-method queue not auto-started here:
  - SRC-MT
  - RankMatch-style FixMatch
  - Zeng et al.
  - Universal SSL for Medical Image Classification
These need their own launcher/eval wrappers to avoid blocking the USB pipeline.
EOF

echo "[$(date)] ===== TN5000 compare pipeline done ====="
