#!/usr/bin/env bash
set -euo pipefail

cd /home/xiexiaozheng/Semi-supervised-learning
TRAIN_PID=3548888
TRAIN_LOG=logs/refixmatch_fixed_base_s5_resume_20260721.log

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

grep -q "GPU 0 training is FINISHED" "$TRAIN_LOG"

/home/xiexiaozheng/anaconda3/envs/wssl/bin/python eval_sup.py \
  --dataset bus --num_classes 2 \
  --summary_csv results/refixmatch_fixed_base_summary.csv \
  --net resnet18 --model_key ema_model --data_dir ../uda_data \
  --batch_size 16 --num_labels 878 --eval_dest test \
  --lpath ../data_split/28/labeled_images_20_9.pth \
  --ulpath ../data_split/28/unlabeled_images_80_9.pth \
  --load_glob 'saved_models/usb_cv/refixmatch_fixed_base_s[12345]/latest_model.pth' \
  --method_suffix refixmatch_fixed_base_latest

touch REFIXMATCH_FIXED_BASE_DONE
echo "[$(date)] corrected ReFixMatch Base 5-seed resume and summary complete"
