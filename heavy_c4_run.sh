#!/usr/bin/env bash
# 重型 Captum TracIn 版 SimMatch, c4(multiplyo λ15/T0.05/强/w1), 3 seed。对标闭式c4/基线。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SUMMARY=results/heavy_c4_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/heavy_c4_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== 重型 c4 (Captum TracIn) 开始 ====="

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
CFG="ifrank_mode=add ifrank_combine=multiplyo corrT=0.05 use_strong_if=True if_lambda=15 csim_lambda=1 ifrank_loss_weight=1 num_references=4 ref_select=by_instance ref_cand_k=8"

for s in 1 2 3; do
  sn="simmatch_if_bus_hvc4_s${s}"
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] $sn"; continue; fi
  tmp="config/_hv_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" algorithm simmatch_if_heavy; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  for kv in $CFG; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train $sn ==="
  python train_simmatch_if_heavy.py --c "$tmp" || { echo "[warn] $sn 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if_heavy.py --c "$tmp" || echo "[warn] $sn 跳过"; }
  rm -f "$tmp"
done

for kind in best latest; do
  [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/simmatch_if_bus_hvc4_*/${ck}" --method_suffix "hvc4_${kind}" || echo "[warn] eval $kind"
done
touch results/ALL_DONE_HEAVY_C4
echo "[$(date)] ===== 重型 c4 完成. $SUMMARY (对标 闭式c4 3seed0.8792/5seed0.8753, 基线0.8771/0.8119) ====="
