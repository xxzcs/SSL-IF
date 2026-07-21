#!/usr/bin/env bash
# 等软目标 FreeMatch/FlexMatch 跑完 -> 跑 c4(multiplyo λ15/T0.05/强/soft) + 置信度mask, 3seed。
# 对照: c4 无mask(faithful5) 3seed 0.8792/0.8142 ; 基线 0.8771/0.8119。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SUMMARY=results/c4mask_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/c4mask_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] 等待 软目标复跑完成 (ALL_DONE_GENERALITY_SOFT)..."
while [ ! -f results/ALL_DONE_GENERALITY_SOFT ]; do sleep 120; done
echo "[$(date)] 软目标已完成, 开始 c4+mask"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
CFG="ifrank_mode=add ifrank_combine=multiplyo corrT=0.05 use_strong_if=True if_lambda=15 csim_lambda=1 ifrank_loss_weight=1 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=soft if_mean_reduce=True if_mask=True"

for s in 1 2 3; do
  sn="simmatch_if_bus_c4mask_s${s}"
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] $sn"; continue; fi
  tmp="config/_cm_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" algorithm simmatch_if; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$s"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  for kv in $CFG; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train $sn ==="
  python train_simmatch_if.py --c "$tmp" || { echo "[warn] $sn 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if.py --c "$tmp" || echo "[warn] $sn 跳过"; }
  rm -f "$tmp"
done
for kind in best latest; do
  [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/simmatch_if_bus_c4mask_*/${ck}" --method_suffix "c4mask_${kind}" || echo "[warn] eval $kind"
done
touch results/ALL_DONE_C4MASK
echo "[$(date)] ===== c4+mask 完成. $SUMMARY (对照 c4无mask 0.8792/0.8142, 基线 0.8771/0.8119) ====="
