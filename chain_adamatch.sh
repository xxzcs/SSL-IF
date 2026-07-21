#!/usr/bin/env bash
# 等 SoftMatch 跑完 -> AdaMatch base + AdaMatch+IF#2(soft, 权重{1,2}), 各3seed。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/adamatch/adamatch_bus_878_0.yaml
SUMMARY=results/adamatch_if_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/adamatch_if_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] 等待 SoftMatch 完成 (ALL_DONE_SOFTMATCH)..."
while [ ! -f results/ALL_DONE_SOFTMATCH ]; do sleep 120; done
echo "[$(date)] SoftMatch 已完成, 开始 AdaMatch base + IF"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
IF2="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=soft if_mean_reduce=True"

run_one(){ local launcher=$1 algo=$2 sn=$3 seed=$4; shift 4
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] $sn"; return 0; fi
  local tmp="config/_am_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train $sn ($algo) ==="
  python "$launcher" --c "$tmp" || { echo "[warn] $sn 重试"; rm -rf "saved_models/usb_cv/${sn}"; python "$launcher" --c "$tmp" || echo "[warn] $sn 跳过"; }
  rm -f "$tmp"
}
eval_group(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1 $kind"; done; }

for s in 1 2 3; do run_one train.py      adamatch      "am_base_s${s}"  "$s"; done
eval_group am_base;  echo "[$(date)] ---- AdaMatch base done ----"
for s in 1 2 3; do run_one train_ifcf.py adamatch_ifcf "am_ifsw1_s${s}" "$s" $IF2 ifrank_loss_weight=1.0; done
eval_group am_ifsw1; echo "[$(date)] ---- AdaMatch +IF w1 done ----"
for s in 1 2 3; do run_one train_ifcf.py adamatch_ifcf "am_ifsw2_s${s}" "$s" $IF2 ifrank_loss_weight=2.0; done
eval_group am_ifsw2; echo "[$(date)] ---- AdaMatch +IF w2 done ----"

touch results/ALL_DONE_ADAMATCH
echo "[$(date)] ===== AdaMatch base+IF 完成. $SUMMARY ====="
