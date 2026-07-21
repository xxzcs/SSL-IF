#!/usr/bin/env bash
# USB 消融: Captum(fixmatch_if,train.py) + 闭式软(fixmatch_ifcf,train_ifcf.py,if_target=soft)。
# 对齐闭式赢家: multiply_balanced/强/corrT0.9/w2, p95(默认), 5 seed。凑齐 base/Captum/闭式软/闭式硬。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/ablation_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
SUMMARY=results/ablation_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
COMMON_IF="combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ifrank_loss_weight=2.0"
COMMON_CF="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=2.0 if_target=soft"
run(){ local sn=$1 algo=$2 launcher=$3 seed=$4; shift 4
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === $sn ($algo) ==="
  python "$launcher" --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python "$launcher" --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"
}
evalg(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}"; done; }
for s in 1 2 3 4 5; do run capt_s${s}   fixmatch_if   train.py      "$s" $COMMON_IF; done
evalg capt
for s in 1 2 3 4 5; do run cfsoft_s${s} fixmatch_ifcf train_ifcf.py "$s" $COMMON_CF; done
evalg cfsoft
touch ABLATION_DONE
echo "[$(date)] ===== ablation done ====="
