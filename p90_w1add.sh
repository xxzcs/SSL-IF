#!/usr/bin/env bash
# 补齐 p90 消融的 w1: Captum(w1) + 闭式软(w1), 5 seed, p_cutoff=0.9, balanced/强/0.9。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/p90w1add_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml; SUMMARY=results/p90_all_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
IF="combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4"
CF="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8"
run(){ local sn=$1 algo=$2 launcher=$3 seed=$4; shift 4
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"; setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === $sn ($algo) ==="
  python "$launcher" --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python "$launcher" --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
evalg(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}"; done; }
for s in 1 2 3 4 5; do run captp90w1_s${s}   fixmatch_if   train.py      "$s" $IF ifrank_loss_weight=1.0; done; evalg captp90w1
for s in 1 2 3 4 5; do run cfsoftp90w1_s${s} fixmatch_ifcf train_ifcf.py "$s" $CF ifrank_loss_weight=1.0 if_target=soft; done; evalg cfsoftp90w1
touch P90_W1ADD_DONE; echo "[$(date)] ===== p90 w1补齐完成 ====="
