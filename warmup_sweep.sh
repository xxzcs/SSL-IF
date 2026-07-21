#!/usr/bin/env bash
# warmup 消融: P1 赢家(闭式硬, 强, corrT0.9, multiply_balanced, w1, p90)上比较 3 档 warmup, 3 seed。
#   ① 只关 epoch0  = 现有 ifcf_p90_w1_s[123] (复用, 不重跑)
#   ② 关前 5 epoch  = warmup_epochs=5  mode=zero
#   ③ 线性 rampup 10 = warmup_epochs=10 mode=rampup
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/warmup_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml; SUMMARY=results/warmup_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
WIN="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard use_strong_if=True corrT=0.9"
run(){ local sn=$1 we=$2 wm=$3 seed=$4
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"; setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in $WIN; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  setkv "$tmp" ifrank_warmup_epochs "$we"; setkv "$tmp" ifrank_warmup_mode "$wm"
  echo "[$(date)] === $sn (warmup_epochs=$we mode=$wm) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
evalg(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}"; done; }
# ② 关前5 epoch
for s in 1 2 3; do run wu_zero5_s${s} 5 zero "$s"; done; evalg wu_zero5
# ③ 线性 rampup 10 epoch
for s in 1 2 3; do run wu_ramp10_s${s} 10 rampup "$s"; done; evalg wu_ramp10
# ① 现状(只关epoch0) 复用 ifcf_p90_w1
for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth --load_glob "saved_models/usb_cv/ifcf_p90_w1_s[123]/${ck}" --method_suffix "wu_zero1_${kind}"; done
touch WARMUP_DONE; echo "[$(date)] ===== warmup 消融 完成 ====="
