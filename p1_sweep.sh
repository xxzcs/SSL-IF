#!/usr/bin/env bash
# P1: 温度 × 弱/强 粗扫。闭式硬, multiply_balanced, w1, p90, 3 seed。
# corrT{0.1,0.5,0.9} × use_strong_if{False=弱,True=强}。 strong+0.9 复用已有 ifcf_p90_w1(不重跑)。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/p1_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml; SUMMARY=results/p1_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
CF="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard"
run(){ local sn=$1 strong=$2 ct=$3 seed=$4
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"; setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in $CF; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  setkv "$tmp" use_strong_if "$strong"; setkv "$tmp" corrT "$ct"
  echo "[$(date)] === $sn (strong=$strong corrT=$ct) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
evalg(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}"; done; }
# 弱支 × 三温度
for s in 1 2 3; do run p1_w_t01_s${s} False 0.1 "$s"; done; evalg p1_w_t01
for s in 1 2 3; do run p1_w_t05_s${s} False 0.5 "$s"; done; evalg p1_w_t05
for s in 1 2 3; do run p1_w_t09_s${s} False 0.9 "$s"; done; evalg p1_w_t09
# 强支 × 0.1,0.5 (强+0.9 复用 ifcf_p90_w1)
for s in 1 2 3; do run p1_s_t01_s${s} True  0.1 "$s"; done; evalg p1_s_t01
for s in 1 2 3; do run p1_s_t05_s${s} True  0.5 "$s"; done; evalg p1_s_t05
# 强+0.9 复用已有(seed1-3)
python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth --load_glob "saved_models/usb_cv/ifcf_p90_w1_s[123]/latest_model.pth" --method_suffix "p1_s_t09_latest"
python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth --load_glob "saved_models/usb_cv/ifcf_p90_w1_s[123]/model_best.pth" --method_suffix "p1_s_t09_best"
touch P1_DONE; echo "[$(date)] ===== P1 温度×弱强粗扫 完成 ====="
