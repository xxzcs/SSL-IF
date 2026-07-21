#!/usr/bin/env bash
# 排队: 等 P90_W1ADD_DONE(当前w1补齐) 完成后, 自动跑 Captum-硬(w1,5seed)。
# 配置对齐: fixmatch_if, if_target=hard, multiply_balanced/强/corrT0.9/w1, p_cutoff=0.9。填满2x2消融。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
# 先等前一批完成(最多等 24h)
for i in $(seq 1 2880); do [ -f P90_W1ADD_DONE ] && break; sleep 30; done
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/capthard_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== 前批完成, 开始 Captum-硬 ====="
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml; SUMMARY=results/p90_all_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
IF="combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ifrank_loss_weight=1.0 if_target=hard"
run(){ local sn=$1 seed=$2
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_if; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"; setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in $IF; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === $sn (Captum-hard) ==="
  python train.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
for s in 1 2 3 4 5; do run capthard_p90w1_s${s} "$s"; done
for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth --load_glob "saved_models/usb_cv/capthard_p90w1_*/${ck}" --method_suffix "capthard_p90w1_${kind}"; done
touch CAPTHARD_DONE; echo "[$(date)] ===== Captum-硬 完成 ====="
