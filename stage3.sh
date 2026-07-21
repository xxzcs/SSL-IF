#!/usr/bin/env bash
# Stage3: (A) 补跑 flexmatch base(config 已修 sched) 5seed;
#         (B) 泛化 IF —— freematch/softmatch/adamatch/flexmatch 各挂闭式 IF 插件 5seed。
# IF 插件赢家配方: 闭式硬 + strong + corrT0.9 + multiply_balanced + w1 + warmup默认(zero1)。
# 各框架保留自己的阈值机制(不强加 p90)/自己的 lr。skip-completed + retry-once。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/stage3_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
SUMMARY=results/stage3_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
EVAL_ARGS='--dataset bus --num_classes 2 --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth'
evalg(){ local glob=$1 suf=$2; for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py $EVAL_ARGS --summary_csv "$SUMMARY" --load_glob "saved_models/usb_cv/${glob}/${ck}" --method_suffix "${suf}_${kind}"; done; }
IFPLUG="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard use_strong_if=True corrT=0.9 ifrank_warmup_epochs=1 ifrank_warmup_mode=zero"

# ---- (A) flexmatch base 补跑 ----
runbase(){ local alg=$1 seed=$2; local sn=${alg}_base_s${seed}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local cfg=config/usb_cv/${alg}/${alg}_bus_878_0.yaml; local tmp=config/_${sn}.yaml; cp "$cfg" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  echo "[$(date)] === base $sn ==="
  python train.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
for s in 1 2 3 4 5; do runbase flexmatch $s; done; evalg "flexmatch_base_s[12345]" "flexmatch_base"

# ---- (B) 泛化 IF ----
runif(){ local alg=$1 seed=$2; local sn=${alg}_ifcf_s${seed}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local cfg=config/usb_cv/${alg}/${alg}_bus_878_0.yaml; local tmp=config/_${sn}.yaml; cp "$cfg" "$tmp"
  setkv "$tmp" algorithm ${alg}_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  for kv in $IFPLUG; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === IF $sn (${alg}_ifcf) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
for alg in freematch softmatch adamatch flexmatch; do
  for s in 1 2 3 4 5; do runif $alg $s; done
  evalg "${alg}_ifcf_s[12345]" "${alg}_ifcf"
done
touch STAGE3_DONE; echo "[$(date)] ===== Stage3 (flexmatch base + 4框架泛化IF) 完成 ====="
