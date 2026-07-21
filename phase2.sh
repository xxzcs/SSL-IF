#!/usr/bin/env bash
# Phase2 (链在 PHASE1_DONE 后): 泛化 IF。warmup 已锁 zero5(warmup_epochs=5/zero)。
# IF 插件配方全框架一致(= FixMatch wu_zero5 验证过的):
#   闭式硬/strong/corrT0.9/multiply_balanced/w1/warmup5-zero/num_ref4/by_instance/ref_cand_k8/λ=1。
# 各框架保留自己的阈值(fix/flex/refix=0.9固定, free/soft自适应, ada相对0.95)与自己的lr。
# 顺序: 先 4 核心框架 +IF(base 已就绪), 再 ReFixMatch(base+IF硬), 最后 ReFixMatch IF软消融。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
while [ ! -f PHASE1_DONE ]; do sleep 120; done
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/phase2_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
SUMMARY=results/generalization_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
EVAL_ARGS='--dataset bus --num_classes 2 --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth'
evalg(){ local glob=$1 suf=$2; for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py $EVAL_ARGS --summary_csv "$SUMMARY" --load_glob "saved_models/usb_cv/${glob}/${ck}" --method_suffix "${suf}_${kind}"; done; }
# 锁定的 IF 插件配方(与 FixMatch wu_zero5 逐字一致)
IFPLUG="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard use_strong_if=True corrT=0.9 ifrank_warmup_epochs=5 ifrank_warmup_mode=zero"

runif(){ local alg=$1 seed=$2 tgt=${3:-hard} sfx=${4:-}
  local sn=${alg}_ifcf${sfx}_s${seed}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local cfg=config/usb_cv/${alg}/${alg}_bus_878_0.yaml; local tmp=config/_${sn}.yaml; cp "$cfg" "$tmp"
  setkv "$tmp" algorithm ${alg}_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  for kv in $IFPLUG; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  setkv "$tmp" if_target "$tgt"
  echo "[$(date)] === IF $sn (${alg}_ifcf, target=$tgt) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
runbase(){ local alg=$1 seed=$2; local sn=${alg}_base_s${seed}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local cfg=config/usb_cv/${alg}/${alg}_bus_878_0.yaml; local tmp=config/_${sn}.yaml; cp "$cfg" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  echo "[$(date)] === base $sn ==="
  python train.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }

# ---- (A) 4 核心框架 +IF (base 已就绪, 直接配对) ----
for alg in freematch softmatch adamatch flexmatch; do
  for s in 1 2 3 4 5; do runif $alg $s hard; done
  evalg "${alg}_ifcf_s[12345]" "${alg}_ifcf"
done

# ---- (B) ReFixMatch: 冒烟守卫 -> base -> IF硬 ----
smoke(){ local alg=$1 launcher=$2 extra=$3; local tmp=config/_smk_${alg}.yaml; cp config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml "$tmp"
  setkv "$tmp" algorithm "$alg"; setkv "$tmp" save_name smk_${alg}; setkv "$tmp" seed 1
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" overwrite True
  setkv "$tmp" epoch 1; setkv "$tmp" num_train_iter 3; setkv "$tmp" num_eval_iter 3; setkv "$tmp" num_log_iter 1
  for kv in $extra; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  timeout 300 python "$launcher" --c "$tmp" >/dev/null 2>&1; local rc=$?
  rm -f "$tmp"; rm -rf saved_models/usb_cv/smk_${alg}; return $rc; }
REFIX_OK=1
smoke refixmatch train.py "" || REFIX_OK=0
smoke refixmatch_ifcf train_ifcf.py "$IFPLUG" || REFIX_OK=0
if [ "$REFIX_OK" = 1 ]; then
  for s in 1 2 3 4 5; do runbase refixmatch $s; done; evalg "refixmatch_base_s[12345]" "refixmatch_base"
  for s in 1 2 3 4 5; do runif refixmatch $s hard "_hard"; done; evalg "refixmatch_ifcf_hard_s[12345]" "refixmatch_ifcf_hard"
  # ---- (C) ReFixMatch IF 软消融(它自带软KL, 看要不要配软IF) ----
  for s in 1 2 3 4 5; do runif refixmatch $s soft "_soft"; done; evalg "refixmatch_ifcf_soft_s[12345]" "refixmatch_ifcf_soft"
else
  echo "[warn] refixmatch 冒烟失败, 跳过 ReFixMatch(不影响4核心框架)"
fi
touch PHASE2_DONE; echo "[$(date)] ===== Phase2 (泛化IF: 4核心 + ReFixMatch base/IF硬/IF软) 完成 ====="
