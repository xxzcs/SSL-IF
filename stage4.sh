#!/usr/bin/env bash
# Stage4 (链在 STAGE3_DONE 后): ReFixMatch 泛化 + hard-vs-soft IF 对照。
#   - refixmatch base 5seed
#   - refixmatch_ifcf 硬(赢家配方) 5seed
#   - refixmatch_ifcf 软(if_target=soft 对照, 回应"自带软约束是否配软IF") 5seed
# base 与 +IF 同用 refixmatch 自己的阈值(p95)/lr, 内部公平对照。
# 开头 3-iter 冒烟守卫: refixmatch / refixmatch_ifcf 任一崩则不进长跑。
# (DeFixMatch 已放弃: BUS labeled loader 不产生 x_lb_s。)
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
while [ ! -f STAGE3_DONE ]; do sleep 120; done
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/stage4_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
SUMMARY=results/stage4_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
EVAL_ARGS='--dataset bus --num_classes 2 --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth'
evalg(){ local glob=$1 suf=$2; for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py $EVAL_ARGS --summary_csv "$SUMMARY" --load_glob "saved_models/usb_cv/${glob}/${ck}" --method_suffix "${suf}_${kind}"; done; }
CFG=config/usb_cv/refixmatch/refixmatch_bus_878_0.yaml
IFPLUG="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 use_strong_if=True corrT=0.9 ifrank_warmup_epochs=1 ifrank_warmup_mode=zero"

# ---- 冒烟守卫 ----
smoke(){ local alg=$1 launcher=$2 extra=$3; local tmp=config/_smk_${alg}.yaml; cp "$CFG" "$tmp"
  setkv "$tmp" algorithm "$alg"; setkv "$tmp" save_name smk_${alg}; setkv "$tmp" seed 1
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" overwrite True
  setkv "$tmp" epoch 1; setkv "$tmp" num_train_iter 3; setkv "$tmp" num_eval_iter 3; setkv "$tmp" num_log_iter 1
  for kv in $extra; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  timeout 300 python "$launcher" --c "$tmp" >/dev/null 2>&1; local rc=$?
  rm -f "$tmp"; rm -rf saved_models/usb_cv/smk_${alg}; return $rc; }
echo "[$(date)] 冒烟: refixmatch ..."; smoke refixmatch train.py "" && echo " base OK" || { echo " base 冒烟失败, 中止"; touch STAGE4_DONE; exit 1; }
echo "[$(date)] 冒烟: refixmatch_ifcf ..."; smoke refixmatch_ifcf train_ifcf.py "$IFPLUG if_target=hard" && echo " ifcf OK" || { echo " ifcf 冒烟失败, 中止"; touch STAGE4_DONE; exit 1; }

# ---- base ----
runbase(){ local seed=$1; local sn=refixmatch_base_s${seed}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$CFG" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  echo "[$(date)] === base $sn ==="
  python train.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
for s in 1 2 3 4 5; do runbase $s; done; evalg "refixmatch_base_s[12345]" "refixmatch_base"

# ---- IF 硬/软 ----
runif(){ local tgt=$1 seed=$2; local sn=refixmatch_ifcf_${tgt}_s${seed}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$CFG" "$tmp"
  setkv "$tmp" algorithm refixmatch_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  for kv in $IFPLUG; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  setkv "$tmp" if_target "$tgt"
  echo "[$(date)] === IF $sn ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
for s in 1 2 3 4 5; do runif hard $s; done; evalg "refixmatch_ifcf_hard_s[12345]" "refixmatch_ifcf_hard"
for s in 1 2 3 4 5; do runif soft $s; done; evalg "refixmatch_ifcf_soft_s[12345]" "refixmatch_ifcf_soft"
touch STAGE4_DONE; echo "[$(date)] ===== Stage4 (ReFixMatch base + IF硬/软) 完成 ====="
