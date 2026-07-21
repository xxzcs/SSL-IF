#!/usr/bin/env bash
# 诊断 (链在 PHASE2_DONE 后): 回答 free/flex 该用啥配置。
#  freematch(自适应,无p_cutoff): soft-w1, soft-w2  (对比已有 hard-w1 + base)
#  flexmatch: base@0.95, hard-w1@0.95, soft-w1@0.95  (看 p_cutoff 0.95 是否恢复base + 同口径软vs硬)
# IF 其余配方与当前泛化一致: 闭式/strong/corrT0.9/multiply_balanced/warmup5-zero。5 seed。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
while [ ! -f PHASE2_DONE ]; do sleep 120; done
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/diag_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
SUMMARY=results/diag_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
EVAL_ARGS='--dataset bus --num_classes 2 --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth'
evalg(){ local glob=$1 suf=$2; for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py $EVAL_ARGS --summary_csv "$SUMMARY" --load_glob "saved_models/usb_cv/${glob}/${ck}" --method_suffix "${suf}_${kind}"; done; }
IFBASE="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 use_strong_if=True corrT=0.9 ifrank_warmup_epochs=5 ifrank_warmup_mode=zero"

# alg=框架, sn=存名, seed, extra=覆盖项(if_target/weight/p_cutoff...)
runif(){ local alg=$1 sn=$2 seed=$3; shift 3
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local cfg=config/usb_cv/${alg}/${alg}_bus_878_0.yaml; local tmp=config/_${sn}.yaml; cp "$cfg" "$tmp"
  setkv "$tmp" algorithm ${alg}_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  for kv in $IFBASE "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === IF $sn ($*) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
runbase(){ local alg=$1 sn=$2 seed=$3; shift 3
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local cfg=config/usb_cv/${alg}/${alg}_bus_878_0.yaml; local tmp=config/_${sn}.yaml; cp "$cfg" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === base $sn ($*) ==="
  python train.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }

# ---- freematch: 软 w1 / 软 w2 ----
for s in 1 2 3 4 5; do runif freematch fm_soft_w1_s${s} $s if_target=soft ifrank_loss_weight=1.0; done; evalg "fm_soft_w1_s[12345]" "fm_soft_w1"
for s in 1 2 3 4 5; do runif freematch fm_soft_w2_s${s} $s if_target=soft ifrank_loss_weight=2.0; done; evalg "fm_soft_w2_s[12345]" "fm_soft_w2"
# ---- flexmatch @0.95: base / 硬w1 / 软w1 ----
for s in 1 2 3 4 5; do runbase flexmatch fx_base_p95_s${s} $s p_cutoff=0.95; done; evalg "fx_base_p95_s[12345]" "fx_base_p95"
for s in 1 2 3 4 5; do runif flexmatch fx_hard_w1_p95_s${s} $s if_target=hard ifrank_loss_weight=1.0 p_cutoff=0.95; done; evalg "fx_hard_w1_p95_s[12345]" "fx_hard_w1_p95"
for s in 1 2 3 4 5; do runif flexmatch fx_soft_w1_p95_s${s} $s if_target=soft ifrank_loss_weight=1.0 p_cutoff=0.95; done; evalg "fx_soft_w1_p95_s[12345]" "fx_soft_w1_p95"
touch DIAG_DONE; echo "[$(date)] ===== 诊断(free软/flex@0.95软硬) 完成 ====="
