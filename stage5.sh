#!/usr/bin/env bash
# Stage5 (链在 STAGE4_DONE 后): 把 warmup ② 关前5epoch (wu_zero5) 补到 5seed。
# 配置严格对齐 warmup_sweep.sh 的 wu_zero5: fixmatch_ifcf, 闭式硬/strong/corrT0.9/
#   multiply_balanced/w1/p90, ifrank_warmup_epochs=5 mode=zero, lr0.0046875。s1-3已有,补s4/s5。
# 完成后重新评估 zero5/zero1/ramp10 三者 5seed, 供统计对比。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
while [ ! -f STAGE4_DONE ]; do sleep 120; done
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/stage5_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
SUMMARY=results/warmup5seed_summary.csv
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
EVAL_ARGS='--dataset bus --num_classes 2 --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth'
evalg(){ local glob=$1 suf=$2; for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py $EVAL_ARGS --summary_csv "$SUMMARY" --load_glob "saved_models/usb_cv/${glob}/${ck}" --method_suffix "${suf}_${kind}"; done; }
WIN="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard use_strong_if=True corrT=0.9"
runif(){ local sn=$1 seed=$2
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"; setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in $WIN; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  setkv "$tmp" ifrank_warmup_epochs 5; setkv "$tmp" ifrank_warmup_mode zero
  echo "[$(date)] === 补 $sn (warmup=5/zero) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
for s in 4 5; do runif wu_zero5_s${s} "$s"; done
# 三档 5seed 汇总(zero1=ifcf_p90_w1, zero5=wu_zero5, ramp10=wu_ramp10)
evalg "wu_zero5_s[12345]"    "wu_zero5_5s"
evalg "ifcf_p90_w1_s[12345]" "wu_zero1_5s"
evalg "wu_ramp10_s[12345]"   "wu_ramp10_5s"
touch STAGE5_DONE; echo "[$(date)] ===== Stage5 (warmup zero5 补满5seed + 三档对比) 完成 ====="
