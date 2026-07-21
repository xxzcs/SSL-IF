#!/usr/bin/env bash
# ============================================================================
# 忠实复刻用户 5 个 FixMatch 有效配置 -> SimMatch (add 模式, 保留 in_loss)。2026-07-06。
# 关键: 这次 IF 用【软目标 g_u=p·Σlogits−logits + batch-mean 规约】(实测与其 Captum corr≈0.905),
#       融合与其 compute_corrfea 一致(balanced=两个都zscore相加; multiplyo=原始相加, 无clamp)。
# 之前 a2 用的是硬 argmax(corr仅0.52, 不忠实) -> 本轮看忠实软目标能否提升。
# 5 配置: #1 bal/T0.5/弱  #2 bal/T0.9/强(=a2位置)  #3/#4/#5 multiplyo/T0.05/强 λ10/15/20。
# 对标: SimMatch 基线 0.8771/0.8119 ; a2(硬,balanced) 0.8794。3 seed 初筛。
# 机器不稳: 跳过已完成(model_best.pth), 崩溃重试一次再跳。config-major, 每组eval。
# ============================================================================
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SEEDS="1 2 3"
SUMMARY=results/faithful5_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/faithful5_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== 忠实5配置(软目标) SimMatch 开始, log=$LOG ====="

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }

COMMON="ifrank_mode=add if_target=soft if_mean_reduce=True num_references=4 ref_select=by_instance ref_cand_k=8"
declare -A CFG
# balanced 四个: 权重 2(a2位置) 和 权重 1(FixMatch原生有效权重) 各两配置
CFG[c2]="ifrank_combine=multiply_balanced   corrT=0.9 use_strong_if=True  if_lambda=1  csim_lambda=1 ifrank_loss_weight=2 $COMMON"
CFG[c1]="ifrank_combine=multiply_balanced   corrT=0.5 use_strong_if=False if_lambda=1  csim_lambda=1 ifrank_loss_weight=2 $COMMON"
CFG[c2w1]="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True  if_lambda=1  csim_lambda=1 ifrank_loss_weight=1 $COMMON"
CFG[c1w1]="ifrank_combine=multiply_balanced corrT=0.5 use_strong_if=False if_lambda=1  csim_lambda=1 ifrank_loss_weight=1 $COMMON"
# multiplyo 三个(原始尺度, rank_loss 较大~4, 权重用 1)
CFG[c5]="ifrank_combine=multiplyo           corrT=0.05 use_strong_if=True if_lambda=20 csim_lambda=1 ifrank_loss_weight=1 $COMMON"
CFG[c3]="ifrank_combine=multiplyo           corrT=0.05 use_strong_if=True if_lambda=10 csim_lambda=1 ifrank_loss_weight=1 $COMMON"
CFG[c4]="ifrank_combine=multiplyo           corrT=0.05 use_strong_if=True if_lambda=15 csim_lambda=1 ifrank_loss_weight=1 $COMMON"
ORDER="c2 c2w1 c1 c1w1 c5 c3 c4"

run_one(){  local sn=$1 seed=$2; shift 2
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_f5_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train ${sn} ($*) ==="
  python train_simmatch_if.py --c "$tmp" || { echo "[warn] ${sn} 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if.py --c "$tmp" || echo "[warn] ${sn} 跳过"; }
  rm -f "$tmp"
}
eval_group(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1 $kind"; done; }

for tag in $ORDER; do
  for seed in $SEEDS; do run_one "simmatch_if_bus_f5_${tag}_s${seed}" "$seed" ${CFG[$tag]}; done
  eval_group "simmatch_if_bus_f5_${tag}"; echo "[$(date)] ---- ${tag} 完成 ----"
done
touch results/ALL_DONE_FAITHFUL5
echo "[$(date)] ===== 全部完成. 结果: $SUMMARY (对标 0.8771/0.8119, a2硬 0.8794) ====="
