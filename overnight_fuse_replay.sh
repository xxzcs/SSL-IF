#!/usr/bin/env bash
# ============================================================================
# 把 IF 思想融进 SimMatch (保留 in_loss!) — 2026-07-03 夜跑, 自主运行, 明早看结果。
# 用户诉求: 不要 replace(那会退回 FixMatch), 要 fuse 进 SimMatch, 偏好 in_fuse。
#
# 6 个方法-配置 (全部保留 in_loss), 每个 3 seed, 按价值排序, 每配置跑完立即 eval:
#   tf2 : teacher_fuse, balanced融合, T0.9, strong-IF, if_fuse_strength=2   (IF 折进 in_loss teacher, 忠实 #2 融合)
#   if3 : in_fuse (类因子路径), if_fuse_strength=3                          (最"名副其实"的折入)
#   tf4 : teacher_fuse, balanced, T0.9, strong, if_fuse_strength=4          (更强折入)
#   a8  : add, balanced, T0.9, strong, ifrank_loss_weight=8                 (IF 排序项外挂, 高权重)
#   a2  : add, balanced, T0.9, strong, ifrank_loss_weight=2                 (中权重)
#   a1w : add, balanced, T0.5, weak-IF (#1融合), ifrank_loss_weight=4       (弱支变体)
#
# 说明: multiplyo(#3/#4/#5) 因闭式 IF 大尺度 ÷T0.05 数值饱和 -> ifrank≡0, 今晚跳过(需先对齐 IF 尺度)。
#       add 用 balanced (zscore, 不饱和)。fold-in(tf/if) 走类因子路径, 不受饱和影响。
# 对照: SimMatch 完整 AUC0.8771/ACC0.8119 ; infuse 最佳 0.8784 ; rep_base(replace无IF) 0.8539。
# 机器不稳: 跳过已完成(model_best.pth), 单次崩溃重试一次再跳。
# ============================================================================
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SEEDS="1 2 3"
SUMMARY=results/overnight2_fuse_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/overnight_fuse_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== 融合复刻 (保留 in_loss) 开始, log=$LOG ====="

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }

COMMON="num_references=4 ref_select=by_instance ref_cand_k=8"
declare -A CFG
CFG[tf2]="ifrank_mode=teacher_fuse ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 if_fuse_strength=2 ifrank_loss_weight=0 $COMMON"
CFG[if3]="ifrank_mode=in_fuse       ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 if_fuse_strength=3 ifrank_loss_weight=0 $COMMON"
CFG[tf4]="ifrank_mode=teacher_fuse ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 if_fuse_strength=4 ifrank_loss_weight=0 $COMMON"
CFG[a8]="ifrank_mode=add            ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 ifrank_loss_weight=8 $COMMON"
CFG[a2]="ifrank_mode=add            ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 ifrank_loss_weight=2 $COMMON"
CFG[a1w]="ifrank_mode=add           ifrank_combine=multiply_balanced corrT=0.5 use_strong_if=False if_lambda=1 csim_lambda=1 ifrank_loss_weight=4 $COMMON"
ORDER="tf2 if3 tf4 a8 a2 a1w"

run_one(){  # <save_name> <seed> <k=v>...
  local sn=$1 seed=$2; shift 2
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_ov2_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"; setkv "$tmp" seed "$seed"
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train ${sn} ($*) ==="
  python train_simmatch_if.py --c "$tmp" || { echo "[warn] ${sn} 首次失败, 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if.py --c "$tmp" || echo "[warn] ${sn} 跳过"; }
  rm -f "$tmp"
}

eval_group(){  # <glob_prefix>
  for kind in best latest; do
    [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
    python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" \
      --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 \
      --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1 $kind"
  done
}

for tag in $ORDER; do
  for seed in $SEEDS; do
    run_one "simmatch_if_bus_ov2_${tag}_s${seed}" "$seed" ${CFG[$tag]}
  done
  eval_group "simmatch_if_bus_ov2_${tag}"
  echo "[$(date)] ---- ${tag} 完成, 已 eval ----"
done

touch results/ALL_DONE_OVERNIGHT2
echo "[$(date)] ===== 全部完成. 结果: $SUMMARY (对照 SimMatch 0.8771/0.8119, infuse 0.8784) ====="
