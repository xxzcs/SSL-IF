#!/usr/bin/env bash
# ============================================================================
# 忠实复刻用户在 FixMatch 上有效的 5 个 IF 融合配置 -> SimMatch (REPLACE 模式)
# 2026-07-03 夜跑。目标: 检验"上次 IF 在 SimMatch 上失效"是否只是没复刻对/权重被压没。
#
# 5 配置 (见记忆 fixmatch-if-effective-configs):
#   c1 balanced 1:1  T0.5  weak-IF   (use_strong_if=False)
#   c2 balanced 1:1  T0.9  strong-IF (use_strong_if=True)
#   c3 multiplyo 10:1 T0.05 strong-IF
#   c4 multiplyo 15:1 T0.05 strong-IF
#   c5 multiplyo 20:1 T0.05 strong-IF   (FixMatch 里 multiplyo 最佳)
#
# 关键修复: REPLACE 模式(丢 in_loss, IF 排序损失当唯一关系项 = FixMatch 里的真实角色) +
#           每配置自动标定 ifrank_loss_weight 使 w·rank_loss ≈ λu·unsup ≈ 1.5 (同量级, 不压没不崩)。
# 对照: rep_base(replace 无 IF) AUC0.8539/ACC0.7967 ; SimMatch 完整 0.8771/0.8119。
#
# 顺序: 按价值 c5,c2,c1,c3,c4, 每配置跑满 3 seed 后立即 eval -> 渐进出结果。
# 机器不稳: 跳过已完成(model_best.pth), 单次崩溃重试一次再跳过。
# ============================================================================
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SEEDS="1 2 3"
TARGET=1.5          # 目标 w·rank_loss 贡献 (≈ λu·unsup)
SUMMARY=results/faithful_replay_replace_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/faithful_replay_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ===== 忠实复刻 (REPLACE) 开始, log=$LOG ====="

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }

# 每配置的忠实参数 (不含 mode/weight/seed/save_name)
COMMON="ifrank_mode=replace num_references=4 ref_select=by_instance ref_cand_k=8"
declare -A CFG
CFG[c5]="ifrank_combine=multiplyo        if_lambda=20 csim_lambda=1 corrT=0.05 use_strong_if=True  $COMMON"
CFG[c2]="ifrank_combine=multiply_balanced if_lambda=1  csim_lambda=1 corrT=0.9  use_strong_if=True  $COMMON"
CFG[c1]="ifrank_combine=multiply_balanced if_lambda=1  csim_lambda=1 corrT=0.5  use_strong_if=False $COMMON"
CFG[c3]="ifrank_combine=multiplyo        if_lambda=10 csim_lambda=1 corrT=0.05 use_strong_if=True  $COMMON"
CFG[c4]="ifrank_combine=multiplyo        if_lambda=15 csim_lambda=1 corrT=0.05 use_strong_if=True  $COMMON"
ORDER="c5 c2 c1 c3 c4"
declare -A WEIGHT

# ---- 标定: smoke 量 rank_loss 量级 (weight=0 -> 纯 replace-noIF 动力学, ifrank_loss 仍被记录) ----
calibrate(){
  local tag=$1; shift
  local sn="_fr_calib_${tag}"; local tmp="config/${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" overwrite True
  setkv "$tmp" num_train_iter 550; setkv "$tmp" num_log_iter 110; setkv "$tmp" num_eval_iter 100000   # 550 可被 epoch=50 整除(660 不行会 assert 崩); 跑过 warmup=220 才有非零 ifrank
  setkv "$tmp" ifrank_loss_weight 0.0; setkv "$tmp" seed 1
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] [calib ${tag}] smoke..."
  timeout 900 python train_simmatch_if.py --c "$tmp" >/dev/null 2>&1
  local med
  med=$(python3 - "saved_models/usb_cv/${sn}/log.txt" <<'PY'
import re,sys,statistics
vals=[]
try:
    for ln in open(sys.argv[1]):
        m=re.search(r'(\d+) iteration',ln); r=re.search(r'ifrank_loss:\s*([0-9.eE+-]+)',ln)
        if m and r:
            it=int(m.group(1)); v=float(r.group(1))
            if it>=220 and v>0: vals.append(v)
except Exception: pass
print(f"{statistics.median(vals):.6g}" if vals else "NA")
PY
)
  rm -rf "saved_models/usb_cv/${sn}" "$tmp"
  if [ "$med" = "NA" ] || [ -z "$med" ]; then echo "[calib ${tag}] 解析失败 -> 回退 weight=1.0"; WEIGHT[$tag]=1.0; return; fi
  local w; w=$(python3 -c "m=$med; w=$TARGET/m if m>0 else 1.0; print(f'{min(max(w,0.02),300.0):.4g}')")
  WEIGHT[$tag]=$w
  echo "[calib ${tag}] median rank_loss=${med} -> ifrank_loss_weight=${w}"
}

run_one(){  # <save_name> <seed> <weight> <k=v>...
  local sn=$1 seed=$2 w=$3; shift 3
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_fr_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" seed "$seed"; setkv "$tmp" ifrank_loss_weight "$w"
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train ${sn} (w=${w}) ==="
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

# ===== 阶段0: 标定全部 5 配置 =====
for tag in $ORDER; do calibrate "$tag" ${CFG[$tag]}; done
echo "[$(date)] ===== 标定完成: $(for t in $ORDER; do echo -n "$t=${WEIGHT[$t]} "; done) ====="

# ===== 阶段1: 忠实复刻, 每配置 3 seed 后立即 eval =====
for tag in $ORDER; do
  w=${WEIGHT[$tag]}
  for seed in $SEEDS; do
    run_one "simmatch_if_bus_fr_${tag}_s${seed}" "$seed" "$w" ${CFG[$tag]}
  done
  eval_group "simmatch_if_bus_fr_${tag}"
  echo "[$(date)] ---- ${tag} 完成 (w=${w}), 已 eval ----"
done

touch results/ALL_DONE_FAITHFUL_REPLAY
echo "[$(date)] ===== 全部完成. 结果: $SUMMARY  (对照 rep_base 0.8539/0.7967, SimMatch 0.8771/0.8119) ====="
