#!/usr/bin/env bash
# ============================================================================
# a2 权重精扫 + IF 开关消融 — 2026-07-03 白天。坐实 a2(add·balanced·T0.9·strong·w2) 超基线的增益。
# a2 已有 seed1-3 (AUC0.8804/ACC0.8153)。本轮:
#   w2 : 补 seed4,5 (复用 ov2_a2 前缀 -> 凑满 5 seed)
#   w1 : 5 seed
#   w3 : 5 seed
#   noIF: w2 但 if_lambda=0 (排序项只用 cosine, 去掉影响力) x5 seed  -> 证明增益来自 influence 而非普通 balanced-cosine
# 对照: SimMatch 完整 0.8771/0.8119 ; a2(w2,3seed) 0.8804/0.8153。
# 机器不稳: 跳过已完成(model_best.pth), 崩溃重试一次再跳。config-major, 每组跑完即 eval。
# ============================================================================
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SUMMARY=results/sweep_a2_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/sweep_a2_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== a2 权重精扫 + noIF 消融 开始, log=$LOG ====="

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }

A2="ifrank_mode=add ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8"

run_one(){  # <save_name> <seed> <k=v>...
  local sn=$1 seed=$2; shift 2
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_sw_${sn}.yaml"; cp "$BASE" "$tmp"
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

# ---- w2: 补 seed4,5 (复用 ov2_a2 前缀), 先把赢家凑满 5 seed ----
for s in 4 5; do run_one "simmatch_if_bus_ov2_a2_s${s}" "$s" $A2 ifrank_loss_weight=2; done
eval_group "simmatch_if_bus_ov2_a2"; echo "[$(date)] ---- w2(5seed) eval 完成 ----"

# ---- w3 ----
for s in 1 2 3 4 5; do run_one "simmatch_if_bus_sw_a2w3_s${s}" "$s" $A2 ifrank_loss_weight=3; done
eval_group "simmatch_if_bus_sw_a2w3"; echo "[$(date)] ---- w3 eval 完成 ----"

# ---- w1 ----
for s in 1 2 3 4 5; do run_one "simmatch_if_bus_sw_a2w1_s${s}" "$s" $A2 ifrank_loss_weight=1; done
eval_group "simmatch_if_bus_sw_a2w1"; echo "[$(date)] ---- w1 eval 完成 ----"

# ---- noIF 消融: w2 但 if_lambda=0 (排序项只剩 cosine) ----
for s in 1 2 3 4 5; do run_one "simmatch_if_bus_sw_a2noif_s${s}" "$s" ifrank_mode=add ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=0 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=2; done
eval_group "simmatch_if_bus_sw_a2noif"; echo "[$(date)] ---- noIF 消融 eval 完成 ----"

touch results/ALL_DONE_SWEEP_A2
echo "[$(date)] ===== 全部完成. 结果: $SUMMARY ====="
