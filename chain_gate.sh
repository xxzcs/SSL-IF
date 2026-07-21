#!/usr/bin/env bash
# 等 c4mask 跑完 -> 跑 gate(相乘门控)融合: cos基座 × softplus(β·zscore(IF)), β=if_lambda∈{1,3}, 3seed。
# 对照 c4(加法 multiplyo) 3seed 0.8792/0.8142, 基线 0.8771/0.8119。verify: 有效分布/尺度无关/无翻转 已过。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SUMMARY=results/gate_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/gate_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] 等待 c4mask 完成 (ALL_DONE_C4MASK)..."
while [ ! -f results/ALL_DONE_C4MASK ]; do sleep 120; done
echo "[$(date)] c4mask 已完成, 开始 gate 融合"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
COMMON="ifrank_mode=add ifrank_combine=gate corrT=0.9 use_strong_if=True csim_lambda=1 ifrank_loss_weight=2 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=soft if_mean_reduce=True"

run_group(){  # <tag> <beta=if_lambda>
  local tag=$1 beta=$2
  for s in 1 2 3; do
    sn="simmatch_if_bus_gate_${tag}_s${s}"
    if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] $sn"; continue; fi
    tmp="config/_gt_${sn}.yaml"; cp "$BASE" "$tmp"
    setkv "$tmp" algorithm simmatch_if; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$s"
    setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
    setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
    for kv in $COMMON; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
    setkv "$tmp" if_lambda "$beta"
    echo "[$(date)] === train $sn (beta=$beta) ==="
    python train_simmatch_if.py --c "$tmp" || { echo "[warn] $sn 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if.py --c "$tmp" || echo "[warn] $sn 跳过"; }
    rm -f "$tmp"
  done
  for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
    python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
      --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
      --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/simmatch_if_bus_gate_${tag}_*/${ck}" --method_suffix "gate_${tag}_${kind}" || echo "[warn] eval $tag $kind"; done
}

run_group b1 1
run_group b3 3
touch results/ALL_DONE_GATE
echo "[$(date)] ===== gate 融合完成. $SUMMARY (对照 c4加法 0.8792/0.8142, 基线 0.8771/0.8119) ====="
