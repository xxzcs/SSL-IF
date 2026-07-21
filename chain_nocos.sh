#!/usr/bin/env bash
# 等 sweep_a2 跑完 -> 补 noCos 消融 (纯影响力排序: if_lambda=1, csim_lambda=0, 其余同 a2)。
# 配合已有的 noIF(cosine-only) + a2(both), 凑齐增益来源 2x2 拆解。写进同一 summary 便于对比。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
SUMMARY=results/sweep_a2_summary.csv
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/chain_nocos_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] 等待 sweep_a2 完成 (ALL_DONE_SWEEP_A2)..."
while [ ! -f results/ALL_DONE_SWEEP_A2 ]; do sleep 120; done
echo "[$(date)] sweep 已完成, 开始 noCos 消融"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
NOCOS="ifrank_mode=add ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=0 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=2"

run_one(){ local sn=$1 seed=$2; shift 2
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_nc_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"; setkv "$tmp" seed "$seed"
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train ${sn} ==="
  python train_simmatch_if.py --c "$tmp" || { echo "[warn] ${sn} 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if.py --c "$tmp" || echo "[warn] ${sn} 跳过"; }
  rm -f "$tmp"
}
eval_group(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1 $kind"; done; }

for s in 1 2 3 4 5; do run_one "simmatch_if_bus_sw_a2nocos_s${s}" "$s" $NOCOS; done
eval_group "simmatch_if_bus_sw_a2nocos"
touch results/ALL_DONE_NOCOS
echo "[$(date)] ===== noCos 消融完成. 结果并入 $SUMMARY ====="
