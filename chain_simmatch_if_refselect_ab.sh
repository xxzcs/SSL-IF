#!/usr/bin/env bash
# A/B: topk vs by_instance 代表性样本选择器 (其余同 phase-2 最佳: add+multiply+strong+corrT0.9)
# 用法: ./chain_simmatch_if_refselect_ab.sh
set -u
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SEEDS="1 2 3"
mkdir -p logs

run_one() {  # save_name seed ref_select ref_cand_k
  local sn=$1 seed=$2 rsel=$3 rck=$4
  local tmp="config/_tmp_${sn}.yaml"
  sed "s|^save_name:.*|save_name: ${sn}|;\
s|^load_path:.*|load_path: ./saved_models/usb_cv/${sn}/latest_model.pth|;\
s/^seed:.*/seed: ${seed}/;\
s/^use_strong_if:.*/use_strong_if: True/;\
s/^corrT:.*/corrT: 0.9/" "$BASE" > "$tmp"
  # 追加选择器参数(base 里没有这两行)
  echo "ref_select: ${rsel}"   >> "$tmp"
  echo "ref_cand_k: ${rck}"    >> "$tmp"
  echo "=== train ${sn} (ref_select=${rsel} cand_k=${rck} seed=${seed}) ==="
  python train_simmatch_if.py --c "$tmp"; local rc=$?
  rm -f "$tmp"; return $rc
}

for seed in $SEEDS; do
  # A: topk (对照, cand_k 对 topk 无意义, 传 8)
  sn="simmatch_if_bus_ab_topk_s${seed}"
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && echo "skip ${sn}" || \
    run_one "$sn" "$seed" "topk" 8 || echo "[warn] ${sn} 失败, 跳过"
  # B: by_instance (cand_k=5 激活弱强一致)
  sn="simmatch_if_bus_ab_byinst_s${seed}"
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && echo "skip ${sn}" || \
    run_one "$sn" "$seed" "by_instance" 5 || echo "[warn] ${sn} 失败, 跳过"
done
echo "=== A/B 全部完成 ==="
