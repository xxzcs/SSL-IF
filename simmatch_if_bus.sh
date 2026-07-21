#!/bin/bash
# SimMatch + 闭式 last-layer TracIn 影响排序一致性 (BUS, da1) 实验矩阵
# 7 组配置 × 5 seed = 35 run。 tag => "mode combine use_strong corrT weight"
#   rep_base_w0     : replace + w0  = SimMatch 去掉 in_loss 的基线 (add+w0 = 原版SimMatch 0.812 已有)
#   *_mulF_*        : multiply, nostrong, T0.5   (你最佳设置之一)
#   *_mulT_*        : multiply, strong,   T0.9   (你最佳设置之一)
#   *_bal_*         : multiply_balanced, T0.5    (IF/余弦都 z-score)
# 经 train_simmatch_if.py 启动。健壮版: 跳过已完成/失败重试。
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
# tag => "mode combine use_strong corrT weight" ; weight 按各融合 rank_loss 量级定, 使贡献≈0.5(可比)
declare -A CFG=(
  [rep_base]="replace multiply False 0.5 0"
  [rep_mulF]="replace multiply False 0.5 400"
  [rep_mulT]="replace multiply True 0.9 25"
  [rep_bal]="replace multiply_balanced False 0.5 0.33"
)

run_one () {
  local save_name=$1 seed=$2 mode=$3 cmb=$4 st=$5 ct=$6 w=$7
  local tmp; tmp=$(mktemp --suffix=.yaml)
  sed "s|^save_name:.*|save_name: ${save_name}|;s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|;s/^seed:.*/seed: ${seed}/;s/^ifrank_mode:.*/ifrank_mode: ${mode}/;s/^ifrank_combine:.*/ifrank_combine: ${cmb}/;s/^use_strong_if:.*/use_strong_if: ${st}/;s/^corrT:.*/corrT: ${ct}/;s/^ifrank_loss_weight:.*/ifrank_loss_weight: ${w}/" "$BASE" > "$tmp"
  python train_simmatch_if.py --c "$tmp"; local rc=$?
  rm -f "$tmp"; return $rc
}

for tag in "${!CFG[@]}"; do
  set -- ${CFG[$tag]}; mode=$1; cmb=$2; st=$3; ct=$4; w=$5
  for seed in 1 2 3 4 5; do
    save_name="simmatch_if_bus_${tag}_${seed}"
    if [ -f "saved_models/usb_cv/${save_name}/model_best.pth" ]; then
      echo "=== skip ${save_name} (已完成) ==="; continue
    fi
    echo "=== train ${save_name} ($mode $cmb strong=$st T=$ct w=$w seed=$seed) ==="
    run_one "$save_name" "$seed" "$mode" "$cmb" "$st" "$ct" "$w"
    if [ $? -ne 0 ]; then
      echo "[warn] ${save_name} 失败, 重试一次"; sleep 5
      rm -rf "saved_models/usb_cv/${save_name}"
      run_one "$save_name" "$seed" "$mode" "$cmb" "$st" "$ct" "$w" || echo "[warn] ${save_name} 重试仍失败, 跳过"
    fi
  done
done
echo "=== SimMatch-IF BUS 训练循环结束 ==="
