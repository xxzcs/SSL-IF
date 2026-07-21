#!/bin/bash
# SimMatch + influence-rank (BUS, 878 标注, da1) 消融对比:
#   ifrank_lambda = 0.0 (=原版 SimMatch 基线) vs 1.0 (加 influence-rank), 各 5 seed = 10 run
# 用独立启动器 train_ifrank.py (注册新算法, 不改原文件)。
# 健壮版: 不用 set -e; 跳过已完成; 失败重试一次。存到 simmatch_ifrank_bus_* 不覆盖。
BASE=config/usb_cv/simmatch_ifrank/simmatch_ifrank_bus_878_0.yaml
LAMBDAS="0.0 1.0"

run_one () {
  local save_name=$1 seed=$2 lam=$3
  local tmp; tmp=$(mktemp)
  sed "s|^save_name:.*|save_name: ${save_name}|" "$BASE" | \
  sed "s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|" | \
  sed "s/^seed:.*/seed: ${seed}/" | \
  sed "s/^ifrank_lambda:.*/ifrank_lambda: ${lam}/" > "$tmp"
  python train_ifrank.py --c "$tmp"; local rc=$?
  rm -f "$tmp"; return $rc
}

for lam in $LAMBDAS; do
  for seed in 1 2 3 4 5; do
    save_name="simmatch_ifrank_bus_878_l${lam}_${seed}"
    if [ -f "saved_models/usb_cv/${save_name}/model_best.pth" ]; then
      echo "=== skip ${save_name} (已完成) ==="; continue
    fi
    echo "=== train ${save_name} (lambda=${lam} seed=${seed}) ==="
    run_one "$save_name" "$seed" "$lam"
    if [ $? -ne 0 ]; then
      echo "[warn] ${save_name} 失败, 重试一次"; sleep 5
      rm -rf "saved_models/usb_cv/${save_name}"
      run_one "$save_name" "$seed" "$lam" || echo "[warn] ${save_name} 重试仍失败, 跳过"
    fi
  done
done
echo "=== SimMatch-ifrank BUS 训练循环结束 ==="
