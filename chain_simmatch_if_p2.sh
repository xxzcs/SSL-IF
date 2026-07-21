#!/bin/bash
# 等 Phase1 (simmatch_if_bus 的 ALL_DONE_SIMMATCH_IF_BUS) 完成后, 自动跑 Phase2:
#   add 模式(2组) + replace 权重扫描(2组), 各5seed=20run, 然后评测。
# 自包含, 不改原文件。
cd /home/xiexiaozheng/Semi-supervised-learning || exit 1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
P1LOG=$(ls -1t logs/simmatch_if_bus_*.log 2>/dev/null | head -1)   # 当前=Phase1 日志

echo "[$(date)] chainP2: 等待 Phase1 完成 (ALL_DONE_SIMMATCH_IF_BUS @ $P1LOG)..."
for i in $(seq 1 240); do
  grep -q "ALL_DONE_SIMMATCH_IF_BUS" "$P1LOG" 2>/dev/null && { echo "[$(date)] chainP2: Phase1 完成"; break; }
  sleep 180
done
while pgrep -f "train_simmatch_if" >/dev/null 2>&1; do sleep 30; done

P2LOG="logs/simmatch_if_bus_p2_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] chainP2: 启动 Phase2 -> $P2LOG"
{
  echo "[$(date)] === Phase2 训练 (add×2 + replace权重扫描×2, 20run) ==="
  # tag => "mode combine use_strong corrT weight"
  declare -A CFG=(
    [add_mulT]="add multiply True 0.9 100"
    [add_bal]="add multiply_balanced False 0.5 1.5"
    [rep_mulT_w5]="replace multiply True 0.9 5"
    [rep_mulT_w100]="replace multiply True 0.9 100"
  )
  run_one () {
    local save_name=$1 seed=$2 mode=$3 cmb=$4 st=$5 ct=$6 w=$7
    local tmp; tmp=$(mktemp --suffix=.yaml)
    sed "s|^save_name:.*|save_name: ${save_name}|;s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|;s/^seed:.*/seed: ${seed}/;s/^ifrank_mode:.*/ifrank_mode: ${mode}/;s/^ifrank_combine:.*/ifrank_combine: ${cmb}/;s/^use_strong_if:.*/use_strong_if: ${st}/;s/^corrT:.*/corrT: ${ct}/;s/^ifrank_loss_weight:.*/ifrank_loss_weight: ${w}/" "$BASE" > "$tmp"
    python train_simmatch_if.py --c "$tmp"; local rc=$?; rm -f "$tmp"; return $rc
  }
  for tag in "${!CFG[@]}"; do
    set -- ${CFG[$tag]}; mode=$1; cmb=$2; st=$3; ct=$4; w=$5
    for seed in 1 2 3 4 5; do
      sn="simmatch_if_bus_${tag}_${seed}"
      [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "skip ${sn}"; continue; }
      echo "=== train ${sn} ($mode $cmb strong=$st T=$ct w=$w seed=$seed) ==="
      run_one "$sn" "$seed" "$mode" "$cmb" "$st" "$ct" "$w" || { echo "[warn] ${sn} 重试"; sleep 5; rm -rf "saved_models/usb_cv/${sn}"; run_one "$sn" "$seed" "$mode" "$cmb" "$st" "$ct" "$w" || echo "[warn] ${sn} 跳过"; }
    done
  done
  echo "[$(date)] === Phase2 评测 ==="
  for tag in add_mulT add_bal rep_mulT_w5 rep_mulT_w100; do
    for kind in best latest; do
      [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
      python3 eval_sup.py --dataset bus --num_classes 2 \
        --summary_csv results/simmatch_if_bus_summary.csv \
        --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 \
        --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
        --load_glob "saved_models/usb_cv/simmatch_if_bus_${tag}_*/${ck}" --method_suffix ${tag}_${kind} || echo "[warn] eval ${tag} ${kind}"
    done
  done
  echo "[$(date)] ALL_DONE_SIMMATCH_IF_BUS_P2"
} > "$P2LOG" 2>&1
echo "[$(date)] chainP2: Phase2 结束, 日志 $P2LOG"
