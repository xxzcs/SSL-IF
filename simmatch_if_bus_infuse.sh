#!/bin/bash
# SimMatch-IF in_fuse 模式 (BUS, 878, da1): 把闭式 last-layer TracIn 影响融进 in_loss 的
# teacher 分布 (teacher_prob ∝ softmax(feat@bank/T) × 语义factor × 影响factor), 不外挂 rank loss。
# if_fuse_strength 扫描; strength=0 应精确复现原版 SimMatch (sanity check)。
# 注: in_fuse 走全 memory bank 路线, 不使用 num_references(k) / 排列, 影响按类别广播到整个 bank。
#
# 用法:
#   ./simmatch_if_bus_infuse.sh infuse_s0                       # 只跑 s0 验证 (先做这一步)
#   ./simmatch_if_bus_infuse.sh infuse_s1 infuse_s2 infuse_s4   # 验证通过后跑余下三组
#   ./simmatch_if_bus_infuse.sh                                 # 跑全部 (s0 s1 s2 s4)
# 健壮版: 不用 set -e; 跳过已完成; 失败重试一次。
cd /home/xiexiaozheng/Semi-supervised-learning || exit 1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate wssl 2>/dev/null || true
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SUMMARY=results/BUS_ALL_EXPERIMENTS.csv

# tag => if_fuse_strength
declare -A STR=(
  [infuse_s0]=0
  [infuse_s1]=1
  [infuse_s2]=2
  [infuse_s3]=3
  [infuse_s4]=4
)
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=(infuse_s0 infuse_s1 infuse_s2 infuse_s3 infuse_s4)
SEEDS="${SEEDS:-1 2 3 4 5}"   # 可用环境变量覆盖, 如 SEEDS="1 2 3"

run_one () {
  local save_name=$1 seed=$2 strength=$3
  local tmp; tmp=$(mktemp --suffix=.yaml)
  sed "s|^save_name:.*|save_name: ${save_name}|;s|^load_path:.*|load_path: ./saved_models/usb_cv/${save_name}/latest_model.pth|;s/^seed:.*/seed: ${seed}/;s/^ifrank_mode:.*/ifrank_mode: in_fuse/" "$BASE" > "$tmp"
  echo "if_fuse_strength: ${strength}" >> "$tmp"   # base yaml 无此键, 追加注入
  python train_simmatch_if.py --c "$tmp"; local rc=$?
  rm -f "$tmp"; return $rc
}

LOG="logs/simmatch_if_bus_infuse_$(date +%Y%m%d_%H%M%S).log"
{
echo "[$(date)] === in_fuse 训练: ${TAGS[*]} ==="
for tag in "${TAGS[@]}"; do
  s=${STR[$tag]}
  if [ -z "$s" ]; then echo "[warn] 未知 tag '$tag' (可选: ${!STR[*]}), 跳过"; continue; fi
  for seed in $SEEDS; do
    sn="simmatch_if_bus_${tag}_${seed}"
    if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then
      echo "=== skip ${sn} (已完成) ==="; continue
    fi
    echo "=== train ${sn} (in_fuse strength=$s seed=$seed) ==="
    run_one "$sn" "$seed" "$s"
    if [ $? -ne 0 ]; then
      echo "[warn] ${sn} 失败, 重试一次"; sleep 5
      rm -rf "saved_models/usb_cv/${sn}"
      run_one "$sn" "$seed" "$s" || echo "[warn] ${sn} 重试仍失败, 跳过"
    fi
  done
done

echo "[$(date)] === in_fuse 评测 -> $SUMMARY ==="
for tag in "${TAGS[@]}"; do
  [ -z "${STR[$tag]}" ] && continue
  for kind in best latest; do
    [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
    python3 eval_sup.py --dataset bus --num_classes 2 \
      --summary_csv "$SUMMARY" \
      --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 \
      --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/simmatch_if_bus_${tag}_*/${ck}" --method_suffix ${tag}_${kind} \
      || echo "[warn] eval ${tag} ${kind}"
  done
done
echo "[$(date)] ALL_DONE_SIMMATCH_IF_BUS_INFUSE (${TAGS[*]})"
} 2>&1 | tee -a "$LOG"
echo "日志: $LOG"
