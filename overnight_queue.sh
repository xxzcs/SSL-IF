#!/usr/bin/env bash
# 通宵队列: 等 A/B 结束 -> [冒烟门] teacher_fuse 扫描 -> by_instance 的 combine 变体(忠实 FixMatch)
# 每阶段自动 eval 出 AUC。跳过已完成(model_best.pth), 单个崩溃重试一次再跳过, 不影响后续。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
SEEDS="1 2 3"
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }

run_one(){  # <save_name> <seed> <k=v>...
  local sn=$1 seed=$2; shift 2
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_ov_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"; setkv "$tmp" seed "$seed"
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train ${sn} ($*) ==="
  python train_simmatch_if.py --c "$tmp" || { echo "[warn] ${sn} 首次失败, 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if.py --c "$tmp" || echo "[warn] ${sn} 跳过"; }
  rm -f "$tmp"
}

eval_group(){  # <glob_prefix> <summary_csv>
  for kind in best latest; do
    [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
    python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$2" \
      --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 \
      --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1 $kind"
  done
}

echo "[$(date)] 等待 A/B(chain_simmatch_if_refselect_ab) 结束..."
while pgrep -f chain_simmatch_if_refselect_ab.sh >/dev/null 2>&1; do sleep 60; done
echo "[$(date)] A/B 已结束, eval A/B"
eval_group simmatch_if_bus_ab_topk   results/BUS_ALL_EXPERIMENTS.csv
eval_group simmatch_if_bus_ab_byinst results/BUS_ALL_EXPERIMENTS.csv

# ---- 冒烟门: 独占 GPU 下确认 teacher_fuse 能跑通(250 iter, epoch1 触发新路径) ----
echo "[$(date)] teacher_fuse 冒烟测试..."
TF_OK=1; SM=config/_ov_smoke_tf.yaml; cp "$BASE" "$SM"
setkv "$SM" save_name _ov_smoke_tf; setkv "$SM" num_train_iter 250; setkv "$SM" num_eval_iter 2000; setkv "$SM" num_log_iter 50
setkv "$SM" ifrank_mode teacher_fuse; setkv "$SM" use_strong_if True; setkv "$SM" corrT 0.9
setkv "$SM" ref_select by_instance; setkv "$SM" ref_cand_k 8; setkv "$SM" if_fuse_strength 2.0
timeout 600 python train_simmatch_if.py --c "$SM" || TF_OK=0
rm -rf saved_models/usb_cv/_ov_smoke_tf "$SM"
[ "$TF_OK" = 1 ] && echo "[$(date)] 冒烟通过, 跑阶段1" || echo "[$(date)] 冒烟失败, 跳过阶段1(teacher_fuse), 直接跑阶段2"

# ===== 阶段1: teacher_fuse (新方法; 冒烟通过才跑) =====
if [ "$TF_OK" = 1 ]; then
  for st in 1 2 4; do
    for seed in $SEEDS; do
      run_one "simmatch_if_bus_tf_st${st}_s${seed}" "$seed" \
        ifrank_mode=teacher_fuse ifrank_combine=multiply use_strong_if=True corrT=0.9 \
        ref_select=by_instance ref_cand_k=8 if_fuse_strength=${st}
    done
  eval_group "simmatch_if_bus_tf_st${st}" results/BUS_ALL_EXPERIMENTS.csv
  done
fi

# ===== 阶段2: by_instance add 模式 combine 变体(忠实 FixMatch 有效设置) =====
for seed in $SEEDS; do
  run_one "simmatch_if_bus_biMulo_s${seed}" "$seed" \
    ifrank_mode=add ifrank_combine=multiplyo if_lambda=10.0 corrT=0.1 use_strong_if=True \
    ref_select=by_instance ref_cand_k=8
done
eval_group simmatch_if_bus_biMulo results/BUS_ALL_EXPERIMENTS.csv
for seed in $SEEDS; do
  run_one "simmatch_if_bus_biBal_s${seed}" "$seed" \
    ifrank_mode=add ifrank_combine=multiply_balanced corrT=0.5 use_strong_if=False \
    ref_select=by_instance ref_cand_k=8
done
eval_group simmatch_if_bus_biBal results/BUS_ALL_EXPERIMENTS.csv

echo "[$(date)] ===== 通宵队列全部完成 ====="
