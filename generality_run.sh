#!/usr/bin/env bash
# ============================================================================
# IF 通用性验证 (2026-07-04): 闭式 IF 排序一致性接入无关系项框架 FreeMatch / FlexMatch。
# 假设: SimMatch 已含关系项(in_loss)->IF 收益小; FreeMatch/FlexMatch 无关系项->IF 收益应更大。
# 每个框架: 基线 vs +IF(#2: balanced/T0.9/强支; 权重 w1 FixMatch档 / w2 a2档)。3 seed。
# 各自 vs 各自基线比 (通用性), 不比 SimMatch。config-major, 每组跑完即 eval。
# 机器不稳: 跳过已完成(model_best.pth), 崩溃重试一次再跳。
# ============================================================================
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
SEEDS="1 2 3"
SUMMARY=results/generality_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/generality_${STAMP}.log
FM=config/usb_cv/freematch/freematch_bus_878_0.yaml
FX=config/usb_cv/flexmatch/flexmatch_bus_878_fixed.yaml
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== IF 通用性验证开始, log=$LOG ====="

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }

# IF #2 配方 (balanced/T0.9/强支)
IF2="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8"
run_one(){  # <launcher.py> <base_cfg> <algorithm> <save_name> <seed> <k=v>...
  local launcher=$1 base=$2 algo=$3 sn=$4 seed=$5; shift 5
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_gen_${sn}.yaml"; cp "$base" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110
  setkv "$tmp" lr 0.0046875   # 与 SimMatch/FixMatch 对齐 (原 freematch/flexmatch 配置误写成 0.009375)
  for kv in "$@"; do
    # 允许含空格的值(lr_drop_iter)
    setkv "$tmp" "${kv%%=*}" "${kv#*=}"
  done
  echo "[$(date)] === train ${sn} (${algo}) ==="
  python "$launcher" --c "$tmp" || { echo "[warn] ${sn} 重试"; rm -rf "saved_models/usb_cv/${sn}"; python "$launcher" --c "$tmp" || echo "[warn] ${sn} 跳过"; }
  rm -f "$tmp"
}

eval_group(){  # <glob_prefix>
  for kind in best latest; do
    [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
    python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
      --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
      --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
      --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1 $kind"
  done
}

# ========== FreeMatch ==========
for s in $SEEDS; do run_one train.py      "$FM" freematch       "gen_fm_base_s${s}"  "$s"; done
eval_group gen_fm_base;  echo "[$(date)] ---- FreeMatch 基线 done ----"
for s in $SEEDS; do run_one train_ifcf.py "$FM" freematch_ifcf  "gen_fm_ifw1_s${s}"  "$s" $IF2 ifrank_loss_weight=1.0; done
eval_group gen_fm_ifw1;  echo "[$(date)] ---- FreeMatch+IF w1 done ----"
for s in $SEEDS; do run_one train_ifcf.py "$FM" freematch_ifcf  "gen_fm_ifw2_s${s}"  "$s" $IF2 ifrank_loss_weight=2.0; done
eval_group gen_fm_ifw2;  echo "[$(date)] ---- FreeMatch+IF w2 done ----"

# ========== FlexMatch (基线配置补 sched) ==========
for s in $SEEDS; do run_one train.py      "$FX" flexmatch       "gen_fx_base_s${s}"  "$s"; done
eval_group gen_fx_base;  echo "[$(date)] ---- FlexMatch 基线 done ----"
for s in $SEEDS; do run_one train_ifcf.py "$FX" flexmatch_ifcf  "gen_fx_ifw1_s${s}"  "$s" $IF2 ifrank_loss_weight=1.0; done
eval_group gen_fx_ifw1;  echo "[$(date)] ---- FlexMatch+IF w1 done ----"
for s in $SEEDS; do run_one train_ifcf.py "$FX" flexmatch_ifcf  "gen_fx_ifw2_s${s}"  "$s" $IF2 ifrank_loss_weight=2.0; done
eval_group gen_fx_ifw2;  echo "[$(date)] ---- FlexMatch+IF w2 done ----"

touch results/ALL_DONE_GENERALITY
echo "[$(date)] ===== 全部完成. 结果: $SUMMARY ====="
