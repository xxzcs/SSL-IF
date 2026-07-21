#!/usr/bin/env bash
# FreeMatch/FlexMatch + IF【软目标】BUS 复跑, #2 配方(balanced/T0.9/强), 权重{1,2}, 3seed。
# 对比之前的硬目标(gen_fm_ifw*/gen_fx_ifw*)。基线复用已有 gen_fm_base/gen_fx_base。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
FM=config/usb_cv/freematch/freematch_bus_878_0.yaml
FX=config/usb_cv/flexmatch/flexmatch_bus_878_fixed.yaml
SEEDS="1 2 3"
SUMMARY=results/generality_soft_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/generality_soft_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== FreeMatch/FlexMatch +IF 软目标 复跑 开始 ====="

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
IF2="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=soft if_mean_reduce=True"

run_one(){ local base=$1 algo=$2 sn=$3 seed=$4; shift 4
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_gs_${sn}.yaml"; cp "$base" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train ${sn} ==="
  python train_ifcf.py --c "$tmp" || { echo "[warn] ${sn} 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_ifcf.py --c "$tmp" || echo "[warn] ${sn} 跳过"; }
  rm -f "$tmp"
}
eval_group(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1 $kind"; done; }

# FreeMatch soft w1/w2
for s in $SEEDS; do run_one "$FM" freematch_ifcf "gen2_fm_ifsw1_s${s}" "$s" $IF2 ifrank_loss_weight=1.0; done
eval_group gen2_fm_ifsw1; echo "[$(date)] ---- FreeMatch soft w1 done ----"
for s in $SEEDS; do run_one "$FM" freematch_ifcf "gen2_fm_ifsw2_s${s}" "$s" $IF2 ifrank_loss_weight=2.0; done
eval_group gen2_fm_ifsw2; echo "[$(date)] ---- FreeMatch soft w2 done ----"
# FlexMatch soft w1/w2
for s in $SEEDS; do run_one "$FX" flexmatch_ifcf "gen2_fx_ifsw1_s${s}" "$s" $IF2 ifrank_loss_weight=1.0; done
eval_group gen2_fx_ifsw1; echo "[$(date)] ---- FlexMatch soft w1 done ----"
for s in $SEEDS; do run_one "$FX" flexmatch_ifcf "gen2_fx_ifsw2_s${s}" "$s" $IF2 ifrank_loss_weight=2.0; done
eval_group gen2_fx_ifsw2; echo "[$(date)] ---- FlexMatch soft w2 done ----"

touch results/ALL_DONE_GENERALITY_SOFT
echo "[$(date)] ===== 完成. $SUMMARY (对比硬: FM +0.004~0.008, FX +0.004~0.007) ====="
