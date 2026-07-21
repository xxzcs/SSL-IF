#!/usr/bin/env bash
# ============================================================================
# P0: fixmatch_ifcf 调参锚点 (2026-07-11)
# 目的: 在 USB-FixMatch 上把闭式 IF-rank 的一套好参数调出来 + 对照 bus_test_if 的 +1.5 AUC 验证端口正确。
#       调好的参数再原样泛化到 flexmatch/freematch/softmatch/adamatch。
# 内容: fixmatch base + fixmatch_ifcf × 权重{0.5,1,2,4} × 5 seed。latest 为主口径, best 旁证。
# 机制照搬 generality_run.sh。skip-completed(model_best.pth) + 崩溃重试一次。
# ============================================================================
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
SEEDS="1 2 3 4 5"
WEIGHTS="0.5 1.0 2.0 4.0"
SUMMARY=results/fixmatch_ifcf_sweep_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/fmifcf_sweep_${STAMP}.log
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ===== fixmatch_ifcf 权重扫描开始, log=$LOG ====="

# ==== P1 赢家参数 (等 seed5 强支结果确认后填) ====
IF_TARGET=hard          # <-- P1 赢家: hard / soft  (默认按 P1 弱支苗头填 hard, 强支确认后改)
USE_STRONG=True         # <-- P1 赢家: True / False
CORRT=0.9               # <-- P1 赢家温度
IFR="ifrank_combine=multiply_balanced corrT=${CORRT} use_strong_if=${USE_STRONG} if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=${IF_TARGET}"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }

run_one(){  # <launcher.py> <base_cfg> <algorithm> <save_name> <seed> <k=v>...
  local launcher=$1 base=$2 algo=$3 sn=$4 seed=$5; shift 5
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_fmcf_${sn}.yaml"; cp "$base" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110
  setkv "$tmp" lr 0.0046875
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
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

# ---- FixMatch 基线 ----
for s in $SEEDS; do run_one train.py "$FIX" fixmatch "fmcf_base_s${s}" "$s"; done
eval_group fmcf_base; echo "[$(date)] ---- base done ----"

# ---- fixmatch_ifcf 权重扫描 ----
for w in $WEIGHTS; do
  wtag=$(echo "$w" | tr -d '.')
  for s in $SEEDS; do run_one train_ifcf.py "$FIX" fixmatch_ifcf "fmcf_w${wtag}_s${s}" "$s" $IFR ifrank_loss_weight=${w}; done
  eval_group "fmcf_w${wtag}"; echo "[$(date)] ---- weight ${w} done ----"
done

touch FMCF_SWEEP_DONE
echo "[$(date)] ===== fixmatch_ifcf 权重扫描全部完成 ====="
