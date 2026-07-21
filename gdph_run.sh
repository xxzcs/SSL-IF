#!/usr/bin/env bash
# GDPH ratio-19(192标注,低标注高headroom): FixMatch base+IF, SimMatch+IF。5fold×5run CV。
# 硬目标(论文首选) IF#2(balanced/T0.9/强/w2)。挂在 BUS gen5_w4 之后。
# 决胜局: SimMatch 在低标注 in_loss 不可靠时, IF 能否稳住/超过它。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
FMCFG=config/usb_cv/fixmatch/fixmatch_gdph_0.yaml
SMCFG=config/usb_cv/simmatch/simmatch_gdph_19_0.yaml
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/gdph_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] 等待 BUS gen5_w4 完成 (ALL_DONE_GEN5)..."
while [ ! -f results/ALL_DONE_GEN5 ]; do sleep 120; done
echo "[$(date)] BUS 完成, 开始 GDPH ratio-19"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
IF2="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=hard if_mean_reduce=True ifrank_loss_weight=2.0"

run_cv(){  # <base_cfg> <launcher> <algo> <prefix> <extra k=v...>
  local base=$1 launcher=$2 algo=$3 prefix=$4; shift 4
  for fold in 0 1 2 3 4; do for run in 1 2 3 4 5; do
    sn="${prefix}_fold${fold}_${run}"
    if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] $sn"; continue; fi
    tmp="config/_gd_${sn}.yaml"; cp "$base" "$tmp"
    setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" fold "$fold"; setkv "$tmp" seed "$run"
    setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
    setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
    for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
    echo "[$(date)] === train $sn ($algo) ==="
    python "$launcher" --c "$tmp" || { echo "[warn] $sn 重试"; rm -rf "saved_models/usb_cv/${sn}"; python "$launcher" --c "$tmp" || echo "[warn] $sn 跳过"; }
    rm -f "$tmp"
  done; done
}
eval_cv(){  # <prefix>
  python3 eval_sup_cv.py --dataset gdph --num_classes 2 --summary_csv results/gdph_if_summary.csv \
    --net resnet18 --model_key model --data_dir ../uda_data/GDPH --batch_size 16 --num_labels 192 \
    --label_ratio 0.1 --eval_dest eval --folds 0 1 2 3 4 \
    --load_glob_template "saved_models/usb_cv/$1_fold{fold}_*/model_best.pth" || echo "[warn] eval $1"
}

# FixMatch base (锚点; GDPH无base需新跑)
run_cv "$FMCFG" train.py       fixmatch      fixmatch_gdph;    eval_cv fixmatch_gdph
# FixMatch + IF
run_cv "$FMCFG" train_ifcf.py  fixmatch_ifcf fmif_gdph  $IF2;  eval_cv fmif_gdph
# SimMatch base (决胜对手): 3090 已训 simmatch_gdph_19_da1_fold*, 不重训, 只用和 +IF 相同的
#   eval(eval_sup_cv.py, model_key=model)重跑一遍 -> 落进同一张 gdph_if_summary.csv, 苹果对苹果。
eval_cv simmatch_gdph_19_da1
# SimMatch + IF
run_cv "$SMCFG" train_simmatch_if.py simmatch_if smif_gdph ifrank_mode=add $IF2; eval_cv smif_gdph

touch results/ALL_DONE_GDPH
echo "[$(date)] ===== GDPH FixMatch±IF + SimMatch+IF 完成. results/gdph_if_summary.csv ====="
