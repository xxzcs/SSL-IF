#!/usr/bin/env bash
# 等 AdaMatch 完 -> ①4框架 baseline+IF-w2 补到5seed(复用已有3seed) ②4框架 IF-w4(3seed, 试更大权重)。
# 框架: FreeMatch/FlexMatch/SoftMatch/AdaMatch (均无关系项)。IF=#2 balanced/T0.9/强/soft。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
SUMMARY=results/BUS_ALL_EXPERIMENTS.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/gen5_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] 等待 AdaMatch 完成 (ALL_DONE_ADAMATCH)..."
while [ ! -f results/ALL_DONE_ADAMATCH ]; do sleep 120; done
echo "[$(date)] AdaMatch 已完成, 开始 5seed + w4"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
IF2="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=soft if_mean_reduce=True"

run_one(){ local cfg=$1 launcher=$2 algo=$3 sn=$4 seed=$5 lr=$6; shift 6
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] $sn"; return 0; fi
  local tmp="config/_g5_${sn}.yaml"; cp "$cfg" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  setkv "$tmp" lr "$lr"
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train $sn ($algo) ==="
  python "$launcher" --c "$tmp" || { echo "[warn] $sn 重试"; rm -rf "saved_models/usb_cv/${sn}"; python "$launcher" --c "$tmp" || echo "[warn] $sn 跳过"; }
  rm -f "$tmp"
}
eval_group(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_${kind}" || echo "[warn] eval $1"; done; }

# 每框架: cfg | base_algo | if_algo | base_dir | 软w2_dir | 硬w2_dir(FM/FX复用旧gen_*_ifw2) | lr
FWS=(
"config/usb_cv/freematch/freematch_bus_878_0.yaml|freematch|freematch_ifcf|gen_fm_base|gen2_fm_ifsw2|gen_fm_ifw2|0.0046875"
"config/usb_cv/flexmatch/flexmatch_bus_878_fixed.yaml|flexmatch|flexmatch_ifcf|gen_fx_base|gen2_fx_ifsw2|gen_fx_ifw2|0.0046875"
"config/usb_cv/softmatch/softmatch_bus_878_0.yaml|softmatch|softmatch_ifcf|sm_base|sm_ifsw2|sm_ifhw2|0.0046875"
"config/usb_cv/adamatch/adamatch_bus_878_0.yaml|adamatch|adamatch_ifcf|am_base|am_ifsw2|am_ifhw2|0.0046875"
)
for rec in "${FWS[@]}"; do
  IFS='|' read -r cfg balgo ialgo bdir w2dir h2dir lr <<< "$rec"
  # baseline -> 5seed
  for s in 1 2 3 4 5; do run_one "$cfg" train.py "$balgo" "${bdir}_s${s}" "$s" "$lr"; done
  eval_group "$bdir"
  # IF-w2 -> 5seed
  for s in 1 2 3 4 5; do run_one "$cfg" train_ifcf.py "$ialgo" "${w2dir}_s${s}" "$s" "$lr" $IF2 ifrank_loss_weight=2.0; done
  eval_group "$w2dir"
  # IF-硬-w2 -> 5seed (FM/FX 复用旧 gen_*_ifw2 的3seed, 只补2; SM/AM 全新5)
  for s in 1 2 3 4 5; do run_one "$cfg" train_ifcf.py "$ialgo" "${h2dir}_s${s}" "$s" "$lr" $IF2 if_target=hard ifrank_loss_weight=2.0; done
  eval_group "$h2dir"
  # 权重扫(w4/w6, 软硬都扫; 最优权重可能依赖目标质量) 只在 SoftMatch(框架损失结构一致, 可迁移)
  if [ "$balgo" = "softmatch" ]; then
    for w in 4.0 6.0; do
      wt=$(printf '%.0f' "$w")
      for tgt in soft hard; do [ "$tgt" = soft ] && tt=ifs || tt=ifh
        for s in 1 2 3; do run_one "$cfg" train_ifcf.py "$ialgo" "sm_${tt}w${wt}_s${s}" "$s" "$lr" $IF2 if_target=$tgt ifrank_loss_weight=$w; done
        eval_group "sm_${tt}w${wt}"
      done
    done
  fi
  echo "[$(date)] ---- $balgo 完成(base/软w2/硬w2 5seed; SoftMatch额外软w4/w6) ----"
done
touch results/ALL_DONE_GEN5
echo "[$(date)] ===== gen5+w4 完成. $SUMMARY ====="
