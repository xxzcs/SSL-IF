#!/usr/bin/env bash
# ============================================================================
# 自动接续: 等 faithful5(7配置×3seed) 跑完 -> 找 best AUC 超过 SimMatch 基线(0.8771)的配置组
#          -> 给每个补跑 seed 4,5 -> 重新 eval 成 5-seed。2026-07-06 夜, 自主运行。
# 结果: 胜出组的 5-seed 写入 results/faithful5_5seed_summary.csv ; 标记 ALL_DONE_EXPAND。
# ============================================================================
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
BASE=config/usb_cv/simmatch_if/simmatch_if_bus_878_0.yaml
BASE_AUC=0.8771
SUMMARY=results/faithful5_5seed_summary.csv
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/expand_winners_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
COMMON="ifrank_mode=add if_target=soft if_mean_reduce=True num_references=4 ref_select=by_instance ref_cand_k=8"
declare -A CFG
CFG[c2]="ifrank_combine=multiply_balanced   corrT=0.9 use_strong_if=True  if_lambda=1  csim_lambda=1 ifrank_loss_weight=2 $COMMON"
CFG[c1]="ifrank_combine=multiply_balanced   corrT=0.5 use_strong_if=False if_lambda=1  csim_lambda=1 ifrank_loss_weight=2 $COMMON"
CFG[c2w1]="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True  if_lambda=1  csim_lambda=1 ifrank_loss_weight=1 $COMMON"
CFG[c1w1]="ifrank_combine=multiply_balanced corrT=0.5 use_strong_if=False if_lambda=1  csim_lambda=1 ifrank_loss_weight=1 $COMMON"
CFG[c5]="ifrank_combine=multiplyo           corrT=0.05 use_strong_if=True if_lambda=20 csim_lambda=1 ifrank_loss_weight=1 $COMMON"
CFG[c3]="ifrank_combine=multiplyo           corrT=0.05 use_strong_if=True if_lambda=10 csim_lambda=1 ifrank_loss_weight=1 $COMMON"
CFG[c4]="ifrank_combine=multiplyo           corrT=0.05 use_strong_if=True if_lambda=15 csim_lambda=1 ifrank_loss_weight=1 $COMMON"

run_one(){ local sn=$1 seed=$2; shift 2
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] ${sn}"; return 0; fi
  local tmp="config/_ex_${sn}.yaml"; cp "$BASE" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train ${sn} ==="
  python train_simmatch_if.py --c "$tmp" || { echo "[warn] ${sn} 重试"; rm -rf "saved_models/usb_cv/${sn}"; python train_simmatch_if.py --c "$tmp" || echo "[warn] ${sn} 跳过"; }
  rm -f "$tmp"
}
eval_group(){ for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/$1_*/${ck}" --method_suffix "$(basename $1)_5seed_${kind}" || echo "[warn] eval $1 $kind"; done; }

echo "[$(date)] 等待 faithful5 完成 (ALL_DONE_FAITHFUL5)..."
while [ ! -f results/ALL_DONE_FAITHFUL5 ]; do sleep 120; done
echo "[$(date)] faithful5 已完成, 解析胜出配置(best AUC > ${BASE_AUC})"

WINNERS=$(python3 - "$BASE_AUC" <<'PY'
import csv, sys
base=float(sys.argv[1]); tags=['c2','c2w1','c1','c1w1','c5','c3','c4']; best={}
try:
    for row in csv.reader(open('results/faithful5_summary.csv')):
        if not row or row[0]=='method': continue
        m=row[0]
        for t in tags:
            if f'_f5_{t}_best' in m:
                try: best[t]=float(row[1].split('±')[0])
                except Exception: pass
except Exception as e: sys.stderr.write(str(e)+'\n')
sys.stderr.write('best AUC per config: '+str(best)+'  baseline='+str(base)+'\n')
print(' '.join(t for t in tags if best.get(t,0.0)>base))
PY
)
echo "[$(date)] 胜出配置(将补到5seed): [${WINNERS}]"
if [ -z "${WINNERS// }" ]; then
  echo "[$(date)] 无配置超过基线, 不补 seed。"; touch results/ALL_DONE_EXPAND; exit 0
fi

for tag in $WINNERS; do
  echo "[$(date)] ---- 扩展 ${tag} 到 5 seed ----"
  for seed in 4 5; do run_one "simmatch_if_bus_f5_${tag}_s${seed}" "$seed" ${CFG[$tag]}; done
  eval_group "simmatch_if_bus_f5_${tag}"   # glob 覆盖 s1..s5 -> 5seed 统计
done
touch results/ALL_DONE_EXPAND
echo "[$(date)] ===== 胜出组 5-seed 扩展完成. 结果: $SUMMARY ====="
