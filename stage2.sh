#!/usr/bin/env bash
# Stage2 自动流水线(等 warmup 完后自动接,卡不空转):
#   1) 按 latest AUC 自动选 warmup 赢家(zero1 现状 / zero5 / ramp10)
#   2) 若赢家非现状 -> 补 seed4/5 凑满 5seed(现状=ifcf_p90_w1 已有5seed)
#   3) 用 base 基线(freematch/softmatch/adamatch/flexmatch × 5seed)垫场——泛化表必需, 与 IF 决策无关
# 全程 skip-completed + retry-once。
set -u; cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
# 等 warmup 完成
while [ ! -f WARMUP_DONE ]; do sleep 120; done
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/stage2_${STAMP}.log; exec > >(tee -a "$LOG") 2>&1
SUMMARY=results/stage2_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
EVAL_ARGS='--dataset bus --num_classes 2 --net resnet18 --model_key ema_model --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth'
evalg(){ local glob=$1 suf=$2; for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py $EVAL_ARGS --summary_csv "$SUMMARY" --load_glob "saved_models/usb_cv/${glob}/${ck}" --method_suffix "${suf}_${kind}"; done; }

# ---- 1) 选 warmup 赢家 ----
WIN_TAG=$(python3 - <<'PY'
import csv,collections
best={}
try:
  for r in csv.DictReader(open('results/warmup_summary.csv')):
    m=r['method']
    if '_latest' not in m: continue
    for tag in ('wu_zero1','wu_zero5','wu_ramp10'):
      if m.startswith(tag) or ('_'+tag+'_') in m:
        auc=float(r['AUC'].split('±')[0]); best[tag]=max(best.get(tag,0),auc)
except Exception as e:
  pass
if not best: print('wu_zero1');
else: print(max(best,key=best.get))
PY
)
echo "[$(date)] warmup 赢家 = $WIN_TAG"
echo "warmup latest AUC 汇总:"; grep -E "wu_(zero1|zero5|ramp10)_latest" "$SUMMARY" 2>/dev/null; grep -E "wu_(zero1|zero5|ramp10)_latest" results/warmup_summary.csv 2>/dev/null

# ---- 2) 赢家补满 5seed ----
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
WIN="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 ifrank_loss_weight=1.0 if_target=hard use_strong_if=True corrT=0.9"
case "$WIN_TAG" in
  wu_zero5)  WE=5;  WM=zero;   PFX=wu_zero5 ;;
  wu_ramp10) WE=10; WM=rampup; PFX=wu_ramp10 ;;
  *)         WE=1;  WM=zero;   PFX=ifcf_p90_w1 ;;  # 现状, 已5seed
esac
runif(){ local sn=$1 seed=$2
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"; setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in $WIN; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  setkv "$tmp" ifrank_warmup_epochs "$WE"; setkv "$tmp" ifrank_warmup_mode "$WM"
  echo "[$(date)] === 补 $sn (warmup=$WE/$WM) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
if [ "$WIN_TAG" != "wu_zero1" ]; then
  for s in 4 5; do runif ${PFX}_s${s} "$s"; done
  evalg "${PFX}_s[12345]" "WINNER5_${WIN_TAG}"
else
  evalg "ifcf_p90_w1_s[12345]" "WINNER5_zero1"
fi

# ---- 3) base 基线垫场(用各框架自己的 config: lr/阈值原值, 不覆盖) ----
runbase(){ local alg=$1 seed=$2; local sn=${alg}_base_s${seed}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; return; }
  local cfg=config/usb_cv/${alg}/${alg}_bus_878_0.yaml; local tmp=config/_${sn}.yaml; cp "$cfg" "$tmp"
  setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True; setkv "$tmp" num_log_iter 110
  echo "[$(date)] === base $sn ==="
  python train.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"; }
for alg in freematch softmatch adamatch flexmatch; do
  for s in 1 2 3 4 5; do runbase $alg $s; done
  evalg "${alg}_base_s[12345]" "${alg}_base"
done
touch STAGE2_DONE; echo "[$(date)] ===== Stage2 (warmup赢家5seed + 4框架base) 完成 ====="
