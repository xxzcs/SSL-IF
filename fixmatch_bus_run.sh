#!/usr/bin/env bash
# FixMatch+IF 忠实复现: 重跑用户 5 个"曾有效"配方(软标签原版)+ 硬标签对照, 看增量在 3090/semilearn 下是否还在。
# 5 配方(记忆 fixmatch-if-effective-configs, 自然权重 ifrank_loss_weight=1.0):
#   c1 balanced/λ1:1/T0.5/弱IF   c2 balanced/λ1:1/T0.9/强IF
#   c3 multiplyo/λ10:1/T0.05/强   c4 multiplyo/λ15:1/T0.05/强   c5 multiplyo/λ20:1/T0.05/强(multiplyo最佳)
# 顺序: ①等孤儿 FlexMatch s5 释放GPU ②FlexMatch s5 收尾+eval ③FixMatch base 5seed ④5软 5seed ⑤5硬 5seed。
# 全 3090/semilearn 同套: lr0.0046875 / ema_model eval / skip-completed + 重试一次。软先出(核心答案), 硬随后。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
SUMMARY=results/fim_summary.csv          # FixMatch base + 5配方 结果
FXSUM=results/gen5_summary.csv           # FlexMatch s5 收尾并入 gen5
FXCFG=config/usb_cv/flexmatch/flexmatch_bus_878_fixed.yaml
FIMCFG=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
LR=0.0046875
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/fim_${STAMP}.log
mkdir -p logs results config
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
COMMON="num_references=4 ref_select=by_instance ref_cand_k=8 if_mean_reduce=True ifrank_loss_weight=1.0"
declare -A CFGV
CFGV[c1]="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 corrT=0.5 use_strong_if=False"
CFGV[c2]="ifrank_combine=multiply_balanced if_lambda=1 csim_lambda=1 corrT=0.9 use_strong_if=True"
CFGV[c3]="ifrank_combine=multiplyo if_lambda=10 csim_lambda=1 corrT=0.05 use_strong_if=True"
CFGV[c4]="ifrank_combine=multiplyo if_lambda=15 csim_lambda=1 corrT=0.05 use_strong_if=True"
CFGV[c5]="ifrank_combine=multiplyo if_lambda=20 csim_lambda=1 corrT=0.05 use_strong_if=True"

setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
run_one(){ local cfg=$1 launcher=$2 algo=$3 sn=$4 seed=$5; shift 5
  if [ -f "saved_models/usb_cv/${sn}/model_best.pth" ]; then echo "[skip] $sn"; return 0; fi
  local tmp="config/_fim_${sn}.yaml"; cp "$cfg" "$tmp"
  setkv "$tmp" algorithm "$algo"; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$seed"
  setkv "$tmp" load_path "./saved_models/usb_cv/${sn}/latest_model.pth"
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  setkv "$tmp" lr "$LR"
  for kv in "$@"; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train $sn ($algo) ==="
  python "$launcher" --c "$tmp" || { echo "[warn] $sn 重试"; rm -rf "saved_models/usb_cv/${sn}"; python "$launcher" --c "$tmp" || echo "[warn] $sn 跳过"; }
  rm -f "$tmp"
}
eval_group(){ local sum=$1 pre=$2; for kind in best latest; do [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$sum" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/${pre}_*/${ck}" --method_suffix "${pre}_${kind}" || echo "[warn] eval $pre"; done; }

# ---- ① 等孤儿 FlexMatch s5(及任何 gen5 残留)跑完释放 GPU ----
echo "[$(date)] 等孤儿 FlexMatch s5 / gen5 残留释放 GPU..."
while pgrep -f 'config/_fim_gen_fx_ifw2_s5' >/dev/null 2>&1 || pgrep -f 'config/_g5_' >/dev/null 2>&1; do sleep 60; done
echo "[$(date)] GPU 空闲, 开始."

# ---- ② FlexMatch 硬-w2 s5 收尾(孤儿已训则自动 skip)+ eval, 并入 gen5 ----
run_one "$FXCFG" train_ifcf.py flexmatch_ifcf gen_fx_ifw2_s5 5 \
  ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 \
  num_references=4 ref_select=by_instance ref_cand_k=8 if_mean_reduce=True if_target=hard ifrank_loss_weight=2.0
eval_group "$FXSUM" gen_fx_ifw2

# ---- ③ FixMatch base 5-seed ----
for s in 1 2 3 4 5; do run_one "$FIMCFG" train.py fixmatch fim_base_s${s} $s; done
eval_group "$SUMMARY" fim_base

# ---- ④ 5 配方 软标签 5-seed (核心: 增量是否还在) ----
for c in c1 c2 c3 c4 c5; do
  for s in 1 2 3 4 5; do run_one "$FIMCFG" train_ifcf.py fixmatch_ifcf "fim_${c}s_s${s}" $s $COMMON ${CFGV[$c]} if_target=soft; done
  eval_group "$SUMMARY" "fim_${c}s"
  echo "[$(date)] ---- 软 $c 完成 ----"
done

# ---- ⑤ 5 配方 硬标签 5-seed (对照) ----
for c in c1 c2 c3 c4 c5; do
  for s in 1 2 3 4 5; do run_one "$FIMCFG" train_ifcf.py fixmatch_ifcf "fim_${c}h_s${s}" $s $COMMON ${CFGV[$c]} if_target=hard; done
  eval_group "$SUMMARY" "fim_${c}h"
  echo "[$(date)] ---- 硬 $c 完成 ----"
done

touch results/ALL_DONE_FIXMATCH
echo "[$(date)] ===== FixMatch base + 5配方(软/硬)BUS 5-seed 完成. $SUMMARY ====="
