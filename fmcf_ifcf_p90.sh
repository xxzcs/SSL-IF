#!/usr/bin/env bash
# p90 下的 +IF(w2): 补齐 p90 阈值的提升, 与 p95 对比"哪个阈值提升更多"。hard/强/0.9/balanced/w2。3 seed。
set -u
cd /home/xiexiaozheng/Semi-supervised-learning
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate wssl 2>/dev/null
STAMP=$(date +%Y%m%d_%H%M%S); LOG=logs/ifcf_p90_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1
FIX=config/usb_cv/fixmatch/fixmatch_bus_878_0.yaml
SUMMARY=results/ifcf_p90_summary.csv
setkv(){ local f=$1 k=$2 v=$3; if grep -qE "^${k}:" "$f"; then sed -i "s|^${k}:.*|${k}: ${v}|" "$f"; else echo "${k}: ${v}" >> "$f"; fi; }
IFR="ifrank_combine=multiply_balanced corrT=0.9 use_strong_if=True if_lambda=1 csim_lambda=1 num_references=4 ref_select=by_instance ref_cand_k=8 if_target=hard ifrank_loss_weight=1.0"
for s in 1 2 3; do
  sn=ifcf_p90_w1_s${s}
  [ -f "saved_models/usb_cv/${sn}/model_best.pth" ] && { echo "[skip] $sn"; continue; }
  tmp=config/_${sn}.yaml; cp "$FIX" "$tmp"
  setkv "$tmp" algorithm fixmatch_ifcf; setkv "$tmp" save_name "$sn"; setkv "$tmp" seed "$s"
  setkv "$tmp" p_cutoff 0.9
  setkv "$tmp" multiprocessing_distributed False; setkv "$tmp" resume False; setkv "$tmp" overwrite True
  setkv "$tmp" num_log_iter 110; setkv "$tmp" lr 0.0046875
  for kv in $IFR; do setkv "$tmp" "${kv%%=*}" "${kv#*=}"; done
  echo "[$(date)] === train $sn (p90, ifcf w2) ==="
  python train_ifcf.py --c "$tmp" || { echo "[retry] $sn"; rm -rf saved_models/usb_cv/$sn; python train_ifcf.py --c "$tmp" || echo "[fail] $sn"; }
  rm -f "$tmp"
done
for kind in best latest; do
  [ "$kind" = best ] && ck=model_best.pth || ck=latest_model.pth
  python3 eval_sup.py --dataset bus --num_classes 2 --summary_csv "$SUMMARY" --net resnet18 --model_key ema_model \
    --data_dir ../uda_data --batch_size 16 --num_labels 878 --eval_dest test \
    --lpath ../data_split/28/labeled_images_20_9.pth --ulpath ../data_split/28/unlabeled_images_80_9.pth \
    --load_glob "saved_models/usb_cv/ifcf_p90_w1_*/${ck}" --method_suffix "ifcf_p90_w1_${kind}"
done
touch IFCF_P90_DONE
echo "[$(date)] ===== ifcf p90 w2 done ====="
