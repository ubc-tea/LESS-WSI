#!/bin/bash
# 5-fold CV across 5 seeds for the CrossViT aggregator (Step 1).
# Run from 1-WSI_aggregation/ (or `cd` into it).
#
# Override these as needed:
DATA_PATH="${DATA_PATH:-PATH_TO_SAVED_FEATURES}"   # Step-0 --feature_root
MODEL="${MODEL:-crossvit_base_224}"                # see models/crossvit.py
VPU_DIM="${VPU_DIM:-384_768}"                      # must match MODEL's embed_dim
DATASET="${DATASET:-urine}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-6}"

cd "$(dirname "$0")/1-WSI_aggregation" || exit 1

for seed in 0 42 212330 2294892 990624; do
  for nth_fold in 0 1 2 3 4; do
    python main.py \
      --model "$MODEL" \
      --features VPU \
      --vpu_dim "$VPU_DIM" \
      --data_set "$DATASET" \
      --data-path "$DATA_PATH" \
      --nth_fold "$nth_fold" \
      --seed "$seed" \
      --batch-size 1 \
      --epochs "$EPOCHS" \
      --opt adam \
      --lr "$LR" \
      --warmup-lr 1e-6 --warmup-epochs 1 \
      --sched cosine \
      --weight-decay 0.1 \
      --drop 0.1 \
      --output_dir "./outputs/${MODEL}_seed${seed}_fold${nth_fold}"
  done
done
