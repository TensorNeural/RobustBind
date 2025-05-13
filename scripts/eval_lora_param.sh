#!/bin/bash
set -e

DATASET="LLVIP"
MODALITY="thermal"
SUFFIX="llvip"
EPSILON="2"
DATASET_ROOT="/home/user/datasets/${DATASET}"
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten.pt"
CENTER_EMB="./centre_embs/${MODALITY}_${SUFFIX}_center_embeddings.pkl"
OUTPUT_DIR="./output"
NUM_WORKERS=2
VAL_ATTACK_LOSS="ce"
TWO_STAGE_ITERS=100

VAL_JSON="./datasets/${DATASET}/val_data.json"
VAL_BATCH_SIZE=70
VAL_MAX_SAMPLES=16974

BASE_DIR="output/lora/eps${EPSILON}"

# === 3 with fixed rank = 4, varying alpha ===
for alpha in 4.0 8.0 16.0; do
  RANK=4
  LORA_WEIGHTS="${BASE_DIR}/lora_r${RANK}_a${alpha}/best_lora_weights.pt"
  echo "🧪 Eval: rank=${RANK}, alpha=${alpha}"
  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_unibind.py \
    --modality "$MODALITY" \
    --dataset_name "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_root "$DATASET_ROOT" \
    --attack_val_json "$VAL_JSON" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --val_max_samples "$VAL_MAX_SAMPLES" \
    --pretrain_weights "$PRETRAIN_WEIGHTS" \
    --center_emb "$CENTER_EMB" \
    --lora_weights_list "$LORA_WEIGHTS" \
    --num_workers "$NUM_WORKERS" \
    --use_flash_attention \
    --val_attack_loss "$VAL_ATTACK_LOSS" \
    --epsilons "$EPSILON" \
    --two_stage_iters "$TWO_STAGE_ITERS"
done

# === 2 with fixed alpha = 8.0, varying rank ===
for rank in 2 8; do
  ALPHA=8.0
  LORA_WEIGHTS="${BASE_DIR}/lora_r${rank}_a${ALPHA}/best_lora_weights.pt"
  echo "🧪 Eval: rank=${rank}, alpha=${ALPHA}"
  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_unibind.py \
    --modality "$MODALITY" \
    --dataset_name "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_root "$DATASET_ROOT" \
    --attack_val_json "$VAL_JSON" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --val_max_samples "$VAL_MAX_SAMPLES" \
    --pretrain_weights "$PRETRAIN_WEIGHTS" \
    --center_emb "$CENTER_EMB" \
    --lora_weights_list "$LORA_WEIGHTS" \
    --num_workers "$NUM_WORKERS" \
    --use_flash_attention \
    --val_attack_loss "$VAL_ATTACK_LOSS" \
    --epsilons "$EPSILON" \
    --two_stage_iters "$TWO_STAGE_ITERS"
done
