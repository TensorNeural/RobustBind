#!/bin/bash
set -e

# === Dataset & Modality Setup ===
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

# === From reference mapping ===
CLEAN_VAL_JSON="./datasets/${DATASET}/val_data.json"
ATTACK_VAL_JSON="./datasets/${DATASET}/val_data.json"
CLEAN_VAL_BATCH_SIZE=1000
ATTACK_VAL_BATCH_SIZE=300
CLEAN_VAL_MAX_SAMPLES=5000
ATTACK_VAL_MAX_SAMPLES=5000

BASE_DIR="output/train/${MODALITY}/${DATASET}__${DATASET}/eps${EPSILON}"

GPUS=$(nvidia-smi -L | wc -l)

COMMON_ARGS="--modality $MODALITY \
  --dataset_name $DATASET \
  --output_dir $OUTPUT_DIR \
  --dataset_root $DATASET_ROOT \
  --clean_val_json $CLEAN_VAL_JSON \
  --attack_val_json $ATTACK_VAL_JSON \
  --clean_val_batch_size $CLEAN_VAL_BATCH_SIZE \
  --attack_val_batch_size $ATTACK_VAL_BATCH_SIZE \
  --clean_val_max_samples $CLEAN_VAL_MAX_SAMPLES \
  --attack_val_max_samples $ATTACK_VAL_MAX_SAMPLES \
  --pretrain_weights $PRETRAIN_WEIGHTS \
  --center_emb $CENTER_EMB \
  --num_workers $NUM_WORKERS \
  --use_flash_attention \
  --val_attack_loss $VAL_ATTACK_LOSS \
  --epsilons $EPSILON \
  --two_stage_iters $TWO_STAGE_ITERS \
  --run_clean_eval"

# === Sweep: fixed rank = 4, varying alpha ===
for alpha in 4.0 8.0 16.0; do
  for rank in 2 4 8; do
    LORA_WEIGHTS="${BASE_DIR}/lora_r${rank}_a${alpha}/best_lora_weights.pt"
    echo "🧪 Eval: rank=${rank}, alpha=${alpha}"
    torchrun --nproc_per_node=$GPUS eval_unibind.py \
      $COMMON_ARGS \
      --lora_weights_list "$LORA_WEIGHTS" \
      --lora_rank "$rank" \
      --lora_alpha "$alpha"
  done
done
