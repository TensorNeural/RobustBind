#!/bin/bash
set -e

# === User-defined paths ===
COCO_JSON="datasets/COCO/caption/train_data.json"
VQA_JSON="datasets/VQA2/train_data.json"
DATASET_ROOT="/home/user/datasets"
PRETRAINED_MODEL="liuhaotian/llava-v1.6-mistral-7b"
UNIBIND_WEIGHTS="ckpts/pretrained_weights_flash_atten.pt"
OUTPUT_DIR="output/llava"

# === Training params ===
BATCH_SIZE=200
NUM_WORKERS=2

# === Launch training ===
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) -m downstream.llava.train_mm_projector \
  --coco_json "$COCO_JSON" \
  --vqa_json "$VQA_JSON" \
  --dataset_root "$DATASET_ROOT" \
  --pretrained_model "$PRETRAINED_MODEL" \
  --unibind_weights "$UNIBIND_WEIGHTS" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS"