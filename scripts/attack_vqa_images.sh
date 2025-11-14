#!/bin/bash
set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)

VAL_JSON=./datasets/VQA2/val_data.json
IMAGE_ROOT=/data/datasets/VQA2
PRETRAIN_WEIGHTS=./ckpts/pretrained_weights_flash_atten_image_patchs.pt

torchrun --nproc_per_node=$NUM_GPUS attack_vqa_images.py \
    --val_json "$VAL_JSON" \
    --image_root "$IMAGE_ROOT" \
    --pretrain_weights "$PRETRAIN_WEIGHTS"
