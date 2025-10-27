#!/bin/bash
set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$NUM_GPUS attack_coco_caption_images.py \
  --caption_json ./datasets/COCO/caption/val_data.json \
  --image_root /data/datasets/COCO/caption \
  --pretrain_weights ./ckpts/pretrained_weights_flash_atten.pt