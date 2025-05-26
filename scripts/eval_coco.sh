#!/bin/bash
set -e

torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) -m downstream.llava.eval_coco \
  --model_dir .cache/liuhaotian--llava-v1.6-mistral-7b \
  --projector_weight ./ckpts/coco_projector.pt \
  --val_json datasets/COCO/caption/val_data.json \
  --image_root /home/user/datasets/COCO/caption \
  --output_dir output/llava/eval/coco \
  --max_samples 20
