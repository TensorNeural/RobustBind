#!/bin/bash
set -e

torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) -m downstream.llava.eval_vqa \
  --model_dir ".cache/liuhaotian--llava-v1.6-mistral-7b" \
  --projector_weight "./ckpts/projector.pt" \
  --val_json "datasets/VQA2/val_data.json" \
  --image_root "/home/user/datasets/VQA2" \
  --output_dir "output/llava/eval/vqa" \
  --max_samples 5000
