#!/bin/bash
set -e
export OMP_NUM_THREADS=$(nproc)
NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$NUM_GPUS -m eval_llava_vqa \
  --model_path liuhaotian/llava-v1.6-mistral-7b \
  --projector_weight "./ckpts/vqa2_projector.pt" \
  --image_root "/data/datasets/VQA2" \
  --output_dir "output/llava/eval/vqa"
