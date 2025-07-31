#!/bin/bash
set -e
export OMP_NUM_THREADS=$(nproc)
NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$NUM_GPUS -m downstream.llava.eval_vqa \
  --model_dir ".cache/liuhaotian--llava-v1.6-mistral-7b" \
  --projector_weight "./ckpts/projector.pt" \
  --image_root "/home/user/datasets/VQA2" \
  --output_dir "output/llava/eval/vqa"
