#!/bin/bash

# Define sample sizes
SIZES="500 1000 3000 5000 8000"

# # Standard inference
# for size in $SIZES; do
#   CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --test_dataset_dir /data/datasets/ImageNet-1K \
#     --test_data_path ./datasets/ImageNet-1K/val_data_${size}.json \
#     --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
#     --pretrain_weights ./ckpts/pretrained_weights.pt \
#     --output_dir ./outputs/ImageNet-1K_val_data_${size}_zero_shot \
#     --modality image \
#     --val_batch_size 2000 \
#     --num_workers 4 \
#     --seed 1234
# done

# Uniform inference
for size in $SIZES; do
  CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_dataset_dir /data/datasets/ImageNet-1K \
    --test_data_path ./datasets/ImageNet-1K/val_data_${size}_uniform.json \
    --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
    --pretrain_weights ./ckpts/pretrained_weights.pt \
    --output_dir ./outputs/ImageNet-1K_val_data_${size}_uniform_zero_shot \
    --modality image \
    --val_batch_size 2000 \
    --num_workers 4 \
    --seed 1234
done
