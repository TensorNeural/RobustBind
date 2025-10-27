#!/usr/bin/env bash

# Each item contains three space-separated parts:
#   1. The .json file name (e.g. val_data.json)
#   2. The directory of the dataset (e.g. /data/datasets/ImageNet-1K)
#   3. The output directory (e.g. ./outputs/ImageNet-1K_val_data_zero_shot)
for combo in \
  "val_data_3000.json /data/datasets/ImageNet-1K ./outputs/ImageNet-1K_val_data_zero_shot"
do
  # Split each line into individual variables
  set -- $combo
  TEST_JSON="$1"
  DATASET_DIR="$2"
  OUTPUT_DIR="$3"

  CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_dataset_dir "$DATASET_DIR" \
    --test_data_path "./datasets/ImageNet-1K/$TEST_JSON" \
    --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
    --pretrain_weights ./ckpts/pretrained_weights_flash_atten.pt \
    --output_dir "$OUTPUT_DIR" \
    --modality image \
    --val_batch_size 2000 \
    --num_workers 4 \
    --seed 1234

done

# for combo in \
#   "val_data.json /data/datasets/ImageNet-1K ./outputs/ImageNet-1K_val_data_zero_shot" \
#   "val_5000_adv_eps2.json /data/datasets/ImageNet-1K/val_5000_adv ./outputs/ImageNet-1K_val_adv_eps2_data_zero_shot" \
#   "val_5000_adv_eps4.json /data/datasets/ImageNet-1K/val_5000_adv ./outputs/ImageNet-1K_val_adv_eps4_data_zero_shot"
# do
#   # Split each line into individual variables
#   set -- $combo
#   TEST_JSON="$1"
#   DATASET_DIR="$2"
#   OUTPUT_DIR="$3"

#   CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --test_dataset_dir "$DATASET_DIR" \
#     --test_data_path "./datasets/ImageNet-1K/$TEST_JSON" \
#     --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
#     --pretrain_weights ./ckpts/pretrained_weights_flash_atten.pt \
#     --output_dir "$OUTPUT_DIR" \
#     --modality image \
#     --val_batch_size 2000 \
#     --num_workers 4 \
#     --seed 1234

# done

# for combo in \
#   "val_data.json /data/datasets/ImageNet-1K ./outputs/ImageNet-1K_val_data_zero_shot" \
#   "val_5000_adv_eps2.json /data/datasets/ImageNet-1K/val_5000_adv ./outputs/ImageNet-1K_val_adv_eps2_data_zero_shot" \
#   "val_5000_adv_eps4.json /data/datasets/ImageNet-1K/val_5000_adv ./outputs/ImageNet-1K_val_adv_eps4_data_zero_shot"
# do
#   # Split each line into individual variables
#   set -- $combo
#   TEST_JSON="$1"
#   DATASET_DIR="$2"
#   OUTPUT_DIR="$3"

#   CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --test_dataset_dir "$DATASET_DIR" \
#     --test_data_path "./datasets/ImageNet-1K/$TEST_JSON" \
#     --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
#     --pretrain_weights ./ckpts/pretrained_weights.pt \
#     --output_dir "$OUTPUT_DIR" \
#     --modality image \
#     --val_batch_size 2000 \
#     --num_workers 4 \
#     --seed 1234

# done
