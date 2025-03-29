#!/usr/bin/env bash

# Each item contains three space-separated parts:
#   1. The .json file name (e.g. val_data.json)
#   2. The directory of the dataset (e.g. /home/user/datasets/Places365)
#   3. The output directory (e.g. ./outputs/Places365_val_data_zero_shot)
for combo in \
  "val_data.json /home/user/datasets/Places365 ./outputs/Places365_val_data_zero_shot" \
  "val_adv_eps2.json /home/user/datasets/Places365/val_adv ./outputs/Places365_val_adv_eps2_data_zero_shot" \
  "val_adv_eps4.json /home/user/datasets/Places365/val_adv ./outputs/Places365_val_adv_eps4_data_zero_shot"
do
  # Split each line into individual variables
  set -- $combo
  TEST_JSON="$1"
  DATASET_DIR="$2"
  OUTPUT_DIR="$3"

  CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_dataset_dir "$DATASET_DIR" \
    --test_data_path "./datasets/Places365/$TEST_JSON" \
    --centre_embeddings_path ./centre_embs/image_p365_center_embeddings.pkl \
    --pretrain_weights ./ckpts/pretrained_weights.pt \
    --output_dir "$OUTPUT_DIR" \
    --modality image \
    --val_batch_size 2000 \
    --num_workers 4 \
    --seed 1234

done
