#!/bin/bash
set -e

# === Paths ===
ROOT_DIR="/home/user/datasets"
CKPT_DIR="./ckpts"
CENTER_EMB_DIR="./centre_embs"
OUTPUT_DIR="./output"

# === Center Embeddings ===
CENTER_AUDIO="${CENTER_EMB_DIR}/audio_esc_center_embeddings.pkl"
CENTER_IMAGE="${CENTER_EMB_DIR}/image_in_center_embeddings.pkl"
CENTER_EVENT="${CENTER_EMB_DIR}/event_nin_center_embeddings.pkl"
CENTER_POINT="${CENTER_EMB_DIR}/point_modelnet40_center_embeddings.pkl"

# === Validation JSONs ===
VAL_AUDIO="${ROOT_DIR}/ESC-50/val_data.json"
VAL_IMAGE="${ROOT_DIR}/ImageNet-1K/val_data.json"
VAL_EVENT="${ROOT_DIR}/N-ImageNet-1K/val_data.json"
VAL_POINT="${ROOT_DIR}/ModelNet40/val_data.json"

# === Run Cross-Modality Search ===
python audio_to_crossmodal.py \
  --dataset_root "${ROOT_DIR}" \
  --pretrain_weights "${CKPT_DIR}/pretrained_weights_flash_atten.pt" \
  --val_json_audio "${VAL_AUDIO}" \
  --val_json_image "${VAL_IMAGE}" \
  --val_json_event "${VAL_EVENT}" \
  --val_json_point "${VAL_POINT}" \
  --center_emb_audio "${CENTER_AUDIO}" \
  --center_emb_image "${CENTER_IMAGE}" \
  --center_emb_event "${CENTER_EVENT}" \
  --center_emb_point "${CENTER_POINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_flash_attention
