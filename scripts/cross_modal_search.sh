#!/bin/bash
set -e

# === Directories ===
ROOT_DIR="/data/datasets"
CKPT_DIR="./ckpts"
CENTER_EMB_DIR="./centre_embs"
OUTPUT_DIR="./output"

# === Center Embedding Files ===
CENTER_AUDIO="${CENTER_EMB_DIR}/audio_esc_center_embeddings.pkl"
CENTER_IMAGE="${CENTER_EMB_DIR}/image_p365_center_embeddings.pkl"
CENTER_EVENT="${CENTER_EMB_DIR}/event_nin_center_embeddings.pkl"
CENTER_POINT="${CENTER_EMB_DIR}/point_modelnet40_center_embeddings.pkl"

# === Validation JSON Metadata ===
VAL_AUDIO="./datasets/ESC-50/train_data.json"
VAL_IMAGE="./datasets/Places365/val_data.json"
VAL_EVENT="./datasets/N-ImageNet-1K/val_data.json"
VAL_POINT="./datasets/ModelNet40/val_data.json"

# === Run Distributed Cross-Modality Search ===
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) -m downstream.cross_modality_search \
  --dataset_root "${ROOT_DIR}" \
  --val_json_audio "${VAL_AUDIO}" \
  --val_json_image "${VAL_IMAGE}" \
  --val_json_event "${VAL_EVENT}" \
  --val_json_point "${VAL_POINT}" \
  --center_emb_audio "${CENTER_AUDIO}" \
  --center_emb_image "${CENTER_IMAGE}" \
  --center_emb_event "${CENTER_EVENT}" \
  --center_emb_point "${CENTER_POINT}" \
  --label_map "./datasets/esc50_label_map.json" \
  --pretrain_weights "${CKPT_DIR}/pretrained_weights_flash_atten_image_patchs.pt" \
  --output_dir "${OUTPUT_DIR}" \
  --use_flash_attention
