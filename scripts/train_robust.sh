#!/bin/bash
set -e

declare -A MODALITY_MAP=(
  [ImageNet-1K]=image
#   [Places365]=image
#   [ModalNet40]=point
#   [ShapeNet]=point
#   [ESC-50]=audio
#   [Urban-Sound-8K]=audio
#   [LLVIP]=thermal
#   [RGB-T]=thermal
#   [MSR-VTT]=video
#   [UCF-101]=video
#   [N-Caltech-101]=event
#   [N-ImageNet-1K]=event
)

declare -A EMB_SUFFIX_MAP=(
  [ImageNet-1K]=in
  [Places365]=p365
  [ModalNet40]=modelnet40
  [ShapeNet]=shapenet
  [ESC-50]=esc
  [Urban-Sound-8K]=us
  [LLVIP]=llvip
  [RGB-T]=rgbt
  [MSR-VTT]=msrvtt
  [UCF-101]=ucf
  [N-Caltech-101]=caltech
  [N-ImageNet-1K]=nin
)

# Config
OUTPUT_DIR=output
EPSILON=4
TRAIN_BATCH_SIZE=70
VAL_BATCH_SIZE=70
NUM_WORKERS=2
TRAIN_MAX_SAMPLES=10
VAL_MAX_SAMPLES=10
TENSORBOARD_DATA_DIR=tensorboard

for dataset in "${!MODALITY_MAP[@]}"; do
  modality="${MODALITY_MAP[$dataset]}"
  suffix="${EMB_SUFFIX_MAP[$dataset]}"

  echo "=== Training: $modality / $dataset ==="

  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
    --output_dir "$OUTPUT_DIR" \
    --modality "$modality" \
    --dataset_name "$dataset" \
    --dataset_root "/home/user/datasets/$dataset" \
    --train_json "./datasets/$dataset/train_data.json" \
    --val_json "./datasets/$dataset/val_data.json" \
    --pretrain_weights "./ckpts/pretrained_weights_flash_atten.pt" \
    --use_flash_attention \
    --center_emb "./centre_embs/${modality}_${suffix}_center_embeddings.pkl" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --train_max_samples "$TRAIN_MAX_SAMPLES" \
    --val_max_samples "$VAL_MAX_SAMPLES" \
    --train_attack_loss "l2" \
    --val_attack_loss "ce" \
    --train_loss "l2" \
    --lora_rank 5 \
    --lora_alpha 10 \
    --epsilon "$EPSILON" \
    --tensorboard_data_dir "$TENSORBOARD_DATA_DIR"
done
