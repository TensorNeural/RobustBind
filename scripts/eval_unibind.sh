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

# Common configuration
OUTPUT_DIR=output
VAL_BATCH_SIZE=64
NUM_WORKERS=2
VAL_MAX_SAMPLES=10
EPSILONS="2,4"
TWO_STAGE_ITERS=100
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten.pt"
LORA_WEIGHTS_LIST="./ckpts/eps2_lora_weights.pt ./ckpts/eps4_lora_weights.pt"

for dataset in "${!MODALITY_MAP[@]}"; do
  modality="${MODALITY_MAP[$dataset]}"
  suffix="${EMB_SUFFIX_MAP[$dataset]}"

  echo "=== Evaluating: $modality / $dataset ==="

  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_unibind.py \
    --modality "$modality" \
    --dataset_name "$dataset" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_root "/home/user/datasets/$dataset" \
    --val_json "./datasets/$dataset/val_data.json" \
    --pretrain_weights "$PRETRAIN_WEIGHTS" \
    --center_emb "./centre_embs/${modality}_${suffix}_center_embeddings.pkl" \
    --lora_weights_list $LORA_WEIGHTS_LIST \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --val_max_samples "$VAL_MAX_SAMPLES" \
    --use_flash_attention \
    --val_attack_loss "ce" \
    --epsilons "$EPSILONS" \
    --run_clean_eval \
    --two_stage_iters "$TWO_STAGE_ITERS"
done
