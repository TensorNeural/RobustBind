#!/bin/bash
set -e

declare -A MODALITY_MAP=(
  [ImageNet-1K]=image
  [Places365]=image
  [ModelNet40]=point
  # [ShapeNet]=point
  [ESC-50]=audio
  [UrbanSound8K]=audio
  [LLVIP]=thermal
  # [RGB-T]=thermal
  # [MSR-VTT]=video
  [UCF-101]=video
  # [N-Caltech-101]=event
  # [N-ImageNet-1K]=event
)

declare -A EMB_SUFFIX_MAP=(
  [ImageNet-1K]=in
  [Places365]=p365
  [ModelNet40]=modelnet40
  [ShapeNet]=shapenet
  [ESC-50]=esc
  [UrbanSound8K]=us
  [LLVIP]=llvip
  [RGB-T]=rgbt
  [MSR-VTT]=msrvtt
  [UCF-101]=ucf
  [N-Caltech-101]=caltech
  [N-ImageNet-1K]=nin
)

declare -A VAL_BATCH_SIZE_MAP=(
  [ImageNet-1K]=70
  [Places365]=70
  [ModelNet40]=64
  [ShapeNet]=64
  [ESC-50]=32
  [UrbanSound8K]=2
  [LLVIP]=16
  [RGB-T]=16
  [MSRVTT]=16
  [UCF-101]=6
  [N-Caltech-101]=70
  [N-ImageNet-1K]=70
)

declare -A VAL_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/val_data_3000.json"
  [Places365]="./datasets/Places365/val_data_3000.json"
  [ModelNet40]="./datasets/ModelNet40/val_data.json"
  [ShapeNet]="./datasets/ShapeNet/val_data.json"
  [ESC-50]="./datasets/ESC-50/val_data.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/val_data.json"
  [LLVIP]="./datasets/LLVIP/val_data.json"
  [RGB-T]="./datasets/RGB-T/val_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/val_data.json"
  [UCF-101]="./datasets/UCF-101/val_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/val_data_3000.json"
  [N-ImageNet-1K]="./datasets/N-ImageNet-1K/val_data_3000.json"
)

declare -A VAL_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=2
  [Places365]=2
  [ModelNet40]=2
  [ShapeNet]=2048
  [ESC-50]=2
  [UrbanSound8K]=2
  [LLVIP]=2
  [RGB-T]=500
  [MSR-VTT]=2
  [UCF-101]=2
  [N-Caltech-101]=3000
  [N-ImageNet-1K]=3000
)

# Common configuration
OUTPUT_DIR=output
NUM_WORKERS=2
EPSILONS="2,4"
TWO_STAGE_ITERS=100
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten.pt"
LORA_WEIGHTS_LIST="./ckpts/eps2_lora_weights.pt ./ckpts/eps4_lora_weights.pt"

for dataset in "${!MODALITY_MAP[@]}"; do
  modality="${MODALITY_MAP[$dataset]}"
  suffix="${EMB_SUFFIX_MAP[$dataset]}"
  val_batch_size="${VAL_BATCH_SIZE_MAP[$dataset]}"
  val_json="${VAL_JSON_MAP[$dataset]}"
  val_max_samples="${VAL_MAX_SAMPLES_MAP[$dataset]}"

  echo "=== Evaluating: $modality / $dataset (Batch size: $val_batch_size, Max samples: $val_max_samples) ==="

  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_unibind.py \
    --modality "$modality" \
    --dataset_name "$dataset" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_root "/home/user/datasets/$dataset" \
    --val_json "$val_json" \
    --pretrain_weights "$PRETRAIN_WEIGHTS" \
    --center_emb "./centre_embs/${modality}_${suffix}_center_embeddings.pkl" \
    --lora_weights_list $LORA_WEIGHTS_LIST \
    --val_batch_size "$val_batch_size" \
    --num_workers "$NUM_WORKERS" \
    --val_max_samples "$val_max_samples" \
    --use_flash_attention \
    --val_attack_loss "ce" \
    --epsilons "$EPSILONS" \
    --run_clean_eval \
    --two_stage_iters "$TWO_STAGE_ITERS"
done
