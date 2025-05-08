#!/bin/bash
set -e

declare -A MODALITY_MAP=(
  # [ImageNet-1K]=image
  # [Places365]=image
  # [ModelNet40]=point
  # [ShapeNet]=point
  # [ESC-50]=audio
  # [UrbanSound8K]=audio
  # [LLVIP]=thermal
  # [RGB-T]=thermal
  # [MSRVTT]=video
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
  [MSRVTT]=msrvtt
  [UCF-101]=ucf
  [N-Caltech-101]=caltech
  [N-ImageNet-1K]=nin
)

declare -A TRAIN_BATCH_SIZE_MAP=(
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

declare -A TRAIN_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/train_data.json"
  [Places365]="./datasets/Places365/train_data.json"
  [ModelNet40]="./datasets/ModelNet40/train_data.json"
  [ShapeNet]="./datasets/ShapeNet/train_data.json"
  [ESC-50]="./datasets/ESC-50/train_data.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/train_data.json"
  [LLVIP]="./datasets/LLVIP/train_data.json"
  [RGB-T]="./datasets/RGB-T/train_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/train_data.json"
  [UCF-101]="./datasets/UCF-101/train_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/train_data.json"
  [N-ImageNet-1K]="./datasets/N-ImageNet-1K/train_data.json"
)

declare -A VAL_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/val_data_3000.json"
  [Places365]="./datasets/Places365/val_data.json"
  [ModelNet40]="./datasets/ModelNet40/val_data.json"
  [ShapeNet]="./datasets/ShapeNet/val_data.json"
  [ESC-50]="./datasets/ESC-50/val_data.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/val_data.json"
  [LLVIP]="./datasets/LLVIP/val_data.json"
  [RGB-T]="./datasets/RGB-T/val_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/val_data.json"
  [UCF-101]="./datasets/UCF-101/val_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/val_data.json"
  [N-ImageNet-1K]="./datasets/N-ImageNet-1K/val_data.json"
)

declare -A TRAIN_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=1281167        # from train_data.json
  [Places365]=0                # no train_data.json reported
  [ModelNet40]=9843            # from train_data.json
  [ShapeNet]=20480             # unchanged (not reported)
  [ESC-50]=1600                # from train_data.json
  [UrbanSound8K]=7079          # from train_data.json
  [LLVIP]=15000                # from train_data.json
  [RGB-T]=800                  # placeholder
  [MSR-VTT]=1000               # placeholder
  [UCF-101]=9537               # from train_data.json
  [N-Caltech-101]=3060         # from actual bin-to-png split
  [N-ImageNet-1K]=128000       # placeholder
)

declare -A VAL_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=3000           # from val_data_3000.json
  [Places365]=3000             # from val_data_3000.json
  [ModelNet40]=2468            # from val_data.json
  [ShapeNet]=2048              # placeholder
  [ESC-50]=400                 # from val_data.json
  [UrbanSound8K]=1653          # from val_data.json
  [LLVIP]=21354                # from val_data.json
  [RGB-T]=500                  # placeholder
  [MSR-VTT]=500                # placeholder
  [UCF-101]=3783               # from val_data.json
  [N-Caltech-101]=3000         # from actual bin-to-png split
  [N-ImageNet-1K]=3000         # placeholder
)

# Config
OUTPUT_DIR=output
EPSILON=4
NUM_WORKERS=2
TENSORBOARD_DATA_DIR=tensorboard

for dataset in "${!MODALITY_MAP[@]}"; do
  modality="${MODALITY_MAP[$dataset]}"
  suffix="${EMB_SUFFIX_MAP[$dataset]}"
  train_batch_size="${TRAIN_BATCH_SIZE_MAP[$dataset]}"
  val_batch_size="${VAL_BATCH_SIZE_MAP[$dataset]}"
  train_json="${TRAIN_JSON_MAP[$dataset]}"
  val_json="${VAL_JSON_MAP[$dataset]}"
  train_max_samples="${TRAIN_MAX_SAMPLES_MAP[$dataset]}"
  val_max_samples="${VAL_MAX_SAMPLES_MAP[$dataset]}"

  echo "=== Training: $modality / $dataset (Train BS: $train_batch_size, Val BS: $val_batch_size) ==="

  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
    --output_dir "$OUTPUT_DIR" \
    --modality "$modality" \
    --dataset_name "$dataset" \
    --dataset_root "/home/user/datasets/$dataset" \
    --train_json "$train_json" \
    --val_json "$val_json" \
    --pretrain_weights "./ckpts/pretrained_weights_flash_atten.pt" \
    --use_flash_attention \
    --center_emb "./centre_embs/${modality}_${suffix}_center_embeddings.pkl" \
    --train_batch_size "$train_batch_size" \
    --val_batch_size "$val_batch_size" \
    --num_workers "$NUM_WORKERS" \
    --train_max_samples "$train_max_samples" \
    --val_max_samples "$val_max_samples" \
    --train_attack_loss "l2" \
    --val_attack_loss "ce" \
    --train_loss "l2" \
    --lora_rank 5 \
    --lora_alpha 10 \
    --epsilon "$EPSILON" \
    --tensorboard_data_dir "$TENSORBOARD_DATA_DIR"
done
