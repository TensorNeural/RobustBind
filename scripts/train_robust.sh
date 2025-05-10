#!/bin/bash
set -e

# === Model type to modalities ===
declare -A MODEL_TYPE_TO_MODALITIES=(
  # [vision]="image video event"
  # [audio]="audio"
  [thermal]="thermal"
  # [point]="point"
)

# === Active training dataset per modality ===
declare -A TRAIN_MODALITY_TO_DATASET=(
  # --- Vision ---
  [image]="ImageNet-1K"
  # [image]="Places365"
  # [video]="UCF-101"
  # [video]="MSR-VTT"
  # [event]="N-Caltech-101"
  # [event]="N-ImageNet-1K"

  # --- Audio ---
  [audio]="FSD-50K"
  # [audio]="ESC-50"
  # [audio]="UrbanSound8K"

  # --- Thermal ---
  [thermal]="LLVIP"
  # [thermal]="RGB-T"

  # --- Point ---
  [point]="ModelNet40"
  # [point]="ShapeNet"
)

# === Active validation dataset per modality ===
declare -A VAL_MODALITY_TO_DATASET=(
  # --- Vision ---
  # [image]="ImageNet-1K"
  # [image]="Places365"
  [video]="UCF-101"
  # [video]="MSR-VTT"
  # [event]="N-Caltech-101"
  # [event]="N-ImageNet-1K"

  # --- Audio ---
  [audio]="ESC-50"
  # [audio]="UrbanSound8K"

  # --- Thermal ---
  [thermal]="LLVIP"
  # [thermal]="RGB-T"

  # --- Point ---
  [point]="ModelNet40"
)

# === Dataset-specific config ===

declare -A DATASET_TO_BATCH_SIZE=(
  # --- Vision ---
  [ImageNet-1K]=1
  [Places365]=70
  [UCF-101]=6
  [MSR-VTT]=6
  [N-Caltech-101]=70
  [N-ImageNet-1K]=70

  # --- Audio ---
  [FSD-50K]=90
  [ESC-50]=90
  [UrbanSound8K]=2

  # --- Thermal ---
  [LLVIP]=280
  [RGB-T]=16

  # --- Point ---
  [ModelNet40]=64
  [ShapeNet]=64
)

declare -A TRAIN_MAX_SAMPLES_MAP=(
  # --- Vision ---
  # [ImageNet-1K]=1281167
  [ImageNet-1K]=18
  [Places365]=0
  [UCF-101]=9537
  [MSR-VTT]=3000
  [N-Caltech-101]=3060
  [N-ImageNet-1K]=1281167

  # --- Audio ---
  [FSD-50K]=36796
  [ESC-50]=1600
  [UrbanSound8K]=7079

  # --- Thermal ---
  [LLVIP]=67900
  [RGB-T]=800

  # --- Point ---
  [ModelNet40]=9843
  [ShapeNet]=20480
)

declare -A VAL_MAX_SAMPLES_MAP=(
  # --- Vision ---
  # [ImageNet-1K]=3000
  [ImageNet-1K]=6
  [Places365]=3000
  # [UCF-101]=3783
  [UCF-101]=6
  [MSR-VTT]=3000
  [N-Caltech-101]=3000
  [N-ImageNet-1K]=3000

  # --- Audio ---
  [ESC-50]=400
  [UrbanSound8K]=1653

  # --- Thermal ---
  [LLVIP]=16974
  [RGB-T]=500

  # --- Point ---
  [ModelNet40]=2468
  [ShapeNet]=2048
)

declare -A TRAIN_JSON_MAP=(
  # --- Vision ---
  [ImageNet-1K]="./datasets/ImageNet-1K/train_data.json"
  [Places365]="./datasets/Places365/train_data.json"
  [UCF-101]="./datasets/UCF-101/train_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/train_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/train_data.json"
  [N-ImageNet-1K]="./datasets/N-ImageNet-1K/train_data.json"

  # --- Audio ---
  [FSD-50K]="./datasets/FSD-50K/train_data.json"
  [ESC-50]="./datasets/ESC-50/train_data.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/train_data.json"

  # --- Thermal ---
  [LLVIP]="./datasets/LLVIP/train_data.json"
  [RGB-T]="./datasets/RGB-T/train_data.json"

  # --- Point ---
  [ModelNet40]="./datasets/ModelNet40/train_data.json"
  [ShapeNet]="./datasets/ShapeNet/train_data.json"
)

declare -A VAL_JSON_MAP=(
  # --- Vision ---
  [ImageNet-1K]="./datasets/ImageNet-1K/val_data_3000.json"
  [Places365]="./datasets/Places365/val_data.json"
  [UCF-101]="./datasets/UCF-101/val_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/val_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/val_data.json"
  [N-ImageNet-1K]="./datasets/N-ImageNet-1K/val_data.json"

  # --- Audio ---
  [ESC-50]="./datasets/ESC-50/val_data.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/val_data.json"

  # --- Thermal ---
  [LLVIP]="./datasets/LLVIP/val_data.json"
  [RGB-T]="./datasets/RGB-T/val_data.json"

  # --- Point ---
  [ModelNet40]="./datasets/ModelNet40/val_data.json"
  [ShapeNet]="./datasets/ShapeNet/val_data.json"
)

declare -A EMB_SUFFIX_MAP=(
  # --- Vision ---
  [ImageNet-1K]=in
  [Places365]=p365
  [UCF-101]=ucf
  [MSR-VTT]=msr
  [N-Caltech-101]=caltech
  [N-ImageNet-1K]=nin

  # --- Audio ---
  [ESC-50]=esc
  [UrbanSound8K]=us

  # --- Thermal ---
  [LLVIP]=llvip
  [RGB-T]=rgbt

  # --- Point ---
  [ModelNet40]=modelnet40
  [ShapeNet]=shapenet
)

# === Constants ===
OUTPUT_DIR=output
NUM_WORKERS=2
TENSORBOARD_DATA_DIR=tensorboard
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten.pt"
EPSILONS=(2 4)

# === Main loop ===

for model_type in "${!MODEL_TYPE_TO_MODALITIES[@]}"; do
  for train_modality in ${MODEL_TYPE_TO_MODALITIES[$model_type]}; do
    train_dataset="${TRAIN_MODALITY_TO_DATASET[$train_modality]}"
    [[ -z "$train_dataset" ]] && continue

    for val_modality in ${MODEL_TYPE_TO_MODALITIES[$model_type]}; do
      val_dataset="${VAL_MODALITY_TO_DATASET[$val_modality]}"
      [[ -z "$val_dataset" ]] && continue

      train_bs="${DATASET_TO_BATCH_SIZE[$train_dataset]}"
      val_bs="${DATASET_TO_BATCH_SIZE[$val_dataset]}"
      train_max="${TRAIN_MAX_SAMPLES_MAP[$train_dataset]}"
      val_max="${VAL_MAX_SAMPLES_MAP[$val_dataset]}"
      train_json="${TRAIN_JSON_MAP[$train_dataset]}"
      val_json="${VAL_JSON_MAP[$val_dataset]}"
      emb_suffix="${EMB_SUFFIX_MAP[$val_dataset]}"
      emb_path="./centre_embs/${val_modality}_${emb_suffix}_center_embeddings.pkl"

      for eps in "${EPSILONS[@]}"; do
        echo "=== $model_type | $train_dataset => $val_dataset | eps=$eps ==="
        echo "=== Embedding path: $emb_path ==="

        torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
          --model_type "$model_type" \
          --train_modality "$train_modality" \
          --val_modality "$val_modality" \
          --train_dataset_name "$train_dataset" \
          --val_dataset_name "$val_dataset" \
          --train_dataset_root "/home/user/datasets/$train_dataset" \
          --val_dataset_root "/home/user/datasets/$val_dataset" \
          --train_json "$train_json" \
          --val_json "$val_json" \
          --pretrain_weights "$PRETRAIN_WEIGHTS" \
          --center_emb "$emb_path" \
          --train_batch_size "$train_bs" \
          --val_batch_size "$val_bs" \
          --num_workers "$NUM_WORKERS" \
          --train_max_samples "$train_max" \
          --val_max_samples "$val_max" \
          --train_attack_loss "l2" \
          --val_attack_loss "ce" \
          --train_loss "l2" \
          --lora_rank 4 \
          --lora_alpha 8 \
          --epsilon "$eps" \
          --use_flash_attention \
          --tensorboard_data_dir "$TENSORBOARD_DATA_DIR" \
          --output_dir "$OUTPUT_DIR"
      done
    done
  done
done
