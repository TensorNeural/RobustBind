#!/bin/bash
set -e

# Shared constants
OUTPUT_DIR=output
NUM_WORKERS=4
TENSORBOARD_DATA_DIR=tensorboard
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten.pt"

declare -A MODEL_TYPE_TO_MODALITIES=(
  [vision]="image video event"
  [audio]="audio"
  [thermal]="thermal"
  [point]="point"
)

# Robust config
ROBUST_EPSILONS=(2)
ROBUST_LORA_RANKS=(2 4 8)
ROBUST_LORA_ALPHAS=(4 8 16)
ROBUST_EPOCHS=2
ROBUST_MODES=(lora full_fine_tune)

declare -A ROBUST_TRAIN_MODALITY_TO_DATASET=(
  # [image]="ImageNet-1K"
  # [image]="Places365"
  [video]="Kinetics-400"
  # [video]="UCF-101"
  # [video]="MSR-VTT"
  # [event]="N-Caltech-101"
  # [event]="N-ImageNet-1K"
  [audio]="FSD-50K"
  # [audio]="ESC-50"
  # [audio]="UrbanSound8K"
  [thermal]="LLVIP"
  # [thermal]="RGB-T"
  [point]="ModelNet40"
  # [point]="ShapeNet"
)

declare -A ROBUST_VAL_MODALITY_TO_DATASET=(
  # [image]="ImageNet-1K"
  # [image]="Places365"
  # [video]="UCF-101"
  [video]="MSR-VTT"
  # [event]="N-Caltech-101"
  # [event]="N-ImageNet-1K"
  [audio]="ESC-50"
  # [audio]="UrbanSound8K"
  [thermal]="LLVIP"
  # [thermal]="RGB-T"
  [point]="ModelNet40"
)

declare -A ROBUST_DATASET_TO_BATCH_SIZE=(
  [ImageNet-1K]=1
  [Places365]=70
  [Kinetics-400]=25
  [UCF-101]=6
  [MSR-VTT]=6
  [N-Caltech-101]=70
  [N-ImageNet-1K]=70
  [FSD-50K]=90
  [ESC-50]=90
  [UrbanSound8K]=2
  [LLVIP]=280
  [RGB-T]=16
  [ModelNet40]=64
  [ShapeNet]=64
)

declare -A ROBUST_TRAIN_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=18
  [Places365]=0
  [Kinetics-400]=241258
  [UCF-101]=9537
  [MSR-VTT]=2990
  [N-Caltech-101]=3060
  [N-ImageNet-1K]=1281167
  [FSD-50K]=36796
  [ESC-50]=1600
  [UrbanSound8K]=7079
  [LLVIP]=67900
  [RGB-T]=800
  [ModelNet40]=9843
  [ShapeNet]=20480
)

declare -A ROBUST_VAL_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=6
  [Places365]=3000
  [MSR-VTT]=2990
  [N-Caltech-101]=3000
  [N-ImageNet-1K]=3000
  [ESC-50]=400
  [UrbanSound8K]=1653
  [LLVIP]=16974
  [RGB-T]=500
  [ModelNet40]=2468
  [ShapeNet]=2048
)

declare -A ROBUST_TRAIN_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/train_data.json"
  [Places365]="./datasets/Places365/train_data.json"
  [Kinetics-400]="./datasets/Kinetics-400/train_data.json"
  [UCF-101]="./datasets/UCF-101/train_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/train_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/train_data.json"
  [N-ImageNet-1K]="./datasets/N-ImageNet-1K/train_data.json"
  [FSD-50K]="./datasets/FSD-50K/train_data.json"
  [ESC-50]="./datasets/ESC-50/train_data.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/train_data.json"
  [LLVIP]="./datasets/LLVIP/train_data.json"
  [RGB-T]="./datasets/RGB-T/train_data.json"
  [ModelNet40]="./datasets/ModelNet40/train_data.json"
  [ShapeNet]="./datasets/ShapeNet/train_data.json"
)

declare -A ROBUST_VAL_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/val_data_3000.json"
  [Places365]="./datasets/Places365/val_data.json"
  [UCF-101]="./datasets/UCF-101/val_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/val_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/val_data.json"
  [N-ImageNet-1K]="./datasets/N-ImageNet-1K/val_data.json"
  [ESC-50]="./datasets/ESC-50/val_data.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/val_data.json"
  [LLVIP]="./datasets/LLVIP/val_data.json"
  [RGB-T]="./datasets/RGB-T/val_data.json"
  [ModelNet40]="./datasets/ModelNet40/val_data.json"
  [ShapeNet]="./datasets/ShapeNet/val_data.json"
)

declare -A ROBUST_EMB_SUFFIX_MAP=(
  [ImageNet-1K]=in
  [Places365]=p365
  [UCF-101]=ucf
  [MSR-VTT]=msrvtt
  [N-Caltech-101]=caltech
  [N-ImageNet-1K]=nin
  [ESC-50]=esc
  [UrbanSound8K]=us
  [LLVIP]=llvip
  [RGB-T]=rgbt
  [ModelNet40]=modelnet40
  [ShapeNet]=shapenet
)

# # Update loop logic to use --robust_training_mode
# for model_type in "${!MODEL_TYPE_TO_MODALITIES[@]}"; do
#   for modality in ${MODEL_TYPE_TO_MODALITIES[$model_type]}; do
#     train_dataset="${ROBUST_TRAIN_MODALITY_TO_DATASET[$modality]}"; [[ -z "$train_dataset" ]] && continue
#     val_dataset="${ROBUST_VAL_MODALITY_TO_DATASET[$modality]}"; [[ -z "$val_dataset" ]] && continue

#     train_bs="${ROBUST_DATASET_TO_BATCH_SIZE[$train_dataset]}"
#     val_bs="${ROBUST_DATASET_TO_BATCH_SIZE[$val_dataset]}"
#     train_max="${ROBUST_TRAIN_MAX_SAMPLES_MAP[$train_dataset]}"
#     val_max="${ROBUST_VAL_MAX_SAMPLES_MAP[$val_dataset]}"
#     train_json="${ROBUST_TRAIN_JSON_MAP[$train_dataset]}"
#     val_json="${ROBUST_VAL_JSON_MAP[$val_dataset]}"
#     emb_suffix="${ROBUST_EMB_SUFFIX_MAP[$val_dataset]}"
#     emb_path="./centre_embs/${modality}_${emb_suffix}_center_embeddings.pkl"

#     for eps in "${ROBUST_EPSILONS[@]}"; do
#       for mode in "${ROBUST_MODES[@]}"; do
#         echo "=== (ROBUST) $model_type | $train_dataset => $val_dataset | modality=$modality | eps=$eps | mode=$mode ==="
#         torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
#           --model_type "$model_type" \
#           --train_modality "$modality" \
#           --val_modality "$modality" \
#           --train_dataset_name "$train_dataset" \
#           --val_dataset_name "$val_dataset" \
#           --train_dataset_root "/home/user/datasets/$train_dataset" \
#           --val_dataset_root "/home/user/datasets/$val_dataset" \
#           --train_json "$train_json" \
#           --val_json "$val_json" \
#           --pretrain_weights "$PRETRAIN_WEIGHTS" \
#           --center_emb "$emb_path" \
#           --train_batch_size "$train_bs" \
#           --val_batch_size "$val_bs" \
#           --num_workers "$NUM_WORKERS" \
#           --train_max_samples "$train_max" \
#           --val_max_samples "$val_max" \
#           --robust_train_attack_loss "l2" \
#           --robust_val_attack_loss "ce" \
#           --robust_train_loss "l2" \
#           --robust_lora_rank "$ROBUST_LORA_RANKS" \
#           --robust_lora_alpha "$ROBUST_LORA_ALPHAS" \
#           --robust_epsilon "$eps" \
#           --robust_training_mode "$mode" \
#           --epochs "$ROBUST_EPOCHS" \
#           --use_flash_attention \
#           --tensorboard_data_dir "$TENSORBOARD_DATA_DIR" \
#           --output_dir "$OUTPUT_DIR"
#       done
#     done
#   done
# done

# Alignment config
DO_ALIGNMENT=1
ALIGN_EPOCHS=1

declare -A ALIGN_TRAIN_MODALITY_TO_DATASET=(
  [image]="ImageNet-1K"
  [video]="MSR-VTT"
  [audio]="ESC-50"
  [thermal]="LLVIP"
  [event]="N-Caltech-101"
)

declare -A ALIGN_VAL_MODALITY_TO_DATASET=(
  [image]="ImageNet-1K"
  [video]="MSR-VTT"
  [audio]="ESC-50"
  [thermal]="LLVIP"
  [event]="N-Caltech-101"
)

declare -A ALIGN_DATASET_TO_BATCH_SIZE=(
  [ImageNet-1K]=2000
  [MSR-VTT]=200
  [ESC-50]=500
  [LLVIP]=2000
  [N-Caltech-101]=500
)

declare -A ALIGN_TRAIN_MAX_SAMPLES_MAP=(
  # [ImageNet-1K]=1281167
  # [MSR-VTT]=7010
  # [ESC-50]=1600
  # [LLVIP]=67900
  # [N-Caltech-101]=6139
  [ImageNet-1K]=4
  [MSR-VTT]=4
  [ESC-50]=4
  [LLVIP]=4
  [N-Caltech-101]=4
)

declare -A ALIGN_VAL_MAX_SAMPLES_MAP=(
  # [ImageNet-1K]=50000
  # [MSR-VTT]=2990
  # [ESC-50]=400
  # [LLVIP]=16974
  # [N-Caltech-101]=2570
  [ImageNet-1K]=4
  [MSR-VTT]=4
  [ESC-50]=4
  [LLVIP]=4
  [N-Caltech-101]=4
)

declare -A ALIGN_TRAIN_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/train_data_align.json"
  [MSR-VTT]="./datasets/MSR-VTT/train_data_align.json"
  [ESC-50]="./datasets/ESC-50/train_data_align.json"
  [LLVIP]="./datasets/LLVIP/train_data_align.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/train_data_align.json"
)

declare -A ALIGN_VAL_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/val_data.json"
  [MSR-VTT]="./datasets/MSR-VTT/val_data.json"
  [ESC-50]="./datasets/ESC-50/val_data.json"
  [LLVIP]="./datasets/LLVIP/val_data.json"
  [N-Caltech-101]="./datasets/N-Caltech-101/val_data.json"
)

declare -A ALIGN_EMB_SUFFIX_MAP=(
  [ImageNet-1K]=in
  [MSR-VTT]=msrvtt
  [ESC-50]=esc
  [LLVIP]=llvip
  [N-Caltech-101]=caltech
)

if [[ "$DO_ALIGNMENT" -eq 1 ]]; then
  echo "=== Alignment runs (epochs=$ALIGN_EPOCHS) ==="

  for model_type in "${!MODEL_TYPE_TO_MODALITIES[@]}"; do
    for modality in ${MODEL_TYPE_TO_MODALITIES[$model_type]}; do
      train_dataset="${ALIGN_TRAIN_MODALITY_TO_DATASET[$modality]}"
      val_dataset="${ALIGN_VAL_MODALITY_TO_DATASET[$modality]}"
      [[ -z "$train_dataset" || -z "$val_dataset" ]] && continue

      train_bs="${ALIGN_DATASET_TO_BATCH_SIZE[$train_dataset]}"
      val_bs="${ALIGN_DATASET_TO_BATCH_SIZE[$val_dataset]}"
      train_max="${ALIGN_TRAIN_MAX_SAMPLES_MAP[$train_dataset]}"
      val_max="${ALIGN_VAL_MAX_SAMPLES_MAP[$val_dataset]}"
      train_json="${ALIGN_TRAIN_JSON_MAP[$train_dataset]}"
      val_json="${ALIGN_VAL_JSON_MAP[$val_dataset]}"
      emb_suffix="${ALIGN_EMB_SUFFIX_MAP[$val_dataset]}"
      emb_path="./centre_embs/${modality}_${emb_suffix}_center_embeddings.pkl"
      [[ -z "$train_bs" || -z "$val_bs" || -z "$train_max" || -z "$val_max" || -z "$train_json" || -z "$val_json" || -z "$emb_suffix" ]] && continue

      echo "[ALIGN] $model_type | $train_dataset ($modality) => $val_dataset"
      torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
        --training_mode alignment \
        --model_type "$model_type" \
        --train_modality "$modality" \
        --val_modality "$modality" \
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
        --epochs "$ALIGN_EPOCHS" \
        --use_flash_attention \
        --tensorboard_data_dir "$TENSORBOARD_DATA_DIR" \
        --output_dir "$OUTPUT_DIR"
    done
  done
fi