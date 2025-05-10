#!/bin/bash
set -e

# === Modality Mapping ===
declare -A MODALITY_MAP=(
  [ImageNet-1K]=image
  # [Places365]=image

  # [ModelNet40]=point
  # [ShapeNet]=point

  # [ESC-50]=audio
  # [UrbanSound8K]=audio

  # [LLVIP]=thermal
  # [RGB-T]=thermal

  # [MSR-VTT]=video
  # [UCF-101]=video

  # [N-Caltech-101]=event
  # [N-ImageNet-1K]=event
)

# === Embedding Suffix Mapping ===
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

# === Clean Validation JSON Mapping ===
declare -A CLEAN_VAL_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/val_data.json"
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

# === Attack Validation JSON Mapping ===
declare -A ATTACK_VAL_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/val_data.json"
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

# === Clean Validation Batch Size Mapping ===
declare -A CLEAN_VAL_BATCH_SIZE_MAP=(
  [ImageNet-1K]=1000
  [Places365]=1000
  [ModelNet40]=64
  [ShapeNet]=64
  [ESC-50]=90
  [UrbanSound8K]=90
  [LLVIP]=1000
  [RGB-T]=16
  [MSR-VTT]=30
  [UCF-101]=30
  [N-Caltech-101]=1000
  [N-ImageNet-1K]=1000
)

# === Attack Validation Batch Size Mapping ===
declare -A ATTACK_VAL_BATCH_SIZE_MAP=(
  [ImageNet-1K]=70
  [Places365]=70
  [ModelNet40]=64
  [ShapeNet]=64
  [ESC-50]=90
  [UrbanSound8K]=90
  [LLVIP]=70
  [RGB-T]=16
  [MSR-VTT]=6
  [UCF-101]=6
  [N-Caltech-101]=70
  [N-ImageNet-1K]=70
)

# === Clean Validation Max Samples Mapping ===
declare -A CLEAN_VAL_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=50000
  [Places365]=36500
  [ModelNet40]=2
  [ShapeNet]=2048
  [ESC-50]=400
  [UrbanSound8K]=1653
  [LLVIP]=16974
  [RGB-T]=500
  [MSR-VTT]=2990
  [UCF-101]=3783
  [N-Caltech-101]=2613
  [N-ImageNet-1K]=3000
)

# === Attack Validation Max Samples Mapping ===
declare -A ATTACK_VAL_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=50000
  [Places365]=36500
  [ModelNet40]=2
  [ShapeNet]=2048
  [ESC-50]=400
  [UrbanSound8K]=1653
  [LLVIP]=16974
  [RGB-T]=500
  [MSR-VTT]=2990
  [UCF-101]=3783
  [N-Caltech-101]=2613
  [N-ImageNet-1K]=3000
)

# === LoRA Weights List Mapping by Modality ===
declare -A LORA_WEIGHTS_LIST_MAP=(
  [image]="./ckpts/vision_eps2_lora_weights.pt ./ckpts/vision_eps4_lora_weights.pt"
  [audio]="./ckpts/audio_eps2_lora_weights.pt ./ckpts/audio_eps4_lora_weights.pt"
  [thermal]="./ckpts/thermal_eps2_lora_weights.pt ./ckpts/thermal_eps4_lora_weights.pt"
  [point]="./ckpts/point_eps2_lora_weights.pt ./ckpts/point_eps4_lora_weights.pt"
  [video]="./ckpts/vision_eps2_lora_weights.pt ./ckpts/vision_eps4_lora_weights.pt"
  [event]="./ckpts/vision_eps2_lora_weights.pt ./ckpts/vision_eps4_lora_weights.pt"
)

# === Common Config ===
OUTPUT_DIR=output
NUM_WORKERS=2
EPSILONS="2,4"
TWO_STAGE_ITERS=100
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten.pt"

# === Eval Loop ===
for dataset in "${!MODALITY_MAP[@]}"; do
  modality="${MODALITY_MAP[$dataset]}"
  suffix="${EMB_SUFFIX_MAP[$dataset]}"
  clean_val_json="${CLEAN_VAL_JSON_MAP[$dataset]}"
  attack_val_json="${ATTACK_VAL_JSON_MAP[$dataset]}"
  clean_val_batch_size="${CLEAN_VAL_BATCH_SIZE_MAP[$dataset]}"
  attack_val_batch_size="${ATTACK_VAL_BATCH_SIZE_MAP[$dataset]}"
  clean_val_max_samples="${CLEAN_VAL_MAX_SAMPLES_MAP[$dataset]}"
  attack_val_max_samples="${ATTACK_VAL_MAX_SAMPLES_MAP[$dataset]}"
  lora_weights_list="${LORA_WEIGHTS_LIST_MAP[$modality]}"

  echo "=== Evaluating: $modality / $dataset (Clean BS: $clean_val_batch_size, Attack BS: $attack_val_batch_size) ==="

  torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_unibind.py \
    --modality "$modality" \
    --dataset_name "$dataset" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_root "/home/user/datasets/$dataset" \
    --clean_val_json "$clean_val_json" \
    --attack_val_json "$attack_val_json" \
    --clean_val_batch_size "$clean_val_batch_size" \
    --attack_val_batch_size "$attack_val_batch_size" \
    --clean_val_max_samples "$clean_val_max_samples" \
    --attack_val_max_samples "$attack_val_max_samples" \
    --pretrain_weights "$PRETRAIN_WEIGHTS" \
    --center_emb "./centre_embs/${modality}_${suffix}_center_embeddings.pkl" \
    --lora_weights_list $lora_weights_list \
    --num_workers "$NUM_WORKERS" \
    --use_flash_attention \
    --val_attack_loss "ce" \
    --epsilons "$EPSILONS" \
    --run_clean_eval \
    --two_stage_iters "$TWO_STAGE_ITERS"
done
