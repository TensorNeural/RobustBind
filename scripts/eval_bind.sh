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

# === Modality to Supported Binds ===
declare -A MODALITY_TO_BINDS=(
  [image]="LanguageBind ImageBind"
  # [image]="UniBind"
  # [audio]="LanguageBind ImageBind"
  # [audio]="UniBind"
  # [video]="UniBind LanguageBind ImageBind"
  # [video]="LanguageBind ImageBind"
  # [video]="UniBind"
  # [depth]="LanguageBind ImageBind"
  # [imu]="LanguageBind ImageBind"
  # [thermal]="LanguageBind ImageBind"
  # [thermal]="UniBind"
  # [point]="LanguageBind"
  # [event]="UniBind"
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

# === Validation JSON Mapping ===
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

declare -A CLASSES_JSON_MAP=(
  [ImageNet-1K]="./datasets/ImageNet-1K/classes_imagenet.json"
  [Places365]="./datasets/Places365/classes_places365.json"
  [LLVIP]="./datasets/LLVIP/classes_llvip.json"
  [ModelNet40]="./datasets/ModelNet40/classes_modelnet40.json"
  [ESC-50]="./datasets/ESC-50/classes.json"
  [UrbanSound8K]="./datasets/UrbanSound8K/classes.json"
  [UCF-101]="./datasets/UCF-101/classes.json"
  [MSR-VTT]="./datasets/MSR-VTT/classes.json"
)

declare -A ATTACK_VAL_JSON_MAP

for key in "${!CLEAN_VAL_JSON_MAP[@]}"; do
  ATTACK_VAL_JSON_MAP[$key]="${CLEAN_VAL_JSON_MAP[$key]}"
done

# === Batch Size Mapping ===
declare -A CLEAN_VAL_BATCH_SIZE_MAP=(
  [ImageNet-1K]=2000
  [Places365]=2000
  [ModelNet40]=64
  [ShapeNet]=64
  [ESC-50]=50
  [UrbanSound8K]=50
  [LLVIP]=2000
  [RGB-T]=16
  [MSR-VTT]=100
  [UCF-101]=100
  [N-Caltech-101]=500
  [N-ImageNet-1K]=500
)

declare -A ATTACK_VAL_BATCH_SIZE_MAP=(
  [ImageNet-1K]=70
  [Places365]=70
  [ModelNet40]=64
  [ShapeNet]=64
  [ESC-50]=50
  [UrbanSound8K]=50
  [LLVIP]=70
  [RGB-T]=16
  [MSR-VTT]=15
  [UCF-101]=15
  [N-Caltech-101]=70
  [N-ImageNet-1K]=70
)

# === Max Sample Mapping ===
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
  [N-ImageNet-1K]=50000
)

declare -A ATTACK_VAL_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=5000
  [Places365]=36500
  [ModelNet40]=2
  [ShapeNet]=2048
  [ESC-50]=400
  [UrbanSound8K]=1653
  # [LLVIP]=16974
  [LLVIP]=5000
  [RGB-T]=500
  # [MSR-VTT]=2990
  # [UCF-101]=3783
  [MSR-VTT]=800
  [UCF-101]=800
  [N-Caltech-101]=2613
  [N-ImageNet-1K]=50000
)

# === Config ===
OUTPUT_DIR=/data/output
# Session timestamp used to group all eval outputs under a single run directory
SESSION_TS=$(date +%F_%H-%M-%S)
NUM_WORKERS=2
# EPSILONS="2,4"
EPSILONS=""
TWO_STAGE_ITERS=100
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten_image_patchs.pt"
LORA_RANK=4
LORA_ALPHA=8.0

# === Eval Loop ===
for dataset in "${!MODALITY_MAP[@]}"; do
  modality="${MODALITY_MAP[$dataset]}"
  supported_binds="${MODALITY_TO_BINDS[$modality]}"

  suffix="${EMB_SUFFIX_MAP[$dataset]}"
  clean_val_json="${CLEAN_VAL_JSON_MAP[$dataset]}"
  attack_val_json="${ATTACK_VAL_JSON_MAP[$dataset]}"
  clean_val_batch_size="${CLEAN_VAL_BATCH_SIZE_MAP[$dataset]}"
  attack_val_batch_size="${ATTACK_VAL_BATCH_SIZE_MAP[$dataset]}"
  clean_val_max_samples="${CLEAN_VAL_MAX_SAMPLES_MAP[$dataset]}"
  attack_val_max_samples="${ATTACK_VAL_MAX_SAMPLES_MAP[$dataset]}"
  center_emb="./centre_embs/${modality}_${suffix}_center_embeddings.pkl"
  dataset_root="/data/datasets/$dataset"
  classes_json="${CLASSES_JSON_MAP[$dataset]}"

  for model_type in UniBind LanguageBind ImageBind; do
    if [[ ! " $supported_binds " =~ " $model_type " ]]; then
      echo "Skipping $model_type on $dataset (modality=$modality) — not supported."
      continue
    fi

    echo "=== Evaluating: $model_type / $modality / $dataset ==="

    # Non-UniBind models: run a single evaluation (clean + eps2/eps4 attacks) without alignment
    if [[ "$model_type" != "UniBind" ]]; then
      echo "Running clean + eps2/eps4 attacks for $model_type (no alignment) ..."
      torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_bind.py \
        --model_type "$model_type" \
        --modality "$modality" \
        --dataset_name "$dataset" \
        --output_dir "$OUTPUT_DIR" \
        --session_timestamp "$SESSION_TS" \
        --dataset_root "$dataset_root" \
        --clean_val_json "$clean_val_json" \
        --attack_val_json "$attack_val_json" \
        --clean_val_batch_size "$clean_val_batch_size" \
        --attack_val_batch_size "$attack_val_batch_size" \
        --clean_val_max_samples "$clean_val_max_samples" \
        --attack_val_max_samples "$attack_val_max_samples" \
        --classes_json "$classes_json" \
        --pretrain_weights "$PRETRAIN_WEIGHTS" \
        --center_emb "$center_emb" \
        --num_workers "$NUM_WORKERS" \
        --use_flash_attention \
        --val_attack_loss "ce" \
        --epsilons "$EPSILONS" \
        --run_clean_eval \
        --two_stage_iters "$TWO_STAGE_ITERS"
      continue
    fi

    # LoRA weights generated by train_robust_unibind.py (best copies under ./ckpts)
    # Filename pattern: robust_<modality>_lora_r${LORA_RANK}a${LORA_ALPHA}_eps{2|4}.pt
    LORA_EPS2="./ckpts/robust_${modality}_lora_r${LORA_RANK}a${LORA_ALPHA}_eps2.pt"
    LORA_EPS4="./ckpts/robust_${modality}_lora_r${LORA_RANK}a${LORA_ALPHA}_eps4.pt"

    # Alignment MLP weights (best-effort; eval code will warn if missing)
    ALIGN_MLP_WEIGHTS="./ckpts/align_${modality}.pt"

    # 1) UniBind attack without align
    torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_bind.py \
      --model_type "$model_type" \
      --modality "$modality" \
      --dataset_name "$dataset" \
      --output_dir "$OUTPUT_DIR" \
      --session_timestamp "$SESSION_TS" \
      --dataset_root "$dataset_root" \
      --clean_val_json "$clean_val_json" \
      --attack_val_json "$attack_val_json" \
      --clean_val_batch_size "$clean_val_batch_size" \
      --attack_val_batch_size "$attack_val_batch_size" \
      --clean_val_max_samples "$clean_val_max_samples" \
      --attack_val_max_samples "$attack_val_max_samples" \
      --classes_json "$classes_json" \
      --pretrain_weights "$PRETRAIN_WEIGHTS" \
      --center_emb "$center_emb" \
      --num_workers "$NUM_WORKERS" \
      --use_flash_attention \
      --val_attack_loss "ce" \
      --epsilons "$EPSILONS" \
      --run_clean_eval \
      --two_stage_iters "$TWO_STAGE_ITERS"

    # # 2) UniBind attack with align
    # torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_bind.py \
    #   --model_type "$model_type" \
    #   --modality "$modality" \
    #   --dataset_name "$dataset" \
    #   --output_dir "$OUTPUT_DIR" \
    #   --session_timestamp "$SESSION_TS" \
    #   --dataset_root "$dataset_root" \
    #   --clean_val_json "$clean_val_json" \
    #   --attack_val_json "$attack_val_json" \
    #   --clean_val_batch_size "$clean_val_batch_size" \
    #   --attack_val_batch_size "$attack_val_batch_size" \
    #   --clean_val_max_samples "$clean_val_max_samples" \
    #   --attack_val_max_samples "$attack_val_max_samples" \
    #   --classes_json "$classes_json" \
    #   --pretrain_weights "$PRETRAIN_WEIGHTS" \
    #   --center_emb "$center_emb" \
    #   --num_workers "$NUM_WORKERS" \
    #   --use_flash_attention \
    #   --val_attack_loss "ce" \
    #   --epsilons "$EPSILONS" \
    #   --run_clean_eval \
    #   --two_stage_iters "$TWO_STAGE_ITERS" \
    #   --use_modality_head_mlp \
    #   --modality_head_mlp_weights "$ALIGN_MLP_WEIGHTS"

    # # 3) robustbind^2 attack without align
    # torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_bind.py \
    #   --model_type "$model_type" \
    #   --modality "$modality" \
    #   --dataset_name "$dataset" \
    #   --output_dir "$OUTPUT_DIR" \
    #   --session_timestamp "$SESSION_TS" \
    #   --dataset_root "$dataset_root" \
    #   --clean_val_json "$clean_val_json" \
    #   --attack_val_json "$attack_val_json" \
    #   --clean_val_batch_size "$clean_val_batch_size" \
    #   --attack_val_batch_size "$attack_val_batch_size" \
    #   --clean_val_max_samples "$clean_val_max_samples" \
    #   --attack_val_max_samples "$attack_val_max_samples" \
    #   --classes_json "$classes_json" \
    #   --pretrain_weights "$PRETRAIN_WEIGHTS" \
    #   --center_emb "$center_emb" \
    #   --num_workers "$NUM_WORKERS" \
    #   --use_flash_attention \
    #   --val_attack_loss "ce" \
    #   --epsilons "$EPSILONS" \
    #   --two_stage_iters "$TWO_STAGE_ITERS" \
    #   --skip_original \
    #   --lora_weights_list "$LORA_EPS2"

    # # 4) robustbind^2 attack with align
    # torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_bind.py \
    #   --model_type "$model_type" \
    #   --modality "$modality" \
    #   --dataset_name "$dataset" \
    #   --output_dir "$OUTPUT_DIR" \
    #   --session_timestamp "$SESSION_TS" \
    #   --dataset_root "$dataset_root" \
    #   --clean_val_json "$clean_val_json" \
    #   --attack_val_json "$attack_val_json" \
    #   --clean_val_batch_size "$clean_val_batch_size" \
    #   --attack_val_batch_size "$attack_val_batch_size" \
    #   --clean_val_max_samples "$clean_val_max_samples" \
    #   --attack_val_max_samples "$attack_val_max_samples" \
    #   --classes_json "$classes_json" \
    #   --pretrain_weights "$PRETRAIN_WEIGHTS" \
    #   --center_emb "$center_emb" \
    #   --num_workers "$NUM_WORKERS" \
    #   --use_flash_attention \
    #   --val_attack_loss "ce" \
    #   --epsilons "$EPSILONS" \
    #   --two_stage_iters "$TWO_STAGE_ITERS" \
    #   --skip_original \
    #   --use_modality_head_mlp \
    #   --modality_head_mlp_weights "$ALIGN_MLP_WEIGHTS" \
    #   --lora_weights_list "$LORA_EPS2"

    # # 5) robustbind^4 attack without align
    # torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_bind.py \
    #   --model_type "$model_type" \
    #   --modality "$modality" \
    #   --dataset_name "$dataset" \
    #   --output_dir "$OUTPUT_DIR" \
    #   --session_timestamp "$SESSION_TS" \
    #   --dataset_root "$dataset_root" \
    #   --clean_val_json "$clean_val_json" \
    #   --attack_val_json "$attack_val_json" \
    #   --clean_val_batch_size "$clean_val_batch_size" \
    #   --attack_val_batch_size "$attack_val_batch_size" \
    #   --clean_val_max_samples "$clean_val_max_samples" \
    #   --attack_val_max_samples "$attack_val_max_samples" \
    #   --classes_json "$classes_json" \
    #   --pretrain_weights "$PRETRAIN_WEIGHTS" \
    #   --center_emb "$center_emb" \
    #   --num_workers "$NUM_WORKERS" \
    #   --use_flash_attention \
    #   --val_attack_loss "ce" \
    #   --epsilons "$EPSILONS" \
    #   --two_stage_iters "$TWO_STAGE_ITERS" \
    #   --skip_original \
    #   --lora_weights_list "$LORA_EPS4"

    # # 6) robustbind^4 attack with align
    # torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) eval_bind.py \
    #   --model_type "$model_type" \
    #   --modality "$modality" \
    #   --dataset_name "$dataset" \
    #   --output_dir "$OUTPUT_DIR" \
    #   --dataset_root "$dataset_root" \
    #   --clean_val_json "$clean_val_json" \
    #   --attack_val_json "$attack_val_json" \
    #   --clean_val_batch_size "$clean_val_batch_size" \
    #   --attack_val_batch_size "$attack_val_batch_size" \
    #   --clean_val_max_samples "$clean_val_max_samples" \
    #   --attack_val_max_samples "$attack_val_max_samples" \
    #   --classes_json "$classes_json" \
    #   --pretrain_weights "$PRETRAIN_WEIGHTS" \
    #   --center_emb "$center_emb" \
    #   --num_workers "$NUM_WORKERS" \
    #   --use_flash_attention \
    #   --val_attack_loss "ce" \
    #   --epsilons "$EPSILONS" \
    #   --two_stage_iters "$TWO_STAGE_ITERS" \
    #   --skip_original \
    #   --use_modality_head_mlp \
    #   --modality_head_mlp_weights "$ALIGN_MLP_WEIGHTS" \
    #   --lora_weights_list "$LORA_EPS4"
  done
done
