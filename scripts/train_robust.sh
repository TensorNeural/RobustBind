#!/bin/bash
set -e

# Shared constants
SESSION_TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
SESSION_OUTPUT_DIR="output/train/${SESSION_TIMESTAMP}"
TENSORBOARD_ROOT="output/tensorboard"
OUTPUT_DIR="$SESSION_OUTPUT_DIR"
NUM_WORKERS=4
TENSORBOARD_DATA_DIR=tensorboard
PRETRAIN_WEIGHTS="./ckpts/pretrained_weights_flash_atten.pt"

mkdir -p "$SESSION_OUTPUT_DIR"
mkdir -p "$TENSORBOARD_ROOT"

DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"

# Helper: format seconds to HH:MM:SS
fmt_duration() {
  local total=$1
  local h=$(( total / 3600 ))
  local m=$(( (total % 3600) / 60 ))
  local s=$(( total % 60 ))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

post_discord() {
  local message="$1"
  [[ -z "$DISCORD_WEBHOOK_URL" ]] && return 0
  local attempt=0
  local backoff=1
  while true; do
    attempt=$((attempt + 1))
    # Render escape sequences (e.g., \n) to real characters
    local rendered
    rendered=$(printf '%b' "$message")
    # Use form-encoded content to avoid JSON escaping issues; bypass proxies
    http_code=$(curl -sS -o /dev/null -w "%{http_code}" --noproxy '*' \
      -H 'User-Agent: RobustBind/1.0' \
      --data-urlencode "content=${rendered}" \
      -X POST "$DISCORD_WEBHOOK_URL" || echo "000")
    case "$http_code" in
      2*) return 0 ;;
      *)
  echo "[WARN] Discord notify failed (attempt ${attempt}, code=$http_code), retrying in ${backoff}s…" >&2
        sleep "$backoff"
        backoff=$(( backoff * 2 ))
        ;;
    esac
  done
}

# --- Session-level notification: start ---
SESSION_START_EPOCH=$(date +%s)
post_discord "🚀 Session started\n• **Session:** \`${SESSION_TIMESTAMP}\`\n• **Output:** \`${SESSION_OUTPUT_DIR}\`"

declare -A MODEL_TYPE_TO_MODALITIES=(
  [vision]="image video event"
  [audio]="audio"
  [thermal]="thermal"
  [point]="point"
)

# Alignment config
ALIGN_EPOCHS=50  # default epochs if not overridden per-dataset

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
  [ImageNet-1K]=1281167
  [MSR-VTT]=7010
  [ESC-50]=1600
  [LLVIP]=67900
  [N-Caltech-101]=6139
)

declare -A ALIGN_VAL_MAX_SAMPLES_MAP=(
  [ImageNet-1K]=50000
  [MSR-VTT]=2990
  [ESC-50]=400
  [LLVIP]=16974
  [N-Caltech-101]=2570
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

declare -A ALIGN_DATASET_TO_EPOCHS=(
  [ImageNet-1K]=6
  [MSR-VTT]=30
  [ESC-50]=100
  [LLVIP]=8
  [N-Caltech-101]=25
)

declare -A ALIGN_EMB_SUFFIX_MAP=(
  [ImageNet-1K]=in
  [MSR-VTT]=msrvtt
  [ESC-50]=esc
  [LLVIP]=llvip
  [N-Caltech-101]=caltech
)

# --- Alignment notifications: start ---
ALIGN_START_EPOCH=$(date +%s)
post_discord "🚦 Alignment started\n• **Session:** \`${SESSION_TIMESTAMP}\`"

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

  # Per-dataset epochs with fallback to default
  dataset_epochs="${ALIGN_DATASET_TO_EPOCHS[$train_dataset]}"
  if [[ -z "$dataset_epochs" ]]; then dataset_epochs="$ALIGN_EPOCHS"; fi
    torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
      --training_mode alignment \
      --model_type "$model_type" \
      --train_modality "$modality" \
      --val_modality "$modality" \
      --train_dataset_name "$train_dataset" \
      --val_dataset_name "$val_dataset" \
      --train_dataset_root "/data/datasets/$train_dataset" \
      --val_dataset_root "/data/datasets/$val_dataset" \
      --train_json "$train_json" \
      --val_json "$val_json" \
      --pretrain_weights "$PRETRAIN_WEIGHTS" \
      --val_center_emb "$emb_path" \
      --train_batch_size "$train_bs" \
      --val_batch_size "$val_bs" \
      --num_workers "$NUM_WORKERS" \
      --train_max_samples "$train_max" \
      --val_max_samples "$val_max" \
  --epochs "$dataset_epochs" \
      --use_flash_attention \
      --tensorboard_data_dir "$TENSORBOARD_DATA_DIR" \
      --output_dir "$SESSION_OUTPUT_DIR" \
      --session_output_dir "$SESSION_OUTPUT_DIR" \
      --session_timestamp "$SESSION_TIMESTAMP" \
      --tensorboard_root "$TENSORBOARD_ROOT"
  done
done

# --- Alignment notifications: end ---
ALIGN_END_EPOCH=$(date +%s)
ALIGN_ELAPSED=$(( ALIGN_END_EPOCH - ALIGN_START_EPOCH ))
ALIGN_DUR=$(fmt_duration "$ALIGN_ELAPSED")
post_discord "🏁 Alignment finished\n• **Session:** \`${SESSION_TIMESTAMP}\`\n• **Duration:** \`${ALIGN_DUR}\`"

# Robust config
ROBUST_EPSILONS=(2 4)
ROBUST_LORA_RANKS=(
  4 
  # 4 
  # 8
)
ROBUST_LORA_ALPHAS=(
  8 
  # 8 
  # 16
)
ROBUST_EPOCHS=2
ROBUST_MODES=(lora full_fine_tune)

declare -A ROBUST_TRAIN_MODALITY_TO_DATASET=(
  [image]="ImageNet-1K"
  # [image]="Places365"
  [video]="Kinetics-400"
  # [video]="UCF-101"
  # [video]="MSR-VTT"
  [event]="N-Caltech-101"
  # [event]="N-ImageNet-1K"
  [audio]="FSD-50K"
  # [audio]="ESC-50"
  # [audio]="UrbanSound8K"
  [thermal]="LLVIP"
  # [thermal]="RGB-T"
  # [point]="ModelNet40"
  # [point]="ShapeNet"
)

declare -A ROBUST_VAL_MODALITY_TO_DATASET=(
  [image]="ImageNet-1K"
  # [image]="Places365"
  # [video]="UCF-101"
  [video]="MSR-VTT"
  [event]="N-Caltech-101"
  [audio]="ESC-50"
  [thermal]="LLVIP"
)

declare -A ROBUST_DATASET_TO_BATCH_SIZE=(
  [ImageNet-1K]=70
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
  [ImageNet-1K]=1281167
  [Places365]=0
  [Kinetics-400]=239783
  [UCF-101]=9537
  [MSR-VTT]=7010
  [N-Caltech-101]=6139
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
  [ImageNet-1K]=3000
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

declare -A ROBUST_TRAIN_EMB_SUFFIX_MAP=(
  [ImageNet-1K]=in
  [Places365]=p365
  # [Kinetics-400]=kin400
  [UCF-101]=ucf
  [MSR-VTT]=msrvtt
  [N-Caltech-101]=caltech
  [N-ImageNet-1K]=nin
  # [FSD-50K]=fsd50k
  [ESC-50]=esc
  [UrbanSound8K]=us
  [LLVIP]=llvip
  [RGB-T]=rgbt
  [ModelNet40]=modelnet40
  [ShapeNet]=shapenet
)

declare -A ROBUST_VAL_EMB_SUFFIX_MAP=(
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

# Robust runs
# --- Robust notifications: start ---
ROBUST_START_EPOCH=$(date +%s)
post_discord "🚦 Robust started\n• **Session:** \`${SESSION_TIMESTAMP}\`"

for model_type in "${!MODEL_TYPE_TO_MODALITIES[@]}"; do
  for modality in ${MODEL_TYPE_TO_MODALITIES[$model_type]}; do
    train_dataset="${ROBUST_TRAIN_MODALITY_TO_DATASET[$modality]}"; [[ -z "$train_dataset" ]] && continue
    val_dataset="${ROBUST_VAL_MODALITY_TO_DATASET[$modality]}"; [[ -z "$val_dataset" ]] && continue

    train_bs="${ROBUST_DATASET_TO_BATCH_SIZE[$train_dataset]}"
    val_bs="${ROBUST_DATASET_TO_BATCH_SIZE[$val_dataset]}"
    train_max="${ROBUST_TRAIN_MAX_SAMPLES_MAP[$train_dataset]}"
    val_max="${ROBUST_VAL_MAX_SAMPLES_MAP[$val_dataset]}"
    train_json="${ROBUST_TRAIN_JSON_MAP[$train_dataset]}"
    val_json="${ROBUST_VAL_JSON_MAP[$val_dataset]}"

    train_emb_suffix="${ROBUST_TRAIN_EMB_SUFFIX_MAP[$train_dataset]}"
    val_emb_suffix="${ROBUST_VAL_EMB_SUFFIX_MAP[$val_dataset]}"

    if [[ -n "$train_emb_suffix" ]]; then
      train_emb_path="./centre_embs/${modality}_${train_emb_suffix}_center_embeddings.pkl"
      TRAIN_CENTER_ARG=(--train_center_emb "$train_emb_path")
    else
      TRAIN_CENTER_ARG=()
    fi

    if [[ -n "$val_emb_suffix" ]]; then
      val_emb_path="./centre_embs/${modality}_${val_emb_suffix}_center_embeddings.pkl"
      VAL_CENTER_ARG=(--val_center_emb "$val_emb_path")
    else
      echo "[WARN] No val emb suffix for $val_dataset ($modality); skipping run (val centers required)." >&2
      continue
    fi

    for eps in "${ROBUST_EPSILONS[@]}"; do
      for mode in "${ROBUST_MODES[@]}"; do
        if [[ "$mode" == "lora" ]]; then
          for rank in "${ROBUST_LORA_RANKS[@]}"; do
            for alpha in "${ROBUST_LORA_ALPHAS[@]}"; do
              torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
                --training_mode robust \
                --model_type "$model_type" \
                --train_modality "$modality" \
                --val_modality "$modality" \
                --train_dataset_name "$train_dataset" \
                --val_dataset_name "$val_dataset" \
                --train_dataset_root "/data/datasets/$train_dataset" \
                --val_dataset_root "/data/datasets/$val_dataset" \
                --train_json "$train_json" \
                --val_json "$val_json" \
                --pretrain_weights "$PRETRAIN_WEIGHTS" \
                "${TRAIN_CENTER_ARG[@]}" \
                "${VAL_CENTER_ARG[@]}" \
                --train_batch_size "$train_bs" \
                --val_batch_size "$val_bs" \
                --num_workers "$NUM_WORKERS" \
                --train_max_samples "$train_max" \
                --val_max_samples "$val_max" \
                --robust_train_attack_loss l2 \
                --robust_val_attack_loss ce \
                --robust_train_loss l2 \
                --robust_lora_rank "$rank" \
                --robust_lora_alpha "$alpha" \
                --robust_epsilon "$eps" \
                --robust_training_mode "$mode" \
                --epochs "$ROBUST_EPOCHS" \
                --use_flash_attention \
                --tensorboard_data_dir "$TENSORBOARD_DATA_DIR" \
                --output_dir "$SESSION_OUTPUT_DIR" \
                --session_output_dir "$SESSION_OUTPUT_DIR" \
                --session_timestamp "$SESSION_TIMESTAMP" \
                --tensorboard_root "$TENSORBOARD_ROOT"
            done
          done
        else
          torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_robust_unibind.py \
            --training_mode robust \
            --model_type "$model_type" \
            --train_modality "$modality" \
            --val_modality "$modality" \
            --train_dataset_name "$train_dataset" \
            --val_dataset_name "$val_dataset" \
            --train_dataset_root "/data/datasets/$train_dataset" \
            --val_dataset_root "/data/datasets/$val_dataset" \
            --train_json "$train_json" \
            --val_json "$val_json" \
            --pretrain_weights "$PRETRAIN_WEIGHTS" \
            "${TRAIN_CENTER_ARG[@]}" \
            "${VAL_CENTER_ARG[@]}" \
            --train_batch_size "$train_bs" \
            --val_batch_size "$val_bs" \
            --num_workers "$NUM_WORKERS" \
            --train_max_samples "$train_max" \
            --val_max_samples "$val_max" \
            --robust_train_attack_loss l2 \
            --robust_val_attack_loss ce \
            --robust_train_loss l2 \
            --robust_epsilon "$eps" \
            --robust_training_mode "$mode" \
            --epochs "$ROBUST_EPOCHS" \
            --use_flash_attention \
            --tensorboard_data_dir "$TENSORBOARD_DATA_DIR" \
            --output_dir "$SESSION_OUTPUT_DIR" \
            --session_output_dir "$SESSION_OUTPUT_DIR" \
            --session_timestamp "$SESSION_TIMESTAMP" \
            --tensorboard_root "$TENSORBOARD_ROOT"
        fi
      done
    done
  done
done

# --- Robust notifications: end ---
ROBUST_END_EPOCH=$(date +%s)
ROBUST_ELAPSED=$(( ROBUST_END_EPOCH - ROBUST_START_EPOCH ))
ROBUST_DUR=$(fmt_duration "$ROBUST_ELAPSED")
post_discord "🏁 Robust finished\n• **Session:** \`${SESSION_TIMESTAMP}\`\n• **Duration:** \`${ROBUST_DUR}\`"

# --- Session-level notification: end ---
SESSION_END_EPOCH=$(date +%s)
SESSION_ELAPSED=$(( SESSION_END_EPOCH - SESSION_START_EPOCH ))
SESSION_DUR=$(fmt_duration "$SESSION_ELAPSED")
post_discord "✅ Session finished\n• **Session:** \`${SESSION_TIMESTAMP}\`\n• **Duration:** \`${SESSION_DUR}\`\n• **Output:** \`${SESSION_OUTPUT_DIR}\`"