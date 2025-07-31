#!/bin/bash
set -e

python filter_unchanged_questions.py \
  --clean_json output/llava/eval/vqa/2025-07-30_17-58/unibind/vqa_results.json \
  --random_json output/llava/eval/vqa/2025-07-30_18-37/unibind/vqa_results.json \
  --raw_result_jsons \
    output/llava/eval/vqa/2025-07-30_17-58/unibind/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_17-58/robustbind2/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_17-58/robustbind4/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_08-54/unibind/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_09-06/robustbind2/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_09-19/robustbind4/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_09-33/unibind/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_09-45/robustbind2/vqa_results.json \
    output/llava/eval/vqa/2025-07-30_09-58/robustbind4/vqa_results.json \
  --output_dir output/llava/eval/vqa/filtered
