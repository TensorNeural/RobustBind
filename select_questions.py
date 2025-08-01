import os
import json
import argparse


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved {len(data)} entries to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqa_result', type=str,
                        default='output/llava/eval/vqa/2025-07-31_09-08/eps2/unibind/vqa_results.json',
                        help='Adversarial result JSON with question_ids to keep')
    parser.add_argument('--val_data', type=str,
                        default='datasets/VQA2/val_data.json',
                        help='Path to VQA2 validation data')
    parser.add_argument('--output', type=str,
                        default='datasets/VQA2/val_data_filtered.json',
                        help='Path to save filtered ground-truth')
    args = parser.parse_args()

    vqa_result = load_json(args.vqa_result)
    val_data = load_json(args.val_data)

    keep_ids = {item['question_id'] for item in vqa_result}
    print(f"[✓] Loaded {len(keep_ids)} question_ids from {args.vqa_result}")

    filtered_val = [item for item in val_data if item['question_id'] in keep_ids]
    print(f"[✓] Selected {len(filtered_val)} matching entries from {args.val_data}")

    save_json(filtered_val, args.output)


if __name__ == '__main__':
    main()
