import json
import csv
import argparse
from datetime import datetime


def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    result_map = {}
    for item in data:
        key = (item['image_id'], item['question'].strip().lower())
        result_map[key] = {
            'pred': item['predicted_answer'].strip().lower(),
            'acc': float(item['vqa_soft_accuracy']),
            'question_id': item['question_id']
        }
    return result_map


def compare_results(clean_res, adv_res, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image ID", "Question ID", "Question",
            "Clean Answer", "Adv Answer",
            "Clean Acc", "Adv Acc", "Acc Change"
        ])

        for key in clean_res:
            if key not in adv_res:
                continue

            image_id, question = key
            clean = clean_res[key]
            adv = adv_res[key]

            writer.writerow([
                image_id,
                clean['question_id'],
                question,
                clean['pred'],
                adv['pred'],
                f"{clean['acc']:.2f}",
                f"{adv['acc']:.2f}",
                f"{adv['acc'] - clean['acc']:.2f}",
            ])


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clean_json',
        type=str,
        default="/data/output/llava/eval/vqa/2025-07-30_05-27/vqa_results.json",
        help='Path to clean vqa_results.json'
    )
    parser.add_argument(
        '--adv_json',
        type=str,
        default="/data/output/llava/eval/vqa/2025-07-30_08-02/robustbind4/vqa_results.json",
        help='Path to adversarial vqa_results.json'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=f'/data/output/llava/eval/vqa/comparison_{timestamp}.csv',
        help='Output CSV path'
    )
    args = parser.parse_args()

    clean_res = load_results(args.clean_json)
    adv_res = load_results(args.adv_json)
    compare_results(clean_res, adv_res, args.output_csv)

    print(f"[✓] Comparison complete. CSV saved to {args.output_csv}")


if __name__ == '__main__':
    main()
