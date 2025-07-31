import os
import json
import csv
import argparse
from datetime import datetime


def load_result_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    result_map = {}
    for item in data:
        key = (item['image_id'], item['question'].strip().lower())
        result_map[key] = {
            'acc': float(item['vqa_soft_accuracy']),
            'question_id': item['question_id'],
            'full': item
        }
    return result_map


def compute_soft_score(data):
    if len(data) == 0:
        return 0.0
    return sum(float(item["vqa_soft_accuracy"]) for item in data) / len(data)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved {len(data)} entries to {os.path.abspath(path)}")


def extract_unchanged_questions(clean_path, random_path, output_json):
    clean_map = load_result_json(clean_path)
    rand_map = load_result_json(random_path)

    unchanged = []
    for key in clean_map:
        if key in rand_map and clean_map[key]['acc'] == rand_map[key]['acc']:
            unchanged.append({
                'image_id': key[0],
                'question': key[1],
                'question_id': clean_map[key]['question_id']
            })

    save_json(unchanged, output_json)
    return unchanged


def filter_json_by_excluding_questions(raw_data, unchanged_set):
    filtered = []
    unchanged_count = 0
    for item in raw_data:
        key = (item['image_id'], item['question'].strip().lower())
        if key in unchanged_set:
            unchanged_count += 1
        else:
            filtered.append(item)
    return filtered, unchanged_count


def write_missing_unchanged_csv(missing, output_path, raw_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question_id', 'image_id', 'question', 'raw_json_path'])
        for item in missing:
            writer.writerow([
                item['question_id'],
                item['image_id'],
                item['question'],
                raw_path
            ])
    print(f"[✓] Saved {len(missing)} missing unchanged entries to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_json', type=str, required=True,
                        help='Path to clean unibind VQA results')
    parser.add_argument('--random_json', type=str, required=True,
                        help='Path to random image VQA results')
    parser.add_argument('--raw_result_jsons', type=str, nargs='+', required=True,
                        help='List of raw VQA result JSONs to filter')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for all filtered results and unchanged questions')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    log_path = os.path.join(args.output_dir, f"log_{timestamp}.txt")
    log_file = open(log_path, "w")

    # Extract unchanged and keep both the set and raw list
    unchanged_path = os.path.join(args.output_dir, f'unchanged_questions_{timestamp}.json')
    unchanged_raw = extract_unchanged_questions(args.clean_json, args.random_json, unchanged_path)
    unchanged_set = set((x['image_id'], x['question']) for x in unchanged_raw)

    log_file.write("=== UNCHANGED QUESTIONS ===\n")
    log_file.write(f"Clean JSON:   {os.path.abspath(args.clean_json)}\n")
    log_file.write(f"Random JSON:  {os.path.abspath(args.random_json)}\n")
    log_file.write(f"Unchanged JSON Output: {os.path.abspath(unchanged_path)}\n")
    log_file.write(f"Total Unchanged: {len(unchanged_set)}\n\n")

    log_file.write("=== FILTERED RESULTS ===\n")

    for file_path in args.raw_result_jsons:
        tag = os.path.basename(os.path.dirname(file_path))
        out_file = os.path.join(args.output_dir, f'vqa_results_{tag}_filtered_{timestamp}.json')
        missing_csv = os.path.join(args.output_dir, f'missing_unchanged_in_{tag}_{timestamp}.csv')

        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        total = len(raw_data)
        old_score = compute_soft_score(raw_data)

        # Determine missing unchanged questions
        raw_keys_set = set((item['image_id'], item['question'].strip().lower()) for item in raw_data)
        missing_unchanged = [
            x for x in unchanged_raw
            if (x['image_id'], x['question']) not in raw_keys_set
        ]
        write_missing_unchanged_csv(missing_unchanged, missing_csv, os.path.abspath(file_path))

        # Filter unchanged
        filtered_data, num_matched = filter_json_by_excluding_questions(raw_data, unchanged_set)
        new_score = compute_soft_score(filtered_data)
        num_kept = len(filtered_data)
        diff = new_score - old_score

        save_json(filtered_data, out_file)

        log_file.write(f"Model Tag: {tag}\n")
        log_file.write(f"Input JSON:    {os.path.abspath(file_path)}\n")
        log_file.write(f"Filtered JSON: {os.path.abspath(out_file)}\n")
        log_file.write(f"Missing Unchanged CSV: {os.path.abspath(missing_csv)}\n")
        log_file.write(f"  Total Raw Entries:     {total}\n")
        log_file.write(f"  Unchanged Matched:     {num_matched}\n")
        log_file.write(f"  Entries Kept:          {num_kept}\n")
        log_file.write(f"  Original Score:        {old_score:.4f}\n")
        log_file.write(f"  Filtered Score:        {new_score:.4f}\n")
        log_file.write(f"  Difference:            {diff:+.4f}\n\n")

    log_file.close()
    print(f"[✓] Logged results to {os.path.abspath(log_path)}")


if __name__ == '__main__':
    main()
