import os
import json
import argparse
from datetime import datetime
from itertools import combinations
import re

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_question_ids(path):
    data = load_json(path)
    return set(e['question_id'] for e in data)

def tag_from_path(path):
    # Use folder name as tag, fallback to file stem
    base = os.path.basename(os.path.dirname(path))
    if base: return base
    return os.path.splitext(os.path.basename(path))[0]

def extract_run_metadata(result_json_path):
    result_dir = os.path.dirname(result_json_path)
    log_path = os.path.join(result_dir, "eval_rank0.log")
    if not os.path.isfile(log_path):
        return None, None

    cli_json_lines = []
    in_block = False

    with open(log_path, "r") as f:
        for line in f:
            if '[INFO] - CLI Arguments:' in line:
                match = re.search(r'\{.*', line)
                if match:
                    cli_json_lines.append(match.group(0))
                    in_block = True
                continue
            if in_block:
                cli_json_lines.append(line.strip())
                if line.strip().endswith("}"):
                    break

    try:
        cli_str = "\n".join(cli_json_lines)
        args = json.loads(cli_str)
        val_json = args.get("val_json_template", None)
        model_tags = args.get("model_tags", [])
        if isinstance(model_tags, list):
            model_tags = ",".join(model_tags)
        return val_json, model_tags
    except Exception as e:
        print(f"Warning: Failed to parse CLI JSON from {log_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_jsons', type=str, nargs='+', default=[
        '/data/output/llava/eval/vqa/2025-07-31_09-08/clean/unibind/vqa_results.json',
        # '/data/output/llava/eval/vqa/2025-07-31_09-08/clean/robustbind2/vqa_results.json',
        # '/data/output/llava/eval/vqa/2025-07-31_09-08/clean/robustbind4/vqa_results.json',
        '/data/output/llava/eval/vqa/2025-07-31_09-08/eps2/unibind/vqa_results.json',
        # '/data/output/llava/eval/vqa/2025-07-31_09-08/eps2/robustbind2/vqa_results.json',
        # '/data/output/llava/eval/vqa/2025-07-31_09-08/eps2/robustbind4/vqa_results.json',
        # '/data/output/llava/eval/vqa/2025-07-31_09-08/eps4/unibind/vqa_results.json',
        # '/data/output/llava/eval/vqa/2025-07-31_09-08/eps4/robustbind2/vqa_results.json',
        # '/data/output/llava/eval/vqa/2025-07-31_09-08/eps4/robustbind4/vqa_results.json'
    ], help="List of VQA result JSON files to compare.")
    parser.add_argument('--output_dir', type=str,
        default="/data/output/llava/eval/vqa/filtered_pairwise",
        help="Directory to save pairwise comparison results")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Comparing {len(args.result_jsons)} result files...")

    for f1, f2 in combinations(args.result_jsons, 2):
        tag1 = tag_from_path(f1)
        tag2 = tag_from_path(f2)

        val1, tag_meta1 = extract_run_metadata(f1)
        val2, tag_meta2 = extract_run_metadata(f2)

        qids1 = get_question_ids(f1)
        qids2 = get_question_ids(f2)

        unchanged_ids = sorted(qids1 & qids2)
        added_ids = sorted(qids2 - qids1)
        removed_ids = sorted(qids1 - qids2)

        print(f"\n== Comparing ==")
        print(f"  file1: {f1}")
        print(f"    model_tags: {tag_meta1} | val_json_template: {val1}")
        print(f"  file2: {f2}")
        print(f"    model_tags: {tag_meta2} | val_json_template: {val2}")
        print(f"  Shared question_ids (unchanged): {len(unchanged_ids)}")
        print(f"  Only in file1 (removed): {len(removed_ids)}")
        print(f"  Only in file2 (added):   {len(added_ids)}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(
            args.output_dir,
            f"diff_{tag1}_vs_{tag2}_{timestamp}.json"
        )
        with open(out_path, "w") as f:
            json.dump({
                "file1": f1,
                "file2": f2,
                "file1_metadata": {
                    "val_json_template": val1,
                    "model_tags": tag_meta1
                },
                "file2_metadata": {
                    "val_json_template": val2,
                    "model_tags": tag_meta2
                },
                "unchanged_ids": unchanged_ids,
                "added_ids": added_ids,
                "removed_ids": removed_ids
            }, f, indent=2)
        print(f"  → Saved diff to {out_path}")

if __name__ == "__main__":
    main()
