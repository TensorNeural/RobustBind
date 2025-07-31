import os
import json
import argparse
from datetime import datetime
from itertools import combinations, permutations

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_keys(entries, mode):
    if mode == "vqa":
        return { (e['image_id'], e['question_id']) : e for e in entries }
    elif mode == "caption":
        return { e['image_id']: e for e in entries }
    else:
        raise ValueError("Mode must be 'vqa' or 'caption'")

def compare_entries(file1, file2, mode):
    entries1 = load_json(file1)
    entries2 = load_json(file2)

    index1 = extract_keys(entries1, mode)
    index2 = extract_keys(entries2, mode)

    keys1 = set(index1.keys())
    keys2 = set(index2.keys())

    added_keys = keys2 - keys1
    removed_keys = keys1 - keys2
    unchanged_keys = keys1 & keys2

    added = [index2[k] for k in added_keys]
    removed = [index1[k] for k in removed_keys]
    unchanged = [index1[k] for k in unchanged_keys]

    added_ids = [k[1] if isinstance(k, tuple) else k for k in added_keys]
    removed_ids = [k[1] if isinstance(k, tuple) else k for k in removed_keys]
    unchanged_ids = [k[1] if isinstance(k, tuple) else k for k in unchanged_keys]

    return {
        "file1": file1,
        "file2": file2,
        "added_ids": sorted(added_ids),
        "removed_ids": sorted(removed_ids),
        "unchanged_ids": sorted(unchanged_ids),
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
    }

def save_result(result, tag1, tag2, out_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"diff_{tag1}_vs_{tag2}_{timestamp}.json"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  → Diff saved to {out_path}")

def tag_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str,
        default="./datasets/VQA2/val_data.json",
        help="Path to the base val_data.json")
    parser.add_argument('--file2', type=str, nargs='+', default=[
        "./datasets/VQA2/val_data_adv_eps2_robustbind2.json",
        "./datasets/VQA2/val_data_adv_eps2_robustbind4.json",
        "./datasets/VQA2/val_data_adv_eps2_unibind.json",
        "./datasets/VQA2/val_data_adv_eps4_robustbind2.json",
        "./datasets/VQA2/val_data_adv_eps4_robustbind4.json",
        "./datasets/VQA2/val_data_adv_eps4_unibind.json"
    ], help="List of val_data_adv JSON files to compare against")
    parser.add_argument('--mode', type=str, choices=['vqa', 'caption'], default="vqa", help="Data format type")
    parser.add_argument('--compare_file2_pairs', action='store_true', default=True,
        help="If set, compare each pair of file2 entries (e.g. robustbind2 vs unibind)")
    parser.add_argument('--symmetric_pairs', action='store_true', default=True,
        help="If set, do unordered pairwise comparison (A vs B only once). If not set, ordered A vs B and B vs A.")

    args = parser.parse_args()
    out_dir = os.path.join("output", "diff")
    os.makedirs(out_dir, exist_ok=True)

    if args.compare_file2_pairs:
        # === Compare file2[i] vs file2[j] ===
        print("\n=== Pairwise comparisons among --file2 files ===")
        pair_generator = combinations if args.symmetric_pairs else permutations
        for f1, f2 in pair_generator(args.file2, 2):
            print(f"\n--- Comparing: {f1} vs {f2} ---")
            result = compare_entries(f1, f2, args.mode)
            print(f"  Added: {len(result['added'])}, Removed: {len(result['removed'])}, Unchanged: {len(result['unchanged'])}")
            save_result(result, tag_from_path(f1), tag_from_path(f2), out_dir)
    else:
        # === Compare file1 vs each file2 ===
        for file2 in args.file2:
            print(f"\n=== Comparing base: {args.file1} vs {file2} ===")
            result = compare_entries(args.file1, file2, args.mode)
            print(f"  Added: {len(result['added'])}, Removed: {len(result['removed'])}, Unchanged: {len(result['unchanged'])}")
            save_result(result, tag_from_path(args.file1), tag_from_path(file2), out_dir)

if __name__ == "__main__":
    main()
