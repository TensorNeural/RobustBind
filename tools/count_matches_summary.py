import os
import json
import argparse
from collections import defaultdict

# All required variants
ALL_VARIANTS = {
    "clean__clean",
    "clean__eps2",
    "clean__eps4",
    "lora_robust2__clean",
    "lora_robust2__eps2",
    "lora_robust2__eps4",
    "lora_robust4__clean",
    "lora_robust4__eps2",
    "lora_robust4__eps4",
}

def load_query_match_counts(base_dir, variant):
    """Load match counts per query for a given variant by reading all rank*/top5_results.json"""
    variant_dir = os.path.join(base_dir, variant)
    match_counts = {}

    if not os.path.isdir(variant_dir):
        print(f"[Warning] Missing variant directory: {variant_dir}")
        return match_counts

    for rank_name in os.listdir(variant_dir):
        rank_path = os.path.join(variant_dir, rank_name)
        if not os.path.isdir(rank_path) or not rank_name.startswith("rank"):
            continue

        result_path = os.path.join(rank_path, "top5_results.json")
        if not os.path.isfile(result_path):
            print(f"[Warning] Missing: {result_path}")
            continue

        with open(result_path, "r") as f:
            results = json.load(f)

        grouped = defaultdict(list)
        for entry in results:
            grouped[entry["query_index"]].append(entry)

        for qid, entries in grouped.items():
            match_counts[qid] = sum(1 for e in entries if e.get("match", False))

    return match_counts

def main(search_dir):
    print(f"\n[INFO] Loading from: {search_dir}")
    variant_counts = {v: load_query_match_counts(search_dir, v) for v in ALL_VARIANTS}

    base_qids = sorted(variant_counts["clean__clean"].keys())
    print(f"[INFO] Found {len(base_qids)} queries in clean__clean\n")

    print(f"{'Query':>5} | {'cc':>2} | {'ce2':>3} | {'ce4':>3} | "
          f"{'r2c':>3} | {'r2e2':>4} | {'r2e4':>4} | "
          f"{'r4c':>3} | {'r4e2':>4} | {'r4e4':>4} | Win ✔")
    print("-" * 90)

    win_count = 0

    for qid in base_qids:
        cc   = variant_counts["clean__clean"].get(qid, 0)
        ce2  = variant_counts["clean__eps2"].get(qid, 0)
        ce4  = variant_counts["clean__eps4"].get(qid, 0)

        r2c  = variant_counts["lora_robust2__clean"].get(qid, 0)
        r2e2 = variant_counts["lora_robust2__eps2"].get(qid, 0)
        r2e4 = variant_counts["lora_robust2__eps4"].get(qid, 0)

        r4c  = variant_counts["lora_robust4__clean"].get(qid, 0)
        r4e2 = variant_counts["lora_robust4__eps2"].get(qid, 0)
        r4e4 = variant_counts["lora_robust4__eps4"].get(qid, 0)

        passed = (
            cc > 3 and
            ce2 < 2 and ce4 < 2 and
            all(x > 3 for x in [r2c, r2e2, r2e4, r4c, r4e2, r4e4])
        )

        mark = "✔" if passed else ""
        if passed:
            win_count += 1

        print(f"{qid:5d} | {cc:2d} | {ce2:3d} | {ce4:3d} | "
              f"{r2c:3d} | {r2e2:4d} | {r2e4:4d} | "
              f"{r4c:3d} | {r4e2:4d} | {r4e4:4d} | {mark:>6}")

    print(f"\n✅ Total queries satisfying win condition: {win_count} / {len(base_qids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_dir",
        required=True,
        help="Path to cross_modality_search/{timestamp}, e.g., /data/output/cross_modality_search/2025-05-20_07-04-54"
    )
    args = parser.parse_args()
    main(args.search_dir)
