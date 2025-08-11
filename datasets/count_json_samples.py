#!/usr/bin/env python3

import os
import json

def count_json_samples(root_dir="."):
    total = 0
    counts = {}
    missing_description_counts = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".json"):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                try:
                    with open(full_path, "r") as f:
                        data = json.load(f)
                    count = len(data)
                    counts[rel_path] = count
                    total += count

                    # Check for missing descriptions in '_align.json' files
                    if "_align.json" in fname:
                        missing_count = sum(1 for item in data if "description" not in item or not item["description"])
                        missing_description_counts[rel_path] = missing_count
                except Exception as e:
                    print(f"❌ Failed to read {rel_path}: {e}")

    print("\n📦 Sample counts per JSON file:\n")
    print(f"{'File Path':<50} {'Total Samples':>15} {'Missing Descriptions':>20}")
    print("-" * 85)
    for path in sorted(counts.keys()):
        total_samples = counts[path]
        missing_descriptions = missing_description_counts.get(path, 0)
        print(f"{path:<50} {total_samples:>15} {missing_descriptions:>20}")

    print(f"\n✅ Total samples across all JSON files: {total:,}")

if __name__ == "__main__":
    count_json_samples(".")
