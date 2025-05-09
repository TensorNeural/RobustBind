#!/usr/bin/env python3

import os
import json

def count_json_samples(root_dir="."):
    total = 0
    counts = {}

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
                except Exception as e:
                    print(f"❌ Failed to read {rel_path}: {e}")

    print("\n📦 Sample counts per JSON file:\n")
    for path, count in sorted(counts.items()):
        print(f"{path:<50} {count:>6} samples")

    print(f"\n✅ Total samples across all JSON files: {total:,}")

if __name__ == "__main__":
    count_json_samples(".")
