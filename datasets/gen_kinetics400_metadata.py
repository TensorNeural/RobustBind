import os
import csv
import json
import argparse
from collections import defaultdict


def collect_all_labels(csv_paths):
    label_set = set()
    for path in csv_paths:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_set.add(row["label"].strip())
    return sorted(label_set)


def parse_csv(csv_path, split_name, dataset_root, label_to_id):
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"].strip() != split_name:
                continue
            label = row["label"].strip()
            youtube_id = row["youtube_id"].strip()
            start = int(row["time_start"])
            end = int(row["time_end"])
            fname = f"{youtube_id}_{start:06d}_{end:06d}.mp4"
            rel_path = os.path.join(split_name, fname)
            abs_path = os.path.join(dataset_root, rel_path)
            if os.path.exists(abs_path):
                entries.append({
                    "data": rel_path,
                    "label": label,
                    "label_id": label_to_id[label]
                })
    return entries


def write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} entries → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kinetics-400 metadata with label IDs.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root path containing train/, val/, test/, annotations/")
    parser.add_argument("--output_dir", type=str, default="Kinetics-400", help="Directory to save JSON outputs")
    args = parser.parse_args()

    ann_dir = os.path.join(args.dataset_root, "annotations")
    train_csv = os.path.join(ann_dir, "train.csv")
    val_csv = os.path.join(ann_dir, "val.csv")
    test_csv = os.path.join(ann_dir, "test.csv")

    # Build label → ID mapping
    all_labels = collect_all_labels([train_csv, val_csv, test_csv])
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    write_json(label_to_id, os.path.join(args.output_dir, "label_to_id.json"))

    # Generate metadata
    for split, csv_file in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        if os.path.exists(csv_file):
            entries = parse_csv(csv_file, split, args.dataset_root, label_to_id)
            write_json(entries, os.path.join(args.output_dir, f"{split}_data.json"))
        else:
            print(f"⚠️ Skipping missing file: {csv_file}")

    print("🎉 All metadata and label mapping generated.")
