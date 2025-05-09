#!/usr/bin/env python3

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

def generate_metadata_flat(dataset_root, output_dir="FSD-50K"):
    """
    Generates train_data.json and val_data.json for FSD50K flat layout.
    Each entry has:
      - 'data': relative path (e.g., 'train/123.wav')
      - 'label': comma-separated string of labels
    """
    csv_path = os.path.join(dataset_root, "FSD50K.ground_truth", "dev.csv")
    if not os.path.exists(csv_path):
        print("Missing dev.csv. Please verify extracted ground truth.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"Missing directory: {split_dir}")
            continue

        split_df = df[df["split"] == split]
        metadata = []

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Generating {split}_data.json"):
            fname = f"{row['fname']}.wav"
            label_str = str(row["labels"])
            file_path = os.path.join(split, fname)

            if not os.path.exists(os.path.join(dataset_root, file_path)):
                print(f"WARNING: missing file {file_path}")
                continue

            metadata.append({
                "data": file_path,
                "label": label_str
            })

        out_path = os.path.join(output_dir, f"{split}_data.json")
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {len(metadata)} entries to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate FSD50K metadata with flat folder layout and string labels.")
    parser.add_argument("DATASET_ROOT", help="Path to FSD50K root directory")
    args = parser.parse_args()
    root = os.path.abspath(args.DATASET_ROOT)

    generate_metadata_flat(root)

if __name__ == "__main__":
    main()
