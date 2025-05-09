#!/usr/bin/env python3

import os
import shutil
import argparse
import csv
from tqdm import tqdm

def organize_urbansound8k(dataset_root):
    audio_dir = os.path.join(dataset_root, "audio")
    metadata_path = os.path.join(dataset_root, "metadata", "UrbanSound8K.csv")

    if not os.path.exists(audio_dir) or not os.path.exists(metadata_path):
        print("Missing audio/ directory or metadata file.")
        return

    out_train = os.path.join(dataset_root, "train")
    out_test = os.path.join(dataset_root, "test")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Processing clips"):
            fold = int(row["fold"])
            class_label = row["class"]
            file_name = row["slice_file_name"]
            file_path = os.path.join(audio_dir, f"fold{fold}", file_name)

            dest_root = out_train if fold <= 8 else out_test
            dest_dir = os.path.join(dest_root, class_label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(dest_dir, file_name))

    print("UrbanSound8K processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Organize UrbanSound8K into train/test.")
    parser.add_argument("DATASET_ROOT", help="Root directory of UrbanSound8K (with audio/ and metadata/)")

    args = parser.parse_args()
    organize_urbansound8k(args.DATASET_ROOT)

if __name__ == "__main__":
    main()
