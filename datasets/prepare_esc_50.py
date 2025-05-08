#!/usr/bin/env python3

import os
import shutil
import argparse
import csv
from tqdm import tqdm
from collections import defaultdict

def prepare_esc50_standard_split(dataset_root):
    audio_dir = os.path.join(dataset_root, "audio")
    meta_csv = os.path.join(dataset_root, "meta", "esc50.csv")

    if not os.path.exists(audio_dir) or not os.path.exists(meta_csv):
        print("Error: 'audio/' folder or 'meta/esc50.csv' missing.")
        return

    out_train = os.path.join(dataset_root, "train")
    out_test = os.path.join(dataset_root, "test")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    folds = defaultdict(list)
    with open(meta_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            fold = int(row["fold"])
            category = row["category"]
            folds[fold].append((filename, category))

    train_files = folds[1] + folds[2] + folds[3] + folds[4]
    test_files = folds[5]

    def copy_files(file_list, target_root):
        for fname, label in tqdm(file_list, desc=f"Copying to {os.path.basename(target_root)}"):
            src = os.path.join(audio_dir, fname)
            class_dir = os.path.join(target_root, label)
            os.makedirs(class_dir, exist_ok=True)
            dst = os.path.join(class_dir, fname)
            shutil.copy2(src, dst)

    copy_files(train_files, out_train)
    copy_files(test_files, out_test)

    print(f"Done: train={len(train_files)}, test={len(test_files)}")

def main():
    parser = argparse.ArgumentParser(description="Prepare ESC-50 using folds 1–4 for training and fold 5 for test.")
    parser.add_argument("DATASET_ROOT", help="Root of ESC-50 dataset (with meta/ and audio/)")
    args = parser.parse_args()

    prepare_esc50_standard_split(args.DATASET_ROOT)

if __name__ == "__main__":
    main()
