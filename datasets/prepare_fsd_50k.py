#!/usr/bin/env python3

import os
import zipfile
import argparse
import shutil
from tqdm import tqdm
import subprocess
import pandas as pd

def extract_multi_part_zip(base_name, zip_dir):
    """Use 7z to extract split archives"""
    zip_path = os.path.join(zip_dir, base_name)
    if not os.path.exists(zip_path):
        print(f"Missing archive: {zip_path}")
        return
    print(f"Extracting {base_name} ...")
    subprocess.run(["7z", "x", zip_path], cwd=zip_dir, check=True)

def unzip_file(zip_path, extract_dir):
    print(f"Unzipping {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def organize_audio(dataset_root):
    meta_path = os.path.join(dataset_root, "FSD50K.metadata", "FSD50K.dev.csv")
    audio_dir = os.path.join(dataset_root, "FSD50K.dev_audio")
    if not os.path.exists(meta_path) or not os.path.exists(audio_dir):
        print("Missing metadata or audio folder.")
        return

    df = pd.read_csv(meta_path)
    os.makedirs(os.path.join(dataset_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, "val"), exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing audio"):
        split = row["split"]
        fname = row["fname"]
        label_dir = os.path.join(dataset_root, split, str(row["labels"]))
        os.makedirs(label_dir, exist_ok=True)

        src_path = os.path.join(audio_dir, fname)
        dst_path = os.path.join(label_dir, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"WARNING: Missing file {src_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare FSD50K: unzip and organize into train/ and val/")
    parser.add_argument("DATASET_ROOT", help="Path to directory containing the FSD50K zip parts and metadata")
    args = parser.parse_args()

    root = os.path.abspath(args.DATASET_ROOT)

    # Step 1: extract multi-part audio archives
    extract_multi_part_zip("FSD50K.dev_audio.zip", root)
    extract_multi_part_zip("FSD50K.eval_audio.zip", root)

    # Step 2: unzip metadata and ground truth
    unzip_file(os.path.join(root, "FSD50K.metadata.zip"), root)
    unzip_file(os.path.join(root, "FSD50K.ground_truth.zip"), root)

    # Step 3: organize dev set into train/ and val/
    organize_audio(root)

    print("FSD50K preparation complete.")

if __name__ == "__main__":
    main()
