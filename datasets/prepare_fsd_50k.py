#!/usr/bin/env python3

import os
import argparse
import shutil
import zipfile
import pandas as pd
from tqdm import tqdm

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {os.path.basename(zip_path)} into {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Done.\n")

def organize_audio_flat(dataset_root):
    meta_path = os.path.join(dataset_root, "FSD50K.ground_truth", "dev.csv")
    audio_dir = os.path.join(dataset_root, "FSD50K.dev_audio")
    if not os.path.exists(meta_path) or not os.path.exists(audio_dir):
        print("Missing metadata or dev audio folder.")
        return

    df = pd.read_csv(meta_path)
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying audio"):
        split = row["split"]
        fname = f"{row['fname']}.wav"
        src_path = os.path.join(audio_dir, fname)
        dst_path = os.path.join(dataset_root, split, fname)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"WARNING: missing file {src_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare FSD50K: copy audio into flat train/ and val/ folders.")
    parser.add_argument("DATASET_ROOT", help="Path to FSD50K root directory")
    args = parser.parse_args()
    root = os.path.abspath(args.DATASET_ROOT)

    # Optional: unzip archive files (commented out)
    # unzip_file(os.path.join(root, "FSD50K.dev_audio.zip"), root)
    # unzip_file(os.path.join(root, "FSD50K.eval_audio.zip"), root)
    # unzip_file(os.path.join(root, "FSD50K.metadata.zip"), root)
    # unzip_file(os.path.join(root, "FSD50K.ground_truth.zip"), root)

    organize_audio_flat(root)
    print("FSD50K preparation complete.")

if __name__ == "__main__":
    main()
