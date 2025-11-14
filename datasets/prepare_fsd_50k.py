#!/usr/bin/env python3

import os
import argparse
import shutil
import zipfile
import subprocess
import glob
import sys
import pandas as pd
from tqdm import tqdm

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {os.path.basename(zip_path)} into {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Done.\n")


def _has_multipart_zip(base_zip: str) -> bool:
    """Return True if alongside base_zip there are .z01 parts (split zip)."""
    base = os.path.splitext(base_zip)[0]  # remove .zip
    return any(os.path.exists(f"{base}.z{str(i).zfill(2)}") for i in range(1, 30))


def _extract_with_7z(zip_path: str, out_dir: str) -> bool:
    """Try to extract (including multi-part zips) using 7z. Returns True on success."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        cmd = ["7z", "x", zip_path, f"-o{out_dir}", "-y"]
        print(f"Using 7z to extract: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res.stdout)
        return res.returncode == 0
    except FileNotFoundError:
        print("ERROR: '7z' not found. Please install p7zip-full or p7zip to extract split archives.")
        return False


def ensure_unpacked_archives(root: str):
    """If FSD50K zip parts are present, unpack them into the dataset root.
    Handles split zips using 7z; extracts metadata and ground truth zips too.
    """
    # Paths to key outputs
    dev_audio_dir = os.path.join(root, "FSD50K.dev_audio")
    eval_audio_dir = os.path.join(root, "FSD50K.eval_audio")
    gt_dir = os.path.join(root, "FSD50K.ground_truth")
    meta_dir = os.path.join(root, "FSD50K.metadata")

    # 1) DEV audio split zip
    dev_zip = os.path.join(root, "FSD50K.dev_audio.zip")
    if (not os.path.isdir(dev_audio_dir)) and os.path.exists(dev_zip):
        if _has_multipart_zip(dev_zip):
            ok = _extract_with_7z(dev_zip, root)
            if not ok:
                print("Failed to extract multi-part dev audio zip.")
        else:
            unzip_file(dev_zip, root)

    # 2) EVAL audio split zip
    eval_zip = os.path.join(root, "FSD50K.eval_audio.zip")
    if (not os.path.isdir(eval_audio_dir)) and os.path.exists(eval_zip):
        if _has_multipart_zip(eval_zip):
            ok = _extract_with_7z(eval_zip, root)
            if not ok:
                print("Failed to extract multi-part eval audio zip.")
        else:
            unzip_file(eval_zip, root)

    # 3) Ground truth and metadata (regular zips)
    gt_zip = os.path.join(root, "FSD50K.ground_truth.zip")
    if (not os.path.isdir(gt_dir)) and os.path.exists(gt_zip):
        unzip_file(gt_zip, root)

    meta_zip = os.path.join(root, "FSD50K.metadata.zip")
    if (not os.path.isdir(meta_dir)) and os.path.exists(meta_zip):
        unzip_file(meta_zip, root)


def _safe_unlink(path: str):
    try:
        os.remove(path)
        print(f"Deleted: {os.path.basename(path)}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: failed to delete {path}: {e}")


def cleanup_archives(root: str):
    """Remove zip and split parts after successful extraction.
    We only delete archives if their extracted directories exist.
    """
    # Paired zips with split parts
    pairs = [
        (os.path.join(root, "FSD50K.dev_audio"), os.path.join(root, "FSD50K.dev_audio")),
        (os.path.join(root, "FSD50K.eval_audio"), os.path.join(root, "FSD50K.eval_audio")),
    ]
    for extracted_dir, base in pairs:
        if os.path.isdir(extracted_dir):
            # Delete base.zip and base.z01, .z02, ... if present
            for pat in [f"{base}.zip", f"{base}.z??", f"{base}.z*"]:
                for z in glob.glob(pat):
                    _safe_unlink(z)

    # Standalone zips
    if os.path.isdir(os.path.join(root, "FSD50K.ground_truth")):
        _safe_unlink(os.path.join(root, "FSD50K.ground_truth.zip"))
    if os.path.isdir(os.path.join(root, "FSD50K.metadata")):
        _safe_unlink(os.path.join(root, "FSD50K.metadata.zip"))
    # Optional docs
    if os.path.exists(os.path.join(root, "FSD50K.doc.zip")):
        _safe_unlink(os.path.join(root, "FSD50K.doc.zip"))

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
    parser = argparse.ArgumentParser(description="Prepare FSD50K: unpack archives and copy audio into flat train/ and val/ folders.")
    parser.add_argument("DATASET_ROOT", nargs="?", default="/data/datasets/FSD-50K", help="Path to FSD50K root directory")
    args = parser.parse_args()
    print(f"Preparing FSD50K dataset in {args.DATASET_ROOT}...")

    root = os.path.abspath(args.DATASET_ROOT)

    # Step 1: Ensure all required archives are unpacked (handles split zips via 7z)
    ensure_unpacked_archives(root)

    # Step 2: Organize dev audio according to dev.csv into train/ and val/
    organize_audio_flat(root)
    # Step 3: Clean up archives after successful preparation
    cleanup_archives(root)
    print("FSD50K preparation complete.")

if __name__ == "__main__":
    main()
