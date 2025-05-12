#!/usr/bin/env python3

import os
import tarfile
import requests
import shutil
import argparse
from tqdm import tqdm

K400_SPLITS = ["train", "val", "test"]
REPLACEMENT_URL = "https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz"

TAR_URLS = {
    "train": "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt",
    "val": "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt",
    "test": "https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt"
}

ANNOTATIONS = {
    "train": "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
    "val": "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
    "test": "https://s3.amazonaws.com/kinetics/400/annotations/test.csv",
}

README_URL = "http://s3.amazonaws.com/kinetics/400/readme.md"

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def download_file(url, dest):
    if os.path.exists(dest):
        return
    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {os.path.basename(dest)}") as pbar:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def download_tar_list(txt_url, out_dir):
    r = requests.get(txt_url)
    r.raise_for_status()
    for url in r.text.strip().splitlines():
        filename = os.path.basename(url)
        out_path = os.path.join(out_dir, filename)
        download_file(url, out_path)

def extract_archives(src_dir, dst_dir, extensions=(".tar.gz", ".tgz")):
    safe_mkdir(dst_dir)
    for fname in os.listdir(src_dir):
        if fname.endswith(extensions):
            full_path = os.path.join(src_dir, fname)
            print(f"Extracting {fname}")
            with tarfile.open(full_path, "r:gz") as tar:
                tar.extractall(path=dst_dir)

def organize_split(split_dir):
    """
    After extracting, move all files into class folders under train/val/test.
    """
    subdirs = [os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    for subdir in subdirs:
        for fname in os.listdir(subdir):
            src = os.path.join(subdir, fname)
            if not os.path.isfile(src):
                continue
            class_name = os.path.basename(subdir)
            class_dir = os.path.join(split_dir, class_name)
            safe_mkdir(class_dir)
            shutil.move(src, os.path.join(class_dir, fname))
        shutil.rmtree(subdir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Prepare Kinetics-400: download, extract, and organize.")
    parser.add_argument('--dataset_root', required=True, help='Root path for the Kinetics-400 dataset')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dataset_root)
    tar_dir = os.path.join(base_dir, "targz")

    safe_mkdir(base_dir)
    safe_mkdir(tar_dir)

    # Download annotations
    ann_dir = os.path.join(base_dir, "annotations")
    safe_mkdir(ann_dir)
    for split, url in ANNOTATIONS.items():
        out_path = os.path.join(ann_dir, f"{split}.csv")
        download_file(url, out_path)

    # Download readme
    download_file(README_URL, os.path.join(base_dir, "readme.md"))

    # Download tarballs
    for split in K400_SPLITS:
        split_tar_dir = os.path.join(tar_dir, split)
        safe_mkdir(split_tar_dir)
        download_tar_list(TAR_URLS[split], split_tar_dir)

    # Download replacement
    repl_dir = os.path.join(tar_dir, "replacement")
    safe_mkdir(repl_dir)
    download_file(REPLACEMENT_URL, os.path.join(repl_dir, os.path.basename(REPLACEMENT_URL)))

    # Extract and organize splits
    for split in K400_SPLITS:
        tar_split_dir = os.path.join(tar_dir, split)
        extract_dir = os.path.join(base_dir, split)
        extract_archives(tar_split_dir, extract_dir)
        organize_split(extract_dir)

    # Extract and organize replacements
    repl_extract = os.path.join(base_dir, "replacement")
    extract_archives(os.path.join(tar_dir, "replacement"), repl_extract)
    organize_split(repl_extract)

    print(f"\n✅ Kinetics-400 is fully prepared under: {base_dir}")

if __name__ == "__main__":
    main()
