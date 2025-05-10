#!/usr/bin/env python3

import os
import shutil
import argparse
from tqdm import tqdm
import concurrent.futures


def move_files(src_dir, dst_dir, max_workers=None):
    """
    Move all files from src_dir to dst_dir using multiple workers.
    """
    if not os.path.exists(src_dir):
        print(f"Source directory '{src_dir}' not found. Skipping.")
        return

    os.makedirs(dst_dir, exist_ok=True)

    file_pairs = []
    for fname in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src):
            file_pairs.append((src, dst))

    if not file_pairs:
        print(f"No files found in {src_dir}.")
        return

    print(f"Moving {len(file_pairs)} files from {src_dir} to {dst_dir}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda p: shutil.move(*p), file_pairs), total=len(file_pairs)))

    print(f"Done moving files to '{dst_dir}'.\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare MSR-VTT dataset.")
    parser.add_argument("DATASET_ROOT", nargs="?", default=".", help="Path to the MSR-VTT dataset root directory.")
    parser.add_argument("--max-workers", type=int, default=None, help="Number of worker threads for file moving.")
    args = parser.parse_args()

    root = os.path.abspath(args.DATASET_ROOT)
    train_src = os.path.join(root, "TrainValVideo")
    test_src = os.path.join(root, "TestVideo")
    train_dst = os.path.join(root, "train")
    test_dst = os.path.join(root, "val")

    move_files(train_src, train_dst, max_workers=args.max_workers)
    move_files(test_src, test_dst, max_workers=args.max_workers)

    print("MSR-VTT dataset preparation complete.")


if __name__ == "__main__":
    main()
