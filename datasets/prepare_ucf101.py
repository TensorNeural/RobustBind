#!/usr/bin/env python3

import os
import shutil
import argparse
from tqdm import tqdm

def parse_split_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip().split()[0] for line in f if line.strip()]

def resolve_split_files(split_dir: str, split_id: int):
    """Return paths to trainlistXX.txt and testlistXX.txt for the requested split id (1-3).
    Falls back to split 1 if requested files are missing.
    """
    sid = f"{split_id:02d}"
    train_split = os.path.join(split_dir, f"trainlist{sid}.txt")
    test_split = os.path.join(split_dir, f"testlist{sid}.txt")
    if not (os.path.isfile(train_split) and os.path.isfile(test_split)):
        # Fallback to split 1
        fallback_train = os.path.join(split_dir, "trainlist01.txt")
        fallback_test = os.path.join(split_dir, "testlist01.txt")
        if os.path.isfile(fallback_train) and os.path.isfile(fallback_test):
            print(f"[UCF101] Requested split {split_id} not found. Falling back to split 1.")
            return fallback_train, fallback_test
        raise FileNotFoundError(f"UCF101 split files not found for split {split_id} in {split_dir}")
    return train_split, test_split


def organize_ucf101(dataset_root, split_id: int = 1):
    video_root = os.path.join(dataset_root)
    split_dir = os.path.join(dataset_root, "ucfTrainTestlist")
    train_split, test_split = resolve_split_files(split_dir, split_id)
    print(f"[UCF101] Using train split: {train_split}")
    print(f"[UCF101] Using test split: {test_split}")

    train_videos = parse_split_file(train_split)
    test_videos = parse_split_file(test_split)

    out_train = os.path.join(dataset_root, "train")
    out_test = os.path.join(dataset_root, "test")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    def copy_videos(video_list, dest_root):
        for rel_path in tqdm(video_list, desc=f"Copying to {os.path.basename(dest_root)}"):
            class_label = os.path.dirname(rel_path)
            src = os.path.join(video_root, rel_path)
            if not os.path.exists(src):
                print(f"WARNING: Missing file {src}")
                continue
            dst_dir = os.path.join(dest_root, class_label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))

    copy_videos(train_videos, out_train)
    copy_videos(test_videos, out_test)

    print("UCF101 organization complete.")

def main():
    parser = argparse.ArgumentParser(description="Organize UCF101 into train/test using trainlistXX.txt and testlistXX.txt from ucfTrainTestlist/ (splits 1-3)")
    parser.add_argument("DATASET_ROOT", help="Path to the root directory (contains UCF-101/ and ucfTrainTestlist/)")
    parser.add_argument("--split-id", type=int, choices=[1, 2, 3], default=1, help="Which split list to use (default: 1)")
    args = parser.parse_args()
    organize_ucf101(args.DATASET_ROOT, split_id=args.split_id)

if __name__ == "__main__":
    main()
