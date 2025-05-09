#!/usr/bin/env python3

import os
import shutil
import argparse
from tqdm import tqdm

def parse_split_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip().split()[0] for line in f if line.strip()]

def organize_ucf101(dataset_root):
    video_root = os.path.join(dataset_root, "UCF-101")
    split_dir = os.path.join(dataset_root, "ucfTrainTestlist")
    train_split = os.path.join(split_dir, "trainlist01.txt")
    test_split = os.path.join(split_dir, "testlist01.txt")

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
    parser = argparse.ArgumentParser(description="Organize UCF101 into train/test using trainlist01.txt and testlist01.txt from ucfTrainTestlist/")
    parser.add_argument("DATASET_ROOT", help="Path to the root directory (contains UCF-101/ and ucfTrainTestlist/)")
    args = parser.parse_args()
    organize_ucf101(args.DATASET_ROOT)

if __name__ == "__main__":
    main()
