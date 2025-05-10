#!/usr/bin/env python3

import os
import argparse
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_events_from_bin(file_path):
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    event_bytes = byte_array.reshape((-1, 5))

    x = event_bytes[:, 0].astype(np.uint32)
    y = event_bytes[:, 1].astype(np.uint32)
    polarity = (event_bytes[:, 2] >> 7) & 0x01

    timestamp = (
        ((event_bytes[:, 2] & 0x7F).astype(np.uint32) << 16) |
        (event_bytes[:, 3].astype(np.uint32) << 8) |
        event_bytes[:, 4].astype(np.uint32)
    )
    return x, y, polarity, timestamp

def plot_event_image(x, y, polarity, output_path):
    try:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        xs = ((x - x_min) / max(x_max - x_min, 1e-5)) * 223
        ys = ((y - y_min) / max(y_max - y_min, 1e-5)) * 223

        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 224)
        ax.set_ylim(0, 224)
        ax.axis('off')
        ax.set_facecolor((0, 0, 0, 0))
        ax.invert_yaxis()

        pos_mask = polarity > 0
        neg_mask = ~pos_mask
        ax.scatter(xs[pos_mask], ys[pos_mask], c='blue', s=1.0, alpha=0.35, edgecolors='none')
        ax.scatter(xs[neg_mask], ys[neg_mask], c='red', s=1.0, alpha=0.35, edgecolors='none')

        plt.savefig(output_path, dpi=100, transparent=True, pad_inches=0, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Failed to render {output_path}: {e}")
        return False

def gather_all_bin_paths(data_root):
    bin_paths = []
    for class_name in sorted(os.listdir(data_root)):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(".bin"):
                rel_path = os.path.join(class_name, fname)
                bin_paths.append(rel_path)
    return bin_paths

def split_and_copy(all_paths, data_root, output_root, train_ratio=0.7):
    random.seed(42)
    random.shuffle(all_paths)
    train_count = int(len(all_paths) * train_ratio)
    train_split = all_paths[:train_count]
    val_split = all_paths[train_count:]

    for split_name, split_paths in [('train', train_split), ('val', val_split)]:
        for rel_path in split_paths:
            src = os.path.join(data_root, rel_path)
            dst = os.path.join(output_root, split_name, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    print(f"✅ Split complete: {len(train_split)} train, {len(val_split)} val")

def convert_bin_dir(split_dir):
    print(f"🎨 Converting .bin → .png in: {split_dir}")
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in tqdm(os.listdir(class_dir), desc=class_name):
            if fname.endswith(".bin"):
                bin_path = os.path.join(class_dir, fname)
                png_path = bin_path.replace(".bin", ".png")
                try:
                    x, y, p, _ = read_events_from_bin(bin_path)
                    if plot_event_image(x, y, p, png_path):
                        os.remove(bin_path)
                except Exception as e:
                    print(f"❌ Failed: {bin_path} — {e}")
    print(f"✅ Finished rendering for: {split_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare N-Caltech101-style dataset (split + render .bin → .png)")
    parser.add_argument("--dataset_root", required=True, help="Path to root folder (must contain data/CLASS/*.bin)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train/val split ratio (default: 0.7)")
    args = parser.parse_args()

    raw_data_root = os.path.join(args.dataset_root, "data")
    if not os.path.exists(raw_data_root):
        raise FileNotFoundError(f"❌ Expected directory: {raw_data_root}")

    all_paths = gather_all_bin_paths(raw_data_root)
    if not all_paths:
        raise ValueError("❌ No .bin files found under data/")

    # Step 1: Split .bin files into train/ and val/
    split_and_copy(all_paths, raw_data_root, args.dataset_root, train_ratio=args.train_ratio)

    # Step 2: Convert .bin to .png and delete .bin
    for split in ["train", "val"]:
        split_dir = os.path.join(args.dataset_root, split)
        if os.path.isdir(split_dir):
            convert_bin_dir(split_dir)
        else:
            print(f"❌ Missing split directory: {split_dir}")

    print("🎉 Done: All .bin files split and rendered to PNGs.")

if __name__ == "__main__":
    main()
