#!/usr/bin/env python3

import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

def create_unibind_style_event_image(x, y, polarity, timestamp, num_samples=50000):
    if len(x) == 0:
        raise ValueError("Empty event stream")

    np.random.seed(42)
    sample_indices = np.random.choice(len(x), size=min(num_samples, len(x)), replace=False)

    x_sample = x[sample_indices]
    y_sample = y[sample_indices]
    pol_sample = polarity[sample_indices]
    ts_sample = timestamp[sample_indices]

    ts_norm = (ts_sample - ts_sample.min()) / (ts_sample.max() - ts_sample.min() + 1e-8)

    hsv = np.zeros((len(x_sample), 3))
    hsv[:, 0] = ts_norm
    hsv[:, 1] = 1.0
    hsv[:, 2] = np.where(pol_sample == 1, 1.0, 0.6)

    rgb = mcolors.hsv_to_rgb(hsv)
    return x_sample, y_sample, rgb

def save_event_image_scatter(x, y, rgb, output_path):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(x, y, c=rgb, s=0.5, edgecolors='none')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.gca().set_facecolor((0, 0, 0, 0))
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

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

def save_split_file(path_list, output_txt):
    with open(output_txt, "w") as f:
        for path in path_list:
            f.write(f"{path}\n")

def convert_split(txt_file, split_name, data_root, output_root):
    with open(txt_file, "r") as f:
        rel_paths = [line.strip() for line in f.readlines()]

    for rel_path in tqdm(rel_paths, desc=f"Processing {split_name}"):
        class_name = rel_path.split("/")[0]
        file_stem = os.path.splitext(os.path.basename(rel_path))[0]
        src_path = os.path.join(data_root, rel_path)
        dst_dir = os.path.join(output_root, split_name, class_name)
        dst_path = os.path.join(dst_dir, file_stem + ".png")
        os.makedirs(dst_dir, exist_ok=True)

        try:
            x, y, p, t = read_events_from_bin(src_path)
            xs, ys, rgb = create_unibind_style_event_image(x, y, p, t)
            save_event_image_scatter(xs, ys, rgb, dst_path)
        except Exception as e:
            print(f"Failed: {src_path} — {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("DATASET_ROOT", help="Path to dataset (must contain data/)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion of data to use for training (default: 0.7)")
    args = parser.parse_args()

    data_root = os.path.join(args.DATASET_ROOT, "data")
    train_txt = os.path.join(args.DATASET_ROOT, "train.txt")
    test_txt = os.path.join(args.DATASET_ROOT, "test.txt")

    if not os.path.exists(train_txt) or not os.path.exists(test_txt):
        all_paths = gather_all_bin_paths(data_root)
        total = len(all_paths)
        if total == 0:
            raise ValueError("No .bin files found.")

        random.seed(42)
        random.shuffle(all_paths)

        train_count = int(total * args.train_ratio)
        train_split = all_paths[:train_count]
        test_split = all_paths[train_count:]

        save_split_file(train_split, train_txt)
        save_split_file(test_split, test_txt)

        print(f"Generated splits: train={len(train_split)}, test={len(test_split)}")

    convert_split(train_txt, "train", data_root, args.DATASET_ROOT)
    convert_split(test_txt, "test", data_root, args.DATASET_ROOT)

    print("All conversions complete.")

if __name__ == "__main__":
    main()
