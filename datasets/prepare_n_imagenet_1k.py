#!/usr/bin/env python3

import os
import zipfile
import tarfile
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

def unzip_file(zip_path, extract_to):
    print(f"📦 Unzipping {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

def extract_tar_safe(tar_path):
    try:
        with tarfile.open(tar_path, 'r:gz') as tf:
            tf.extractall(path=os.path.dirname(tar_path))
        os.remove(tar_path)
        return True
    except Exception as e:
        print(f"❌ Failed to extract {tar_path}: {e}")
        return False

def extract_all_tar_gz_in_dir_parallel(directory, max_workers=12):
    tar_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tar.gz")]
    if not tar_paths:
        return
    print(f"🗜️ Extracting {len(tar_paths)} .tar.gz files in {directory} ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(extract_tar_safe, tar_paths),
                  total=len(tar_paths), desc="Extracting", unit="file"))

def move_class_folders_to_train(part_path, train_dir):
    os.makedirs(train_dir, exist_ok=True)
    for class_name in os.listdir(part_path):
        src_class = os.path.join(part_path, class_name)
        if not os.path.isdir(src_class) or not class_name.startswith("n"):
            continue
        dst_class = os.path.join(train_dir, class_name)
        os.makedirs(dst_class, exist_ok=True)
        for f in os.listdir(src_class):
            shutil.move(os.path.join(src_class, f), os.path.join(dst_class, f))

def plot_event_image(event_path, output_path):
    try:
        ev = np.load(event_path)
        ev = ev['event_data']
        xs, ys, ps = ev['x'], ev['y'], ev['p']

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        xs_norm = (xs - x_min) / max(x_max - x_min, 1e-5) * 223
        ys_norm = (ys - y_min) / max(y_max - y_min, 1e-5) * 223

        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(xs_norm[ps > 0], ys_norm[ps > 0], c='b', s=0.1)
        ax.scatter(xs_norm[ps <= 0], ys_norm[ps <= 0], c='r', s=0.1)
        ax.set_xlim(0, 224)
        ax.set_ylim(0, 224)
        ax.set_axis_off()
        ax.invert_yaxis()
        plt.savefig(output_path, dpi=100, transparent=True)
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Failed to render {event_path}: {e}")
        return False

def convert_npz_to_png_in_dir(data_split_dir):
    print(f"🎨 Processing .npz in: {data_split_dir}")
    for class_name in sorted(os.listdir(data_split_dir)):
        class_dir = os.path.join(data_split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in sorted(os.listdir(class_dir)):
            if not f.endswith(".npz"):
                continue
            npz_path = os.path.join(class_dir, f)
            png_path = npz_path.replace(".npz", ".png")
            if plot_event_image(npz_path, png_path):
                try:
                    os.remove(npz_path)
                except Exception as e:
                    print(f"❌ Failed to delete {npz_path}: {e}")
    print(f"✅ Finished rendering under: {data_split_dir}\n")

def process_part_zip(zip_path, root, train_dir, max_workers):
    unzip_file(zip_path, root)
    part_name = os.path.basename(zip_path).replace(".zip", "")
    part_path = os.path.join(root, part_name)
    extract_all_tar_gz_in_dir_parallel(part_path, max_workers=max_workers)
    move_class_folders_to_train(part_path, train_dir)
    shutil.rmtree(part_path)
    print(f"🧹 Deleted {part_path}")

def prepare_validation(val_zip_path, val_dir):
    print(f"📦 Unzipping validation from {val_zip_path} ...")
    os.makedirs(val_dir, exist_ok=True)
    with zipfile.ZipFile(val_zip_path, 'r') as zf:
        zf.extractall(val_dir)

def main():
    parser = argparse.ArgumentParser(description="Prepare N-ImageNet-1K → PNGs in train/ and val/")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    root = os.path.abspath(args.dataset_root)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    val_zip = os.path.join(root, "extracted_val.zip")

    # STEP 1 — unzip and prepare training
    for fname in sorted(os.listdir(root)):
        if fname.startswith("Part_") and fname.endswith(".zip"):
            process_part_zip(os.path.join(root, fname), root, train_dir, args.max_workers)

    # STEP 2 — prepare validation
    if os.path.exists(val_zip):
        prepare_validation(val_zip, val_dir)
    else:
        print(f"⚠️ Missing val zip: {val_zip}")

    # STEP 3 — convert all .npz → .png
    for split_dir in [train_dir, val_dir]:
        if os.path.isdir(split_dir):
            convert_npz_to_png_in_dir(split_dir)
        else:
            print(f"⚠️ Split missing: {split_dir}")

    print("🎉 Done: N-ImageNet-1K extracted, visualized, and cleaned!")

if __name__ == "__main__":
    main()
