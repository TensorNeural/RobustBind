#!/usr/bin/env python3

import os
import zipfile
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_event_image(event_path, output_path):
    try:
        ev = np.load(event_path)
        ev = ev['event_data']
        xs, ys, ps = ev['x'], ev['y'], ev['p']

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        xs = ((xs - x_min) / max(x_max - x_min, 1e-5)) * 223
        ys = ((ys - y_min) / max(y_max - y_min, 1e-5)) * 223

        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 224)
        ax.set_ylim(0, 224)
        ax.axis('off')
        ax.invert_yaxis()

        pos_mask = ps > 0
        neg_mask = ~pos_mask
        ax.scatter(xs[pos_mask], ys[pos_mask], c='blue', s=1.0, alpha=0.35, edgecolors='none')
        ax.scatter(xs[neg_mask], ys[neg_mask], c='red', s=1.0, alpha=0.35, edgecolors='none')

        plt.savefig(output_path, dpi=100, transparent=True, pad_inches=0, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Failed to render {event_path}: {e}")
        return False

def convert_npz_to_png_in_dir(data_split_dir):
    print(f"🎨 Rendering .npz in: {data_split_dir}")
    for class_name in sorted(os.listdir(data_split_dir)):
        class_dir = os.path.join(data_split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in sorted(os.listdir(class_dir)):
            if f.endswith(".npz"):
                npz_path = os.path.join(class_dir, f)
                png_path = npz_path.replace(".npz", ".png")
                if plot_event_image(npz_path, png_path):
                    try:
                        os.remove(npz_path)
                    except Exception as e:
                        print(f"❌ Failed to delete {npz_path}: {e}")
    print(f"✅ Finished rendering PNGs in: {data_split_dir}\n")

def prepare_validation(val_zip_path, val_dir):
    print(f"📦 Unzipping validation from {val_zip_path} ...")
    os.makedirs(val_dir, exist_ok=True)
    with zipfile.ZipFile(val_zip_path, 'r') as zf:
        zf.extractall(val_dir)

    # Fix structure if zipped into val/extracted_val/
    extracted_subdir = os.path.join(val_dir, "extracted_val")
    if os.path.isdir(extracted_subdir):
        print(f"📂 Moving class dirs up from: {extracted_subdir}")
        for class_dir in os.listdir(extracted_subdir):
            src = os.path.join(extracted_subdir, class_dir)
            dst = os.path.join(val_dir, class_dir)
            if os.path.exists(dst):
                print(f"⚠️ Skipping {class_dir}, already exists.")
                continue
            shutil.move(src, dst)
        shutil.rmtree(extracted_subdir)
        print(f"🧹 Deleted: {extracted_subdir}")

def main():
    parser = argparse.ArgumentParser(description="Render N-ImageNet validation .npz to PNGs (paper style)")
    parser.add_argument("--dataset_root", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.dataset_root)
    val_dir = os.path.join(root, "val")
    val_zip = os.path.join(root, "extracted_val.zip")

    if not os.path.exists(val_zip):
        print(f"❌ Missing val zip: {val_zip}")
        return

    prepare_validation(val_zip, val_dir)
    convert_npz_to_png_in_dir(val_dir)

    print("🎉 Done: Validation PNGs generated in paper format.")

if __name__ == "__main__":
    main()
