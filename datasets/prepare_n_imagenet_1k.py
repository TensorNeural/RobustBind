#!/usr/bin/env python3
import os, argparse, shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_npz(path):
    ev = np.load(path)['event_data']
    if ev['x'].size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return ev['x'], ev['y'], ev['p'], ev['t']

def render_scatter_vis(x, y, p, out_path, size=224):
    if len(x) == 0:
        print(f"⚠️ Skipping empty vis: {out_path}")
        return
    p = ((p + 1) // 2).astype(np.uint8) if p.min() < 0 else (p > 0).astype(np.uint8)
    x = ((x - x.min()) / (x.ptp() + 1e-5) * (size - 1))
    y = ((y - y.min()) / (y.ptp() + 1e-5) * (size - 1))
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.axis('off')
    ax.set_facecolor("black")
    ax.invert_yaxis()
    ax.scatter(x[p == 1], y[p == 1], c='blue', s=1.0, alpha=0.35, edgecolors='none')
    ax.scatter(x[p == 0], y[p == 0], c='red',  s=1.0, alpha=0.35, edgecolors='none')
    plt.savefig(out_path, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

def move_static_dirs(root):
    val_dir = os.path.join(root, "val")
    vis_dir = os.path.join(root, "vis")
    static_val = os.path.join(root, "static", "val")
    static_vis = os.path.join(root, "static", "vis")

    if os.path.exists(val_dir):
        os.makedirs(os.path.dirname(static_val), exist_ok=True)
        shutil.move(val_dir, static_val)
        print(f"✅ Moved existing val/ → static/val/")
    if os.path.exists(vis_dir):
        os.makedirs(os.path.dirname(static_vis), exist_ok=True)
        shutil.move(vis_dir, static_vis)
        print(f"✅ Moved existing vis/ → static/vis/")

def render_all(npz_root, vis_root):
    for cls in sorted(os.listdir(npz_root)):
        cls_dir = os.path.join(npz_root, cls)
        vis_dir = os.path.join(vis_root, cls)
        os.makedirs(vis_dir, exist_ok=True)
        for fname in tqdm(sorted(os.listdir(cls_dir)), desc=cls):
            if not fname.endswith(".npz"): continue
            path = os.path.join(cls_dir, fname)
            x, y, p, t = load_npz(path)
            render_scatter_vis(x, y, p, os.path.join(vis_dir, fname.replace(".npz", "_vis.png")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.dataset_root)
    npz_root = os.path.join(root, "data", "val")
    vis_root = os.path.join(root, "vis")

    move_static_dirs(root)
    render_all(npz_root, vis_root)
    print("🎉 N-ImageNet: scatter-only visualizations complete. Previous outputs saved under static/.")

if __name__ == "__main__":
    main()
