#!/usr/bin/env python3
import os, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_npz(path):
    ev = np.load(path)['event_data']
    if ev['x'].size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return ev['x'], ev['y'], ev['p'], ev['t']

def render_model_input(x, y, p, t, out_path, size=224):
    if len(x) == 0:
        print(f"⚠️ Skipping empty model input: {out_path}")
        return
    H, W = size, size
    pos, neg, ts = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    x = ((x - x.min()) / (x.ptp() + 1e-5) * (W - 1)).astype(int)
    y = ((y - y.min()) / (y.ptp() + 1e-5) * (H - 1)).astype(int)
    t = (t - t.min()) / (t.ptp() + 1e-5)
    p = ((p + 1) // 2).astype(np.uint8) if p.min() < 0 else p
    for xi, yi, pi, ti in zip(x, y, p, t):
        if pi: pos[yi, xi] += 1
        else: neg[yi, xi] += 1
        ts[yi, xi] = max(ts[yi, xi], ti)
    pos_img = 255 * np.log1p(pos) / np.log1p(pos.max()) if pos.max() > 0 else pos
    neg_img = 255 * np.log1p(neg) / np.log1p(neg.max()) if neg.max() > 0 else neg
    ts_img = (255 * ts).astype(np.uint8)
    ts_img[(pos + neg) == 0] = 0
    rgb = np.stack([pos_img, neg_img, ts_img], axis=-1).astype(np.uint8)
    plt.imsave(out_path, rgb)

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

def render_all(npz_root, val_root, vis_root):
    for cls in sorted(os.listdir(npz_root)):
        cls_dir = os.path.join(npz_root, cls)
        val_dir = os.path.join(val_root, cls)
        vis_dir = os.path.join(vis_root, cls)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith(".npz"): continue
            npz_path = os.path.join(cls_dir, fname)
            x, y, p, t = load_npz(npz_path)
            render_model_input(x, y, p, t, os.path.join(val_dir, fname.replace(".npz", ".png")))
            render_scatter_vis(x, y, p, os.path.join(vis_dir, fname.replace(".npz", "_vis.png")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.dataset_root)
    npz_root = os.path.join(root, "data", "val")
    val_root = os.path.join(root, "val")
    vis_root = os.path.join(root, "vis")
    render_all(npz_root, val_root, vis_root)
    print("🎉 N-ImageNet: model + vis rendering complete.")

if __name__ == "__main__":
    main()
