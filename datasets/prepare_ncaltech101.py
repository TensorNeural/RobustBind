#!/usr/bin/env python3
import os, argparse, shutil, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_events_from_bin(path):
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 5)
    if raw.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    x, y = raw[:, 0].astype(np.uint32), raw[:, 1].astype(np.uint32)
    p = (raw[:, 2] >> 7) & 1
    t = ((raw[:, 2] & 0x7F) << 16) | (raw[:, 3] << 8) | raw[:, 4]
    return x, y, p, t

def render_model_input(x, y, p, t, out_path, size=224):
    if len(x) == 0:
        print(f"⚠️ Skipping empty model input: {out_path}")
        return
    H, W = size, size
    pos, neg, ts = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    x = ((x - x.min()) / (x.ptp() + 1e-5) * (W - 1)).astype(int)
    y = ((y - y.min()) / (y.ptp() + 1e-5) * (H - 1)).astype(int)
    t = (t - t.min()) / (t.ptp() + 1e-5)
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
    p = (p > 0).astype(np.uint8)
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

def split_val(data_root, out_root, ratio=0.3):
    all_paths = []
    for cls in sorted(os.listdir(data_root)):
        for f in os.listdir(os.path.join(data_root, cls)):
            if f.endswith(".bin"):
                all_paths.append(os.path.join(cls, f))
    random.seed(42)
    random.shuffle(all_paths)
    split = int(len(all_paths) * ratio)
    val_paths = all_paths[:split]
    for rel in val_paths:
        src = os.path.join(data_root, rel)
        dst = os.path.join(out_root, "val", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    print(f"✅ Copied {len(val_paths)} val files.")

def render_all(val_dir, vis_dir):
    for cls in sorted(os.listdir(val_dir)):
        cls_dir = os.path.join(val_dir, cls)
        vis_cls = os.path.join(vis_dir, cls)
        os.makedirs(vis_cls, exist_ok=True)
        for fname in tqdm(os.listdir(cls_dir), desc=cls):
            if not fname.endswith(".bin"): continue
            path = os.path.join(cls_dir, fname)
            x, y, p, t = read_events_from_bin(path)
            render_model_input(x, y, p, t, path.replace(".bin", ".png"))
            render_scatter_vis(x, y, p, os.path.join(vis_cls, fname.replace(".bin", "_vis.png")))
            try: os.remove(path)
            except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    args = parser.parse_args()

    raw = os.path.join(args.dataset_root, "data")
    val = os.path.join(args.dataset_root, "val")
    vis = os.path.join(args.dataset_root, "vis")

    split_val(raw, args.dataset_root, ratio=args.val_ratio)
    render_all(val, vis)
    print("🎉 N-Caltech101: model + vis rendering complete.")

if __name__ == "__main__":
    main()
