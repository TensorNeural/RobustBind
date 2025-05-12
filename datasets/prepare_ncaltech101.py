#!/usr/bin/env python3
import os, argparse, shutil, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

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

def split_val(data_root, val_root, ratio=0.3):
    total = 0
    for cls in sorted(os.listdir(data_root)):
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir): continue
        files = [f for f in os.listdir(cls_dir) if f.endswith(".bin")]
        random.seed(42)
        random.shuffle(files)
        split = int(len(files) * ratio)
        for fname in files[:split]:
            src = os.path.join(cls_dir, fname)
            dst = os.path.join(val_root, "val", cls, fname)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            total += 1
    print(f"✅ Copied {total} .bin files to val/ (class-balanced)")

def process_bin_file(args):
    path, cls, static_val, static_vis = args
    try:
        x, y, p, t = read_events_from_bin(path)
        base = os.path.basename(path).replace(".bin", "")
        out_val = os.path.join(static_val, cls, base + ".png")
        out_vis = os.path.join(static_vis, cls, base + "_vis.png")
        os.makedirs(os.path.dirname(out_val), exist_ok=True)
        os.makedirs(os.path.dirname(out_vis), exist_ok=True)
        render_model_input(x, y, p, t, out_val)
        render_scatter_vis(x, y, p, out_vis)
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")

def render_all(val_dir, static_val, static_vis, num_workers=12):
    tasks = []
    for cls in sorted(os.listdir(val_dir)):
        cls_dir = os.path.join(val_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith(".bin"):
                path = os.path.join(cls_dir, fname)
                tasks.append((path, cls, static_val, static_vis))
    print(f"🚀 Rendering {len(tasks)} files with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_bin_file, tasks), total=len(tasks)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()

    raw = os.path.join(args.dataset_root, "data")
    val = os.path.join(args.dataset_root, "val")
    static_val = os.path.join(args.dataset_root, "static", "val")
    static_vis = os.path.join(args.dataset_root, "static", "vis")

    split_val(raw, args.dataset_root, ratio=args.val_ratio)
    render_all(val, static_val, static_vis, num_workers=args.num_workers)
    print("🎉 N-Caltech101 static render complete: model inputs + scatter plots.")

if __name__ == "__main__":
    main()
