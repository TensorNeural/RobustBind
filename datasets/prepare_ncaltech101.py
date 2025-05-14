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

def render_eventbind_images(x, y, p, t, out_dir, size=224, T=8):
    if len(x) == 0:
        print(f"⚠️ Skipping empty input: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Normalize coordinates and time
    x = ((x - x.min()) / (x.ptp() + 1e-5) * (size - 1)).astype(int)
    y = ((y - y.min()) / (y.ptp() + 1e-5) * (size - 1)).astype(int)
    t_norm = (t - t.min()) / (t.ptp() + 1e-5)

    bins = np.linspace(0, 1, T + 1)
    for i in range(T):
        idx = (t_norm >= bins[i]) & (t_norm < bins[i + 1])
        if idx.sum() == 0:
            continue
        xx, yy, pp, tt = x[idx], y[idx], p[idx], t_norm[idx]

        pos = np.zeros((size, size))
        neg = np.zeros((size, size))
        ts = np.zeros((size, size))

        for xi, yi, pi, ti in zip(xx, yy, pp, tt):
            if pi:
                pos[yi, xi] += 1
            else:
                neg[yi, xi] += 1
            ts[yi, xi] = max(ts[yi, xi], ti)

        # Color channels
        red   = (255 * pos / (pos.max() + 1e-5)).astype(np.uint8)
        green = (255 * ts).astype(np.uint8)
        blue  = (255 * neg / (neg.max() + 1e-5)).astype(np.uint8)

        rgb = np.stack([red, green, blue], axis=-1)
        frame_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        plt.imsave(frame_path, rgb)

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
    path, cls, static_val, size, T = args
    try:
        x, y, p, t = read_events_from_bin(path)
        base = os.path.basename(path).replace(".bin", "")
        out_dir = os.path.join(static_val, cls, base)
        render_eventbind_images(x, y, p, t, out_dir, size=size, T=T)
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")

def render_all(val_dir, static_val, size=224, T=8, num_workers=12):
    tasks = []
    for cls in sorted(os.listdir(val_dir)):
        cls_dir = os.path.join(val_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith(".bin"):
                path = os.path.join(cls_dir, fname)
                tasks.append((path, cls, static_val, size, T))
    print(f"🚀 Rendering {len(tasks)} samples into EventBind-style RGB images with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_bin_file, tasks), total=len(tasks)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    raw = os.path.join(args.dataset_root, "data")
    val = os.path.join(args.dataset_root, "val")
    static_val = os.path.join(args.dataset_root, "static", "val")

    split_val(raw, args.dataset_root, ratio=args.val_ratio)
    render_all(val, static_val, size=args.frame_size, T=args.num_frames, num_workers=args.num_workers)
    print("🎉 EventBind-style RGB rendering complete.")

if __name__ == "__main__":
    main()
