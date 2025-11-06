#!/usr/bin/env python3
# File: render_events_static.py
"""
Render .bin event files using either:
- 'eventbind' (multi-frame, cyan/yellow Eq. 8)
- 'scatter'   (single PNG) — EXACT match to Script 1's scatter behavior.
"""

import os
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import random

# ---------------------------
# Reader (.bin only)
# ---------------------------

def read_events_from_bin(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        buf = f.read()
    n = len(buf) // 5
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    raw = np.frombuffer(buf, dtype=np.uint8, count=n * 5).reshape(-1, 5)
    x = raw[:, 0].astype(np.uint32)
    y = raw[:, 1].astype(np.uint32)
    p = ((raw[:, 2] >> 7) & 1).astype(np.uint8)
    t = (((raw[:, 2] & 0x7F).astype(np.uint32) << 16) |
         (raw[:, 3].astype(np.uint32) << 8) |
          raw[:, 4].astype(np.uint32))
    return x, y, p, t

# ---------------------------
# EventBind colorization
# ---------------------------

def _eventbind_colorize(pos: np.ndarray, neg: np.ndarray, gain: float = 2.0, gamma: float = 0.7) -> np.ndarray:
    pmax = float(pos.max()) if pos.size and pos.max() > 0 else 1.0
    nmax = float(neg.max()) if neg.size and neg.max() > 0 else 1.0
    p = np.power(np.clip(pos / pmax, 0, 1) * gain, gamma)
    n = np.power(np.clip(neg / nmax, 0, 1) * gain, gamma)
    R = n * 255.0
    G = (p + n).clip(0, 1) * 255.0
    B = p * 255.0
    return np.stack([R, G, B], axis=-1).clip(0, 255).astype(np.uint8)

# ---------------------------
# EVENTBIND (multi-frame)
# ---------------------------

def render_eventbind_images(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    out_dir: str,
    size: int = 224,
    T: int = 8,
    group_by: str = "time",
    total_events: Optional[int] = None,
    events_per_frame: Optional[int] = None,
    gain: float = 2.0,
    gamma: float = 0.7,
):
    if len(x) == 0:
        return

    # Normalize coordinates into [0, size-1]
    x = ((x - x.min()) / (x.ptp() + 1e-5) * (size - 1)).astype(np.int32)
    y = ((y - y.min()) / (y.ptp() + 1e-5) * (size - 1)).astype(np.int32)

    # Sort by time
    order = np.argsort(t, kind="stable")
    x, y, p, t = x[order], y[order], p[order], t[order]

    os.makedirs(out_dir, exist_ok=True)

    if group_by == "count":
        N = len(x)
        P = min(total_events, N) if total_events is not None else N
        Q = max(1, P // max(1, T)) if not events_per_frame else events_per_frame
        T_eff = max(1, P // Q)
        N_used = T_eff * Q
        x, y, p = x[:N_used], y[:N_used], p[:N_used]

        for i in range(T_eff):
            s, e = i * Q, (i + 1) * Q
            pos = np.zeros((size, size), dtype=np.float32)
            neg = np.zeros((size, size), dtype=np.float32)
            mpos = (p[s:e] == 1)
            if np.any(mpos):
                np.add.at(pos, (y[s:e][mpos], x[s:e][mpos]), 1.0)
            if np.any(~mpos):
                np.add.at(neg, (y[s:e][~mpos], x[s:e][~mpos]), 1.0)
            rgb = _eventbind_colorize(pos, neg, gain=gain, gamma=gamma)
            Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{i:03d}.png"), format="PNG", compress_level=6)
    else:
        t_norm = (t - t.min()) / (t.ptp() + 1e-5)
        edges = np.linspace(0.0, 1.0, T + 1, dtype=np.float32)
        fidx = 0
        for i in range(T):
            m = (t_norm >= edges[i]) & (t_norm < edges[i + 1])
            if not np.any(m):
                continue
            pos = np.zeros((size, size), dtype=np.float32)
            neg = np.zeros((size, size), dtype=np.float32)
            mpos = (p[m] == 1)
            if np.any(mpos):
                np.add.at(pos, (y[m][mpos], x[m][mpos]), 1.0)
            if np.any(~mpos):
                np.add.at(neg, (y[m][~mpos], x[m][~mpos]), 1.0)
            rgb = _eventbind_colorize(pos, neg, gain=gain, gamma=gamma)
            Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{fidx:03d}.png"), format="PNG", compress_level=6)
            fidx += 1

# ---------------------------
# SCATTER (EXACT match to Script 1)
# ---------------------------

def render_scatter_image_exact_like_script1(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    out_path: str,
) -> bool:
    """
    EXACT rendering parity with the original Script 1 scatter:
      - Scale to 0..223 (hardcoded)
      - Figure size 2.24 x 2.24 at dpi=100 (→ 224x224)
      - Two scatters: pos->blue, neg->red, s=0.1
      - Axis off, inverted Y
      - xlim/ylim 0..224
      - savefig(..., dpi=100, transparent=True)
      - try/except returns True/False
    """
    try:
        if x.size == 0:
            return False

        xs = x.astype(np.float32)
        ys = y.astype(np.float32)
        pos = (p > 0)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        xs_norm = (xs - x_min) / max(x_max - x_min, 1e-5) * 223.0
        ys_norm = (ys - y_min) / max(y_max - y_min, 1e-5) * 223.0

        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(xs_norm[pos],  ys_norm[pos],  c='b', s=0.1)
        ax.scatter(xs_norm[~pos], ys_norm[~pos], c='r', s=0.1)
        ax.set_xlim(0, 224)
        ax.set_ylim(0, 224)
        ax.set_axis_off()
        ax.invert_yaxis()
        plt.savefig(out_path, dpi=100, transparent=True)
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Failed to render (scatter) {out_path}: {e}")
        return False

# ---------------------------
# Workers / orchestration
# ---------------------------

def process_event_file(args):
    (path, cls, static_root, size, T, skip_existing,
     group_by, total_events, events_per_frame,
     method, gain, gamma) = args

    base, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() != ".bin":
        return  # Only .bin supported

    if method == "scatter":
        out_path = os.path.join(static_root, cls, base + ".png")
        if skip_existing and os.path.exists(out_path):
            return
    else:
        out_path = os.path.join(static_root, cls, base)
        if skip_existing and os.path.isdir(out_path) and any(n.startswith("frame_") for n in os.listdir(out_path)):
            return

    x, y, p, t = read_events_from_bin(path)

    if method == "scatter":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        _ = render_scatter_image_exact_like_script1(x, y, p, out_path)
    else:
        os.makedirs(out_path, exist_ok=True)
        render_eventbind_images(
            x, y, p, t, out_path,
            size=size, T=T,
            group_by=group_by, total_events=total_events, events_per_frame=events_per_frame,
            gain=gain, gamma=gamma
        )

def render_all(split_dir, static_root, size=224, T=8, num_workers=12, skip_existing=True,
               group_by="time", total_events=None, events_per_frame=None,
               method="eventbind", gain=2.0, gamma=0.7):
    tasks = []
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.endswith(".bin"):  # Only .bin files
                tasks.append((
                    os.path.join(cls_dir, fname), cls, static_root, size, T,
                    skip_existing, group_by, total_events, events_per_frame,
                    method, gain, gamma
                ))
    if not tasks:
        return

    chunksize = max(1, len(tasks) // (num_workers * 8) if num_workers > 0 else 1)

    if num_workers <= 1:
        for _ in tqdm(map(process_event_file, tasks), total=len(tasks)):
            pass
    else:
        with mp.Pool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(process_event_file, tasks, chunksize=chunksize), total=len(tasks)):
                pass

# ---------------------------
# CLI
# ---------------------------

def _is_class_dir(path: str) -> bool:
    try:
        if not os.path.isdir(path):
            return False
        for fn in os.listdir(path):
            if fn.lower().endswith('.bin'):
                return True
        return False
    except Exception:
        return False


def _discover_classes_root(dataset_root: str) -> Optional[str]:
    # Prefer dataset_root directly if it contains class dirs
    candidates = [d for d in os.listdir(dataset_root)
                  if os.path.isdir(os.path.join(dataset_root, d))
                  and d not in ("train", "val", "static")]
    class_like = [os.path.join(dataset_root, d) for d in candidates if _is_class_dir(os.path.join(dataset_root, d))]
    if class_like:
        return dataset_root
    # Else, if there is exactly one subdir, try inside it
    if len(candidates) == 1:
        inner = os.path.join(dataset_root, candidates[0])
        inner_cands = [d for d in os.listdir(inner) if os.path.isdir(os.path.join(inner, d))]
        inner_class_like = [os.path.join(inner, d) for d in inner_cands if _is_class_dir(os.path.join(inner, d))]
        if inner_class_like:
            return inner
    return None


def ensure_train_val_split(dataset_root: str, train_ratio: float = 0.8, seed: int = 42) -> None:
    """Create train/ and val/ splits if missing by moving .bin files from class dirs.
    - Detect class directories either directly under dataset_root or one level deeper.
    - Uses per-class 80/20 random split (seeded).
    - Idempotent: if train/ and val/ already exist, does nothing.
    """
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        return

    classes_root = _discover_classes_root(dataset_root)
    if not classes_root:
        print(f"[N-Caltech-101] Could not find class directories under {dataset_root}; skipping split.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    rng = random.Random(seed)
    moved_any = False
    for cls in sorted(os.listdir(classes_root)):
        cls_path = os.path.join(classes_root, cls)
        if not _is_class_dir(cls_path):
            continue
        files = [f for f in os.listdir(cls_path) if f.lower().endswith('.bin')]
        if not files:
            continue
        rng.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        out_train = os.path.join(train_dir, cls)
        out_val = os.path.join(val_dir, cls)
        os.makedirs(out_train, exist_ok=True)
        os.makedirs(out_val, exist_ok=True)

        # Move files into split folders
        for fn in train_files:
            src = os.path.join(cls_path, fn)
            dst = os.path.join(out_train, fn)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved_any = True
        for fn in val_files:
            src = os.path.join(cls_path, fn)
            dst = os.path.join(out_val, fn)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved_any = True

        # Clean up empty class dir
        try:
            if not any(os.scandir(cls_path)):
                os.rmdir(cls_path)
        except Exception:
            pass

    if moved_any:
        print(f"[N-Caltech-101] Created train/val split at {dataset_root} with ratio {train_ratio:.2f}.")
    else:
        print(f"[N-Caltech-101] No .bin files moved; train/val may already exist or dataset is empty.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/data/datasets/N-Caltech-101")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--frame_size", type=int, default=224)  # used by eventbind only
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_seed", type=int, default=42)

    # Method
    parser.add_argument("--method", choices=["eventbind", "scatter"], default="scatter")

    # EventBind grouping controls
    parser.add_argument("--group_by", choices=["time", "count"], default="time")
    parser.add_argument("--total_events", type=int, default=None)
    parser.add_argument("--events_per_frame", type=int, default=None)
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.7)

    args = parser.parse_args()

    # Create train/val split if missing
    ensure_train_val_split(args.dataset_root, train_ratio=args.train_ratio, seed=args.split_seed)

    # Train
    static_train = os.path.join(args.dataset_root, "static", "train")
    render_all(os.path.join(args.dataset_root, "train"), static_train,
               size=args.frame_size, T=args.num_frames,
               num_workers=args.num_workers, skip_existing=args.skip_existing,
               group_by=args.group_by, total_events=args.total_events,
               events_per_frame=args.events_per_frame,
               method=args.method, gain=args.gain, gamma=args.gamma)

    # Val
    static_val = os.path.join(args.dataset_root, "static", "val")
    render_all(os.path.join(args.dataset_root, "val"), static_val,
               size=args.frame_size, T=args.num_frames,
               num_workers=args.num_workers, skip_existing=args.skip_existing,
               group_by=args.group_by, total_events=args.total_events,
               events_per_frame=args.events_per_frame,
               method=args.method, gain=args.gain, gamma=args.gamma)

if __name__ == "__main__":
    main()
