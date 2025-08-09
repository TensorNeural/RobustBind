#!/usr/bin/env python3
# File: prepare_n_imagenet_pngs.py

import os
import zipfile
import tarfile
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
from PIL import Image

# =========================
# Archive helpers (unchanged)
# =========================

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

# =========================
# Scatter rendering (original Script 1)
# =========================

def _read_npz(path):
    ev = np.load(path)
    if "event_data" in ev:
        xs, ys, ps = ev["event_data"]["x"], ev["event_data"]["y"], ev["event_data"]["p"]
    else:
        xs, ys, ps = ev["x"], ev["y"], ev["p"]
    return xs.astype(np.float32), ys.astype(np.float32), (ps > 0)

def plot_event_image_scatter(npz_path, output_path):
    try:
        xs, ys, pos = _read_npz(npz_path)
        if xs.size == 0:
            return False

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
        plt.savefig(output_path, dpi=100, transparent=True)
        plt.close()
        return True
    except Exception as e:
        tqdm.write(f"❌ Failed to render scatter {npz_path}: {e}")
        return False

# =========================
# EventBind rendering
# =========================

def _eventbind_colorize(pos, neg, gain=2.0, gamma=0.7):
    pmax = float(pos.max()) if pos.size and pos.max() > 0 else 1.0
    nmax = float(neg.max()) if neg.size and neg.max() > 0 else 1.0
    p = np.power(np.clip(pos / pmax, 0, 1) * gain, gamma)
    n = np.power(np.clip(neg / nmax, 0, 1) * gain, gamma)
    R = n * 255.0
    G = (p + n).clip(0, 1) * 255.0
    B = p * 255.0
    return np.stack([R, G, B], axis=-1).clip(0, 255).astype(np.uint8)

def render_eventbind_frames_from_npz(npz_path, out_dir, size=224, T=8,
                                     group_by="time", total_events=None, events_per_frame=None,
                                     gain=2.0, gamma=0.7):
    xs, ys, pos = _read_npz(npz_path)
    if xs.size == 0:
        return

    t = np.arange(xs.shape[0], dtype=np.float32)
    xs = ((xs - xs.min()) / (xs.ptp() + 1e-5) * (size - 1)).astype(np.int32)
    ys = ((ys - ys.min()) / (ys.ptp() + 1e-5) * (size - 1)).astype(np.int32)

    os.makedirs(out_dir, exist_ok=True)

    if group_by == "count":
        N = xs.shape[0]
        P = min(total_events, N) if total_events is not None else N
        Q = max(1, P // max(1, T)) if not events_per_frame else events_per_frame
        T_eff = max(1, P // Q)
        N_used = T_eff * Q
        xs, ys, pos = xs[:N_used], ys[:N_used], pos[:N_used]

        for i in range(T_eff):
            s, e = i * Q, (i + 1) * Q
            pmap = np.zeros((size, size), dtype=np.float32)
            nmap = np.zeros((size, size), dtype=np.float32)
            mpos = pos[s:e]
            if np.any(mpos):
                np.add.at(pmap, (ys[s:e][mpos], xs[s:e][mpos]), 1.0)
            if np.any(~mpos):
                np.add.at(nmap, (ys[s:e][~mpos], xs[s:e][~mpos]), 1.0)
            rgb = _eventbind_colorize(pmap, nmap, gain=gain, gamma=gamma)
            Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{i:03d}.png"), format="PNG")
    else:
        t_norm = (t - t.min()) / (t.ptp() + 1e-5)
        edges = np.linspace(0.0, 1.0, T + 1)
        fidx = 0
        for i in range(T):
            m = (t_norm >= edges[i]) & (t_norm < edges[i + 1])
            if not np.any(m):
                continue
            pmap = np.zeros((size, size), dtype=np.float32)
            nmap = np.zeros((size, size), dtype=np.float32)
            if np.any(pos[m]):
                np.add.at(pmap, (ys[m][pos[m]], xs[m][pos[m]]), 1.0)
            negm = ~pos[m]
            if np.any(negm):
                np.add.at(nmap, (ys[m][negm], xs[m][negm]), 1.0)
            rgb = _eventbind_colorize(pmap, nmap, gain=gain, gamma=gamma)
            Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{fidx:03d}.png"), format="PNG")
            fidx += 1

# =========================
# Directory traversal + PROGRESS
# =========================

def _gather_npz(data_split_dir):
    """Return list of (class_name, npz_abs_path)."""
    items = []
    for class_name in sorted(os.listdir(data_split_dir)):
        class_dir = os.path.join(data_split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in sorted(os.listdir(class_dir)):
            if f.endswith(".npz"):
                items.append((class_name, os.path.join(class_dir, f)))
    return items

def convert_npz_to_outputs(data_split_dir, vis_dir, val_dir,
                           scatter_size=224, eventbind_size=224, T=8,
                           group_by="time", total_events=None, events_per_frame=None,
                           gain=2.0, gamma=0.7):
    print(f"🎨 Processing .npz in: {data_split_dir}")

    items = _gather_npz(data_split_dir)
    if not items:
        print("⚠️ No .npz files found.")
        return

    ok_scatter = ok_eventbind = 0
    with tqdm(total=len(items), desc="Rendering (scatter + EventBind)", unit="file") as pbar:
        for class_name, npz_path in items:
            base = os.path.splitext(os.path.basename(npz_path))[0]

            # Scatter → static/vis
            scatter_out = os.path.join(vis_dir, class_name, base + ".png")
            os.makedirs(os.path.dirname(scatter_out), exist_ok=True)
            ok_scatter += 1 if plot_event_image_scatter(npz_path, scatter_out) else 0

            # EventBind → static/val
            eb_out_dir = os.path.join(val_dir, class_name, base)
            try:
                render_eventbind_frames_from_npz(
                    npz_path, eb_out_dir,
                    size=eventbind_size, T=T,
                    group_by=group_by, total_events=total_events, events_per_frame=events_per_frame,
                    gain=gain, gamma=gamma
                )
                ok_eventbind += 1
            except Exception as e:
                tqdm.write(f"❌ Failed to render EventBind {npz_path}: {e}")

            pbar.set_postfix(scatter=ok_scatter, eventbind=ok_eventbind)
            pbar.update(1)

    print(f"✅ Finished rendering under vis={vis_dir} and val={val_dir}")
    print(f"   Scatter OK:    {ok_scatter}/{len(items)}")
    print(f"   EventBind OK:  {ok_eventbind}/{len(items)}\n")

# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Prepare N-ImageNet-1K static/vis (scatter) and static/val (EventBind)")
    parser.add_argument("--dataset_root", default="/home/user/datasets/N-ImageNet-1K", type=str)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--group_by", choices=["time", "count"], default="time")
    parser.add_argument("--total_events", type=int, default=None)
    parser.add_argument("--events_per_frame", type=int, default=None)
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    args = parser.parse_args()

    root = os.path.abspath(args.dataset_root)
    val_dir_in = os.path.join(root, "val")
    static_vis = os.path.join(root, "static", "vis")
    static_val = os.path.join(root, "static", "val")

    if os.path.isdir(val_dir_in):
        convert_npz_to_outputs(
            val_dir_in,
            vis_dir=static_vis,
            val_dir=static_val,
            scatter_size=args.frame_size,
            eventbind_size=args.frame_size,
            T=args.num_frames,
            group_by=args.group_by,
            total_events=args.total_events,
            events_per_frame=args.events_per_frame,
            gain=args.gain,
            gamma=args.gamma
        )
    else:
        print(f"⚠️ Split missing: {val_dir_in}")

    print("🎉 Done: static/vis (scatter) and static/val (EventBind) created!")

if __name__ == "__main__":
    main()
