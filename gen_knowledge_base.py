#!/usr/bin/env python3
import os
import json
import time
import random
import logging
import tempfile
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import numpy as np

from google import genai
from google.genai import types


# ==========================================================
# Logger setup
# ==========================================================

def setup_logger(output_path: Path):
    logger = logging.getLogger(str(output_path))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(output_path)
    file_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(file_handler)
    return logger


# ==========================================================
# Utility
# ==========================================================

def load_input_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def save_output_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved {len(data)} entries to {path}")

def read_file_bytes(file_path: Path):
    with open(file_path, "rb") as f:
        return f.read()

def guess_mime_type(modality: str, file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in (".jpg", ".jpeg"): return "image/jpeg"
    if ext == ".png": return "image/png"
    if ext == ".webp": return "image/webp"
    if ext in (".mp4", ".m4v"): return "video/mp4"
    if ext == ".mov": return "video/mov"
    if ext == ".avi": return "video/avi"
    if ext == ".webm": return "video/webm"
    if ext == ".mp3": return "audio/mp3"
    if ext == ".wav": return "audio/wav"
    if ext == ".flac": return "audio/flac"
    return {
        "image": "image/jpeg",
        "thermal": "image/png",
        "event": "image/png",
        "video": "video/mp4",
        "audio": "audio/mp3",
    }.get(modality, "application/octet-stream")


# ==========================================================
# Event rendering (EventBind-style frames: pos→cyan, neg→yellow)
# ==========================================================

def read_events_from_bin(path: Path):
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

def read_events_from_npz(path: Path):
    data = np.load(str(path))
    if "event_data" in data:
        ev = data["event_data"]
        x = ev["x"].astype(np.uint32)
        y = ev["y"].astype(np.uint32)
        p = (ev["p"] > 0).astype(np.uint8)
        t = ev["t"].astype(np.uint32) if "t" in ev.dtype.names else np.arange(len(x), dtype=np.uint32)
    else:
        x = data["x"].astype(np.uint32)
        y = data["y"].astype(np.uint32)
        p = (data["p"] > 0).astype(np.uint8)
        t = data["t"].astype(np.uint32) if "t" in data.files else np.arange(len(x), dtype=np.uint32)
    return x, y, p, t

def _eventbind_colorize(pos: np.ndarray, neg: np.ndarray, gain: float = 1.5, gamma: float = 0.8) -> np.ndarray:
    """
    EventBind Eq.(8) mapping with optional gain/gamma to improve visibility:
      R = neg, G = pos + neg, B = pos  (scaled 0..255)
    """
    pmax = float(pos.max()) if pos.size and pos.max() > 0 else 1.0
    nmax = float(neg.max()) if neg.size and neg.max() > 0 else 1.0
    p = np.power(np.clip(pos / pmax, 0, 1) * gain, gamma)
    n = np.power(np.clip(neg / nmax, 0, 1) * gain, gamma)

    R = n * 255.0
    G = (p + n).clip(0, 1) * 255.0
    B = p * 255.0
    return np.stack([R, G, B], axis=-1).clip(0, 255).astype(np.uint8)

def _accumulate_histogram(x, y, p, H, W):
    pos = np.zeros((H, W), dtype=np.float32)
    neg = np.zeros((H, W), dtype=np.float32)
    mpos = (p == 1)
    if np.any(mpos):
        np.add.at(pos, (y[mpos], x[mpos]), 1.0)
    if np.any(~mpos):
        np.add.at(neg, (y[~mpos], x[~mpos]), 1.0)
    return pos, neg

def _normalize_coords(x, y, size):
    x = ((x - x.min()) / (x.ptp() + 1e-5) * (size - 1)).astype(np.int32)
    y = ((y - y.min()) / (y.ptp() + 1e-5) * (size - 1)).astype(np.int32)
    return x, y

def render_event_to_frames(event_abs_path: Path, target_size: int = 224, T: int = 8,
                           group_by: str = "time") -> List[Path]:
    """
    Read .bin or .npz and render ALL frames (frame_000.png ..) to temp files.
    - EventBind colorization: pos→cyan, neg→yellow
    - group_by='time': split by normalized timestamps into T equal bins (default)
    Returns list of frame PNG Paths.
    """
    ext = event_abs_path.suffix.lower()
    if ext == ".bin":
        x, y, p, t = read_events_from_bin(event_abs_path)
    elif ext == ".npz":
        x, y, p, t = read_events_from_npz(event_abs_path)
    else:
        return []

    if len(x) == 0:
        # still produce a single empty frame to keep pipeline consistent
        tmp = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
        Image.fromarray(np.zeros((target_size, target_size, 3), dtype=np.uint8)).save(tmp, format="PNG", compress_level=6)
        return [tmp]

    # Sort temporally
    order = np.argsort(t, kind="stable")
    x, y, p, t = x[order], y[order], p[order], t[order]

    # Build at supersampled grid for clarity, then downsample
    supersample = 3
    ssz = target_size * supersample
    x_ss, y_ss = _normalize_coords(x, y, ssz)

    frame_paths: List[Path] = []
    if group_by == "time":
        t_norm = (t - t.min()) / (t.ptp() + 1e-5)
        edges = np.linspace(0.0, 1.0, T + 1, dtype=np.float32)
        fidx = 0
        for i in range(T):
            m = (t_norm >= edges[i]) & (t_norm < edges[i + 1])
            if not np.any(m):
                continue
            pos, neg = _accumulate_histogram(x_ss[m], y_ss[m], p[m], ssz, ssz)
            rgb_hi = _eventbind_colorize(pos, neg, gain=1.5, gamma=0.8)
            img = Image.fromarray(rgb_hi, mode="RGB").resize((target_size, target_size), Image.LANCZOS)

            tmp = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            img.save(tmp, format="PNG", compress_level=6)
            frame_paths.append(tmp)
            fidx += 1
    else:
        # future: count-based grouping can be added if needed
        return render_event_to_frames(event_abs_path, target_size=target_size, T=T, group_by="time")

    # If no non-empty bins (extreme edge case), at least one empty
    if not frame_paths:
        tmp = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
        Image.fromarray(np.zeros((target_size, target_size, 3), dtype=np.uint8)).save(tmp, format="PNG", compress_level=6)
        frame_paths = [tmp]

    return frame_paths


# ==========================================================
# Gemini call with label hint + backoff (DI: client passed in)
# ==========================================================

def _build_event_prompt(label: Optional[str], num_frames: int) -> str:
    label_part = f"This event sample belongs to the class '{label}'. " if label else ""
    # Special prompt explaining rendering so Gemini knows what it's looking at.
    return (
        f"{label_part}"
        "You are given a sequence of event frames rendered from a neuromorphic event stream. "
        "Frames are constructed by partitioning the timeline into equal temporal bins and aggregating per-pixel event counts. "
        "Colorization follows the EventBind mapping: positive events are mapped to cyan (green+blue channels), "
        "negative events to yellow (red+green channels); brighter colors indicate higher event density. "
        f"There are {num_frames} frames in temporal order. "
        "Describe the persistent scene/content succinctly, focusing on stable structure across frames. "
        "Start directly (no 'This image shows...'). Use one short, precise sentence."
    )

def gemini_describe(client: genai.Client,
                    files: Union[Path, List[Path]],
                    modality: str,
                    model: str,
                    logger: logging.Logger,
                    label: Optional[str] = None):
    """
    If `files` is a Path: single-file behavior (old flow).
    If `files` is a list[Path] (event): upload all frames + special prompt.
    """
    is_multi = isinstance(files, list)
    parts = []
    total_bytes = 0

    if is_multi:
        # Build parts for each frame
        for fp in files:
            bs = read_file_bytes(fp)
            total_bytes += len(bs)
            parts.append(types.Part.from_bytes(data=bs, mime_type="image/png"))

        prompt = _build_event_prompt(label, num_frames=len(files))
        contents = parts + [prompt]

    else:
        file_abs_path: Path = files
        file_size = file_abs_path.stat().st_size
        mime_type = guess_mime_type(modality, file_abs_path)

        if file_size <= 20 * 1024 * 1024:
            part = types.Part.from_bytes(data=read_file_bytes(file_abs_path), mime_type=mime_type)
            contents = [part]
        else:
            uploaded = client.files.upload(file=str(file_abs_path))
            contents = [uploaded]

        # Use generic (non-event) prompt wording
        if label:
            prompt = (
                f"This {modality} belongs to the class '{label}'. "
                f"Provide a concise, factual description of the main content in this {modality}. "
                "Start directly without phrases like 'This image shows' or 'This video contains'. "
                "Use one short, precise sentence focused only on relevant details."
            )
        else:
            prompt = (
                f"Provide a concise, factual description of the main content in this {modality}. "
                "Start directly without phrases like 'This image shows' or 'This video contains'. "
                "Use one short, precise sentence focused only on relevant details."
            )
        contents.append(prompt)

    # Call model
    t0_predict = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    predict_latency = time.perf_counter() - t0_predict

    description = (getattr(resp, "text", None) or "").strip()

    # Logging
    if is_multi:
        logger.info(f"OK | {len(files)} event frames | label='{label}' | predict={predict_latency:.3f}s")
    else:
        logger.info(f"OK | {files} | label='{label}' | predict={predict_latency:.3f}s")

    usage_info = {
        "model": model,
        "created_at": datetime.utcnow().isoformat(),
        "predict_latency_sec": round(predict_latency, 4),
        "num_frames": len(files) if is_multi else 1
    }
    return description, usage_info


# ==========================================================
# Processing
# ==========================================================

def partition_indices(n_items: int, num_shards: int, shard_index: int):
    if num_shards <= 1:
        return list(range(n_items))
    return [i for i in range(n_items) if (i % num_shards) == shard_index]

def describe_one(client: genai.Client, entry: Tuple[int, dict], dataset_root: Path,
                 modality: str, model: str, logger: logging.Logger):
    idx, payload = entry
    rel_path = payload["data"]
    label = payload.get("label", "")
    abs_path = dataset_root / rel_path

    # Prepare media
    if modality == "event":
        if not abs_path.exists():
            logger.error(f"Missing event file | {abs_path}")
            return None
        
        frame_paths = render_event_to_frames(abs_path, target_size=224, T=8, group_by="time")
        if not frame_paths:
            logger.error(f"Failed to render event frames | {abs_path}")
            return None

        files_for_model = frame_paths  # include ALL frames

    else:
        # other modalities: just pass through original file
        files_for_model = abs_path
        if not abs_path.exists():
            logger.error(f"Missing media file | {abs_path}")
            return None

    # Inference with retries
    max_tries = 5
    for attempt in range(max_tries):
        try:
            desc_txt, usage = gemini_describe(client, files_for_model, modality, model, logger, label)
            return idx, rel_path, label, desc_txt, usage
        
        except Exception as e:
            logger.warning(f"Retry {attempt+1}/{max_tries} after error: {e}")
            time.sleep((2 ** attempt) + random.uniform(0, 0.5))

    logger.error(f"Max retries exceeded | {abs_path}")
    return None

def process_dataset(client: genai.Client, json_path: Path, dataset_root: Path, modality: str, model: str,
                    limit: Optional[int], num_threads: int, num_shards: int, shard_index: int, shard_dir: Path):
    entries = load_input_json(json_path)
    if limit is not None:
        entries = entries[:limit]

    idxs = partition_indices(len(entries), num_shards, shard_index)
    shard_entries = [(i, entries[i]) for i in idxs]

    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = shard_dir / f"{json_path.parent.name}_{modality}_shard{shard_index}-{num_shards}.log"
    logger = setup_logger(log_path)

    output_rows, meta_rows = [], []
    desc = f"{json_path.parent.name} ({modality}) [shard {shard_index+1}/{num_shards}]"

    with ThreadPoolExecutor(max_workers=num_threads) as ex, \
         tqdm(total=len(shard_entries), desc=desc, unit="file", position=shard_index, leave=True, dynamic_ncols=True) as pbar:
        futures = [ex.submit(describe_one, client, e, dataset_root, modality, model, logger) for e in shard_entries]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                idx, rel_path, label, desc_txt, usage = res
                output_rows.append((idx, {"data": rel_path, "description": desc_txt, "label": label}))
                meta_rows.append((idx, {"data": rel_path, "label": label, "usage": usage}))
            pbar.update(1)

    output_rows.sort(key=lambda x: x[0])
    meta_rows.sort(key=lambda x: x[0])

    save_output_json(shard_dir / f"{json_path.parent.name}_{modality}_align_shard{shard_index}-{num_shards}.json",
                     [row for _, row in output_rows])
    save_output_json(shard_dir / f"{json_path.parent.name}_{modality}_align_meta_shard{shard_index}-{num_shards}.json",
                     [row for _, row in meta_rows])


# ==========================================================
# Merge shards
# ==========================================================

def merge_shards(json_path: Path, modality: str, num_shards: int, output_base: Path):
    stem = f"{json_path.parent.name}_{modality}"
    print(f"[INFO] Merging {num_shards} shards for {stem}...")

    parts, parts_meta = [], []

    for i in range(num_shards):
        shard_dir = output_base / f"{stem}_shard{i}-{num_shards}"
        json_file = shard_dir / f"{stem}_align_shard{i}-{num_shards}.json"
        meta_file = shard_dir / f"{stem}_align_meta_shard{i}-{num_shards}.json"
        if json_file.exists() and meta_file.exists():
            parts.append(load_input_json(json_file))
            parts_meta.append(load_input_json(meta_file))
        else:
            print(f"[WARN] Missing output for shard {i}, skipping merge.")

    if not parts:
        print(f"[ERR] No shard outputs found for {stem}")
        return

    merged = [item for part in parts for item in part]
    merged_meta = [item for part in parts_meta for item in part]

    save_output_json(json_path.parent / f"train_data_align.json", merged)
    save_output_json(json_path.parent / f"train_data_align_meta.json", merged_meta)


# ==========================================================
# Shard runner (per-process DI: create client once here)
# ==========================================================

def run_one_shard(json_path, dataset_path, modality, model, limit, num_threads, total_shards, shard_index, shard_dir):
    # DI: one client per process
    client = genai.Client()
    process_dataset(client, json_path, dataset_path, modality, model, limit, num_threads, total_shards, shard_index, shard_dir)


# ==========================================================
# Main
# ==========================================================

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Generate dataset descriptions with Gemini + label hints.")
    parser.add_argument("--dataset_root", default="/home/user/datasets")
    parser.add_argument("--json_root", default="./datasets")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_modalities", nargs="*", default=[
        "image",
        "audio",
        "thermal",
        "video",
        "event"
        ])
    parser.add_argument("--per_process_threads", type=int, default=1)
    parser.add_argument("--max_cores", type=int, default=None)

    parser.add_argument("--event_frames", type=int, default=8, help="Number of temporal bins for event rendering.")
    args = parser.parse_args()

    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_base = Path("output/knowledge_base") / run_timestamp
    output_base.mkdir(parents=True, exist_ok=True)

    dataset_root_base = Path(args.dataset_root)
    json_root_base = Path(args.json_root)

    mapping = [
        ("image",  "ImageNet-1K"),
        ("audio",  "ESC-50"),
        ("thermal", "LLVIP"),
        ("video",  "MSR-VTT"),
        ("event",  "N-Caltech-101"),
    ]
    mapping = [m for m in mapping if m[0] not in args.skip_modalities]

    if not os.getenv("GEMINI_API_KEY"):
        print("[ERR] GEMINI_API_KEY is not set.")
        return

    total_cores = os.cpu_count() or 1
    if args.max_cores is not None:
        total_cores = max(1, min(total_cores, args.max_cores))

    for modality, dataset_name in mapping:
        json_path = json_root_base / dataset_name / "train_data.json"
        dataset_path = dataset_root_base / dataset_name
        if not json_path.exists():
            print(f"[WARN] Missing JSON: {json_path} — skipping")
            continue
        if not dataset_path.exists():
            print(f"[WARN] Missing dataset root: {dataset_path} — skipping")
            continue

        print(f"[INFO] Processing {dataset_name} ({modality}) with {total_cores} shards, {args.per_process_threads} threads/process")

        procs = []
        try:
            for i in range(total_cores):
                shard_dir = output_base / f"{dataset_name}_{modality}_shard{i}-{total_cores}"
                p = mp.Process(target=run_one_shard,
                               args=(json_path, dataset_path, modality, args.model, args.limit,
                                     args.per_process_threads, total_cores, i, shard_dir))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

        except KeyboardInterrupt:
            print("\n[!] Ctrl-C received — terminating shard processes…")
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()
            break

        merge_shards(json_path, modality, total_cores, output_base)

    print(f"[✓] All datasets processed. Output saved in {output_base}")


if __name__ == "__main__":
    main()
