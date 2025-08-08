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
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# Event rendering
# ==========================================================

def render_event_npz_to_png(event_abs_path: Path) -> Path:
    data = np.load(str(event_abs_path))
    if "event_data" in data:
        ev = data["event_data"]
        xs, ys, ps = ev["x"], ev["y"], ev["p"]
    else:
        xs, ys, ps = data["x"], data["y"], data["p"]

    xs_norm = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-5) * 223
    ys_norm = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-5) * 223

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(xs_norm[ps > 0], ys_norm[ps > 0], c="b", s=0.1)
    ax.scatter(xs_norm[ps <= 0], ys_norm[ps <= 0], c="r", s=0.1)
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 224)
    ax.set_axis_off()
    ax.invert_yaxis()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=100, transparent=True)
    plt.close(fig)
    return Path(tmp.name)


# ==========================================================
# Gemini call with label hint + backoff (DI: client passed in)
# ==========================================================

def gemini_describe(client: genai.Client, file_abs_path: Path, modality: str, model: str,
                    logger: logging.Logger, label: Optional[str] = None):
    if label:
        prompt = (
            f"This {modality} belongs to the class '{label}'. "
            f"Provide a concise, factual description of the main content in this {modality}. "
            "Start directly with the subject matter without phrases like "
            "'This image shows', 'This audio recording features', 'The audio features' or 'The video contains'. "
            "Describe the content with as much detail as possible."
            "Don't include unncessary words or phrases which is unrelated to the content directly"
            "Use 1 short, clear sentence and focus only on relevant details."
        )
    else:
        prompt = (
            f"Provide a concise, factual description of the main content in this {modality}. "
            "Start directly with the subject matter without phrases like "
            "'This image shows', 'This audio recording features', 'The audio features' or 'The video contains'. "
            "Describe the content with as much detail as possible."
            "Don't include unncessary words or phrases which is unrelated to the content directly"
            "Use 1 short, clear sentence and focus only on relevant details."
        )

    file_size = file_abs_path.stat().st_size
    mime_type = guess_mime_type(modality, file_abs_path)

    t0_upload = time.perf_counter()
    if file_size <= 20 * 1024 * 1024:
        part = types.Part.from_bytes(data=read_file_bytes(file_abs_path), mime_type=mime_type)
        contents = [part, prompt]
    else:
        uploaded = client.files.upload(file=str(file_abs_path))
        contents = [uploaded, prompt]
    upload_latency = time.perf_counter() - t0_upload

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

    logger.info(f"OK | {file_abs_path} | label='{label}' | upload={upload_latency:.3f}s | predict={predict_latency:.3f}s")

    usage_info = {
        "model": model,
        "created_at": datetime.utcnow().isoformat(),
        "upload_latency_sec": round(upload_latency, 4),
        "predict_latency_sec": round(predict_latency, 4)
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

    working_path = abs_path
    if modality == "event" and abs_path.suffix.lower() == ".npz":
        if abs_path.exists():
            working_path = render_event_npz_to_png(abs_path)
        else:
            logger.error(f"Missing event file | {abs_path}")
            return None

    if not working_path.exists():
        logger.error(f"Missing media file | {working_path}")
        return None

    max_tries = 5
    for attempt in range(max_tries):
        try:
            return idx, rel_path, label, *gemini_describe(client, working_path, modality, model, logger, label)
        except Exception as e:
            logger.warning(f"Retry {attempt+1}/{max_tries} after error: {e}")
            time.sleep((2 ** attempt) + random.uniform(0, 0.5))

    logger.error(f"Max retries exceeded | {working_path}")
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

    save_output_json(output_base / f"{stem}_align.json", merged)
    save_output_json(output_base / f"{stem}_align_meta.json", merged_meta)


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
        # "image", 
        # "audio",
        # "thermal", 
        # "video", 
        "event"
        ])
    parser.add_argument("--per_process_threads", type=int, default=1)
    parser.add_argument("--max_cores", type=int, default=None)
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
        ("event",  "N-ImageNet-1K"),
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
