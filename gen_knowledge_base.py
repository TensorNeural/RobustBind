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
import re

from model import MODALITY_TEMPLATES
from shared_types import Modality
matplotlib.use("Agg")
from PIL import Image
import numpy as np
import shutil

from google import genai
from google.genai import types


# Shared mapping for modality detection
MODALITY_MAPPING = {
    "ImageNet-1K": "image",
    "ESC-50": "audio",
    "LLVIP": "thermal",
    "MSR-VTT": "video",
    "N-Caltech-101": "event",
}


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
    if ext == ".mov": return "video/quicktime"
    if ext == ".avi": return "video/avi"
    if ext == ".webm": return "video/webm"
    if ext == ".mp3": return "audio/mpeg"
    if ext == ".wav": return "audio/wav"
    if ext == ".flac": return "audio/flac"
    return {
        "image": "image/jpeg",
        "thermal": "image/png",
        "event": "image/png",
        "video": "video/mp4",
        "audio": "audio/mpeg",
    }.get(modality, "application/octet-stream")

def _safe_template_for_modality(modality: str) -> str:
    """Return a fallback template safely even if Modality(modality) fails."""
    try:
        return MODALITY_TEMPLATES.get(Modality(modality), "a {}")
    except Exception:
        return "a {}"

def _cleanup_temp_paths(files: Union[Path, List[Path]]):
    """Remove temporary files generated for event frames. No-op for single Path."""
    try:
        if isinstance(files, list):
            for fp in files:
                try:
                    if isinstance(fp, Path) and fp.exists():
                        fp.unlink()
                except Exception:
                    pass
    except Exception:
        # Best-effort cleanup only
        pass

def sanitize_for_prompt(text: Optional[str]) -> str:
    """Sanitize text to avoid special characters or regex-like symbols in model prompts.
    - Keep ASCII letters, digits, space, period, comma, and hyphen only
    - Drop other characters and collapse spaces
    """
    if not text:
        return ""
    # Remove non-ASCII
    s = str(text).encode("ascii", "ignore").decode()
    # Keep only allowed characters
    s = re.sub(r"[^A-Za-z0-9 \.,\-]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_label(label: Optional[str]) -> str:
    """Normalize raw label strings for consistent appearance in descriptions.
    Heuristics:
    - Remove WordNet-style synset ids (e.g., 'n01234567')
    - Replace `_`, `/`, `-` with spaces
    - If multiple parts (commas/semicolons), prefer the first common-name part
    - Remove parenthetical scientific names or qualifiers
    - Collapse spaces; Title Case for readability
    """
    if not label:
        return ""
    s = str(label).strip()
    # Drop WordNet synset prefix if present
    s = re.sub(r"^n\d{8}[ _-]*", "", s)
    # Replace common separators with spaces
    s = s.replace("_", " ").replace("/", " ").replace("-", " ")
    # Prefer the first part before comma or semicolon
    s = re.split(r"[;,]", s, maxsplit=1)[0]
    # Remove any parenthetical content
    s = re.sub(r"\([^)]*\)", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Title Case
    s = s.title()
    return s

def ensure_label_in_description(text: str, norm_label: str) -> str:
    """Ensure the normalized label is included and prefixed (no colon).
    - If text begins with '<Label>:' or '<Label> -/–', normalize to '<Label> '.
    - If text already begins with '<Label> ' (case-insensitive), keep it.
    - Else, prefix '<Label> ' before the text.
    """
    if not norm_label:
        return text
    if not text:
        return norm_label
    # Normalize any colon/dash after the label to a single space (case-insensitive)
    pattern = r"^" + re.escape(norm_label) + r"\s*[:\-–]\s*"
    text_norm = re.sub(pattern, f"{norm_label} ", text, count=1, flags=re.IGNORECASE)
    # If already starts with label + space, keep
    if text_norm.lower().startswith(norm_label.lower() + " "):
        return text_norm
    # Otherwise, prefix label and a space
    return f"{norm_label} {text_norm}"

def prepare_files_for_model(entry: dict, dataset_root: Path, modality: str, logger: logging.Logger, event_frames: int):
    rel_path = entry["data"]
    abs_path = dataset_root / rel_path

    if modality == "event":
        if not abs_path.exists():
            logger.error(f"Missing event file | {abs_path}")
            return None

        frame_paths = render_event_to_frames(abs_path, target_size=224, T=event_frames, group_by="time")
        if not frame_paths:
            logger.error(f"Failed to render event frames | {abs_path}")
            return None

        return frame_paths  # include ALL frames

    else:
        if not abs_path.exists():
            logger.error(f"Missing media file | {abs_path}")
            return None

        return abs_path

# ==========================================================
# Event rendering (EventBind-style frames: pos→cyan, neg→yellow)
# ==========================================================

def read_events_from_bin(file_path: Path):
    with open(file_path, "rb") as file:
        buffer = file.read()

    num_events = len(buffer) // 5
    if num_events == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    raw_data = np.frombuffer(buffer, dtype=np.uint8, count=num_events * 5).reshape(-1, 5)
    x_coords = raw_data[:, 0].astype(np.uint32)
    y_coords = raw_data[:, 1].astype(np.uint32)
    polarities = ((raw_data[:, 2] >> 7) & 1).astype(np.uint8)
    timestamps = (((raw_data[:, 2] & 0x7F).astype(np.uint32) << 16) |
                  (raw_data[:, 3].astype(np.uint32) << 8) |
                   raw_data[:, 4].astype(np.uint32))
    return x_coords, y_coords, polarities, timestamps

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

def _accumulate_histogram(x_coords, y_coords, polarities, height, width):
    positive_events = np.zeros((height, width), dtype=np.float32)
    negative_events = np.zeros((height, width), dtype=np.float32)
    positive_mask = (polarities == 1)
    if np.any(positive_mask):
        np.add.at(positive_events, (y_coords[positive_mask], x_coords[positive_mask]), 1.0)
    if np.any(~positive_mask):
        np.add.at(negative_events, (y_coords[~positive_mask], x_coords[~positive_mask]), 1.0)
    return positive_events, negative_events

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
    norm_label = normalize_label(label)
    full_label_note = f" Full label (verbatim): '{label}'." if label else ""
    norm_label_note = f" Normalized label: '{norm_label}'." if norm_label else ""
    label_part = f"This event sample belongs to the class '{norm_label or label}'." if (label or norm_label) else ""
    # Keep explanation brief; provide BOTH full and normalized labels for disambiguation; require starting with normalized label.
    # Build using sanitized components and ASCII-only wording
    safe_label_part = sanitize_for_prompt(label_part)
    safe_full_note = sanitize_for_prompt(full_label_note)
    safe_norm_note = sanitize_for_prompt(norm_label_note)
    preface = (safe_label_part + " " + safe_full_note + " " + safe_norm_note).strip()
    guidance = (
        "You are given a sequence of event frames rendered from a neuromorphic event stream. "
        f"There are {num_frames} frames in temporal order. "
        "Begin your sentence with the normalized class label without punctuation. "
        "Only describe the characteristics of the labeled object or action. "
        "Only include attributes that are directly observable in the example; do not infer or guess. "
        "Do not add any information that is not seen or heard in the example. "
        "If the label has multiple parts separated by commas, only include the first part in the description. "
        "For objects describe identity, shape, color or pattern, distinctive parts, pose. "
        "For actions describe the action, the subject, and salient manner. "
        "Exclude unrelated details like background, lighting, composition, counts, other objects, borders, on screen text, watermarks, timestamps, or UI. "
        "Do not use the phrase is a. "
        "Write one concise sentence with at most 16 words. Your output must contain exactly one sentence. "
        "Use the full label and the visual example together to resolve ambiguity."
    )
    prompt = ((preface + " ") if preface else "") + guidance
    return sanitize_for_prompt(prompt)

def remove_isa_phrases(text: str) -> str:
    """Remove the phrase 'is a' (and 'is an') case-insensitively to satisfy style constraints.
    Keeps grammar by collapsing to ' is '.
    """
    if not text:
        return text
    text = re.sub(r"(?i)\bis a\b", " is ", text)
    text = re.sub(r"(?i)\bis an\b", " is ", text)
    # Collapse any doubled spaces that may result
    text = re.sub(r"\s+", " ", text).strip()
    return text

def enforce_single_sentence(text: str) -> str:
    """Ensure the output contains exactly one sentence.
    - Keep content up to and including the first sentence terminator (., !, or ?)
    - If none found, keep text as-is and append a period
    - Collapse spaces and ensure a single trailing period
    """
    if not text:
        return text
    # Find first terminator
    m = re.search(r"[.!?]", text)
    if m:
        text = text[: m.start() + 1]
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Ensure ends with a single period
    text = re.sub(r"[.!?]+$", ".", text)
    if not text.endswith("."):
        text += "."
    return text

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
        contents = parts + [sanitize_for_prompt(prompt)]
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
            norm_label = normalize_label(label)
            safe_modality = sanitize_for_prompt(modality)
            safe_label = sanitize_for_prompt(label)
            safe_norm_label = sanitize_for_prompt(norm_label)
            prompt = (
                f"This {safe_modality} belongs to the class {safe_norm_label or safe_label}. "
                f"Full label verbatim: {safe_label}. "
                + (f"Normalized label: {safe_norm_label}. " if safe_norm_label else "")
                + "Begin your sentence with the normalized class label without punctuation. "
                + "Only describe the characteristics of the labeled object or action. "
                + "Only include attributes that are directly observable in the example; do not infer or guess. "
                + "Do not add any information that is not seen or heard in the example. "
                + "If the label has multiple parts separated by commas, only include the first part in the description. "
                + "For objects describe identity, shape, color or pattern, distinctive parts, pose. "
                + "For actions describe the action, the subject, and salient manner. "
                + "Exclude unrelated details like background, lighting, composition, counts, other objects, borders, on screen text, watermarks, timestamps, or UI. "
                + "Do not use the phrase is a. "
                + "Write one concise sentence with at most 16 words. Your output must contain exactly one sentence. "
                + "Use the full label and the example together to resolve ambiguity."
            )
        else:
            safe_modality = sanitize_for_prompt(modality)
            prompt = (
                f"Write one concise sentence with at most 16 words describing the main content in this {safe_modality}. "
                "Your output must contain exactly one sentence. Start directly and avoid filler. Do not use the phrase is a. "
                "Only include details that are directly observable in the example; do not infer or guess. "
                "Do not add any information that is not seen or heard in the example."
            )
        contents.append(sanitize_for_prompt(prompt))

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
    # Enforce style: remove 'is a'/'is an' phrases
    description = remove_isa_phrases(description)

    pf = getattr(resp, "prompt_feedback", None)
    block_reason = getattr(pf, "block_reason", None) if pf is not None else None
    if block_reason is not None:
        logger.info(f"❗️ Gemini blocked the request: {block_reason}")
        template = _safe_template_for_modality(modality)
        norm_label = normalize_label(label)
        if norm_label:
            description = f"{norm_label} {template.format(norm_label)}".strip()
        else:
            description = template.format(label).strip()
    elif not description:
        logger.info("❗️ Gemini returned an empty description, using fallback template.")
        template = _safe_template_for_modality(modality)
        norm_label = normalize_label(label)
        if norm_label:
            description = f"{norm_label} {template.format(norm_label)}".strip()
        else:
            description = template.format(label).strip()

    # Enforce style again after fallbacks
    description = remove_isa_phrases(description)

    # Logging
    # Enforce inclusion and prefix of normalized label when provided
    if label:
        description = ensure_label_in_description(description, normalize_label(label))
    # Enforce exactly one sentence at the end
    description = enforce_single_sentence(description)

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
                 modality: str, model: str, logger: logging.Logger, event_frames: int):
    idx, payload = entry
    label = payload.get("label", "")

    # Use shared helper to prepare files_for_model
    files_for_model = prepare_files_for_model(payload, dataset_root, modality, logger, event_frames)
    if files_for_model is None:
        return None

    # Infinite exponential backoff for retries
    attempt = 0
    try:
        while True:
            try:
                desc_txt, usage = gemini_describe(client, files_for_model, modality, model, logger, label)
                logger.info(f"[✓] Processed {payload['data']} | label='{label}' | description='{desc_txt}'")
                return idx, payload["data"], label, desc_txt, usage
            except Exception as e:
                attempt += 1
                backoff_time = max(0.02, (2 ** attempt) + random.uniform(0, 0.1))  # Slower increase
                logger.info(f"Error processing {payload['data']}: {e}. Retrying in {backoff_time:.2f}s...")
                time.sleep(backoff_time)
    finally:
        _cleanup_temp_paths(files_for_model)

def process_shard(shard_entries, dataset_root, dataset_dir_name, modality, event_frames, model, deduplicated_align_data, output_base, shard_index):
    log_path = output_base / f"{dataset_dir_name}_shard_{shard_index}.log"
    logger = setup_logger(log_path)

    client = genai.Client()
    with tqdm(total=len(shard_entries), desc=f"Processing shard {shard_index} for {dataset_dir_name}", unit="entry", dynamic_ncols=True) as pbar:
        logger.info(f"[INFO] Processing {len(shard_entries)} entries in shard {shard_index} for {dataset_dir_name}...")
        for entry in shard_entries:
            files_for_model = prepare_files_for_model(entry, dataset_root / dataset_dir_name, modality, logger, event_frames)
            if files_for_model is None:
                logger.warning(f"Skipping entry {entry['data']} due to missing file")
                pbar.update(1)
                continue

            label = entry.get("label", "")
            logger.info(f"Processing {entry['data']} | label='{label}'")

            # Infinite exponential backoff for retries
            attempt = 0
            try:
                while True:
                    try:
                        desc_txt, _ = gemini_describe(client, files_for_model, modality, model, logger, label)
                        entry["description"] = desc_txt
                        deduplicated_align_data[entry["data"]] = entry  # Replace the original entry
                        logger.info(f"Processed {entry['data']} | label='{label}' | description='{desc_txt}' | ")
                        break
                    except Exception as e:
                        attempt += 1
                        backoff_time = max(0.02, (2 ** attempt) + random.uniform(0, 0.1))
                        logger.info(f"Error processing {entry['data']}: {e}. Retrying in {backoff_time:.2f}s...")
                        time.sleep(backoff_time)
            finally:
                _cleanup_temp_paths(files_for_model)
            pbar.update(1)

# ==========================================================
# Dataset processing
# ==========================================================

def process_dataset(client: genai.Client, json_path: Path, dataset_root: Path, modality: str, model: str,
                    limit: Optional[int], num_threads: int, num_shards: int, shard_index: int, shard_dir: Path,
                    event_frames: int):
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
    logger.info(f"[INFO] processing shard {shard_index+1}/{num_shards} with {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as ex, \
         tqdm(total=len(shard_entries), desc=desc, unit="file", position=shard_index, leave=True, dynamic_ncols=True) as pbar:
        futures = [ex.submit(describe_one, client, e, dataset_root, modality, model, logger, event_frames) for e in shard_entries]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                idx, rel_path, label, desc_txt, usage = res
                output_rows.append((idx, {"data": rel_path, "description": desc_txt, "label": label}))
                meta_rows.append((idx, {"data": rel_path, "label": label, "usage": usage}))
            pbar.update(1)

    output_rows.sort(key=lambda x: x[0])
    meta_rows.sort(key=lambda x: x[0])

    align_json_path = shard_dir / "aligned.json"
    meta_json_path = shard_dir / "align_meta.json"

    save_output_json(align_json_path, [row for _, row in output_rows])
    save_output_json(meta_json_path, [row for _, row in meta_rows])


# ==========================================================
# Merge shards
# ==========================================================

def merge_shards(json_path: Path, modality: str, num_shards: int, use_json_root: bool, json_root: Path, output_base: Path, logger: logging.Logger):
    stem = f"{json_path.parent.name}_{modality}"
    logger.info(f"[INFO] Merging {num_shards} shards for {stem}...")

    parts, parts_meta = [], []

    for i in range(num_shards):
        shard_dir = output_base / f"{stem}_shard{i}-{num_shards}"
        json_file = shard_dir / "aligned.json"
        meta_file = shard_dir / "align_meta.json"
        if json_file.exists() and meta_file.exists():
            logger.info(f"[INFO] Loading shard {i} outputs from {json_file} and {meta_file}")
            parts.append(load_input_json(json_file))
            parts_meta.append(load_input_json(meta_file))
        else:
            logger.info(f"[WARN] Missing output for shard {i}, skipping merge.")

    if not parts:
        logger.info(f"[ERR] No shard outputs found for {stem}")
        return

    merged = [item for part in parts for item in part]
    merged_meta = [item for part in parts_meta for item in part]

    if use_json_root:
        output_dir = json_root / json_path.parent.name
        align_json_path = output_dir / "train_data_align.json"
        meta_json_path = output_dir / "train_data_align_meta.json"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        align_json_path = output_base / f"{stem}_data_align.json"
        meta_json_path = output_base / f"{stem}_data_align_meta.json"

    save_output_json(align_json_path, merged)
    save_output_json(meta_json_path, merged_meta)

# ==========================================================
# Shard runner (per-process DI: create client once here)
# ==========================================================

def run_one_shard(json_path, dataset_path, modality, model, limit, num_threads, total_shards, shard_index, shard_dir, event_frames):
    # DI: one client per process
    client = genai.Client()
    process_dataset(client, json_path, dataset_path, modality, model, limit, num_threads, total_shards, shard_index, shard_dir, event_frames)


# ==========================================================
# Main
# ==========================================================

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Generate dataset descriptions with Gemini + label hints.")
    parser.add_argument("--dataset_root", default="/data/datasets")
    parser.add_argument("--json_root", default="./datasets")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of entries per dataset (for testing).")
    parser.add_argument("--skip_modalities", nargs="*", default=[
        # "image",
        # "audio",
        # "thermal",
        # "video",
        # "event"
        ])
    parser.add_argument("--per_process_threads", type=int, default=1)
    parser.add_argument("--max_cores", type=int, default=None)
    parser.add_argument("--event_frames", type=int, default=8, help="Number of temporal bins for event rendering.")
    parser.add_argument("--use_json_root", action="store_true", default=True, help="Flag to toggle output location between json_root and output_base.")
    parser.add_argument("--scan_and_fix", action="store_true", default=False, help="Scan for missing entries and descriptions in train_data.json and train_data_align.json, and fix them.")
    parser.add_argument("--copy_to_json_root", action="store_true", default=False, help="Copy JSON files from output directories into --json_root with the correct names.")
    args = parser.parse_args()

    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_base = Path("output/knowledge_base")
    output_base.mkdir(parents=True, exist_ok=True)

    # Initialize a global run logger early so we can log errors/info
    logger = setup_logger(output_base / "run.log")

    if args.copy_to_json_root:
        copy_json_to_root(output_base, Path(args.json_root))
        return
    
    output_base = output_base / run_timestamp
    output_base.mkdir(parents=True, exist_ok=True)

    if args.scan_and_fix:
        log_path = output_base / "scan_and_fix.log"
        logger = setup_logger(log_path)
        scan_and_fix_entries(Path(args.json_root), Path(args.dataset_root), logger, args.model, args.event_frames, args.limit, output_base)
        return

    dataset_root_base = Path(args.dataset_root)
    json_root_base = Path(args.json_root)

    # Generate mapping list dynamically from MODALITY_MAPPING
    mapping = [(modality, dataset) for dataset, modality in MODALITY_MAPPING.items()]
    mapping = [m for m in mapping if m[0] not in args.skip_modalities]

    if not os.getenv("GOOGLE_API_KEY"):
        logger.info("[ERR] GOOGLE_API_KEY is not set.")
        return

    total_cores = os.cpu_count() or 1
    if args.max_cores is not None:
        total_cores = max(1, min(total_cores, args.max_cores))

    for modality, dataset_name in mapping:
        json_path = json_root_base / dataset_name / "train_data.json"
        dataset_path = dataset_root_base / dataset_name
        if not json_path.exists():
            logger.info(f"[WARN] Missing JSON: {json_path} — skipping")
            continue
        if not dataset_path.exists():
            logger.info(f"[WARN] Missing dataset root: {dataset_path} — skipping")
            continue

        logger.info(f"[INFO] Processing {dataset_name} ({modality}) with {total_cores} shards, {args.per_process_threads} threads/process")

        procs = []
        try:
            for i in range(total_cores):
                shard_dir = output_base / f"{dataset_name}_{modality}_shard{i}-{total_cores}"
                p = mp.Process(target=run_one_shard,
                               args=(json_path, dataset_path, modality, args.model, args.limit,
                                     args.per_process_threads, total_cores, i, shard_dir, args.event_frames))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

        except KeyboardInterrupt:
            logger.info("\n[!] Ctrl-C received — terminating shard processes…")
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()
            break

        merge_shards(json_path, modality, total_cores, args.use_json_root, json_root_base, output_base, logger)

    logger.info(f"[✓] All datasets processed. Output saved in {output_base}")

def copy_json_to_root(output_base: Path, json_root: Path):
    json_root.mkdir(parents=True, exist_ok=True)
    # Copy per-shard outputs to consistent names under json_root
    # aligned.json -> train_data_align.json
    # align_meta.json -> train_data_align_meta.json
    for aligned_file in output_base.rglob("aligned.json"):
        dataset_segment = aligned_file.parent.name  # e.g., "MSR-VTT_video_shard0-8"
        dataset_name = dataset_segment.split("_")[0]
        target_path = json_root / dataset_name / "train_data_align.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(aligned_file, target_path)
        print(f"[✓] Copied {aligned_file} to {target_path}")

    for meta_file in output_base.rglob("align_meta.json"):
        dataset_segment = meta_file.parent.name
        dataset_name = dataset_segment.split("_")[0]
        target_path = json_root / dataset_name / "train_data_align_meta.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(meta_file, target_path)
        print(f"[✓] Copied {meta_file} to {target_path}")

def scan_and_fix_entries(json_root: Path, dataset_root: Path, logger: logging.Logger, model: str, event_frames: int, limit: Optional[int] = None, output_base: Optional[Path] = None):
    total_cores = os.cpu_count() or 1

    for dataset_dir in tqdm(json_root.iterdir(), desc="Scanning datasets", unit="dataset", dynamic_ncols=True):
        if not dataset_dir.is_dir():
            continue

        train_data_path = dataset_dir / "train_data.json"
        align_data_path = dataset_dir / "train_data_align.json"

        if not train_data_path.exists() or not align_data_path.exists():
            logger.warning(f"Missing train_data.json or train_data_align.json in {dataset_dir}")
            continue

        train_data = load_input_json(train_data_path)
        align_data = load_input_json(align_data_path)

        align_data_map = {entry["data"]: entry for entry in align_data}

        missing_entries = []
        missing_record_entries = []
        missing_descriptions_entries = []

        logger.info(f"Processing {dataset_dir.name} with {len(train_data)} train entries and {len(align_data)} align entries")
        for entry in tqdm(train_data, desc=f"Processing {dataset_dir.name}", unit="entry", leave=False, dynamic_ncols=True):
            data_key = entry["data"]
            if data_key not in align_data_map:
                missing_entries.append(entry)
                missing_record_entries.append(entry)
            elif not align_data_map[data_key].get("description"):
                missing_entries.append(entry)
                missing_descriptions_entries.append(entry)

        logger.info(f"Found {len(missing_record_entries)} missing record entries in align_data")
        logger.info(f"Found {len(missing_descriptions_entries)} missing description entries in align_data")

        # Deduplicate entries in align_data, preferring those with descriptions
        deduplicated_align_data = {}
        for entry in align_data:
            data_key = entry["data"]
            if data_key not in deduplicated_align_data or not deduplicated_align_data[data_key].get("description"):
                deduplicated_align_data[data_key] = entry

        align_data = list(deduplicated_align_data.values())

        # Generate descriptions for missing entries using sharding
        if missing_entries:
            logger.info(f"Found {len(missing_entries)} missing entries in {dataset_dir}")
            modality = MODALITY_MAPPING.get(dataset_dir.name, "unknown")

            # Split missing entries into shards and process in parallel
            shard_size = max(1, len(missing_entries) // total_cores)
            missing_entries = missing_entries[:limit] if limit else missing_entries

            shards = [missing_entries[i:i + shard_size] for i in range(0, len(missing_entries), shard_size)]
            logger.info(f"[INFO] Processing {len(shards)} shards for {dataset_dir.name} with {total_cores} cores...")

            with mp.Manager() as manager:
                shared_deduplicated_align_data = manager.dict(deduplicated_align_data)

                with mp.Pool(total_cores) as pool:
                    try:
                        pool.starmap(
                            process_shard,
                            [(shard, dataset_root, dataset_dir.name, modality, event_frames, model, shared_deduplicated_align_data, output_base, i)
                             for i, shard in enumerate(shards)]
                        )
                    finally:
                        pool.close()
                        pool.join()

                # Update the original deduplicated_align_data with the shared dictionary
                deduplicated_align_data.update(shared_deduplicated_align_data)

        # Save the updated align_data
        align_data = list(deduplicated_align_data.values())
        save_output_json(align_data_path, align_data)
        logger.info(f"Updated {align_data_path} with deduplicated and fixed entries.")

if __name__ == "__main__":
    main()
