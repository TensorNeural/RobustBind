import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue, get_context
from datetime import datetime

from shared_types import Modality

MODALITY_COLOR = {
    Modality.IMAGE: "blue",
    Modality.AUDIO: "red",
    # Modality.EVENT: "green",
    Modality.POINT: "purple",
}


def log(step, msg):
    print(f"[{step}] {msg}", flush=True)


def create_output_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("/data/output", "similarity", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log("MAIN", f"Saving to {output_dir}")
    return output_dir


def load_embeddings(embedding_dir):
    emb_dict = defaultdict(dict)
    count = 0
    for fname in os.listdir(embedding_dir):
        if fname.endswith(".npy"):
            modname, lbl = fname.replace(".npy", "").split("_", 1)
            try:
                mod = Modality(modname)
            except ValueError:
                log("WARN", f"Unknown modality prefix in filename: {fname}")
                continue
            emb_dict[mod][lbl] = np.load(os.path.join(embedding_dir, fname))
            count += 1
    log("LOAD", f"Loaded {count} .npy files from {embedding_dir}")
    for mod in emb_dict:
        log("LOAD", f"{mod.name}: {len(emb_dict[mod])} labels loaded: {list(emb_dict[mod].keys())[:5]}...")
    return emb_dict


def to_gpu_norm(x, device):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device, dtype=torch.float32)
    return F.normalize(x, dim=1)


def compute_similarity(A, B, metric, device):
    A, B = to_gpu_norm(A, device), to_gpu_norm(B, device)
    if metric == "cosine":
        return torch.mm(A, B.T).mean().item()
    elif metric == "centroid":
        return -torch.norm(A.mean(dim=0) - B.mean(dim=0)).item()
    elif metric == "chamfer":
        D = torch.cdist(A, B)
        return -(D.min(1).values.mean() + D.min(0).values.mean()).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def worker(mod, base_label, compare_labels, emb_dict, metric, device_id, queue):
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    A = emb_dict[mod][base_label]
    results = []
    for lbl in tqdm(compare_labels, desc=f"[GPU {device_id}] {mod.name} ↔ all", position=device_id):
        B = emb_dict[mod][lbl]
        score = compute_similarity(A, B, metric, device)
        results.append((score, lbl))
    queue.put(results)


def rank_all_labels_parallel(mod, base_label, emb_dict, metric):
    all_labels = list(emb_dict[mod].keys())
    log("DEBUG", f"{mod.name}: base_label = '{base_label}', available labels = {len(all_labels)}")
    if base_label not in all_labels:
        raise ValueError(f"Label '{base_label}' not found in {mod.name} embeddings")

    compare_labels = [lbl for lbl in all_labels if lbl != base_label]
    log("DEBUG", f"{mod.name}: comparing '{base_label}' to {len(compare_labels)} other labels")

    device_count = torch.cuda.device_count()
    chunk_size = (len(compare_labels) + device_count - 1) // device_count

    queue = get_context("spawn").Queue()
    procs = []

    for i in range(device_count):
        chunk = compare_labels[i * chunk_size:(i + 1) * chunk_size]
        log("SPAWN", f"GPU {i} handling {len(chunk)} comparisons for {mod.name}")
        p = Process(target=worker, args=(mod, base_label, chunk, emb_dict, metric, i, queue))
        p.start()
        procs.append(p)

    all_results = []
    for _ in range(device_count):
        all_results.extend(queue.get())

    for p in procs:
        p.join()

    all_results.sort(reverse=True)
    return all_results


def save_results(mod, base_label, results, output_dir, metric):
    fname = os.path.join(output_dir, f"{mod.name.lower()}__{base_label.replace(' ', '_')}_{metric}.txt")
    with open(fname, "w") as f:
        f.write(f"Base label: {base_label} ({mod.name})\n")
        f.write(f"Metric: {metric}\n\n")
        for rank, (score, lbl) in enumerate(results, 1):
            f.write(f"{rank:03d}. {base_label} ↔ {lbl}: {score:.4f}\n")
    log("SAVE", f"Saved to {fname}")


def main(args):
    log("MAIN", f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    log("MAIN", f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    emb_dict = load_embeddings(args.embedding_dir)
    output_dir = create_output_dir()

    input_labels = {
        Modality.IMAGE: args.image_label.lower(),
        Modality.AUDIO: args.audio_label.lower(),
        Modality.POINT: args.point_label.lower(),
    }

    for mod in MODALITY_COLOR:
        base = input_labels.get(mod)
        log("MAIN", f"▶ {mod.name} - base label: '{base}'")

        if base is None or base.strip() == "":
            log("SKIP", f"{mod.name}: empty base label. Skipping.")
            continue
        if mod not in emb_dict or not emb_dict[mod]:
            log("SKIP", f"{mod.name}: no embeddings found. Skipping.")
            continue

        if base not in emb_dict[mod]:
            log("ERROR", f"{mod.name}: base label '{base}' not found in loaded embeddings")
            log("HINT", f"{mod.name} labels available: {list(emb_dict[mod].keys())[:10]}")
            continue

        log("MAIN", f"Ranking all labels by similarity to '{base}' ({mod.name}) using {args.metric}")
        results = rank_all_labels_parallel(mod, base, emb_dict, args.metric)
        for rank, (score, lbl) in enumerate(results[:10], 1):  # top-10 preview
            print(f"{rank:03d}. {base} ↔ {lbl}: {score:.4f}")
        save_results(mod, base, results, output_dir, args.metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str,
                        default="/data/output/alignment/2025-07-27_23-27-43",
                        help="Directory with saved .npy embeddings")
    parser.add_argument("--metric", type=str,
                        choices=["cosine", "centroid", "chamfer"],
                        default="cosine",
                        help="Similarity metric")
    parser.add_argument("--image_label", type=str,
                        default="sports car, sport car",
                        help="Image label to compare")
    parser.add_argument("--audio_label", type=str,
                        default="car_horn",
                        help="Audio label to compare")
    parser.add_argument("--point_label", type=str,
                        default="car",
                        help="Point cloud label to compare")
    args = parser.parse_args()
    main(args)
