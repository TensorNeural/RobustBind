import os
import json
import argparse
import random
import numpy as np
from tools.tsne_visualization import plot_center_embeddings_tsne
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from torch.utils.data import Subset, DataLoader
from multiprocessing import get_context, Process, set_start_method
from datetime import datetime
import torch.nn.functional as F
import itertools
import heapq
from collections import defaultdict

from model import UniBindClassifier, ForwardMode
from data_util import JsonDataset, get_transform_fn, load_label_mapping
from shared_types import Modality
set_start_method("spawn", force=True)

# === Constants
NUM_SAMPLES = 60
PERPLEXITY = 5
MODALITY_COLOR = {
    Modality.IMAGE: "blue",
    # Modality.EVENT: "green",
    Modality.AUDIO: "red",
    Modality.POINT: "purple",
}
MODALITY_BATCH_SIZE = {
    Modality.IMAGE: 2000,
    # Modality.EVENT: 2000,
    Modality.AUDIO: 1000,
    Modality.POINT: 100,
}

try:
    from cuml.manifold import TSNE as CuMLTSNE
    import cupy as cp
    TSNE_BACKEND = "cuml"
except ImportError:
    from openTSNE import TSNE as OpenTSNE
    TSNE_BACKEND = "openTSNE"

def log(step, msg):
    print(f"[{step}] {msg}", flush=True)

def load_json_safe(path):
    return json.load(open(path)) if os.path.exists(path) else []

def sample_class_indices(label, val_data, train_data):
    def match(data): return [i for i, e in enumerate(data) if label in e["label"].lower()]
    for data in [train_data, val_data, val_data + train_data]:
        indices = match(data)
        if len(indices) >= NUM_SAMPLES:
            return random.sample(indices, NUM_SAMPLES), data
    return match(train_data), train_data

def compute_embeddings(model, subset, device, modality):
    loader = DataLoader(subset, batch_size=MODALITY_BATCH_SIZE[modality])
    embs = []
    for x, _ in loader:
        with torch.no_grad():
            embs.append(model(x.to(device), ForwardMode.EMBEDDINGS).cpu())
    return torch.cat(embs, dim=0).numpy()

def extract_worker(device_id, config):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    modality_config, weights, use_flash, dataset_root, label_shards, tmp_json, save_dir = config

    for mod in MODALITY_COLOR:
        log(f"GPU {device_id}", f"Loading {mod.name}")
        dataset_name, _, _, center_path = modality_config[mod]
        dataset = JsonDataset(
            dataset_root=os.path.join(dataset_root, dataset_name),
            data_json_path=tmp_json[mod],
            transform=get_transform_fn(mod),
            label_to_index={}
        )
        centre_emb, centre_labels, label_to_index, _ = load_label_mapping(center_path, device)
        model = UniBindClassifier(
            device, weights, mod,
            centre_embeddings=centre_emb,
            centre_labels=centre_labels,
            label_to_index=label_to_index,
            logger=None,
            use_flash_attention=use_flash,
            use_modality_head_mlp=False
        ).to(device)

        selected_labels = set(label_shards[mod])
        idx_and_labels = [(i, lbl.lower()) for i, (_, lbl) in enumerate(dataset.samples) if lbl.lower() in selected_labels]
        if not idx_and_labels:
            log(f"GPU {device_id}", f"No samples to process for {mod.name}")
            continue

        indices, merged_labels = zip(*idx_and_labels)
        subset = Subset(dataset, list(indices))
        log(f"GPU {device_id}", f"Computing embeddings for {mod.name} ({len(indices)} samples)")
        embs = compute_embeddings(model, subset, device, mod)

        grouped = {}
        for emb, lbl in zip(embs, merged_labels):
            grouped.setdefault(lbl, []).append(emb)
        for lbl, emb_list in grouped.items():
            np.save(os.path.join(save_dir, f"{mod.name.lower()}_{lbl}.npy"), np.stack(emb_list))
        torch.cuda.empty_cache()
        log(f"GPU {device_id}", f"Saved embeddings for {mod.name}")

def chamfer(A, B):
    A = torch.tensor(A).float().cuda()
    B = torch.tensor(B).float().cuda()
    D = torch.cdist(A, B)
    return (D.min(1).values.mean() + D.min(0).values.mean()).item()

def draw_arrows(A, B, A_coords, B_coords, max_lines=10):
    A_t = torch.tensor(A).float().cuda()
    B_t = torch.tensor(B).float().cuda()
    D = torch.cdist(A_t, B_t).cpu().numpy()
    for i in np.random.choice(len(A), size=min(max_lines, len(A)), replace=False):
        j = D[i].argmin()
        x1, y1 = A_coords[i]
        x2, y2 = B_coords[j]
        plt.arrow(x1, y1, x2 - x1, y2 - y1,
                  alpha=0.2, color='gray', linewidth=0.5, head_width=0.01)

def fit_tsne(X, perplexity):
    if TSNE_BACKEND == "cuml":
        tsne = CuMLTSNE(
            n_components=2,            # 2D plot
            perplexity=PERPLEXITY,              # ↓ lower = more local structure → more mixing
            # learning_rate=800,         # ↑ higher = more spread out
            # early_exaggeration=4.0,    # ↓ lower = less tight clustering
            # n_iter=300,                # ↓ fewer = prevents full separation
            random_state=42,           # reproducibility
            # metric='cosine',        # you can try 'cosine' if embeddings are cosine-aligned
            # init='random',             # or 'pca' – 'random' introduces more overlap
            # method='fft',       # or 'fft' if dataset is large
        )
        coords = tsne.fit_transform(cp.asarray(X))
        coords = cp.asnumpy(coords)
    else:
        tsne = OpenTSNE(n_components=2, perplexity=perplexity, n_jobs=-1, random_state=42)
        coords = tsne.fit(X)
    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0))
    return coords

def plot_tsne_triplet(embedding_tuples, rank, output_dir,
                      center_emb_dict=None, use_center_only=False):
    """
    Args:
        embedding_tuples: List of (modality, label, embeddings ndarray)
        rank: Triplet rank
        output_dir: Save dir
        center_emb_dict: Optional centers
        use_center_only: Whether only centers were used
    """
    plt.figure(figsize=(8, 6))

    # Flatten for t-SNE
    X = np.concatenate([e for (_, _, e) in embedding_tuples], axis=0).astype(np.float32)
    colors = [MODALITY_COLOR[m] for m, _, e in embedding_tuples for _ in range(len(e))]
    perplexity = max(1, min(PERPLEXITY, len(X) - 1))
    coords = fit_tsne(X, perplexity)

    offset = 0
    coord_dict = {}

    for modality, label, emb in embedding_tuples:
        count = len(emb)
        current_coords = coords[offset:offset + count]
        current_colors = colors[offset:offset + count]

        plt.scatter(current_coords[:, 0], current_coords[:, 1],
                    c=current_colors, label=f"{modality.name}: {label}", s=40)
        coord_dict[modality] = current_coords
        offset += count

    # === Chamfer annotation ===
    chamfer_lines = ["Chamfer Distances:"]
    for (i, j) in itertools.combinations(range(len(embedding_tuples)), 2):
        mod_i, _, emb_i = embedding_tuples[i]
        mod_j, _, emb_j = embedding_tuples[j]
        score = chamfer(emb_i, emb_j)
        chamfer_lines.append(f"{mod_i.name}↔{mod_j.name}: {score:.2f}")
    chamfer_text = "\n".join(chamfer_lines)

    plt.text(1.02, 0.5, chamfer_text, transform=plt.gca().transAxes,
             fontsize=10, va='center', ha='left',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fname = f"{rank:03d}_" + "_".join(label for _, label, _ in embedding_tuples).replace(" ", "_") + ".png"
    plt.title(f"Triplet Rank {rank} ({TSNE_BACKEND})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()

def to_gpu(t):
    if isinstance(t, torch.Tensor):
        x = t.to(device='cuda')
    elif isinstance(t, list):
        if isinstance(t[0], torch.Tensor):
            x = torch.stack(t).to(device='cuda')
        elif isinstance(t[0], (np.ndarray, list)):
            x = torch.stack([torch.tensor(v, device='cuda') for v in t])
        else:
            x = torch.tensor(t, device='cuda')
    elif isinstance(t, np.ndarray):
        x = torch.tensor(t, device='cuda')
    else:
        raise TypeError(f"Unsupported input type for to_gpu: {type(t)}")

    if x.ndim == 1:
        x = x.unsqueeze(0)
    return F.normalize(x, dim=1)

def cosine_score(A, B):
    return torch.mm(A, B.T).mean()

def centroid_score(A, B):
    return torch.norm(torch.mean(A, dim=0) - torch.mean(B, dim=0))

def chamfer_score(A, B):
    D = torch.cdist(A, B)
    return (D.min(1).values.mean() + D.min(0).values.mean())

def score_triplet(triplet, metric):
    labels = [(mod, lbl) for mod, lbl, _ in triplet]
    embeddings = [to_gpu(emb) for _, _, emb in triplet]

    scores = []
    for A, B in itertools.combinations(embeddings, 2):
        if metric == "cosine":
            scores.append(cosine_score(A, B))
        elif metric == "centroid":
            scores.append(centroid_score(A, B))
        elif metric == "chamfer":
            scores.append(chamfer_score(A, B))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    scores = torch.stack(scores)

    return (labels, scores.mean().item(), scores)


def gpu_score_worker(rank, chunk, metric, queue, top_k):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    print(f"[GPU WORKER {rank}] Scoring {len(chunk)} triplets...", flush=True)
    heap = []
    for triplet in tqdm(chunk, desc=f"[GPU {rank}] Scoring", position=rank):
        label_triplet, score, scores = score_triplet(triplet, metric)
        heapq.heappush(heap, (score, (label_triplet, scores.cpu().tolist())))
        if len(heap) > top_k * 2:
            heap = heapq.nlargest(top_k, heap)
    queue.put(heap)

def filter_image_class(emb_dict, center_emb_dict, image_class, use_center_only):
    label = image_class.lower()

    # print("Available image classes in emb_dict:", list(emb_dict[Modality.IMAGE].keys()))

    if use_center_only:
        if label not in center_emb_dict[Modality.IMAGE].keys():
            raise ValueError(f"Image class '{label}' not found in center embeddings.")
        center_emb_dict[Modality.IMAGE] = {label: center_emb_dict[Modality.IMAGE][label]}
    else:
        if label not in emb_dict[Modality.IMAGE]:
            raise ValueError(f"Image class '{label}' not found in sample embeddings.")
        
        emb_dict[Modality.IMAGE] = {label: emb_dict[Modality.IMAGE][label]}

    log("MAIN", f"Scoring only for image class: {label}")

def plot_individual_center_tsnes(center_emb_dict, output_dir):
    log("MAIN", "Drawing TSNE of center embeddings for each (modality, label)...")

    if TSNE_BACKEND == "cuml":
        log("MAIN", "Using GPU-accelerated cuML TSNE")
    else:
        log("MAIN", "Using CPU-based openTSNE")

    all_items = [
        (mod, label, vecs)
        for mod in MODALITY_COLOR
        for label, vecs in center_emb_dict[mod].items()
        if len(vecs) >= 2
    ]

    for mod, label, vecs in tqdm(all_items, desc="[MAIN] Center TSNEs", unit="class"):
        log("CENTER", f"Plotting {mod.name} - '{label}' with {len(vecs)} centers")

        vecs_np = np.stack(vecs).astype(np.float32)
        perplexity = max(1, min(PERPLEXITY, len(vecs_np) - 1))

        if TSNE_BACKEND == "cuml":
            coords = CuMLTSNE(n_components=2, perplexity=perplexity).fit_transform(cp.asarray(vecs_np))
            coords = cp.asnumpy(coords)
        else:
            tsne = OpenTSNE(n_components=2, perplexity=perplexity, n_jobs=-1, random_state=42)
            coords = tsne.fit(vecs_np)

        coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0))

        plt.figure(figsize=(6, 5))
        plt.scatter(coords[:, 0], coords[:, 1],
                    c=MODALITY_COLOR[mod], marker='X', s=60,
                    edgecolors='black', linewidths=0.8, alpha=0.6)
        plt.title(f"{mod.name}: {label} (center embeddings)")
        plt.tight_layout()
        fname = f"centers_{mod.name.lower()}_{label.replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()


def create_output_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("output", "alignment", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log("MAIN", f"Saving to {output_dir}")
    return output_dir

def load_all_center_embeddings(args):
    center_emb_dict = {}
    for mod in MODALITY_COLOR:
        path = getattr(args, f"centre_emb_{mod.name.lower()}")
        emb, labels, _, _ = load_label_mapping(path, torch.device("cpu"))
        mod_dict = defaultdict(list)
        for lbl, vec in zip(labels, emb):
            mod_dict[lbl.lower()].append(vec)
        center_emb_dict[mod] = dict(mod_dict)
    return center_emb_dict

def load_precomputed_embeddings(embedding_dir):
    emb_dict = {mod: {} for mod in MODALITY_COLOR}
    for fname in os.listdir(embedding_dir):
        if fname.endswith(".npy"):
            modname, lbl = fname.replace(".npy", "").split("_", 1)
            mod = Modality(modname)
            if mod not in MODALITY_COLOR:
                # log("MAIN", f"Skipping unknown modality: {modname}")
                continue

            emb_dict[mod][lbl] = np.load(os.path.join(embedding_dir, fname))
    return emb_dict

def sample_and_extract_embeddings(args, output_dir, center_emb_dict):
    tmp_json = {
        Modality.IMAGE: os.path.join(output_dir, "tmp_image.json"),
        # Modality.EVENT: os.path.join(output_dir, "tmp_event.json"),
        Modality.AUDIO: os.path.join(output_dir, "tmp_audio.json"),
        Modality.POINT: os.path.join(output_dir, "tmp_point.json")
    }
    modality_config = {
        Modality.IMAGE: ("ImageNet-1K", args.val_json_image, args.train_json_image, args.centre_emb_image),
        # Modality.EVENT: ("N-ImageNet-1K", args.val_json_event, args.train_json_event, args.centre_emb_event),
        Modality.AUDIO: ("ESC-50", args.val_json_audio, args.train_json_audio, args.centre_emb_audio),
        Modality.POINT: ("ModelNet40", args.val_json_point, args.train_json_point, args.centre_emb_point)
    }

    label_map = {}
    for mod in MODALITY_COLOR:
        _, val_json, train_json, _ = modality_config[mod]
        val_data = load_json_safe(val_json)
        train_data = load_json_safe(train_json)
        centre_labels = list(center_emb_dict[mod].keys())
        examples, labels = [], []
        log("MAIN", f"Sampling {mod.name}")
        for lbl in tqdm(sorted(centre_labels), desc=f"[MAIN] {mod.name}"):
            idx, used = sample_class_indices(lbl, val_data, train_data)
            if idx is None:
                continue
            examples.extend([used[i] for i in idx])
            labels.append(lbl)
        with open(tmp_json[mod], "w") as f:
            json.dump(examples, f, indent=2)
        label_map[mod] = labels

    tasks = []
    device_count = torch.cuda.device_count()
    for device_id in range(device_count):
        label_shards = {
            mod: [lbl for i, lbl in enumerate(label_map[mod]) if i % device_count == device_id]
            for mod in MODALITY_COLOR
        }
        config = (
            modality_config,
            args.pretrain_weights,
            args.use_flash_attention,
            args.dataset_root,
            label_shards,
            tmp_json,
            output_dir
        )
        tasks.append((device_id, config))

    log("MAIN", f"Extracting embeddings using {device_count} GPUs")
    with get_context("spawn").Pool(device_count) as pool:
        pool.starmap(extract_worker, tasks)

    return load_precomputed_embeddings(output_dir)

def score_triplets_parallel(triplet_data, metric, top_k, top_prefer):
    if metric == "cosine":
        prefer_high = (top_prefer == "close")  # high similarity is good
    else:
        prefer_high = (top_prefer == "far")    # high distance is bad (close → low)
        
    queue = get_context("spawn").Queue()
    device_count = torch.cuda.device_count()
    chunk_size = (len(triplet_data) + device_count - 1) // device_count
    processes = []

    for rank in range(device_count):
        chunk = triplet_data[rank * chunk_size:(rank + 1) * chunk_size]
        p = Process(target=gpu_score_worker, args=(rank, chunk, metric, queue, top_k))
        p.start()
        processes.append(p)

    heap = []
    for _ in range(device_count):
        local = queue.get()
        for score, (labels, scores) in local:
            score_key = score if prefer_high else -score
            heapq.heappush(heap, (score_key, (labels, scores)))
            if len(heap) > top_k:
                heapq.heappop(heap)

    for p in processes:
        p.join()

    return sorted(heap, reverse=True)

def score_center_triplets_parallel(center_emb_dict, metric, top_k, top_prefer):
    if metric == "cosine":
        prefer_high = (top_prefer == "close")  # high similarity is good
    else:
        prefer_high = (top_prefer == "far")    # high distance is bad (close → low)

    triplet_data = list(product(
        center_emb_dict[Modality.IMAGE].items(),
        center_emb_dict[Modality.AUDIO].items(),
        center_emb_dict[Modality.POINT].items()
    ))

    heap = []
    for triplet in tqdm(triplet_data, desc="[MAIN] Scoring center triplets"):
        labels, score, scores = score_triplet(triplet, metric)
        score_key = score if prefer_high else -score
        heapq.heappush(heap, (score_key, (labels, scores)))
        if len(heap) > top_k:
            heapq.heappop(heap)
    return sorted(heap, reverse=True)

def plot_top_triplets(top_triplets, emb_dict, center_emb_dict, output_dir, use_center_only):
    print(f"Top triplets: {len(top_triplets)}")

    for rank, (score, (labels, scores)) in enumerate(tqdm(top_triplets, desc="[MAIN] Plotting", unit="triplet")):
        print(f"Plotting triplet {rank + 1}/{len(top_triplets)}: {labels} (score: {score:.2f}, all scores: {scores})")
        embedding_tuples = []
        for (modality, label) in labels:
            emb = (np.stack(center_emb_dict[modality][label])
                if use_center_only else emb_dict[modality][label])
            embedding_tuples.append((modality, label, emb))

        plot_tsne_triplet(
            embedding_tuples=embedding_tuples,
            rank=rank,
            output_dir=output_dir,
            center_emb_dict=center_emb_dict,
            use_center_only=use_center_only
        )

def main(args):
    output_dir = create_output_dir()

    log("MAIN", f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    log("MAIN", f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    center_emb_dict = load_all_center_embeddings(args)
    if args.use_center_only:
        print("Using center embeddings only for triplet scoring and visualization")

        if args.image_class:
            filter_image_class(
                emb_dict={Modality.IMAGE: {}},
                center_emb_dict=center_emb_dict,
                image_class=args.image_class,
                use_center_only=args.use_center_only
            )

        print(f"Center embeddings loaded for {center_emb_dict[Modality.IMAGE].keys()} image classes")
        top_triplets = score_center_triplets_parallel(center_emb_dict, args.score_metric, args.top_k_plots, args.top_prefer)
        print(f"Found {len(top_triplets)} top triplets using center embeddings")
        plot_top_triplets(
            top_triplets,
            emb_dict=None,
            center_emb_dict=center_emb_dict,
            output_dir=output_dir,
            use_center_only=True
        )
        log("MAIN", f"✅ Center-only t-SNEs saved to {output_dir}")
        return

    emb_dict = (load_precomputed_embeddings(args.embedding_dir)
                if args.skip_embed else sample_and_extract_embeddings(args, output_dir, center_emb_dict))

    if args.image_class:
        filter_image_class(emb_dict, center_emb_dict, args.image_class, args.use_center_only)

    triplet_source = center_emb_dict if args.use_center_only else emb_dict
    triplet_data = [
        [
            (Modality.IMAGE, lbl_i, emb_i),
            # (Modality.EVENT, lbl_e, emb_e),
            (Modality.AUDIO, lbl_a, emb_a),
            (Modality.POINT, lbl_p, emb_p),
        ]
        for (lbl_i, emb_i), (lbl_a, emb_a), (lbl_p, emb_p) in product(
            triplet_source[Modality.IMAGE].items(),
            # triplet_source[Modality.EVENT].items(),
            triplet_source[Modality.AUDIO].items(),
            triplet_source[Modality.POINT].items(),
        )
    ]

    top_triplets = score_triplets_parallel(triplet_data, args.score_metric, args.top_k_plots, args.top_prefer)

    log("MAIN", f"Plotting t-SNE for top {len(top_triplets)} triplets...")
    plot_top_triplets(top_triplets, emb_dict, center_emb_dict, output_dir, args.use_center_only)

    log("MAIN", f"✅ Done. Top {args.top_k_plots} plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/data/datasets")
    parser.add_argument("--val_json_image", default="./datasets/ImageNet-1K/val_data.json")
    parser.add_argument("--train_json_image", default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--val_json_event", default="./datasets/N-ImageNet-1K/val_data.json")
    parser.add_argument("--train_json_event", default="./datasets/N-ImageNet-1K/train_data.json")
    parser.add_argument("--val_json_audio", default="./datasets/ESC-50/val_data.json")
    parser.add_argument("--train_json_audio", default="./datasets/ESC-50/train_data.json")
    parser.add_argument("--val_json_point", default="./datasets/ModelNet40/val_data.json")
    parser.add_argument("--train_json_point", default="./datasets/ModelNet40/train_data.json")
    parser.add_argument("--centre_emb_image", default="./centre_embs/image_in_center_embeddings.pkl")
    parser.add_argument("--centre_emb_event", default="./centre_embs/event_nin_center_embeddings.pkl")
    parser.add_argument("--centre_emb_audio", default="./centre_embs/audio_esc_center_embeddings.pkl")
    parser.add_argument("--centre_emb_point", default="./centre_embs/point_modelnet40_center_embeddings.pkl")
    parser.add_argument("--pretrain_weights", default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt")
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--skip_embed", action="store_true", default=True, help="Skip embedding & sampling")
    parser.add_argument("--embedding_dir", type=str, default="output/alignment/2025-07-27_23-27-43", help="Path to precomputed .npy embeddings")
    parser.add_argument("--top_k_plots", type=int, default=5)
    parser.add_argument("--top_prefer", choices=["close", "far"], default="close")
    parser.add_argument("--score_metric", choices=["cosine", "centroid", "chamfer"], default="centroid")
    parser.add_argument("--image_class", type=str, default="sports car, sport car", help="If set, only use this image class label for scoring")
    parser.add_argument("--use_center_only", action="store_true", default=False, help="If set, rank and plot using only class center embeddings.")

    args = parser.parse_args()
    main(args)
