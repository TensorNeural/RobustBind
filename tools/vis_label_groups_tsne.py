import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared_types import Modality
from model import UniBindClassifier
from data_util import load_and_transform_text


# -------------------------
# Data structures
# -------------------------

@dataclass
class LabelItem:
    group_id: int
    dataset: str
    label: str


# -------------------------
# IO helpers
# -------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def slugify(s: str) -> str:
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s.strip("_")


def load_label_groups(path: str) -> List[LabelItem]:
    with open(path, "r") as f:
        data = json.load(f)
    items: List[LabelItem] = []
    groups = data.get("groups", [])
    for g in groups:
        gid = int(g.get("group_id"))
        for d in g.get("datasets", []):
            ds = str(d.get("dataset", ""))
            for lab in d.get("labels", []):
                items.append(LabelItem(group_id=gid, dataset=ds, label=str(lab)))
    return items


# -------------------------
# Embeddings
# -------------------------

def build_unibind_text_encoder(
    device: torch.device,
    pretrain_weights: str,
) -> UniBindClassifier:
    # We only need the text encoder path of UniBindClassifier
    model = UniBindClassifier(
        device=device,
        pretrain_weights=pretrain_weights,
        modality=Modality.IMAGE,  # modality choice doesn't affect encode_text
        centre_embeddings=None,
        centre_labels=None,
        label_to_index=None,
        use_lora=False,
        lora_weights=None,
        use_flash_attention=True,
    )
    model.eval()
    return model.to(device)


def encode_labels_to_embeddings(
    labels: List[str],
    model: UniBindClassifier,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            chunk = labels[i : i + batch_size]
            tokens = load_and_transform_text(chunk, device=device)
            v = model.encode_text(tokens)  # (B, D)
            v = v.detach().cpu().numpy()
            # L2 normalize rows
            n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            v = v / n
            vecs.append(v)
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)


# -------------------------
# TSNE
# -------------------------

def run_opentsne(
    X: np.ndarray,
    perplexity: float = 30.0,
    metric: str = "cosine",
    random_state: int = 42,
    n_iter: int = 1000,
    early_exaggeration: float = 8.0,
    learning_rate: float = 50.0,
    initialization: str = "pca",
) -> np.ndarray:
    if X.shape[0] <= 2:
        # Degenerate: lay points on a line/diagonal
        return np.hstack([
            np.linspace(0.25, 0.75, X.shape[0]).reshape(-1, 1),
            np.linspace(0.25, 0.75, X.shape[0]).reshape(-1, 1),
        ])

    from openTSNE import TSNE  # type: ignore

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        n_iter=n_iter,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        initialization=initialization,
        n_jobs=-1,
    )
    emb2d = np.asarray(tsne.fit(X))
    # Normalize for stable plotting margins
    emb2d = (emb2d - emb2d.min(0)) / (emb2d.max(0) - emb2d.min(0) + 1e-9)
    return emb2d


# -------------------------
# Plotting
# -------------------------

def assign_colors(group_ids: List[int]) -> Dict[int, Tuple[float, float, float]]:
    # Use tab20 then tab20b/tab20c to accommodate many groups
    palettes = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    unique = sorted(set(group_ids))
    colors: Dict[int, Tuple[float, float, float]] = {}
    idx = 0
    for gid in unique:
        cmap = palettes[idx // 20 % len(palettes)]
        colors[gid] = tuple(cmap(idx % 20)[:3])  # RGB
        idx += 1
    return colors


def compute_group_centroids(coords: np.ndarray, items: List[LabelItem]) -> Dict[int, np.ndarray]:
    """Return centroid (mean) of points for each group_id in 2D space."""
    centroids: Dict[int, np.ndarray] = {}
    for gid in sorted({it.group_id for it in items}):
        idx = [i for i, it in enumerate(items) if it.group_id == gid]
        if idx:
            centroids[gid] = coords[idx].mean(axis=0)
    return centroids


def compute_outliers_by_group(
    coords: np.ndarray,
    items: List[LabelItem],
    per_group: int = 5,
) -> List[Tuple[int, int, float]]:
    """Compute top-k farthest points from their group's centroid.
    Returns list of tuples (global_index, group_id, distance), sorted by group then distance desc.
    """
    results: List[Tuple[int, int, float]] = []
    centroids = compute_group_centroids(coords, items)
    for gid, center in centroids.items():
        idx = [i for i, it in enumerate(items) if it.group_id == gid]
        if not idx:
            continue
        dists = np.linalg.norm(coords[idx] - center[None, :], axis=1)
        order = np.argsort(-dists)  # descending
        k = min(per_group, len(idx))
        for j in order[:k]:
            results.append((idx[j], gid, float(dists[j])))
    # sort for stable output: by group_id, then distance desc
    results.sort(key=lambda t: (t[1], -t[2]))
    return results


def save_outliers_tsv(
    path: str,
    outliers: List[Tuple[int, int, float]],
    items: List[LabelItem],
    coords: Optional[np.ndarray] = None,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        f.write("group_id\tdistance\tdataset\tlabel\tindex")
        if coords is not None:
            f.write("\tx\ty")
        f.write("\n")
        for idx, gid, dist in outliers:
            it = items[idx]
            if coords is not None:
                x, y = coords[idx, 0], coords[idx, 1]
                f.write(f"{gid}\t{dist:.6f}\t{it.dataset}\t{it.label}\t{idx}\t{x:.6f}\t{y:.6f}\n")
            else:
                f.write(f"{gid}\t{dist:.6f}\t{it.dataset}\t{it.label}\t{idx}\n")


def plot_tsne_by_group(
    coords: np.ndarray,
    items: List[LabelItem],
    out_png: str,
    title: str,
    point_size: float = 8.0,
    alpha: float = 1.0,
    annotate: bool = False,
    annotate_outliers: bool = False,
    outliers: Optional[List[Tuple[int, int, float]]] = None,
):
    group_ids = [it.group_id for it in items]
    gid2color = assign_colors(group_ids)

    plt.figure(figsize=(10, 8))
    # Scatter all points, one color per group
    for gid in sorted(set(group_ids)):
        idx = [i for i, it in enumerate(items) if it.group_id == gid]
        if not idx:
            continue
        x = coords[idx, 0]
        y = coords[idx, 1]
        plt.scatter(x, y, s=point_size, c=[gid2color[gid]], alpha=alpha, label=f"Group {gid} (n={len(idx)})", linewidths=0)

    if annotate:
        # Optionally annotate a few random labels per group (can clutter on large sets)
        import random
        rng = random.Random(0)
        for gid in sorted(set(group_ids)):
            idx = [i for i, it in enumerate(items) if it.group_id == gid]
            for i in idx[: max(1, len(idx) // 40)]:  # annotate ~2.5% per group
                it = items[i]
                plt.text(coords[i, 0], coords[i, 1], it.label, fontsize=6, alpha=0.8)

    # Draw centroids as stars
    centroids = compute_group_centroids(coords, items)
    centroid_handle = None
    for gid, c in centroids.items():
        star = plt.scatter([c[0]], [c[1]], s=180, c=[gid2color[gid]], marker='*', edgecolors='k', linewidths=0.8)
        if centroid_handle is None:
            centroid_handle = star

    # Optionally annotate outliers
    if annotate_outliers and outliers:
        for idx, gid, dist in outliers:
            it = items[idx]
            text = f"[{it.dataset}] {it.label}"
            plt.text(coords[idx, 0], coords[idx, 1], text, fontsize=6, alpha=0.95)

    plt.title(title)
    # Add a single centroid legend entry if present
    handles, labels = plt.gca().get_legend_handles_labels()
    if centroid_handle is not None:
        handles.append(centroid_handle)
        labels.append("centroid")
    plt.legend(handles, labels, loc="best", fontsize=8, ncol=2)
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="t-SNE of label groups using OpenTSNE (colors per group)")
    p.add_argument("--input", type=str, default="/data/output/label_groups_gemini.json", help="Path to label groups JSON")
    p.add_argument("--output", type=str, default="/data/output/tsne/label_groups_gemini_tsne.png", help="Path to save PNG")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--early-exaggeration", type=float, default=8.0)
    p.add_argument("--learning-rate", type=float, default=50.0)
    p.add_argument("--init", type=str, default="pca", choices=["pca", "random"])  # noqa
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrain-weights", type=str, default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt")
    p.add_argument("--annotate", action="store_true", help="Annotate a small random subset of labels per group")
    p.add_argument("--annotate-outliers", action="store_true", help="Annotate outlier labels per group on the plot")
    p.add_argument("--outliers-per-group", type=int, default=5, help="Top-K farthest labels to flag per group")
    p.add_argument("--outliers-output", type=str, default=None, help="Optional TSV file to save outliers list")
    p.add_argument("--point-size", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Ensure relative resources (like bpe/ vocab) resolve correctly
    os.chdir(REPO_ROOT)
    items = load_label_groups(args.input)
    if not items:
        raise RuntimeError(f"No labels loaded from {args.input}")

    # Prepare label list for encoding
    labels = [it.label for it in items]

    # Build text encoder and encode
    device = torch.device(args.device)
    model = build_unibind_text_encoder(device=device, pretrain_weights=args.pretrain_weights)
    X = encode_labels_to_embeddings(labels, model, device=device, batch_size=128)

    # TSNE
    coords = run_opentsne(
        X,
        perplexity=args.perplexity,
        metric=args.metric,
        random_state=args.seed,
        n_iter=args.n_iter,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        initialization=args.init,
    )

    # Identify outliers
    outliers = compute_outliers_by_group(coords, items, per_group=args.outliers_per_group) if args.outliers_per_group > 0 else []
    # Save outliers TSV
    outliers_path = args.outliers_output or os.path.splitext(args.output)[0] + "_outliers.tsv"
    save_outliers_tsv(outliers_path, outliers, items, coords)

    # Plot
    title = f"Label Groups t-SNE (OpenTSNE) — {os.path.basename(args.input)}"
    plot_tsne_by_group(
        coords,
        items,
        out_png=args.output,
        title=title,
        point_size=args.point_size,
        alpha=args.alpha,
        annotate=args.annotate,
        annotate_outliers=args.annotate_outliers,
        outliers=outliers,
    )
    print(f"[INFO] Saved t-SNE plot to: {args.output}")
    print(f"[INFO] Saved outliers list to: {outliers_path}")


if __name__ == "__main__":
    main()
