import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import traceback
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp
from itertools import cycle

from shared_types import Modality
from model import UniBindClassifier, ForwardMode, MODALITY_TEMPLATES
from data_util import get_transform_fn, load_and_transform_text, load_label_mapping, get_normalization_tensors
from attack import AttackModel, APGDAttack, two_stage_attack
# Logging setup
LOGGER = logging.getLogger("tools.vis_embs")

# Prefer RAPIDS cuML TSNE if available; fallback to openTSNE. No sklearn TSNE.
# try:
#     from cuml.manifold import TSNE as CuMLTSNE  # type: ignore
#     _HAS_CUML_TSNE = True
# except Exception:
#     CuMLTSNE = None  # type: ignore
_HAS_CUML_TSNE = False

try:
    from openTSNE import TSNE as OpenTSNE  # type: ignore
    _HAS_OPENTSNE = True
except Exception:
    OpenTSNE = None  # type: ignore
    _HAS_OPENTSNE = False

def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort convert various GPU/array types to NumPy on host."""
    try:
        # cupy ndarray
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    try:
        # cuDF objects
        import cudf  # type: ignore
        if isinstance(x, (cudf.Series, cudf.DataFrame)):
            return x.to_numpy()
    except Exception:
        pass
    # Generic adapters
    if hasattr(x, "to_numpy"):
        try:
            return x.to_numpy()  # type: ignore
        except Exception:
            pass
    if hasattr(x, "get"):
        try:
            return x.get()  # type: ignore
        except Exception:
            pass
    return np.asarray(x)

def _select_tsne_backend() -> str:
    """Return 'cuml' if available, else 'opentsne'; raise if none present."""
    if _HAS_CUML_TSNE:
        print("[INFO] TSNE backend selected: cuML")
        return "cuml"
    if _HAS_OPENTSNE:
        print("[INFO] TSNE backend selected: openTSNE")
        return "opentsne"
    raise ImportError(
        "No TSNE backend available. Install RAPIDS cuML (preferred) or openTSNE."
    )

# Default LoRA checkpoints for robust variants (if user doesn't specify lora_weights)
LORA_WEIGHTS_MAP: Dict[str, Dict[str, str]] = {
    "image": {
        "eps2": "./ckpts/image_eps2_lora_weights.pt",
        "eps4": "./ckpts/image_eps4_lora_weights.pt",
    },
    # Provide audio keys for reference, but these files may not exist by default.
    # We'll validate existence at runtime and gracefully fall back to original weights.
    "audio": {
        "eps2": "./ckpts/audio_eps2_lora_weights.pt",
        "eps4": "./ckpts/audio_eps4_lora_weights.pt",
    },
}

def run_tsne(
    X: np.ndarray,
    perplexity: int = 5,
    random_state: int = 42,
    learning_rate: int = 400,
    n_iter: int = 2000,
    early_exaggeration: float = 2.0,
    initialization: str = "random",
    pca_dims: Optional[int] = None,
) -> np.ndarray:
    if X.shape[0] <= 2:
        # Not enough points for TSNE – return trivial layout
        return np.hstack([
            np.linspace(0.25, 0.75, X.shape[0]).reshape(-1, 1),
            np.linspace(0.25, 0.75, X.shape[0]).reshape(-1, 1),
        ])

    backend = _select_tsne_backend()

    # Optional PCA pre-reduction
    X_fit = X
    if pca_dims and 0 < pca_dims < X.shape[1]:
        try:
            from sklearn.decomposition import PCA as SkPCA  # type: ignore
            pca = SkPCA(n_components=pca_dims, random_state=random_state)
            X_fit = pca.fit_transform(X)
        except Exception:
            X_fit = X

    t0 = time.perf_counter()
    if backend == "cuml":
        # cuML uses 'init' instead of 'initialization'
        init = "pca" if initialization.lower() == "pca" else "random"
        tsne = CuMLTSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            early_exaggeration=early_exaggeration,
            init=init,
            random_state=random_state,
            verbose=0,
        )
        coords = _to_numpy(tsne.fit_transform(X_fit))
    else:
        # openTSNE
        tsne = OpenTSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            early_exaggeration=early_exaggeration,
            initialization=initialization,
            random_state=random_state,
            n_jobs=-1,
        )
        coords = np.asarray(tsne.fit(X_fit))
    dt = time.perf_counter() - t0
    print(f"[INFO] TSNE ({backend}) finished: n={X.shape[0]}, d={X.shape[1]}, time={dt:.2f}s")

    # Normalize to [0,1] for consistent plotting margins
    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-9)
    return coords


def _normalize_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize each row vector to unit length."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve similarity transform (scale s, rotation R, translation t) s*source*R + t ~= target.
    source, target: (N, 2)
    Returns: (s, R (2x2), t (1x2))
    """
    # Center
    mu_s = source.mean(axis=0, keepdims=True)
    mu_t = target.mean(axis=0, keepdims=True)
    S0 = source - mu_s
    T0 = target - mu_t
    # Rotation via SVD
    M = S0.T @ T0  # 2x2
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Fix improper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    # Scale
    var_S = (S0 ** 2).sum()
    s = np.trace(R.T @ M) / (var_S + 1e-9)
    # Translation
    t = mu_t - s * mu_s @ R
    return float(s), R, t


def _fit_tsne_and_place_text(
    X_samples: np.ndarray,
    X_text: Optional[np.ndarray],
    tsne_cfg: Dict[str, Any],
    fallback_perplexity: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit t-SNE on samples; place text points using transform if available, else Procrustes alignment.
    Returns (coords_samples, coords_text_or_None).
    """
    if X_text is not None and X_text.ndim == 1:
        X_text = X_text[None, :]

    backend = _select_tsne_backend()

    # Optional PCA pre-reduction prior to TSNE
    X_fit = X_samples
    pca_dims = int(tsne_cfg.get("pca_dims", 50))
    random_state = int(tsne_cfg.get("random_state", 42))
    if pca_dims and 0 < pca_dims < X_samples.shape[1]:
        try:
            from sklearn.decomposition import PCA as SkPCA  # type: ignore
            pca = SkPCA(n_components=pca_dims, random_state=random_state)
            X_fit = pca.fit_transform(X_samples)
            X_text_p = pca.transform(X_text) if X_text is not None else None
        except Exception:
            X_text_p = X_text
    else:
        X_text_p = X_text

    t0 = time.perf_counter()
    if backend == "cuml":
        # cuML has no out-of-sample transform; fit on combined when text provided
        init = "pca"
        tsne = CuMLTSNE(
            n_components=int(tsne_cfg.get("n_components", 2)),
            perplexity=int(tsne_cfg.get("perplexity", fallback_perplexity)),
            learning_rate=int(tsne_cfg.get("learning_rate", 50)),
            n_iter=int(tsne_cfg.get("n_iter", 1000)),
            early_exaggeration=float(tsne_cfg.get("early_exaggeration", 8.0)),
            init=init,
            random_state=random_state,
            verbose=0,
        )
        if X_text_p is not None:
            X_stack = np.vstack([X_fit, X_text_p])
        else:
            X_stack = X_fit
        coords_all = _to_numpy(tsne.fit_transform(X_stack))
        dt = time.perf_counter() - t0
        print(f"[INFO] TSNE (cuml) fit: samples={X_fit.shape[0]}, text={(0 if X_text_p is None else X_text_p.shape[0])}, time={dt:.2f}s")
        # Normalize to [0,1]
        coords_all = (coords_all - coords_all.min(0)) / (coords_all.max(0) - coords_all.min(0) + 1e-9)
        if X_text_p is not None:
            n = X_fit.shape[0]
            return coords_all[:n], coords_all[n:]
        else:
            return coords_all, None
    else:
        # openTSNE supports transform of new points
        tsne = OpenTSNE(
            n_components=int(tsne_cfg.get("n_components", 2)),
            perplexity=int(tsne_cfg.get("perplexity", fallback_perplexity)),
            learning_rate=int(tsne_cfg.get("learning_rate", 50)),
            n_iter=int(tsne_cfg.get("n_iter", 1000)),
            early_exaggeration=float(tsne_cfg.get("early_exaggeration", 8.0)),
            initialization="pca",
            random_state=random_state,
            n_jobs=-1,
        )
        embedding = tsne.fit(X_fit)
        coords_samples = np.asarray(embedding)
        if X_text_p is not None and hasattr(embedding, "transform"):
            coords_text = embedding.transform(X_text_p)
        else:
            coords_text = None
        dt = time.perf_counter() - t0
        print(f"[INFO] TSNE (openTSNE) fit: samples={X_fit.shape[0]}, text={(0 if X_text_p is None else X_text_p.shape[0])}, time={dt:.2f}s")
        # Normalize to [0,1]
        allc = coords_samples if coords_text is None else np.vstack([coords_samples, coords_text])
        allc = (allc - allc.min(0)) / (allc.max(0) - allc.min(0) + 1e-9)
        if coords_text is not None:
            return allc[: len(coords_samples)], allc[len(coords_samples) :]
        else:
            return allc, None


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s.strip("_")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Task:
    modality: str
    dataset_name: str
    dataset_root: str
    val_json: Optional[str]
    train_json: Optional[str]
    classes: List[str]
    samples_per_class: int
    # Plot control: "per_class" (default) or "combined"
    plot_mode: str = "per_class"
    # Matching controls
    match_key: str = "label"        # which field to match: "label" or "data"
    match_mode: str = "contains"     # "contains" | "exact" | "regex"
    # For adversarial attack (2-stage APGD classification attack)
    attack_enabled: bool = False
    attack_epsilon: Optional[float] = None  # e.g., 2/255 or 4/255
    # For robust model (LoRA)
    model_variant: str = "original"  # "original" | "robust"
    robust_level: Optional[str] = None  # "eps2" | "eps4"
    lora_weights: Optional[str] = None
    # Required when attack_enabled=True to compute logits
    centre_emb_path: Optional[str] = None
    # Combined plot support
    plot_name: Optional[str] = None  # used when plot_mode == "combined"
    combine: Optional[List[Dict[str, Any]]] = None  # list of per-entry dicts
    # TSNE params override per task (optional)
    tsne: Optional[Dict[str, Any]] = None


@dataclass
class TaskEntry:
    modality: str
    dataset_name: str
    dataset_root: str
    val_json: Optional[str]
    train_json: Optional[str]
    classes: List[str]
    samples_per_class: int
    match_key: str = "label"
    match_mode: str = "contains"
    attack_enabled: bool = False
    attack_epsilon: Optional[float] = None
    model_variant: str = "original"
    robust_level: Optional[str] = None
    lora_weights: Optional[str] = None
    centre_emb_path: Optional[str] = None
    tsne: Optional[Dict[str, Any]] = None


def load_json_list(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def match_indices(
    records: List[Dict[str, Any]],
    pattern: str,
    key: str = "label",
    mode: str = "contains",
) -> List[int]:
    target = pattern.strip()
    out: List[int] = []
    if mode == "regex":
        try:
            rx = re.compile(target, flags=re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(target), flags=re.IGNORECASE)
        for i, r in enumerate(records):
            val = str(r.get(key, ""))
            if rx.search(val):
                out.append(i)
        return out
    # case-insensitive contains/exact
    target_l = target.lower()
    for i, r in enumerate(records):
        val = str(r.get(key, ""))
        val_l = val.lower()
        if mode == "exact":
            if val_l == target_l:
                out.append(i)
        else:  # contains
            if target_l in val_l:
                out.append(i)
    return out


def sample_file_paths(task: Task) -> Dict[str, List[str]]:
    records_val = load_json_list(task.val_json)
    records_train = load_json_list(task.train_json)
    combined = records_val + records_train

    base_paths: List[List[str]] = []
    for cls in task.classes:
        indices = match_indices(combined, cls, key=task.match_key, mode=task.match_mode)
        if not indices:
            print(f"[WARN] No matches for class '{cls}' in {task.dataset_name} – skipping.")
            base_paths.append([])
            continue
        k = min(task.samples_per_class, len(indices))
        pick = random.sample(indices, k=k)
        paths = [os.path.join(task.dataset_root, combined[i]["data"]) for i in pick]
        print(f"[INFO] Sampled {len(paths)} paths for class='{cls}'")
        base_paths.append(paths)
    return {cls: paths for cls, paths in zip(task.classes, base_paths)}


def build_unibind(
    modality: Modality,
    device: torch.device,
    pretrain_weights: str,
    use_flash_attention: bool,
    centre_embeddings: Optional[torch.Tensor] = None,
    centre_labels: Optional[List[str]] = None,
    label_to_index: Optional[Dict[str, int]] = None,
    use_lora: bool = False,
    lora_weights: Optional[str] = None,
) -> UniBindClassifier:
    model = UniBindClassifier(
        device=device,
        pretrain_weights=pretrain_weights,
        modality=modality,
        centre_embeddings=centre_embeddings,
        centre_labels=centre_labels,
        label_to_index=label_to_index,
        use_lora=use_lora,
        lora_weights=lora_weights,
        use_flash_attention=use_flash_attention,
    )
    model.eval()
    return model.to(device)


def encode_samples(
    model: UniBindClassifier,
    modality: Modality,
    paths: List[str],
    device: torch.device,
    resize: int = 224,
) -> Optional[np.ndarray]:
    if not paths:
        return None
    transform = get_transform_fn(modality)
    with torch.no_grad():
        x = transform(paths, device=device)
        emb = model(x, ForwardMode.EMBEDDINGS)
        return emb.detach().cpu().numpy()


def maybe_run_attack(
    model: UniBindClassifier,
    modality: Modality,
    inputs: torch.Tensor,
    device: torch.device,
    epsilon: float,
) -> torch.Tensor:
    # Build mean/std tensors compatible with modality
    mean, std = get_normalization_tensors(modality, device)
    labels = torch.zeros(inputs.size(0), dtype=torch.long, device=device)
    atk_model = AttackModel(model, mean, std)
    stage1 = APGDAttack(model=atk_model, norm="linf", n_iter=10, n_restarts=1, eps=epsilon, loss_type="ce", device=device, logger=None)
    stage2 = APGDAttack(model=atk_model, norm="linf", n_iter=10, n_restarts=1, eps=epsilon, loss_type="ce", device=device, logger=None)
    adv = two_stage_attack(None, model, inputs, labels, stage1, stage2, mean, std)
    return adv


def _attack_eps_to_float(eps_value: Optional[float | int]) -> Optional[float]:
    """Convert user-provided attack_epsilon to float in [0,1].
    Accepts integers like 0,2,4,8 (interpreted as eps/255), or legacy floats.
    """
    if eps_value is None:
        return None
    if isinstance(eps_value, int):
        return float(eps_value) / 255.0
    if isinstance(eps_value, float):
        return eps_value
    raise ValueError(f"Unsupported attack_epsilon type: {type(eps_value)}")


def encode_text_for_class(model: UniBindClassifier, modality: Modality, cls: str, device: torch.device) -> np.ndarray:
    template = MODALITY_TEMPLATES.get(modality, "a {}")
    prompt = template.format(cls)
    tokens = load_and_transform_text([prompt], device=device)
    with torch.no_grad():
        t = model.encode_text(tokens)
        v = t.detach().cpu().numpy()[0]
        # Ensure unit vector
        n = np.linalg.norm(v) + 1e-9
        return v / n


def plot_tsne_per_class(
    dataset_name: str,
    modality: Modality,
    cls: str,
    sample_emb: np.ndarray,
    text_emb: np.ndarray,
    out_dir: str,
    perplexity: int = 5,
) -> None:
    X = sample_emb
    X_all = np.vstack([X, text_emb[None, :]])
    coords = run_tsne(X_all, perplexity=perplexity)
    n = X.shape[0]

    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:n, 0], coords[:n, 1], s=50, c="#1f77b4", alpha=0.65, label="samples")
    plt.scatter(coords[n:, 0], coords[n:, 1], s=130, c="#ff7f0e", marker="*", edgecolors="k", linewidths=0.7, label="text: " + cls)
    plt.title(f"t-SNE: {dataset_name} [{modality.name}] — {cls}")
    plt.legend(loc="best")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{slugify(cls)}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def process_task(
    task: Task,
    device: torch.device,
    pretrain_weights: str,
    use_flash_attention: bool,
    base_output_dir: str,
    perplexity: int,
) -> None:
    # Handle combined plot mode
    if task.plot_mode == "combined":
        if not task.combine or not isinstance(task.combine, list):
            raise ValueError("For plot_mode='combined', 'combine' must be a list of entries.")

        all_embs: List[np.ndarray] = []
        all_colors: List[str] = []
        text_points: List[tuple] = []  # (text_emb, color, class_label)
        # Assign colors per CLASS (same class shares the same color across datasets/modalities)
        palette = list(plt.get_cmap('tab10').colors) + list(plt.get_cmap('tab20').colors)
        color_cycle = cycle(palette)
        class_color: Dict[str, str] = {}
        seen_text_class: set = set()

        for entry in task.combine:
            te = TaskEntry(
                modality=entry["modality"],
                dataset_name=entry["dataset_name"],
                dataset_root=entry.get("dataset_root", task.dataset_root),
                val_json=entry.get("val_json"),
                train_json=entry.get("train_json"),
                classes=entry["classes"],
                samples_per_class=int(entry.get("samples_per_class", task.samples_per_class)),
                match_key=entry.get("match_key", "label"),
                match_mode=entry.get("match_mode", "contains"),
                attack_enabled=bool(entry.get("attack_enabled", False)),
                attack_epsilon=entry.get("attack_epsilon"),
                model_variant=entry.get("model_variant", "original"),
                robust_level=entry.get("robust_level"),
                lora_weights=entry.get("lora_weights"),
                centre_emb_path=entry.get("centre_emb_path"),
                tsne=entry.get("tsne"),
            )

            modality = Modality(te.modality)
            cls2paths = sample_file_paths(Task(
                modality=te.modality,
                dataset_name=te.dataset_name,
                dataset_root=te.dataset_root,
                val_json=te.val_json,
                train_json=te.train_json,
                classes=te.classes,
                samples_per_class=te.samples_per_class,
                match_key=te.match_key,
                match_mode=te.match_mode,
            ))

            # Optionally load centres for robust/attack
            label_to_index = None
            centre_emb = None
            centre_labels = None
            if te.attack_enabled and te.centre_emb_path:
                centre_emb, centre_labels, label_to_index, _ = load_label_mapping(te.centre_emb_path, device)
            use_lora = te.model_variant == "robust"
            # Select default lora by robust_level if not provided
            lora_w = te.lora_weights
            if use_lora and (lora_w is None) and te.robust_level:
                lora_w = LORA_WEIGHTS_MAP.get(modality.value, {}).get(te.robust_level)
            # Validate LoRA path exists; if missing, fall back to original gracefully
            if use_lora and (not lora_w or not os.path.exists(lora_w)):
                print(
                    "[WARN] "
                    f"LoRA weights not found for {modality.name} (requested level='{te.robust_level}'). "
                    f"Path: {lora_w}. Falling back to original model."
                )
                use_lora = False
                lora_w = None

            model = build_unibind(modality, device, pretrain_weights, use_flash_attention,
                                  centre_embeddings=centre_emb, centre_labels=centre_labels,
                                  label_to_index=label_to_index, use_lora=use_lora, lora_weights=lora_w)

            emb_dir = os.path.join(base_output_dir, "embeddings", f"{te.dataset_name}-{modality.name}")
            ensure_dir(emb_dir)

            for cls, paths in cls2paths.items():
                if not paths:
                    continue
                # Resolve color for this class
                cls_key = cls.strip().lower()
                if cls_key not in class_color:
                    class_color[cls_key] = tuple(next(color_cycle))
                color = class_color[cls_key]
                transform = get_transform_fn(modality)
                with torch.no_grad():
                    x = transform(paths, device=device)
                if te.attack_enabled:
                    eps_float = _attack_eps_to_float(te.attack_epsilon)
                    if not eps_float:
                        raise ValueError("attack_enabled=True requires attack_epsilon (e.g., 2/255 or 4/255)")
                    x = maybe_run_attack(model, modality, x, device, eps_float)
                emb = model(x, ForwardMode.EMBEDDINGS).detach().cpu().numpy()
                emb = _normalize_rows(emb)  # ensure unit vectors
                all_embs.append(emb)
                all_colors += [color] * emb.shape[0]

                # Cache
                cls_slug = slugify(cls)
                np.save(os.path.join(emb_dir, f"{cls_slug}_samples.npy"), emb)
                text_emb = encode_text_for_class(model, modality, cls, device)
                np.save(os.path.join(emb_dir, f"{cls_slug}_text.npy"), text_emb)
                with open(os.path.join(emb_dir, f"{cls_slug}_meta.json"), "w") as f:
                    json.dump({
                        "dataset": te.dataset_name,
                        "modality": modality.name,
                        "class": cls,
                        "num_samples": int(emb.shape[0]),
                        "paths": paths,
                        "attack_enabled": te.attack_enabled,
                        "attack_epsilon": te.attack_epsilon,
                        "model_variant": te.model_variant,
                        "lora_weights": te.lora_weights,
                        "created": datetime.utcnow().isoformat() + "Z",
                    }, f, indent=2)
                print(f"[INFO] Cached embeddings for class='{cls}' in {emb_dir}")
                # Keep one text embedding per class label to overlay and legend
                if cls_key not in seen_text_class:
                    text_points.append((text_emb, color, cls))
                    seen_text_class.add(cls_key)

        if not all_embs:
            return
        X = np.vstack(all_embs)
        tsne_cfg = task.tsne or {}
        X_text = np.vstack([t[0][None, :] for t in text_points]) if text_points else None
        coords, coords_text = _fit_tsne_and_place_text(X, X_text, tsne_cfg, perplexity)

        plt.figure(figsize=(9, 7))
        plt.scatter(coords[:, 0], coords[:, 1], c=all_colors, s=36, alpha=0.65, linewidths=0)

        # Project text embeddings with the same TSNE fit? Simpler: append and re-run for alignment
        if text_points and coords_text is not None:
            for (pt, color, label), (cx, cy) in zip(text_points, coords_text):
                plt.scatter([cx], [cy], c=[color], marker='*', s=140, edgecolors='k', linewidths=0.7, label=label)

        title = task.plot_name or f"combined_tsne_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        plt.title(title)
        plt.legend(loc='best', fontsize=8)
        tsne_dir = os.path.join(base_output_dir, "tsne", "combined")
        ensure_dir(tsne_dir)
        out_path = os.path.join(tsne_dir, f"{slugify(title)}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved combined t-SNE figure: {out_path}")
        return

    # Default: per-class individual plots
    modality = Modality(task.modality) if isinstance(task.modality, str) else task.modality

    # Resolve samples
    cls2paths = sample_file_paths(task)

    # Optional: centres for attack
    label_to_index = None
    centre_emb = None
    centre_labels = None
    if task.attack_enabled and task.centre_emb_path:
        centre_emb, centre_labels, label_to_index, _ = load_label_mapping(task.centre_emb_path, device)

    use_lora = task.model_variant == "robust"
    lora_w = task.lora_weights
    if use_lora and (lora_w is None) and task.robust_level:
        lora_w = LORA_WEIGHTS_MAP.get(modality.value, {}).get(task.robust_level)
    # Validate LoRA path exists; if missing, fall back to original gracefully
    if use_lora and (not lora_w or not os.path.exists(lora_w)):
        print(
            "[WARN] "
            f"LoRA weights not found for {modality.name} (requested level='{task.robust_level}'). "
            f"Path: {lora_w}. Falling back to original model."
        )
        use_lora = False
        lora_w = None

    model = build_unibind(modality, device, pretrain_weights, use_flash_attention,
                          centre_embeddings=centre_emb, centre_labels=centre_labels,
                          label_to_index=label_to_index, use_lora=use_lora, lora_weights=lora_w)

    # Output structure
    emb_dir = os.path.join(base_output_dir, "embeddings", f"{task.dataset_name}-{modality.name}")
    tsne_dir = os.path.join(base_output_dir, "tsne", f"{task.dataset_name}-{modality.name}")
    ensure_dir(emb_dir)
    ensure_dir(tsne_dir)

    for cls, paths in tqdm(cls2paths.items(), desc=f"{task.dataset_name} [{modality.name}]", ncols=80):
        if not paths:
            continue
        transform = get_transform_fn(modality)
        with torch.no_grad():
            x = transform(paths, device=device)
        if task.attack_enabled:
            eps_float = _attack_eps_to_float(task.attack_epsilon)
            if not eps_float:
                raise ValueError("attack_enabled=True requires attack_epsilon (e.g., 2/255 or 4/255)")
            x = maybe_run_attack(model, modality, x, device, eps_float)
            print(f"[INFO] Ran adversarial attack (eps={task.attack_epsilon}) for class='{cls}'")
        # Encode
        sample_emb = model(x, ForwardMode.EMBEDDINGS).detach().cpu().numpy()
        sample_emb = _normalize_rows(sample_emb)  # ensure unit vectors
        if sample_emb is None:
            continue
        text_emb = encode_text_for_class(model, modality, cls, device)

        # Cache embeddings
        cls_slug = slugify(cls)
        np.save(os.path.join(emb_dir, f"{cls_slug}_samples.npy"), sample_emb)
        np.save(os.path.join(emb_dir, f"{cls_slug}_text.npy"), text_emb)
        meta = {
            "dataset": task.dataset_name,
            "modality": modality.name,
            "class": cls,
            "num_samples": int(sample_emb.shape[0]),
            "paths": paths,
            "attack_enabled": task.attack_enabled,
            "attack_epsilon": task.attack_epsilon,
            "model_variant": task.model_variant,
            "lora_weights": task.lora_weights,
            "created": datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(emb_dir, f"{cls_slug}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Cached embeddings for class='{cls}' in {emb_dir}")

        # Plot TSNE per class: fit on samples, then place text
        tsne_cfg = task.tsne or {}
        coords_samples, coords_text = _fit_tsne_and_place_text(sample_emb, text_emb[None, :], tsne_cfg, perplexity)
        n = sample_emb.shape[0]
        plt.figure(figsize=(7, 6))
        plt.scatter(coords_samples[:, 0], coords_samples[:, 1], s=50, c="#1f77b4", alpha=0.65, label="samples")
        if coords_text is not None:
            plt.scatter(coords_text[:, 0], coords_text[:, 1], s=130, c="#ff7f0e", marker="*", edgecolors="k", linewidths=0.7, label="text: " + cls)
        plt.title(f"t-SNE: {task.dataset_name} [{modality.name}] — {cls}")
        plt.legend(loc="best")
        out_path = os.path.join(tsne_dir, f"{slugify(cls)}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved t-SNE: {out_path}")


def _setup_logging(level: str) -> None:
    level_num = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_num,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print(f"[INFO] Logging initialized at level: {level}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample UniBind embeddings and plot per-class t-SNE with text (config-only)")
    # Multi-task config (required)
    p.add_argument("--config", type=str, default="tools/vis_configs/sample.json", help="Path to JSON config with an array of tasks")

    # Model + compute
    p.add_argument("--pretrain-weights", type=str, default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt")
    p.add_argument("--use-flash-attention", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # TSNE & output
    p.add_argument("--perplexity", type=int, default=5)
    p.add_argument("--output-dir", type=str, default="/data/output")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])  # noqa
    return p.parse_args()


def load_tasks_from_config(cfg_path: str) -> List[Task]:
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    tasks: List[Task] = []
    for item in cfg:
        # Support combined mode by passing through 'plot_mode' and 'combine'
        tasks.append(Task(
            modality=item.get("modality", "image"),
            dataset_name=item.get("dataset_name", ""),
            dataset_root=item.get("dataset_root", "/data/datasets"),
            val_json=item.get("val_json"),
            train_json=item.get("train_json"),
            classes=item.get("classes", []),
            samples_per_class=int(item.get("samples_per_class", 40)),
            plot_mode=item.get("plot_mode", "per_class"),
            match_key=item.get("match_key", "label"),
            match_mode=item.get("match_mode", "contains"),
            attack_enabled=bool(item.get("attack_enabled", False)),
            attack_epsilon=item.get("attack_epsilon"),
            model_variant=item.get("model_variant", "original"),
            robust_level=item.get("robust_level"),
            lora_weights=item.get("lora_weights"),
            centre_emb_path=item.get("centre_emb_path"),
            plot_name=item.get("plot_name"),
            combine=item.get("combine"),
            tsne=item.get("tsne"),
        ))
    return tasks


def build_single_task_from_args(args: argparse.Namespace) -> Task:
    raise RuntimeError("Single-task CLI is disabled. Use --config with an array of tasks.")


def run_task_on_gpu(
    task: Task,
    gpu_id: int,
    pretrain_weights: str,
    use_flash_attention: bool,
    base_output_dir: str,
    perplexity: int,
    seed: int,
):
    try:
        # Set default CUDA device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        set_seed(seed + int(gpu_id))
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Starting task on GPU {gpu_id}: {task.dataset_name}-{task.modality}")
        process_task(
            task=task,
            device=device,
            pretrain_weights=pretrain_weights,
            use_flash_attention=use_flash_attention,
            base_output_dir=base_output_dir,
            perplexity=perplexity,
        )
        print(f"[INFO] Completed task on GPU {gpu_id}: {task.dataset_name}-{task.modality}")
    except Exception as e:
        print(f"[ERROR] [GPU {gpu_id}] Task {task.dataset_name}-{task.modality} failed: {e}")
        traceback.print_exc()
        raise


def main():
    args = parse_args()
    _setup_logging(args.log_level)
    # Explicitly report which TSNE backend will be used (INFO level)
    try:
        _backend_name = _select_tsne_backend()
        human = "cuML" if _backend_name == "cuml" else "openTSNE"
        print(f"[INFO] Using TSNE backend: {_backend_name} ({human})")
    except ImportError as e:
        print(f"[ERROR] {str(e)}")
        raise
    set_seed(args.seed)
    device = torch.device(args.device)

    # Build task list (config is required)
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        raise FileNotFoundError(f"Config file not found: {args.config}")
    tasks = load_tasks_from_config(args.config)
    print(f"[INFO] Loaded {len(tasks)} tasks from config: {args.config}")

    # Decide GPU assignment
    num_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_available == 0:
        print("[ERROR] No CUDA device available; multi-GPU execution requires GPUs.")
        raise RuntimeError("No CUDA device available; multi-GPU execution requires GPUs.")

    # Determine GPU IDs to use: use all available by default
    gpu_ids = list(range(num_available))
    print(f"[INFO] Available GPUs: {gpu_ids}")

    # Map tasks to GPUs round-robin
    jobs = []
    for i, t in enumerate(tasks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        print(f"[INFO] Assign task {i}: {t.dataset_name}-{t.modality} -> GPU {gpu_id}")
        jobs.append((t, gpu_id, args.pretrain_weights, args.use_flash_attention, args.output_dir, args.perplexity, args.seed))

    # Parallel execution: one process per GPU (or per task if fewer tasks)
    procs = min(len(gpu_ids), len(jobs))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=procs, maxtasksperchild=1) as pool:
        pool.starmap(run_task_on_gpu, jobs)


if __name__ == "__main__":
    main()
