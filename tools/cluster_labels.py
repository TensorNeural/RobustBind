import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import logging
import time
try:
    import hdbscan  # type: ignore
except Exception as e:
    hdbscan = None  # will error later with a friendly message
from sklearn.cluster import AgglomerativeClustering

import sys
from pathlib import Path

from google import genai as google_genai  # type: ignore

# Ensure repository root is on sys.path when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import UniBindClassifier
from shared_types import Modality
from data_util import load_and_transform_text


DATASET_TO_CLASSES_FILE: Dict[str, str] = {
    # Image
    "ImageNet-1K": "./datasets/ImageNet-1K/classes_imagenet.json",
    "Places365": "./datasets/Places365/classes_places365.json",
    # Video
    "UCF-101": "./datasets/UCF-101/classes.json",
    "MSR-VTT": "./datasets/MSR-VTT/classes.json",
    "Kinetics-400": "./datasets/Kinetics-400/label_to_id.json",
    # Thermal
    "LLVIP": "./datasets/LLVIP/classes_llvip.json",
    # Audio
    "ESC-50": "./datasets/ESC-50/classes.json",
    "UrbanSound8K": "./datasets/UrbanSound8K/classes.json",
    # Point cloud
    "ModelNet40": "./datasets/ModelNet40/classes.json",
    # Event
    "N-Caltech-101": "./datasets/N-Caltech-101/classes.json"  
}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_label(label: str) -> str:
    s = label.strip()
    s = re.sub(r"[_-]+", " ", s)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)  # remove parenthetical notes
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\d+$", "", s)  # drop trailing digits like 'crane2'
    return s.strip().lower()


def load_classes_from_json(path: str) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    # Expect mapping {label: idx} or list of labels
    if isinstance(data, dict):
        return list(data.keys())
    elif isinstance(data, list):
        return list(map(str, data))
    else:
        raise ValueError(f"Unsupported classes.json format in {path}")


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n


def encode_texts(model: UniBindClassifier, labels: List[str], device: torch.device, template: str) -> np.ndarray:
    prompts = [template.format(lbl) for lbl in labels]
    tokens = load_and_transform_text(prompts, device=device)
    with torch.no_grad():
        emb = model.encode_text(tokens).detach().cpu().numpy()
    return _normalize_rows(emb)


def encode_texts_ensemble(
    model: UniBindClassifier,
    labels: List[str],
    device: torch.device,
    templates: List[str],
) -> np.ndarray:
    acc = None
    for tpl in templates:
        emb = encode_texts(model, labels, device=device, template=tpl)
        acc = emb if acc is None else acc + emb
    acc = acc / float(len(templates))
    return _normalize_rows(acc)


@dataclass
class LabelEntry:
    dataset: str
    original_label: str
    clean_label: str


def infer_dataset_name(classes_json_path: str) -> str:
    # Use parent folder name as dataset name by default
    parent = os.path.basename(os.path.dirname(os.path.abspath(classes_json_path)))
    return parent or os.path.splitext(os.path.basename(classes_json_path))[0]


def hdbscan_cluster(
    emb: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.05,
    cluster_selection_method: str = "eom",
) -> Tuple[np.ndarray, np.ndarray]:
    if hdbscan is None:
        raise RuntimeError("hdbscan is not installed. Please install it with: pip install hdbscan")
    # HDBSCAN on L2-normalized embeddings; 'euclidean' is consistent with cosine ranking under normalization
    # Note: sklearn BallTree used under the hood doesn't support 'cosine' here; map to 'euclidean'
    metric_eff = "euclidean" if metric == "cosine" else metric
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric_eff,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(emb)  # -1 denotes noise
    uniq = sorted([int(u) for u in np.unique(labels) if int(u) >= 0])
    if len(uniq) == 0:
        # No clusters formed; return placeholders
        return labels, np.zeros((0, emb.shape[1]), dtype=emb.dtype)
    # Build centroids per cluster label (ordered by sorted unique labels)
    centroids = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        centroid = emb[idx].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        centroids.append(centroid)
    return labels, np.vstack(centroids)


def assign_by_cosine(emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # emb and centroids are unit-normalized
    sims = emb @ centroids.T  # (N, K)
    return np.argmax(sims, axis=1)


def build_groups_from_assignments(entries: List[LabelEntry], assign: np.ndarray) -> List[Dict[str, object]]:
    # Build nested groups: per cluster_id, aggregate labels by dataset
    groups_map: Dict[int, Dict[str, List[Dict[str, str]]]] = {}
    for e, gid in zip(entries, assign):
        gid_i = int(gid)
        ds_map = groups_map.setdefault(gid_i, {})
        ds_map.setdefault(e.dataset, []).append({
            "original_label": e.original_label,
            "clean_label": e.clean_label,
        })

    groups: List[Dict[str, object]] = []
    for gid in sorted(groups_map.keys()):
        ds_entries = groups_map[gid]
        datasets_list = [
            {"dataset": ds, "labels": ds_entries[ds]}
            for ds in sorted(ds_entries.keys())
        ]
        groups.append({"group_id": int(gid), "datasets": datasets_list})
    return groups


def main():
    p = argparse.ArgumentParser(description="Cluster dataset class labels using text embeddings")
    p.add_argument("--algo", type=str, default="agglomerative", choices=["hdbscan", "agglomerative"],
                   help="Clustering algorithm to use")
    # Export-only options (no embedding/clustering)
    p.add_argument("--export-class-list", action="store_true", default=False,
                   help="Export consolidated class list per dataset for LLM-based grouping and exit")
    p.add_argument("--classes-out", type=str, default="./output/classes_all.json",
                   help="Output path for consolidated class list when using --export-class-list")
    p.add_argument("--gemini-prompt-out", type=str, default="./output/gemini_grouping_prompt.txt",
                   help="Output path for a ready-to-use Gemini prompt scaffold")
    # Gemini LLM grouping options
    p.add_argument("--gemini-run", action="store_true", default=True,
                   help="Use Google Gemini to group classes semantically and write label_groups_gemini.json")
    p.add_argument("--gemini-model", type=str, default="gemini-2.5-flash",
                   help="Gemini model name (e.g., gemini-2.5-flash, gemini-1.5-pro)")
    p.add_argument("--gemini-api-key-env", type=str, default="GOOGLE_API_KEY",
                   help="Environment variable name holding the Gemini API key")
    p.add_argument("--gemini-out", type=str, default="./output/label_groups_gemini.json",
                   help="Output JSON path for Gemini-produced groups")
    p.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logs for Gemini path")
    p.add_argument("--pretrain-weights", type=str, default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--template", type=str, default="a {}",
                   help="Prompt template for text encoding, must contain '{}' placeholder")
    p.add_argument("--prompt-ensemble", action="store_true", default=False,
                   help="Use an ensemble of text templates and average embeddings (default: on)")
    p.add_argument("--no-prompt-ensemble", dest="prompt_ensemble", action="store_false",
                   help="Disable prompt ensemble and use a single --template")
    # HDBSCAN parameters
    p.add_argument("--min-cluster-size", type=int, default=8, help="HDBSCAN min_cluster_size")
    p.add_argument("--min-samples", type=int, default=0, help="HDBSCAN min_samples; 0=auto (None)")
    p.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Distance metric for HDBSCAN")
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.05, help="HDBSCAN cluster_selection_epsilon")
    p.add_argument("--cluster-selection-method", type=str, default="eom", choices=["eom", "leaf"], help="HDBSCAN cluster_selection_method")
    # Agglomerative parameters
    p.add_argument("--agg-metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="Agglomerative metric")
    p.add_argument("--agg-linkage", type=str, default="average", choices=["average", "complete", "single"], help="Agglomerative linkage")
    p.add_argument("--agg-distance-threshold", type=float, default=0.5, help="Agglomerative distance_threshold (use with n_clusters=None)")
    p.add_argument("--agg-n-clusters", type=int, default=-1, help="Agglomerative fixed n_clusters; -1 means None (use distance_threshold)")
    p.add_argument("--output-dir", type=str, default="./output")
    p.add_argument("--output-name", type=str, default="label_groups.json")
    p.add_argument("--keep-noise-singletons", action="store_true", default=True,
                   help="Keep HDBSCAN noise points each as their own singleton group (default: on)")
    p.add_argument("--collapse-noise", dest="keep_noise_singletons", action="store_false",
                   help="Collapse all noise points instead of singleton groups")
    args = p.parse_args()

    # Logging setup
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("cluster_labels")

    set_seed(args.seed)
    device = torch.device(args.device)

    # Ensure working directory is repo root so relative assets (e.g., bpe/) resolve
    try:
        os.chdir(str(REPO_ROOT))
    except Exception:
        pass

    # Use static dataset→classes mapping; include only those files that exist
    class_files: List[str] = []
    ds_names: List[str] = []
    for ds, rel_path in DATASET_TO_CLASSES_FILE.items():
        path = rel_path
        if not os.path.isabs(path):
            path = os.path.join(str(REPO_ROOT), rel_path)
        if os.path.exists(path):
            class_files.append(path)
            ds_names.append(ds)
        else:
            print(f"[WARN] Skipping {ds}: classes file not found: {rel_path}")
    if not class_files:
        raise RuntimeError("No classes files found from DATASET_TO_CLASSES_FILE mapping.")

    # Load labels
    entries: List[LabelEntry] = []
    for cj, dn in zip(class_files, ds_names):
        labels = load_classes_from_json(cj)
        for lbl in labels:
            entries.append(LabelEntry(dataset=dn, original_label=str(lbl), clean_label=clean_label(str(lbl))))

    if not entries:
        raise RuntimeError("No labels loaded from provided classes.json files.")

    # Export and exit if requested (no model required)
    if args.export_class_list:
        ensure_dir(os.path.dirname(args.classes_out) or ".")
        classes_payload: Dict[str, List[str]] = {}
        for e in entries:
            classes_payload.setdefault(e.dataset, []).append(e.original_label)
        with open(args.classes_out, "w") as f:
            json.dump(classes_payload, f, indent=2)

        # Write a prompt scaffold for Gemini/manual LLM grouping
        prompt = (
            "You are given multiple datasets, each with a list of class labels. "
            "Group semantically similar labels across datasets into coherent clusters.\n\n"
            "Instructions:\n"
            "- Favor meaning over spelling/casing/underscores.\n"
            "- If a label is ambiguous or an outlier, place it alone in its own cluster.\n"
            "- Do not hallucinate new labels; only group those provided.\n"
            "- Keep dataset provenance: for each cluster, list labels grouped by dataset.\n\n"
            "Output JSON schema:\n"
            "{\n  \"groups\": [\n    {\n      \"group_id\": <int>,\n      \"datasets\": [\n        { \"dataset\": <name>, \"labels\": [<label>, ...] },\n        ...\n      ]\n    },\n    ...\n  ]\n}\n\n"
            "Here are the datasets and their labels (JSON object: dataset -> labels array):\n"
        )
        ensure_dir(os.path.dirname(args.gemini_prompt_out) or ".")
        with open(args.gemini_prompt_out, "w") as f:
            f.write(prompt)
            f.write(json.dumps(classes_payload, indent=2))
            f.write("\n")
        print(f"📄 Wrote class list to {args.classes_out}")
        print(f"📝 Wrote Gemini prompt scaffold to {args.gemini_prompt_out}")
        print("Tip: Paste the prompt into Gemini or call its API with the prompt and expect the specified JSON schema.")
        return

    # Gemini run: build prompt and call the API, then save groups and exit
    if args.gemini_run:
        if google_genai is None:
            raise RuntimeError("google-genai SDK not installed. Install with: pip install google-genai")

        classes_payload: Dict[str, List[str]] = {}
        for e in entries:
            classes_payload.setdefault(e.dataset, []).append(e.original_label)

        # Log basic stats
        num_datasets = len(classes_payload)
        num_labels = sum(len(v) for v in classes_payload.values())
        logger.info(f"Gemini run requested — datasets={num_datasets}, labels={num_labels}")

        prompt = (
            "You are given multiple datasets, each with a list of class labels. "
            "Group semantically similar labels across datasets into coherent clusters.\n\n"
            "Instructions:\n"
            "- Favor meaning over spelling/casing/underscores.\n"
            "- If a label is ambiguous or an outlier, place it alone in its own cluster.\n"
            "- Do not hallucinate new labels; only group those provided.\n"
            "- Keep dataset provenance: for each cluster, list labels grouped by dataset.\n\n"
            "Return ONLY valid JSON, no prose. Schema:\n"
            "{\n  \"groups\": [\n    {\n      \"group_id\": <int>,\n      \"datasets\": [\n        { \"dataset\": <name>, \"labels\": [<label>, ...] },\n        ...\n      ]\n    },\n    ...\n  ]\n}\n\n"
            "Datasets and labels (JSON object: dataset -> labels array):\n" + json.dumps(classes_payload)
        )

        api_key = os.environ.get(args.gemini_api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: set environment variable {args.gemini_api_key_env}")
        logger.info(f"Gemini model={args.gemini_model} | key_env={args.gemini_api_key_env} | prompt_chars={len(prompt)}")

        client = google_genai.Client(api_key=api_key)
        t0 = time.time()
        resp = client.models.generate_content(model=args.gemini_model, contents=prompt)
        dt = time.time() - t0
        logger.info(f"Gemini API call completed in {dt:.2f}s")
        text = getattr(resp, "text", None)
        if not text:
            # Some SDK variants place output in 'candidates'
            try:
                text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
            except Exception:
                raise RuntimeError("Gemini response has no text content")
        logger.info(f"Gemini response length={len(text)} chars")

        # Parse JSON robustly
        def _parse_json_payload(s: str) -> Dict[str, object]:
            s = s.strip()
            try:
                return json.loads(s)
            except Exception:
                # Extract first {...} block
                start = s.find("{")
                end = s.rfind("}")
                if start >= 0 and end > start:
                    try:
                        return json.loads(s[start:end+1])
                    except Exception as e:
                        raise RuntimeError(f"Failed to parse Gemini JSON: {e}")
                raise RuntimeError("No JSON object found in Gemini output")

        logger.info("Parsing Gemini JSON output…")
        groups_obj = _parse_json_payload(text)
        if not isinstance(groups_obj, dict) or "groups" not in groups_obj:
            raise RuntimeError("Gemini output missing 'groups' key in JSON object")

        groups = groups_obj["groups"]
        if not isinstance(groups, list):
            raise RuntimeError("'groups' must be a list of cluster objects")

        ensure_dir(os.path.dirname(args.gemini_out) or ".")
        with open(args.gemini_out, "w") as f:
            json.dump({
                "created": datetime.utcnow().isoformat() + "Z",
                "num_labels": len(entries),
                "num_clusters": len(groups),
                "groups": groups,
            }, f, indent=2)
        logger.info(f"Wrote Gemini groups to {args.gemini_out}")
        print(f"✅ Wrote Gemini label groups to {args.gemini_out} ({len(entries)} labels, {len(groups)} clusters)")
        return

    # Resolve pretrain weights path relative to repo root if needed
    pretrain_weights = args.pretrain_weights
    if not os.path.isabs(pretrain_weights):
        cand = os.path.join(str(REPO_ROOT), pretrain_weights)
        if os.path.exists(cand):
            pretrain_weights = cand

    # Build model (text encoder)
    model = UniBindClassifier(device=device, pretrain_weights=pretrain_weights, modality=Modality.IMAGE, use_flash_attention=True)
    model.eval()
    model.to(device)

    # Encode and normalize (optionally with prompt ensemble)
    clean_texts = [e.clean_label for e in entries]
    if args.prompt_ensemble:
        templates = [
            "a photo of a {}",
            "a picture of {}",
            "a photo of {}",
            "an image of {}",
            "a close-up photo of {}",
        ]
        emb = encode_texts_ensemble(model, clean_texts, device=device, templates=templates)
    else:
        emb = encode_texts(model, clean_texts, device=device, template=args.template)

    # Cluster
    if args.algo == "hdbscan":
        min_samples = None if (args.min_samples is None or int(args.min_samples) <= 0) else int(args.min_samples)
        labels_algo, _centroids = hdbscan_cluster(
            emb,
            min_cluster_size=int(args.min_cluster_size),
            min_samples=min_samples,
            metric=str(args.metric),
            cluster_selection_epsilon=float(args.cluster_selection_epsilon),
            cluster_selection_method=str(args.cluster_selection_method),
        )
    else:
        # AgglomerativeClustering with cosine metric and average linkage by default
        n_clusters = None if int(args.agg_n_clusters) <= 0 else int(args.agg_n_clusters)
        distance_threshold = None if n_clusters is not None else float(args.agg_distance_threshold)
        # sklearn supports metric='cosine' with linkage!='ward'
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=str(args.agg_metric),
            linkage=str(args.agg_linkage),
            distance_threshold=distance_threshold,
            compute_distances=False,
        )
        labels_algo = agg.fit_predict(emb)
    # Build group assignment from raw HDBSCAN labels; do not force-reassign to centroids
    uniq = sorted([int(u) for u in np.unique(labels_algo) if int(u) >= 0])
    label_to_group = {lbl: i for i, lbl in enumerate(uniq)}
    assign = np.empty_like(labels_algo)
    next_gid = len(uniq)
    for i, h in enumerate(labels_algo):
        if h >= 0:
            assign[i] = label_to_group[int(h)]
        else:
            if args.keep_noise_singletons:
                assign[i] = next_gid
                next_gid += 1
            else:
                assign[i] = -1  # will be grouped under a single noise dataset bucket

    # If collapsing noise, map -1 to a final group id
    if not args.keep_noise_singletons and np.any(assign < 0):
        noise_gid = int(next_gid)
        assign = np.where(assign < 0, noise_gid, assign)

    groups = build_groups_from_assignments(entries, assign)
    num_clusters = len(groups)

    ensure_dir(args.output_dir)
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w") as f:
        json.dump({
            "created": datetime.utcnow().isoformat() + "Z",
            "num_labels": len(entries),
            "num_clusters": int(num_clusters),
            "groups": groups,
        }, f, indent=2)

    print(f"✅ Wrote label groups to {out_path} ({len(entries)} labels, {num_clusters} clusters)")


if __name__ == "__main__":
    main()
