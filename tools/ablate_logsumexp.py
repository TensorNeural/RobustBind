#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
import logging
from typing import List, Dict, Tuple
from multiprocessing import get_context

import torch
import torch_scatter
from tqdm import tqdm

from model import UniBindClassifier, Modality
from shared_types import BindModelType
from data_util import JsonDataset, CollateFn
from utils.utils import load_centre_embeddings


# ==================== Notes ====================
# (UniBind optional CLI defaults are inlined in argparse below)

MODALITY_DATASETS: Dict[Modality, Dict[str, str]] = {
    Modality.IMAGE: {
        "dataset_name": "ImageNet-1K",
        "dataset_root": "/data/datasets/ImageNet-1K",
        "val_json": "./datasets/ImageNet-1K/val_data.json",
        "centre_embeddings_path": "./centre_embs/image_in_center_embeddings.pkl",
    },
    Modality.AUDIO: {
        "dataset_name": "ESC-50",
        "dataset_root": "/data/datasets/ESC-50",
        "val_json": "./datasets/ESC-50/val_data.json",
        "centre_embeddings_path": "./centre_embs/audio_esc_center_embeddings.pkl",
    },
    Modality.EVENT: {
        "dataset_name": "N-Caltech-101",
        "dataset_root": "/data/datasets/N-Caltech-101",
        "val_json": "./datasets/N-Caltech-101/val_data.json",
        "centre_embeddings_path": "./centre_embs/event_caltech_center_embeddings.pkl",
    },
    Modality.POINT: {
        "dataset_name": "ModelNet40",
        "dataset_root": "/data/datasets/ModelNet40",
        "val_json": "./datasets/ModelNet40/val_data.json",
        "centre_embeddings_path": "./centre_embs/point_modelnet40_center_embeddings.pkl",
    },
    Modality.VIDEO: {
        "dataset_name": "MSR-VTT",
        "dataset_root": "/data/datasets/MSR-VTT",
        "val_json": "./datasets/MSR-VTT/val_data.json",
        "centre_embeddings_path": "./centre_embs/video_msrvtt_center_embeddings.pkl",
    },
    Modality.THERMAL: {
        "dataset_name": "LLVIP",
        "dataset_root": "/data/datasets/LLVIP",
        "val_json": "./datasets/LLVIP/val_data.json",
        "centre_embeddings_path": "./centre_embs/thermal_llvip_center_embeddings.pkl",
    },
}

CLEAN_VAL_BATCH_SIZE_MAP: Dict[str, int] = {
    "ImageNet-1K": 2000,
    "Places365": 2000,
    "ModelNet40": 64,
    "ShapeNet": 64,
    "ESC-50": 50,
    "UrbanSound8K": 50,
    "LLVIP": 2000,
    "RGB-T": 16,
    "MSR-VTT": 100,
    "UCF-101": 100,
    "N-Caltech-101": 500,
    "N-ImageNet-1K": 500,
}


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("ablate_logsumexp")
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def parse_modality(s: str) -> Modality:
    s = s.strip().lower()
    for m in Modality:
        if m.value == s:
            return m
    raise argparse.ArgumentTypeError(f"Unknown modality: {s}")


def parse_floats(csv: str) -> List[float]:
    vals = []
    for tok in str(csv).split(','):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise argparse.ArgumentTypeError("Provide at least one temperature value")
    return vals


def load_label_mapping(center_emb_path: str, device: torch.device):
    centre_embeddings, labels = load_centre_embeddings(center_emb_path, device)
    centre_embeddings = centre_embeddings / centre_embeddings.norm(dim=-1, keepdim=True)
    unique_lbls = sorted(set(labels))
    label_to_index = {l: i for i, l in enumerate(unique_lbls)}
    index_to_label = {i: l for l, i in label_to_index.items()}
    return centre_embeddings, labels, label_to_index, index_to_label


def _read_dataset_label_set(val_json_path: str) -> set:
    """Read the dataset val JSON and return the set of label strings present."""
    try:
        import json
        with open(val_json_path, 'r') as f:
            items = json.load(f)
        return set([str(it.get('label', '')).strip() for it in items])
    except Exception:
        return set()


def sanity_check_label_overlap(ds_name: str, val_json: str, lbl_to_idx: Dict[str, int], logger: logging.Logger, gpu_prefix: str = "") -> bool:
    """Check overlap between dataset labels in val_json and centre label mapping keys.

    Returns True if overlap is reasonable; logs a clear warning if very low.
    """
    ds_labels = _read_dataset_label_set(val_json)
    if not ds_labels:
        logger.warning(f"{gpu_prefix} [{ds_name}] Could not read dataset labels from {val_json}; skipping overlap check.")
        return True
    centre_labels_set = set(lbl_to_idx.keys())
    overlap = ds_labels & centre_labels_set
    overlap_ratio = (len(overlap) / max(1, len(ds_labels)))
    logger.info(f"{gpu_prefix} [{ds_name}] Label overlap: {len(overlap)}/{len(ds_labels)} = {overlap_ratio:.3f}")
    if overlap_ratio < 0.1:
        logger.error(f"{gpu_prefix} [{ds_name}] Very low label overlap with centre embeddings. Likely wrong centre file is being used. Aborting this dataset.")
        return False
    return True


def build_loader(modality: Modality, dataset_root: str, json_path: str, label_to_index: dict,
                 batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    dataset = JsonDataset(dataset_root, json_path, label_to_index, max_samples=None, debug=False)
    collate = CollateFn(modality, True, BindModelType.UNIBIND)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate,
    )


def compute_logits_lse_scatter(similarity: torch.Tensor, centre_label_indices: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute class logits via masked log-sum-exp without creating a BxCxN tensor.

    similarity: (B, N) cosine similarities
    centre_label_indices: (N,) class index for each centre
    temperature: scalar temperature
    returns: (B, C) logits
    """
    # Match UniBind _scatter_logsumexp implementation (no enforced dtype casting per user request)
    class_raw_scores = torch_scatter.scatter_logsumexp(similarity * temperature, centre_label_indices, dim=1)
    return class_raw_scores / temperature


def top1_accuracy_chunked(
    sims_full_cpu: torch.Tensor,
    labels_full_cpu: torch.Tensor,
    centre_label_indices: torch.Tensor,
    temperature: float,
    device: torch.device,
    row_batch_size: int,
) -> float:
    """Compute top-1 accuracy for large similarity matrices without loading the full (B,N) tensor on GPU.

    This streams rows in chunks to avoid allocating a massive (B,N) tensor on device (e.g. 50k x 50k for ImageNet-1K
    with 50 centres per class). Only the per-chunk similarity rows are transferred.

    Args:
        sims_full_cpu: (B, N) full similarity matrix on CPU.
        labels_full_cpu: (B,) labels corresponding to rows of sims_full_cpu (CPU tensor).
        centre_label_indices: (N,) centre -> class index mapping (device tensor).
        temperature: scalar temperature.
        device: target CUDA device.
        chunk_size: number of rows per streaming chunk.
    Returns:
        top-1 accuracy (float).
    """
    assert sims_full_cpu.device.type == 'cpu', "Expect sims on CPU for streaming"
    B = sims_full_cpu.shape[0]
    correct = 0
    total = labels_full_cpu.numel()
    pbar = tqdm(range(0, B, row_batch_size), desc="stream-chunks", leave=False)
    for start in pbar:
        end = min(start + row_batch_size, B)
        sims_chunk = sims_full_cpu[start:end].to(device, non_blocking=True)
        labels_chunk = labels_full_cpu[start:end].to(device, non_blocking=True)
        logits_chunk = compute_logits_lse_scatter(sims_chunk, centre_label_indices, temperature)
        preds = logits_chunk.argmax(dim=1)
        batch_correct = (preds == labels_chunk).sum().item()
        correct += batch_correct
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(acc=f"{correct/total:.4f}", processed=end, total=B)
        del sims_chunk, labels_chunk, logits_chunk, preds
        torch.cuda.empty_cache()
    if hasattr(pbar, 'close'):
        pbar.close()
    return correct / total if total else 0.0


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.numel() if labels.numel() > 0 else 0.0


def detect_gpu_indices() -> List[int]:
    return list(range(torch.cuda.device_count()))


def validate_cached_tensors(
    sims_full: torch.Tensor,
    labels_full: torch.Tensor,
    centre_emb: torch.Tensor,
    num_classes: int,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, torch.Tensor, bool, str]:
    """Validate cached similarities/labels against current centre embeddings.

    Returns (sims, labels, ok, reason). Light fix-ups are applied when safe.
    """
    try:
        if sims_full is None or labels_full is None:
            return sims_full, labels_full, False, "cache missing"
        if sims_full.ndim != 2:
            return sims_full, labels_full, False, f"sims ndim={sims_full.ndim} != 2"
        if sims_full.shape[1] != centre_emb.shape[0]:
            return sims_full, labels_full, False, f"sims cols {sims_full.shape[1]} != centres {centre_emb.shape[0]}"
        if labels_full.ndim != 1:
            labels_full = labels_full.view(-1)
        if labels_full.dtype != torch.long:
            logger.info("Casting cached labels to torch.long")
            labels_full = labels_full.long()
        if labels_full.numel() != sims_full.shape[0]:
            return sims_full, labels_full, False, f"labels {labels_full.numel()} != sims rows {sims_full.shape[0]}"
        if labels_full.numel() > 0:
            minv = int(labels_full.min().item())
            maxv = int(labels_full.max().item())
            if minv < 0 or maxv >= num_classes:
                return sims_full, labels_full, False, f"label range [{minv},{maxv}] outside [0,{num_classes-1}]"
        # Cast sims to float32 (avoid fp16/float64 issues with scatter kernels)
        if sims_full.dtype != torch.float32:
            logger.info(f"Casting cached sims from {sims_full.dtype} to float32")
            sims_full = sims_full.float()
        return sims_full, labels_full, True, "ok"
    except Exception as e:
        return sims_full, labels_full, False, f"exception: {e}"


def get_unibind_kwargs(args) -> Dict:
    return dict(
        use_flash_attention=bool(args.use_flash_attention),
        use_lora=bool(args.use_lora),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        use_modality_head_mlp=bool(args.use_modality_head_mlp),
        lora_weights=args.lora_weights,
        modality_head_mlp_weights=args.modality_head_mlp_weights,
        use_masked_logsumexp=False,
    )


def build_models_on_devices(
    gpu_indices: List[int],
    centre_emb: torch.Tensor,
    centre_labels: List[str],
    lbl_to_idx: Dict[str, int],
    modality: Modality,
    pretrain_weights: str,
    unibind_kwargs: Dict,
    logger: logging.Logger,
) -> Dict[int, UniBindClassifier]:
    models: Dict[int, UniBindClassifier] = {}
    for gi in gpu_indices:
        device = torch.device(f"cuda:{gi}")
        torch.cuda.set_device(device)
        model = UniBindClassifier(
            device=device,
            pretrain_weights=pretrain_weights,
            modality=modality,
            centre_embeddings=centre_emb.to(device),
            centre_labels=centre_labels,
            label_to_index=lbl_to_idx,
            logger=logger,
            **unibind_kwargs,
        ).to(device)
        model.eval()
        models[gi] = model
    return models


def encode_multi_gpu(models: Dict[int, UniBindClassifier], batch_inputs: torch.Tensor) -> torch.Tensor:
    """Split batch across devices, run encode_vision_with_mlp sequentially, and concatenate results."""
    num_devices = len(models)
    if num_devices == 1:
        # Single GPU fast path
        gi = next(iter(models.keys()))
        device = torch.device(f"cuda:{gi}")
        return models[gi].encode_vision_with_mlp(batch_inputs.to(device))
    chunks = torch.chunk(batch_inputs, num_devices, dim=0)
    embeddings = []
    for (gi, chunk) in zip(models.keys(), chunks):
        device = torch.device(f"cuda:{gi}")
        emb = models[gi].encode_vision_with_mlp(chunk.to(device))
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def run_ablation(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these tests (GPU-only). No CUDA device found.")
    gpu_indices_all = detect_gpu_indices()
    if len(gpu_indices_all) == 0:
        raise RuntimeError("No CUDA devices available.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resolve output + log paths
    if str(args.output).lower().endswith(".csv"):
        out_dir = os.path.dirname(args.output) or "."
        os.makedirs(out_dir, exist_ok=True)
        output_path = args.output
        log_path = os.path.splitext(output_path)[0] + ".log"
        run_dir = out_dir
    else:
        run_dir = os.path.join(args.output, ts)
        os.makedirs(run_dir, exist_ok=True)
        output_path = os.path.join(run_dir, "results.csv")
        log_path = os.path.join(run_dir, "run.log")
    parent_logger = setup_logger(log_path)
    parent_logger.info(f"Detected GPUs: {gpu_indices_all}")

    # Determine modalities
    run_modalities: List[Modality] = list(MODALITY_DATASETS.keys())
    parent_logger.info(f"Run modalities: {[m.value for m in run_modalities]}")
    total_cls_tasks = len(run_modalities) * len(args.temperatures)
    parent_logger.info(f"Total classification tasks (dataset,T): {total_cls_tasks}")

    # Embedding cache options: always cache in the output dir for this run
    reuse_cached = args.reuse_cached_embeddings
    emb_cache_dir = run_dir
    os.makedirs(emb_cache_dir, exist_ok=True)
    parent_logger.info(f"Embedding cache dir (output dir): {emb_cache_dir}")

    # Decide embedding tasks
    to_embed: List[Modality] = []
    if not reuse_cached:
        for mod in run_modalities:
            ds_name = MODALITY_DATASETS[mod]["dataset_name"]
            sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt")
            labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt")
            if not (os.path.isfile(sims_path) and os.path.isfile(labels_path)):
                to_embed.append(mod)
        parent_logger.info(f"Embedding tasks needed: {[MODALITY_DATASETS[m]['dataset_name'] for m in to_embed]}")
    else:
        parent_logger.info("Embedding phase skipped due to --reuse_cached_embeddings.")

    # Embedding phase (multi-GPU parallel) using a global task queue (1 task per dataset globally)
    if to_embed:
        ctx = get_context("spawn")
        embed_queue = ctx.Queue()
        task_queue = ctx.Queue()
        # Enqueue all embedding tasks globally
        for mod in to_embed:
            task_queue.put(mod)
        # Add sentinel per worker
        for _ in gpu_indices_all:
            task_queue.put(None)
        emb_workers = []
        for gi in gpu_indices_all:
            p = ctx.Process(target=_embed_worker, args=(gi, task_queue, args, emb_cache_dir, embed_queue, ts))
            p.start()
            emb_workers.append(p)
            parent_logger.info(f"[gpu{gi}] Launched embed worker")
        done_embed = 0
        while done_embed < len(emb_workers):
            msg = embed_queue.get()
            if msg is None:
                done_embed += 1
                continue
            # Ensure messages carry GPU ID
            parent_logger.info(f"[embed] {msg}")
        for p in emb_workers:
            p.join()
        parent_logger.info("Embedding phase complete.")

    # Classification phase (multi-GPU)
    if len(gpu_indices_all) > 1 and len(run_modalities) > 1:
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        workers = []
        all_tasks: List[Tuple[Modality, float]] = []
        for mod in run_modalities:
            for T in args.temperatures:
                all_tasks.append((mod, T))
        # Log planned classification tasks to parent run.log
        planned_lines = [f"- {MODALITY_DATASETS[m]['dataset_name']} @ T={T}" for m, T in all_tasks]
        parent_logger.info("Planned classification tasks ({}):\n{}".format(len(all_tasks), "\n".join(planned_lines)))
        num_workers = len(gpu_indices_all)
        tasks_per_worker: List[List[Tuple[Modality, float]]] = [[] for _ in range(num_workers)]
        for i, task in enumerate(all_tasks):
            tasks_per_worker[i % num_workers].append(task)
        for wi, gi in enumerate(gpu_indices_all):
            worker_tasks = tasks_per_worker[wi]
            if not worker_tasks:
                continue
            p = ctx.Process(target=_classify_worker, args=(gi, worker_tasks, args, result_queue, ts, emb_cache_dir, reuse_cached))
            p.start()
            workers.append(p)
            worker_task_lines = [f"  - {MODALITY_DATASETS[m]['dataset_name']} @ T={t}" for m, t in worker_tasks]
            parent_logger.info(f"[gpu{gi}] Launched classify worker with {len(worker_tasks)} tasks:\n" + "\n".join(worker_task_lines))
        combined_rows: List[str] = ["modality,dataset,temperature,accuracy"]
        done_count = 0
        while done_count < len(workers):
            rows = result_queue.get()
            if rows is None:
                done_count += 1
                continue
            combined_rows.extend(rows)
        for p in workers:
            p.join()
        # Sort rows (excluding header) by modality, dataset, temperature
        header, *data_rows = combined_rows
        def _row_key(r: str):
            try:
                mod, ds, T, acc = r.split(',')
                return (mod, ds, float(T))
            except Exception:
                return ("~", "~", 1e9)
        data_rows_sorted = sorted(data_rows, key=_row_key)
        csv_str = "\n".join([header] + data_rows_sorted)
        with open(output_path, "w") as f:
            f.write(csv_str + "\n")
        parent_logger.info(f"Saved ablation results (sorted) to {output_path}")
        parent_logger.info(f"Log file: {log_path}")
        print(csv_str)
        return

    # Single-GPU classification path
    device = torch.device(f"cuda:{gpu_indices_all[0]}")
    torch.cuda.set_device(device)
    parent_logger.info(f"[gpu{gpu_indices_all[0]}] Single GPU classification start")
    rows: List[str] = ["modality,dataset,temperature,accuracy"]

    for mod in run_modalities:
        cfg = MODALITY_DATASETS.get(mod)
        if cfg is None:
            parent_logger.warning(f"No dataset mapping for modality {mod.value}; skipping.")
            continue
        ds_name = cfg["dataset_name"]
        batch_size = CLEAN_VAL_BATCH_SIZE_MAP[ds_name]
        centre_path = cfg["centre_embeddings_path"]
        val_json = cfg["val_json"]
        ds_root = cfg["dataset_root"]
        if not os.path.isfile(val_json) or not os.path.isfile(centre_path):
            parent_logger.warning(f"Missing data for {mod.value} (val_json:{val_json}, centre:{centre_path}); skipping.")
            continue
        centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
        parent_logger.info(f"[gpu{gpu_indices_all[0]}] label_to_index ({len(lbl_to_idx)})")
        parent_logger.info(f"[gpu{gpu_indices_all[0]}] index_to_label ({len(idx_to_lbl)})")
        if not sanity_check_label_overlap(ds_name, val_json, lbl_to_idx, parent_logger, gpu_prefix=f"[gpu{gpu_indices_all[0]}]"):
            parent_logger.warning(f"[gpu{gpu_indices_all[0]}] Skipping dataset {ds_name} due to label set mismatch with centres")
            continue
        models = build_models_on_devices([gpu_indices_all[0]], centre_emb, centre_labels, lbl_to_idx, mod, args.pretrain_weights, get_unibind_kwargs(args), parent_logger)
        sims_full = None
        labels_full = None
        sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt") if emb_cache_dir else None
        labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt") if emb_cache_dir else None
        # Prefer cached embeddings when a cache dir is provided. Do not recompute in classification when cache is expected.
        if emb_cache_dir and sims_path and os.path.isfile(sims_path) and os.path.isfile(labels_path):
            sims_try = torch.load(sims_path)
            labels_try = torch.load(labels_path)
            sims_try, labels_try, ok, reason = validate_cached_tensors(sims_try, labels_try, centre_emb.cpu(), len(set(centre_labels)), parent_logger)
            if ok:
                sims_full, labels_full = sims_try.to(device), labels_try.to(device)
                parent_logger.info(f"[gpu{gpu_indices_all[0]}] Loaded cached sims/labels for {ds_name}")
            else:
                parent_logger.warning(f"[gpu{gpu_indices_all[0]}] Cache invalid for {ds_name} ({reason}); skipping classification for this dataset")
                del models
                torch.cuda.empty_cache()
                continue
        elif emb_cache_dir:
            parent_logger.warning(f"[gpu{gpu_indices_all[0]}] Cache missing for {ds_name}; skipping classification for this dataset")
            del models
            torch.cuda.empty_cache()
            continue
        else:
            # No cache dir provided; compute on the fly
            loader = build_loader(mod, ds_root, val_json, lbl_to_idx, batch_size, args.num_workers)
            with torch.no_grad():
                all_labels = []
                all_similarities = []
                for batch in tqdm(loader, desc=f"embed-{ds_name}", leave=False):
                    x = batch["inputs"]
                    y = batch["labels"].to(device, non_blocking=True)
                    emb = encode_multi_gpu(models, x)
                    emb_cpu = emb.cpu() if emb.device.type != 'cpu' else emb
                    sims = emb_cpu @ centre_emb.cpu().t()
                    all_labels.append(y)
                    all_similarities.append(sims)
                labels_full = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long, device=device)
                sims_full = torch.cat(all_similarities, dim=0) if all_similarities else torch.empty(0, centre_emb.shape[0])
                sims_full = sims_full.to(device)
            if emb_cache_dir:
                sims_to_save = sims_full.float().cpu()
                torch.save(sims_to_save, sims_path)
                torch.save(labels_full.cpu(), labels_path)
                parent_logger.info(f"[gpu{gpu_indices_all[0]}] Saved sims/labels cache for {ds_name}")
        label_indices = models[gpu_indices_all[0]].centre_label_indices
        for T in args.temperatures:
            # For very large datasets (e.g. ImageNet-1K with 50 centres/class => 50k x 50k sims) stream rows.
            if sims_full.shape[0] * sims_full.shape[1] > 40_000_000:  # heuristic threshold ~160MB @ float32
                acc = top1_accuracy_chunked(
                    sims_full.cpu(),
                    labels_full.cpu(),
                    label_indices,
                    T,
                    device,
                    row_batch_size=batch_size,
                )
                rows.append(f"{mod.value},{ds_name},{T},{acc:.6f}")
                parent_logger.info(f"[{mod.value}] dataset={ds_name} T={T}: streamed_accuracy={acc:.6f} on {labels_full.numel()} samples (row_batch_size={batch_size})")
            else:
                logits = compute_logits_lse_scatter(sims_full, label_indices, T)
                acc = top1_accuracy(logits, labels_full)
                rows.append(f"{mod.value},{ds_name},{T},{acc:.6f}")
                parent_logger.info(f"[{mod.value}] dataset={ds_name} T={T}: accuracy={acc:.6f} on {labels_full.numel()} samples")
        del models
        torch.cuda.empty_cache()

    # Sort single-GPU rows
    header, *data_rows = rows
    def _row_key_single(r: str):
        try:
            mod, ds, T, acc = r.split(',')
            return (mod, ds, float(T))
        except Exception:
            return ("~", "~", 1e9)
    data_rows_sorted = sorted(data_rows, key=_row_key_single)
    header, *data_rows = rows
    def _row_key_single(r: str):
        try:
            mod, ds, T, acc = r.split(',')
            return (mod, ds, float(T))
        except Exception:
            return ("~", "~", 1e9)
    data_rows_sorted = sorted(data_rows, key=_row_key_single)
    csv_str = "\n".join([header] + data_rows_sorted)
    with open(output_path, "w") as f:
        f.write(csv_str + "\n")
    parent_logger.info(f"Saved ablation results (sorted) to {output_path}")
    parent_logger.info(f"Log file: {log_path}")
    print(csv_str)


def _classify_worker(gpu_index: int, tasks: List[Tuple[Modality, float]], args, result_queue, ts: str, emb_cache_dir: str, reuse_cached: bool):
    """Classification worker consuming (modality, T) tasks; loads cached sims if available else computes and optionally caches."""
    try:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
        # Per-worker log file
        if str(args.output).lower().endswith(".csv"):
            log_dir = os.path.dirname(args.output) or "."
        else:
            log_dir = os.path.join(args.output, ts)
        os.makedirs(log_dir, exist_ok=True)
        worker_log = os.path.join(log_dir, f"gpu{gpu_index}.log")
        logger = setup_logger(worker_log)
        logger.info(f"[gpu{gpu_index}] Worker start with {len(tasks)} classification tasks")
        planned_list_lines = [f"- {MODALITY_DATASETS[m]['dataset_name']} @ T={t}" for (m, t) in tasks]
        logger.info(f"[gpu{gpu_index}] Planned tasks ({len(tasks)}):\n" + "\n".join(planned_list_lines))

        # Group tasks by modality to reuse dataset forward pass per modality
        tasks_by_mod: Dict[Modality, List[float]] = {}
        for mod, temp in tasks:
            tasks_by_mod.setdefault(mod, []).append(float(temp))

        rows: List[str] = []
        done = 0
        total = len(tasks)
        for mod, temps in tasks_by_mod.items():
            cfg = MODALITY_DATASETS.get(mod)
            if cfg is None:
                logger.warning(f"[gpu{gpu_index}] No dataset mapping for modality {mod.value}; skipping.")
                continue
            ds_name = cfg["dataset_name"]
            batch_size = CLEAN_VAL_BATCH_SIZE_MAP[ds_name]
            centre_path = cfg["centre_embeddings_path"]
            val_json = cfg["val_json"]
            ds_root = cfg["dataset_root"]

            if not os.path.isfile(val_json) or not os.path.isfile(centre_path):
                logger.warning(f"[gpu{gpu_index}] Missing data for {mod.value} (val_json: {val_json}, centre: {centre_path}); skipping.")
                continue

            logger.info(f"[gpu{gpu_index}] [{mod.value}] Dataset={ds_name} root={ds_root} val_json={val_json} centres={centre_path} batch_size={batch_size}")
            centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
            num_classes = len(set(centre_labels))
            logger.info(f"[gpu{gpu_index}] [{mod.value}] Loaded centres: {len(centre_labels)}")
            logger.info(f"[gpu{gpu_index}] centre_path({centre_path}), label_to_index ({len(lbl_to_idx)})")
            logger.info(f"[gpu{gpu_index}] centre_path({centre_path}), index_to_label ({len(idx_to_lbl)})")
            # Sanity check label overlap to catch using ImageNet labels on N-Caltech (or vice versa)
            if not sanity_check_label_overlap(ds_name, val_json, lbl_to_idx, logger, gpu_prefix=f"[gpu{gpu_index}]"):
                del centre_emb, centre_labels
                continue
            # Build single-device model
            models = build_models_on_devices(
                [gpu_index],
                centre_emb,
                centre_labels,
                lbl_to_idx,
                mod,
                args.pretrain_weights,
                get_unibind_kwargs(args),
                logger,
            )

            # Attempt to load cached sims/labels first
            labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt") if emb_cache_dir else None
            sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt") if emb_cache_dir else None
            sims_full = None
            labels_full = None
            # In classification phase, prefer cached sims/labels; do not recompute when cache dir is provided
            if emb_cache_dir and sims_path and os.path.isfile(sims_path) and os.path.isfile(labels_path):
                labels_try = torch.load(labels_path)
                sims_try = torch.load(sims_path)
                sims_try, labels_try, ok, reason = validate_cached_tensors(sims_try, labels_try, centre_emb.cpu(), num_classes, logger)
                if ok:
                    labels_full = labels_try.to(device)
                    sims_full = sims_try.to(device)
                    logger.info(f"[gpu{gpu_index}] Loaded cached sims/labels for {ds_name}")
                else:
                    logger.warning(f"[gpu{gpu_index}] Cache invalid for {ds_name} ({reason}); skipping classification for this dataset")
                    del models
                    torch.cuda.empty_cache()
                    continue
            elif emb_cache_dir:
                logger.warning(f"[gpu{gpu_index}] Cache missing for {ds_name}; skipping classification for this dataset")
                del models
                torch.cuda.empty_cache()
                continue
            else:
                # No cache dir provided; compute on the fly
                loader = build_loader(
                    modality=mod,
                    dataset_root=ds_root,
                    json_path=val_json,
                    label_to_index=lbl_to_idx,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                )
                with torch.no_grad():
                    all_labels = []
                    all_similarities = []
                    for batch in tqdm(loader, desc=f"embed-{ds_name}", leave=False):
                        x = batch["inputs"]
                        y = batch["labels"].to(device, non_blocking=True)
                        emb = encode_multi_gpu(models, x)
                        emb_cpu = emb.cpu() if emb.device.type != 'cpu' else emb
                        sims = emb_cpu @ centre_emb.cpu().t()
                        all_labels.append(y)
                        all_similarities.append(sims)
                    labels_full = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long, device=device)
                    sims_full = torch.cat(all_similarities, dim=0) if all_similarities else torch.empty(0, centre_emb.shape[0])
                    sims_full = sims_full.to(device)

            label_indices = models[gpu_index].centre_label_indices
            for T in temps:
                done += 1
                logger.info(f"[gpu{gpu_index}] Progress {done}/{total}: [{mod.value}] dataset={ds_name} T={T}")
                if sims_full.shape[0] * sims_full.shape[1] > 40_000_000:  # streaming path for huge sims matrices
                    acc = top1_accuracy_chunked(
                        sims_full.cpu(),
                        labels_full.cpu(),
                        label_indices,
                        T,
                        device,
                        row_batch_size=batch_size,
                    )
                    rows.append(f"{mod.value},{ds_name},{T},{acc:.6f}")
                    logger.info(f"[gpu{gpu_index}] [{mod.value}] dataset={ds_name} T={T}: streamed_accuracy={acc:.6f} on {labels_full.numel()} samples (row_batch_size={batch_size})")
                else:
                    logits = compute_logits_lse_scatter(sims_full, label_indices, T)
                    acc = top1_accuracy(logits, labels_full)
                    rows.append(f"{mod.value},{ds_name},{T},{acc:.6f}")
                    logger.info(f"[gpu{gpu_index}] [{mod.value}] dataset={ds_name} T={T}: accuracy={acc:.6f} on {labels_full.numel()} samples")
            del models
            torch.cuda.empty_cache()

        result_queue.put(rows)
        result_queue.put(None)
    except Exception as e:
        # Return failure marker to parent
        try:
            result_queue.put([])
            result_queue.put(None)
        except Exception:
            pass
        raise
def _embed_worker(gpu_index: int, task_queue, args, emb_cache_dir: str, queue, ts: str):
    try:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
        # Per-worker log file
        if str(args.output).lower().endswith(".csv"):
            log_dir = os.path.dirname(args.output) or "."
        else:
            log_dir = os.path.join(args.output, ts)
        os.makedirs(log_dir, exist_ok=True)
        worker_log = os.path.join(log_dir, f"embed_gpu{gpu_index}.log")
        logger = setup_logger(worker_log)
        logger.info(f"[gpu{gpu_index}] Embed worker start")

        while True:
            mod = task_queue.get()
            if mod is None:
                break
            cfg = MODALITY_DATASETS.get(mod)
            if cfg is None:
                msg = f"[gpu{gpu_index}] skip:{getattr(mod, 'value', str(mod))}"
                queue.put(msg)
                logger.warning(msg)
                continue
            ds_name = cfg['dataset_name']
            sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt")
            labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt")
            if os.path.isfile(sims_path) and os.path.isfile(labels_path):
                msg = f"[gpu{gpu_index}] cached:{ds_name}"
                queue.put(msg)
                logger.info(msg)
                continue
            batch_size = CLEAN_VAL_BATCH_SIZE_MAP[ds_name]
            centre_path = cfg['centre_embeddings_path']
            val_json = cfg['val_json']
            ds_root = cfg['dataset_root']
            if not os.path.isfile(val_json) or not os.path.isfile(centre_path):
                msg = f"[gpu{gpu_index}] missing:{ds_name}"
                queue.put(msg)
                logger.warning(msg)
                continue
            logger.info(f"[gpu{gpu_index}] [{mod.value}] Dataset={ds_name} root={ds_root} val_json={val_json} centres={centre_path} batch_size={batch_size}")
            centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
            models = build_models_on_devices([gpu_index], centre_emb, centre_labels, lbl_to_idx, mod, args.pretrain_weights, get_unibind_kwargs(args), logger)
            logger.info(f"[gpu{gpu_index}] centre_path({centre_path}), label_to_index ({len(lbl_to_idx)})")
            logger.info(f"[gpu{gpu_index}] centre_path({centre_path}), index_to_label ({len(idx_to_lbl)})")
            # Sanity check label overlap in embedding stage too
            if not sanity_check_label_overlap(ds_name, val_json, lbl_to_idx, logger, gpu_prefix=f"[gpu{gpu_index}]"):
                del models
                torch.cuda.empty_cache()
                continue
            loader = build_loader(mod, ds_root, val_json, lbl_to_idx, batch_size, args.num_workers)
            with torch.no_grad():
                all_labels = []
                all_similarities = []
                for batch in tqdm(loader, desc=f"embed-{ds_name}", leave=False):
                    x = batch['inputs']
                    y = batch['labels'].to(device, non_blocking=True)
                    emb = encode_multi_gpu(models, x)
                    emb_cpu = emb.cpu() if emb.device.type != 'cpu' else emb
                    sims = emb_cpu @ centre_emb.cpu().t()
                    all_labels.append(y)
                    all_similarities.append(sims)
                labels_full = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long, device=device)
                sims_full = torch.cat(all_similarities, dim=0) if all_similarities else torch.empty(0, centre_emb.shape[0])
                sims_full = sims_full.to(device)
                sims_to_save = sims_full.float().cpu()
                torch.save(sims_to_save, sims_path)
                torch.save(labels_full.cpu(), labels_path)
                msg = f"[gpu{gpu_index}] saved:{ds_name}:{labels_full.numel()}samples"
                queue.put(msg)
                logger.info(msg)
            del models
            torch.cuda.empty_cache()
        queue.put(None)
    except Exception as e:
        try:
            queue.put(f"[gpu{gpu_index}] error:{str(e)}")
            queue.put(None)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-modality ablation for log-sum-exp (mask) temperature (GPU-only)")
    ap.add_argument("--run-all-modalities", action="store_true", default=True, help="Run ablation across all predefined modality->dataset mappings")
    ap.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt", help="Path to UniBind pretrain weights")
    ap.add_argument("--temperatures", type=parse_floats, default=[50.0, 100.0, 200.0, 500.0, 1000.0], help="Comma-separated temperatures list")
    ap.add_argument("--output", type=str, default="/data/output/dbam/ablate_logsumexp", help="Optional CSV output path (directory or .csv file)")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--reuse_cached_embeddings", action="store_true", default=False, help="Reuse cached sims/labels instead of recomputing")
    # UniBind configuration (optional)
    ap.add_argument("--use_flash_attention", action="store_true", default=True)
    ap.add_argument("--use_lora", action="store_true", default=False)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--use_modality_head_mlp", action="store_true", default=False)
    ap.add_argument("--lora_weights", type=str, default=None)
    ap.add_argument("--modality_head_mlp_weights", type=str, default=None)
    args = ap.parse_args()
    # If user supplied a string for temperatures via CLI, argparse will apply type=parse_floats.
    # If not supplied, default is already a list; no further processing needed.
    run_ablation(args)
