#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from multiprocessing import get_context

import torch
import torch_scatter
from tqdm import tqdm

from model import UniBindClassifier, Modality
from shared_types import BindModelType
from data_util import JsonDataset, CollateFn
from utils.utils import load_centre_embeddings
from datasets import (
    MODALITY_DATASETS,
    DATASET_TEMPERATURES,
    CLEAN_VAL_BATCH_SIZE_MAP,
    ATTACK_VAL_BATCH_SIZE_MAP,
)

# (UniBind optional CLI defaults are inlined in argparse below)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("classification_methods")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicates on re-run
    for h in list(logger.handlers):
        logger.removeHandler(h)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

# ----------------- Helper Functions -----------------

def parse_modality(s: str) -> Modality:
    s = s.strip().lower()
    for m in Modality:
        if m.value == s:
            return m
    raise argparse.ArgumentTypeError(f"Unknown modality: {s}")


def load_label_mapping(center_emb_path: str, device: torch.device):
    centre_embeddings, labels = load_centre_embeddings(center_emb_path, device)
    centre_embeddings = centre_embeddings / centre_embeddings.norm(dim=-1, keepdim=True)
    unique_lbls = sorted(set(labels))
    label_to_index = {l: i for i, l in enumerate(unique_lbls)}
    index_to_label = {i: l for l, i in label_to_index.items()}
    return centre_embeddings, labels, label_to_index, index_to_label


def _read_dataset_label_set(val_json_path: str) -> set:
    try:
        import json
        with open(val_json_path, 'r') as f:
            items = json.load(f)
        return set([str(it.get('label', '')).strip() for it in items])
    except Exception:
        return set()


def sanity_check_label_overlap(ds_name: str, val_json: str, lbl_to_idx: Dict[str, int], logger: logging.Logger, gpu_prefix: str = "") -> bool:
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


def compute_logits_scatter_max(similarity: torch.Tensor, centre_label_indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    max_vals, _ = torch_scatter.scatter_max(similarity, centre_label_indices, dim=1)
    if max_vals.shape[1] != num_classes:
        pad_cols = num_classes - max_vals.shape[1]
        if pad_cols > 0:
            pad = torch.full((max_vals.shape[0], pad_cols), -1e9, device=max_vals.device)
            max_vals = torch.cat([max_vals, pad], dim=1)
    return max_vals


def compute_logits_logsumexp_mask(similarity: torch.Tensor, centre_label_indices: torch.Tensor, temperature: float) -> torch.Tensor:
    """Match UniBind's _scatter_logsumexp: scatter_logsumexp(sim * T, idx) / T."""
    class_raw_scores = torch_scatter.scatter_logsumexp(similarity * temperature, centre_label_indices, dim=1)
    return class_raw_scores / temperature


def compute_top1_center_pred(similarity: torch.Tensor, centre_label_indices: torch.Tensor) -> torch.Tensor:
    centre_idx = similarity.argmax(dim=1)
    return centre_label_indices[centre_idx]


def top1_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total if total else 0.0


def top1_accuracy_from_preds(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total if total else 0.0


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

    Returns (sims, labels, ok, reason). If ok is False, reason contains a brief message.
    Performs light fix-ups (casting dtypes, squeezing labels) when safe.
    """
    reason = ""
    try:
        if sims_full is None or labels_full is None:
            return sims_full, labels_full, False, "cache missing"
        # Ensure shapes
        if sims_full.ndim != 2:
            return sims_full, labels_full, False, f"sims ndim={sims_full.ndim} != 2"
        if sims_full.shape[1] != centre_emb.shape[0]:
            return sims_full, labels_full, False, f"sims cols {sims_full.shape[1]} != centres {centre_emb.shape[0]}"
        # Labels to 1D long
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
        # Similarity dtype sanity: keep fp16/fp32; downcast fp64
        # Cast sims to float32 (avoid fp16/float64 issues with scatter kernels)
        if sims_full.dtype != torch.float32:
            logger.info(f"Casting cached sims from {sims_full.dtype} to float32")
            sims_full = sims_full.float()
        return sims_full, labels_full, True, "ok"
    except Exception as e:
        reason = f"exception: {e}"
        return sims_full, labels_full, False, reason


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
    num_devices = len(models)
    if num_devices == 1:
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

# ----------------- Main Evaluation -----------------

def evaluate(args) -> None:
    """Two-phase evaluation with optional embedding caching.
    Phase 1 (optional): build and save (similarities, labels) per dataset.
    Phase 2: classification metrics using cached similarities or freshly computed ones.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these tests (GPU-only). No CUDA device found.")
    gpu_indices_all = detect_gpu_indices()
    if len(gpu_indices_all) == 0:
        raise RuntimeError("No CUDA devices available.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    logger = setup_logger(log_path)
    logger.info(f"Detected GPUs: {gpu_indices_all}")
    run_modalities: List[Modality] = list(MODALITY_DATASETS.keys())
    logger.info(f"Run modalities: {[m.value for m in run_modalities]}")
    # Planned tasks with per-dataset temperatures
    planned_lines = []
    for m in run_modalities:
        ds_name = MODALITY_DATASETS[m]["dataset_name"]
        T = DATASET_TEMPERATURES.get(ds_name, 1000.0)
        planned_lines.append(f"- {m.value}: {ds_name} @ T={T}")
    logger.info("Planned classification tasks (dataset, temperature):\n" + "\n".join(planned_lines))

    # Always cache embeddings in the output run directory
    reuse_cached = args.reuse_cached_embeddings
    emb_cache_dir = run_dir
    os.makedirs(emb_cache_dir, exist_ok=True)
    logger.info(f"Embedding cache dir (output dir): {emb_cache_dir}")

    to_embed: List[Modality] = []
    if not reuse_cached:
        for m in run_modalities:
            ds_name = MODALITY_DATASETS[m]["dataset_name"]
            sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt")
            labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt")
            if not (os.path.isfile(sims_path) and os.path.isfile(labels_path)):
                to_embed.append(m)
        if to_embed:
            logger.info("Embedding tasks needed (by dataset):\n" + "\n".join([f"- {MODALITY_DATASETS[m]['dataset_name']}" for m in to_embed]))
        else:
            logger.info("No embedding tasks needed; all caches present.")
    else:
        logger.info("Embedding phase skipped due to --reuse_cached_embeddings.")

    # Phase 1: embedding if needed, using a global task queue (1 task per dataset globally)
    if to_embed:
        ctx = get_context("spawn")
        embed_queue = ctx.Queue()
        task_queue = ctx.Queue()
        for mod in to_embed:
            task_queue.put(mod)
        for _ in gpu_indices_all:
            task_queue.put(None)
        emb_workers = []
        for gi in gpu_indices_all:
            p = ctx.Process(target=_eval_embed_worker, args=(gi, task_queue, args, emb_cache_dir, embed_queue, ts))
            p.start()
            emb_workers.append(p)
            logger.info(f"[gpu{gi}] Launched embed worker")
        done_embed = 0
        while done_embed < len(emb_workers):
            msg = embed_queue.get()
            if msg is None:
                done_embed += 1
                continue
            logger.info(f"[embed] {msg}")
        for p in emb_workers:
            p.join()
        logger.info("Embedding phase complete.")

    # Phase 2: classification (possibly multi-GPU)
    if len(gpu_indices_all) > 1 and len(run_modalities) > 1:
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        workers = []
        num_workers = len(gpu_indices_all)
        # Log planned classification tasks and per-worker assignment for transparency
        planned = "\n".join([f"- {MODALITY_DATASETS[m]['dataset_name']} @ T={DATASET_TEMPERATURES.get(MODALITY_DATASETS[m]['dataset_name'], 1000.0)}" for m in run_modalities])
        logger.info(f"Planned classification tasks ({len(run_modalities)}):\n{planned}")
        for wi, gi in enumerate(gpu_indices_all):
            mods_subset = [m for i, m in enumerate(run_modalities) if i % num_workers == wi]
            if not mods_subset:
                continue
            p = ctx.Process(target=_eval_worker, args=(gi, mods_subset, args, result_queue, ts, emb_cache_dir, reuse_cached))
            p.start()
            workers.append(p)
            subset_lines = [f"- {MODALITY_DATASETS[m]['dataset_name']} @ T={DATASET_TEMPERATURES.get(MODALITY_DATASETS[m]['dataset_name'], 1000.0)}" for m in mods_subset]
            logger.info(f"[gpu{gi}] Launched classify worker with {len(mods_subset)} modalities:\n" + "\n".join(subset_lines))
        combined_rows: List[str] = ["modality,dataset,method,accuracy,temperature"]
        done_count = 0
        while done_count < len(workers):
            rows = result_queue.get()
            if rows is None:
                done_count += 1
                continue
            combined_rows.extend(rows)
        for p in workers:
            p.join()
        # Sort rows globally (skip header) by modality, dataset, temperature
        header, data_rows = combined_rows[0], combined_rows[1:]
        def sort_key(row: str):
            try:
                m, d, _, _, t = row.split(",")
                tval = float(t) if t else float("inf")
                return (m, d, tval)
            except Exception:
                return ("", "", float("inf"))
        data_rows.sort(key=sort_key)
        csv_str = "\n".join([header] + data_rows)
        with open(output_path, "w") as f:
            f.write(csv_str + "\n")
        logger.info(f"Saved evaluation results to {output_path}")
        logger.info(f"Log file: {log_path}")
        print(csv_str)
        return

    # Single GPU classification path
    device = torch.device(f"cuda:{gpu_indices_all[0]}")
    torch.cuda.set_device(device)
    logger.info(f"[gpu{gpu_indices_all[0]}] Single GPU classification start")
    planned_single = "\n".join([f"- {MODALITY_DATASETS[m]['dataset_name']} @ T={DATASET_TEMPERATURES.get(MODALITY_DATASETS[m]['dataset_name'], 1000.0)}" for m in run_modalities])
    logger.info(f"Planned classification tasks ({len(run_modalities)}):\n{planned_single}")
    rows: List[str] = ["modality,dataset,method,accuracy,temperature"]
    total_mods = len(run_modalities)
    for i, mod in enumerate(run_modalities, start=1):
        cfg = MODALITY_DATASETS.get(mod)
        if cfg is None:
            logger.warning(f"No dataset mapping for modality {mod.value}; skipping.")
            continue
        ds_name = cfg["dataset_name"]
        ds_root = cfg["dataset_root"]
        val_json = cfg["val_json"]
        centre_path = cfg["centre_embeddings_path"]
        batch_size = CLEAN_VAL_BATCH_SIZE_MAP[ds_name]
        T = DATASET_TEMPERATURES.get(ds_name, 1000.0)
        if not os.path.isfile(val_json) or not os.path.isfile(centre_path):
            logger.warning(f"Missing data for {mod.value} (val_json:{val_json}, centre:{centre_path}); skipping.")
            continue
        centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
        num_classes = len(set(centre_labels))
        models = build_models_on_devices([gpu_indices_all[0]], centre_emb, centre_labels, lbl_to_idx, mod, args.pretrain_weights, get_unibind_kwargs(args), logger)
        sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt") if emb_cache_dir else None
        labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt") if emb_cache_dir else None
        if not sanity_check_label_overlap(ds_name, val_json, lbl_to_idx, logger, gpu_prefix=f"[gpu{gpu_indices_all[0]}]"):
            logger.warning(f"[gpu{gpu_indices_all[0]}] Skipping dataset {ds_name} due to label set mismatch with centres")
            del models
            torch.cuda.empty_cache()
            continue
        logger.info(f"[gpu{gpu_indices_all[0]}] Starting {i}/{total_mods}: [{mod.value}] dataset={ds_name} at T={T}")
        # Prefer cached embeddings; if missing/invalid and a cache dir is used, skip this dataset
        if emb_cache_dir and sims_path and os.path.isfile(sims_path) and os.path.isfile(labels_path):
            sims_try = torch.load(sims_path)
            labels_try = torch.load(labels_path)
            sims_try, labels_try, ok, reason = validate_cached_tensors(sims_try, labels_try, centre_emb.cpu(), num_classes, logger)
            if ok:
                sims_full_cpu, labels_full_cpu = sims_try.cpu(), labels_try.cpu()
                logger.info(f"[gpu{gpu_indices_all[0]}] Loaded cached sims/labels for {ds_name}")
            else:
                logger.warning(f"[gpu{gpu_indices_all[0]}] Cache invalid for {ds_name} ({reason}); skipping classification for this dataset")
                del models
                torch.cuda.empty_cache()
                continue
        elif emb_cache_dir:
            logger.warning(f"[gpu{gpu_indices_all[0]}] Cache missing for {ds_name}; skipping classification for this dataset")
            del models
            torch.cuda.empty_cache()
            continue
        else:
            loader = build_loader(mod, ds_root, val_json, lbl_to_idx, batch_size, args.num_workers)
            with torch.no_grad():
                all_labels, all_sims = [], []
                for batch in tqdm(loader, desc=f"{ds_name} embed", unit="batch"):
                    x = batch['inputs']
                    y = batch['labels'].to(device, non_blocking=True)
                    emb = encode_multi_gpu(models, x)
                    emb_cpu = emb.cpu() if emb.device.type != 'cpu' else emb
                    sim = emb_cpu @ centre_emb.cpu().t()
                    all_labels.append(y.cpu())
                    all_sims.append(sim)
                labels_full_cpu = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
                sims_full_cpu = torch.cat(all_sims, dim=0) if all_sims else torch.empty(0, centre_emb.shape[0])
        # Stream classification in chunks using dataset batch size
        row_bs = batch_size
        device_local = device
        idx = models[gpu_indices_all[0]].centre_label_indices
        num_rows = labels_full_cpu.numel()
        correct_scatter = 0
        correct_lse = 0
        correct_top1c = 0
        with torch.no_grad():
            for start in tqdm(range(0, num_rows, row_bs), desc=f"{ds_name} classify-chunks", unit="rows"):
                end = min(start + row_bs, num_rows)
                sims_chunk = sims_full_cpu[start:end].to(device_local)
                labels_chunk = labels_full_cpu[start:end].to(device_local)
                logits_max = compute_logits_scatter_max(sims_chunk, idx, num_classes)
                logits_lse = compute_logits_logsumexp_mask(sims_chunk, idx, T)
                preds_top1c = compute_top1_center_pred(sims_chunk, idx)
                correct_scatter += (logits_max.argmax(dim=1) == labels_chunk).sum().item()
                correct_lse += (logits_lse.argmax(dim=1) == labels_chunk).sum().item()
                correct_top1c += (preds_top1c == labels_chunk).sum().item()
        total_samples = int(num_rows)
        acc_scatter_max = correct_scatter / total_samples if total_samples else 0.0
        acc_lse_mask = correct_lse / total_samples if total_samples else 0.0
        acc_top1_center = correct_top1c / total_samples if total_samples else 0.0
        logger.info(f"[{mod.value}] dataset={ds_name} scatter_max={acc_scatter_max:.6f} logsumexp(T={T})={acc_lse_mask:.6f} top1_center={acc_top1_center:.6f} over {total_samples} samples")
        rows.append(f"{mod.value},{ds_name},scatter_max,{acc_scatter_max:.6f},")
        rows.append(f"{mod.value},{ds_name},logsumexp,{acc_lse_mask:.6f},{T}")
        rows.append(f"{mod.value},{ds_name},top1_center,{acc_top1_center:.6f},")
        del models
        torch.cuda.empty_cache()
    # Sort rows before write
    header, data_rows = rows[0], rows[1:]
    def sort_key(row: str):
        try:
            m, d, _, _, t = row.split(",")
            tval = float(t) if t else float("inf")
            return (m, d, tval)
        except Exception:
            return ("", "", float("inf"))
    data_rows.sort(key=sort_key)
    csv_str = "\n".join([header] + data_rows)
    with open(output_path, "w") as f:
        f.write(csv_str + "\n")
    logger.info(f"Saved evaluation results to {output_path}")
    logger.info(f"Log file: {log_path}")
    print(csv_str)


def _eval_embed_worker(gpu_index: int, task_queue, args, emb_cache_dir: str, queue, ts: str):
    """Worker that pulls embedding tasks from a global queue and saves (similarities, labels)."""
    try:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
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
            ds_root = cfg['dataset_root']
            val_json = cfg['val_json']
            centre_path = cfg['centre_embeddings_path']
            batch_size = CLEAN_VAL_BATCH_SIZE_MAP[ds_name]
            sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt")
            labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt")
            if os.path.isfile(sims_path) and os.path.isfile(labels_path):
                msg = f"[gpu{gpu_index}] cached:{ds_name}"
                queue.put(msg)
                logger.info(msg)
                continue
            if not os.path.isfile(val_json) or not os.path.isfile(centre_path):
                msg = f"[gpu{gpu_index}] missing:{ds_name}"
                queue.put(msg)
                logger.warning(msg)
                continue
            logger.info(f"[gpu{gpu_index}] [{mod.value}] Dataset={ds_name} root={ds_root} val_json={val_json} centres={centre_path} batch_size={batch_size}")
            centre_emb, centre_labels, lbl_to_idx, _ = load_label_mapping(centre_path, device)
            models = build_models_on_devices([gpu_index], centre_emb, centre_labels, lbl_to_idx, mod, args.pretrain_weights, get_unibind_kwargs(args), logger)
            loader = build_loader(mod, ds_root, val_json, lbl_to_idx, batch_size, args.num_workers)
            with torch.no_grad():
                all_labels = []
                all_similarities = []
                for batch in tqdm(loader, desc=f"{ds_name} embed", unit="batch"):
                    x = batch['inputs']
                    y = batch['labels'].to(device, non_blocking=True)
                    emb = encode_multi_gpu(models, x)
                    emb_cpu = emb.cpu() if emb.device.type != 'cpu' else emb
                    sims = emb_cpu @ centre_emb.cpu().t()
                    all_labels.append(y.cpu())
                    all_similarities.append(sims)
                labels_full = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
                sims_full = torch.cat(all_similarities, dim=0) if all_similarities else torch.empty(0, centre_emb.shape[0])
                sims_to_save = sims_full.float().cpu()
                torch.save(sims_to_save, sims_path)
                torch.save(labels_full, labels_path)
                msg = f"[gpu{gpu_index}] saved:{ds_name}:{labels_full.numel()}samples"
                queue.put(msg)
                logger.info(msg)
            del models
            torch.cuda.empty_cache()
        queue.put(None)
    except Exception as e:
        try:
            queue.put(f"embed worker error: {e}")
            queue.put(None)
        except Exception:
            pass
        raise

def _eval_worker(gpu_index: int, modalities: List[Modality], args, result_queue, ts: str, emb_cache_dir: str, reuse_cached: bool):
    try:
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
        # Per-worker logger
        if str(args.output).lower().endswith(".csv"):
            log_dir = os.path.dirname(args.output) or "."
        else:
            log_dir = os.path.join(args.output, ts)
        os.makedirs(log_dir, exist_ok=True)
        worker_log = os.path.join(log_dir, f"gpu{gpu_index}.log")
        logger = setup_logger(worker_log)
        # Vertical planned modalities with per-dataset T
        planned = "\n".join([f"- {MODALITY_DATASETS[m]['dataset_name']} @ T={DATASET_TEMPERATURES.get(MODALITY_DATASETS[m]['dataset_name'], 1000.0)}" for m in modalities])
        logger.info(f"[gpu{gpu_index}] Worker start with {len(modalities)} modalities:\n{planned}")
        # Use top-level constants MODALITY_DATASETS and CLEAN_VAL_BATCH_SIZE_MAP

        rows: List[str] = []
        total = len(modalities)
        idx = 0
        for mod in modalities:
            idx += 1
            cfg = MODALITY_DATASETS.get(mod)
            if cfg is None:
                logger.warning(f"No dataset mapping for modality {mod.value}; skipping.")
                continue
            ds_name = cfg["dataset_name"]
            ds_root = cfg["dataset_root"]
            val_json = cfg["val_json"]
            centre_path = cfg["centre_embeddings_path"]
            batch_size = CLEAN_VAL_BATCH_SIZE_MAP[ds_name]
            T = DATASET_TEMPERATURES.get(ds_name, 1000.0)

            if not os.path.isfile(val_json) or not os.path.isfile(centre_path):
                logger.warning(f"[gpu{gpu_index}] Missing data for {mod.value} (val_json: {val_json}, centre: {centre_path}); skipping.")
                continue

            logger.info(f"[gpu{gpu_index}] Starting {idx}/{total}: [{mod.value}] dataset={ds_name} at T={T}")
            logger.info(f"[gpu{gpu_index}] [{mod.value}] Dataset={ds_name} root={ds_root} val_json={val_json} centres={centre_path} batch_size={batch_size}")
            centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
            num_classes = len(set(centre_labels))
            logger.info(f"[{mod.value}] Loaded centres: {len(centre_labels)} centres across {num_classes} classes")
            if not sanity_check_label_overlap(ds_name, val_json, lbl_to_idx, logger, gpu_prefix=f"[gpu{gpu_index}]"):
                continue

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
            sims_path = os.path.join(emb_cache_dir, f"{ds_name}_sims.pt") if emb_cache_dir else None
            labels_path = os.path.join(emb_cache_dir, f"{ds_name}_labels.pt") if emb_cache_dir else None
            loader = None
            sims_full_cpu = None
            labels_full_cpu = None
            # Prefer cached sims/labels; if missing/invalid when using cache dir, skip this dataset
            if emb_cache_dir and sims_path and os.path.isfile(sims_path) and os.path.isfile(labels_path):
                sims_try = torch.load(sims_path)
                labels_try = torch.load(labels_path)
                sims_try, labels_try, ok, reason = validate_cached_tensors(sims_try, labels_try, centre_emb.cpu(), num_classes, logger)
                if ok:
                    sims_full_cpu, labels_full_cpu = sims_try.cpu(), labels_try.cpu()
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
                loader = build_loader(mod, ds_root, val_json, lbl_to_idx, batch_size, args.num_workers)
                sims_all, labels_all = [], []
                with torch.no_grad():
                    for batch in tqdm(loader, desc=f"{ds_name} embed", unit="batch"):
                        x = batch['inputs']
                        y = batch['labels'].to(device, non_blocking=True)
                        emb = encode_multi_gpu(models, x)
                        emb_cpu = emb.cpu() if emb.device.type != 'cpu' else emb
                        sim = emb_cpu @ centre_emb.cpu().t()
                        sims_all.append(sim)
                        labels_all.append(y.cpu())
                labels_full_cpu = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0, dtype=torch.long)
                sims_full_cpu = torch.cat(sims_all, dim=0) if sims_all else torch.empty(0, centre_emb.shape[0])
            total_samples = labels_full_cpu.numel()
            if total_samples == 0:
                logger.warning(f"[{mod.value}] No samples evaluated; skipping row output.")
                del models
                torch.cuda.empty_cache()
                continue
            # Stream classification in chunks
            row_bs = batch_size
            idx_tensor = models[gpu_index].centre_label_indices
            correct_scatter = 0
            correct_lse = 0
            correct_top1c = 0
            with torch.no_grad():
                for start in tqdm(range(0, total_samples, row_bs), desc=f"{ds_name} classify-chunks", unit="rows"):
                    end = min(start + row_bs, total_samples)
                    sims_chunk = sims_full_cpu[start:end].to(device)
                    labels_chunk = labels_full_cpu[start:end].to(device)
                    logits_max = compute_logits_scatter_max(sims_chunk, idx_tensor, num_classes)
                    logits_lse = compute_logits_logsumexp_mask(sims_chunk, idx_tensor, T)
                    preds_top1c = compute_top1_center_pred(sims_chunk, idx_tensor)
                    correct_scatter += (logits_max.argmax(dim=1) == labels_chunk).sum().item()
                    correct_lse += (logits_lse.argmax(dim=1) == labels_chunk).sum().item()
                    correct_top1c += (preds_top1c == labels_chunk).sum().item()
            acc_scatter_max = correct_scatter / total_samples if total_samples else 0.0
            acc_lse_mask = correct_lse / total_samples if total_samples else 0.0
            acc_top1_center = correct_top1c / total_samples if total_samples else 0.0
            logger.info(f"[{mod.value}] dataset={ds_name} scatter_max={acc_scatter_max:.6f} logsumexp(T={T})={acc_lse_mask:.6f} top1_center={acc_top1_center:.6f} over {total_samples} samples")
            rows.append(f"{mod.value},{ds_name},scatter_max,{acc_scatter_max:.6f},")
            rows.append(f"{mod.value},{ds_name},logsumexp,{acc_lse_mask:.6f},{T}")
            rows.append(f"{mod.value},{ds_name},top1_center,{acc_top1_center:.6f},")
            del models
            torch.cuda.empty_cache()

        result_queue.put(rows)
        result_queue.put(None)
    except Exception:
        try:
            result_queue.put([])
            result_queue.put(None)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate classification methods (scatter-max, logsumexp, top1-centre) with optional embedding caching, multi-GPU")
    ap.add_argument("--run-all-modalities", action="store_true", default=True, help="Run evaluation across all predefined modalities")
    ap.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt", help="Path to UniBind pretrain weights")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1000.0, help="Temperature for log-sum-exp mask")
    ap.add_argument("--output", type=str, default="/data/output/dbam/classification_methods", help="CSV output path (directory or .csv file)")
    # Embedding cache flags (cache dir is always the output run directory)
    ap.add_argument("--reuse_cached_embeddings", action="store_true", default=False, help="Reuse existing cached similarities/labels and skip embedding phase")
    # UniBind configuration flags
    ap.add_argument("--use_flash_attention", action="store_true", default=True)
    ap.add_argument("--use_lora", action="store_true", default=False)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--use_modality_head_mlp", action="store_true", default=False)
    ap.add_argument("--lora_weights", type=str, default=None)
    ap.add_argument("--modality_head_mlp_weights", type=str, default=None)
    args = ap.parse_args()
    evaluate(args)
