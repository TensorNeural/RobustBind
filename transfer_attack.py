#!/usr/bin/env python3
"""Generate adversarial examples on UniBind and evaluate transfer accuracy on RobustBind^4.

This script builds UniBindClassifier models for source (UniBind) and target (RobustBind^4 via LoRA
weights), generates adversarial images using a two-stage APGD attack, and reports classification
accuracy for both source and target.
"""

import argparse
import os
from datetime import datetime
import logging

import torch
import math
from tqdm import tqdm
from multiprocessing import get_context

import csv
import numpy as np
from PIL import Image
from transform import unnormalize_inplace

from attack import APGDAttack, AttackModel, two_stage_attack
from model import ForwardMode, Modality, ImageBindClassifier, CLIPClassifier
from data_util import get_normalization_tensors
# For optional uniform sampling of validation set when run_max_samples is provided
from data_util import JsonDataset, CollateFn, BindModelType
import torch.utils.data as data_utils

# Reuse utilities from the classification methods tool
from tools.ablate_classification_methods import (
    build_models_on_devices,
    get_unibind_kwargs,
    load_label_mapping,
    build_loader,
)
from datasets import (
    MODALITY_DATASETS,
    CLEAN_VAL_BATCH_SIZE_MAP,
    ATTACK_VAL_BATCH_SIZE_MAP,
    DATASET_TEMPERATURES,
)

# ------------------------ Constants / experiment grid ------------------------
# Modalities to run (keys should match the dataset/modality names used in MODALITY_DATASETS)
MODALITIES = ["image", "audio", "thermal"]

# Optional overrides for val_json per modality (if different from MODALITY_DATASETS defaults)
VAL_JSONS = {
    "image": "./datasets/ImageNet-1K/val_data.json",
    "audio": "./datasets/ESC-50/val_data.json",
    "thermal": "./datasets/LLVIP/val_data.json",
}

# Per-modality default RobustBind^4 LoRA weights (used when --robust_lora_weights is not provided)
MODALITY_ROBUST_LORA_WEIGHTS = {
    "image": "./ckpts/image_eps4_lora_weights_old.pt",
    "audio": "./ckpts/audio_eps4_lora_weights.pt",
    "thermal": "./ckpts/thermal_eps4_lora_weights.pt",
}

# Default batch size fallback when dataset mapping doesn't provide one
DEFAULT_BATCH_SIZE_FALLBACK = 64

# Per-modality allowed source models. This lets each modality use only compatible encoders.
# Keys are modality names (same as MODALITIES entries). Values are lists of src_model strings.
# (Defined later with expanded formatting) See the later `MODALITY_SRC_MODELS` block for the
# authoritative per-modality allowed source models mapping.

# All known source models (derived from modality mapping)
# NOTE: `MODALITY_SRC_MODELS` is defined below; `ALL_SRC_MODELS` will be populated after that block.

# Eps list (in pixel values out of 255) to run transfer attacks for
EPS_LIST = [2.0, 4.0]

# Default max samples used when running the full grid (can be overridden via CLI)
DEFAULT_RUN_MAX_SAMPLES = 70

# Per-modality allowed source models. This lets each modality use only compatible encoders.
# Keys are modality names (same as MODALITIES entries). Values are lists of src_model strings.
MODALITY_SRC_MODELS = {
    "image": [
        # "unibind", 
        "clip", 
        # "imagebind"
    ],
    # "audio": [
    #     "unibind", 
    #     # "clip-vit-14"
    # ],
    # "thermal": [
    #     "unibind", 
    #     # "imagebind"
    # ],
}

# Populate the derived ALL_SRC_MODELS after the authoritative modality mapping is defined
ALL_SRC_MODELS = sorted({s for lst in MODALITY_SRC_MODELS.values() for s in lst})

# -----------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("transfer_unibind")
    logger.setLevel(logging.INFO)
    # Clear handlers
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
    # Write an initial log entry so the file is created and visible immediately
    logger.info(f"Logger initialized: {log_path}")
    return logger


# Create a logger for a given output directory. Returns a fresh logger instance
# configured to write to console and to out_dir/run.log.
def create_logger(out_dir: str) -> logging.Logger:
    log_path = os.path.join(out_dir, "run.log")
    logger = setup_logger(log_path)
    return logger


def prepare_output_and_logger(args):
    """Create timestamped output dir, adv dir, and initialize a run-local logger.

    Returns: out_dir, results_predictions_path, results_accuracy_path, adv_dir, logger
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        base_out = os.path.join(args.output, ts)
    else:
        base_out = f"output/transfer_unibind_to_robustbind_{ts}"
    os.makedirs(base_out, exist_ok=True)
    out_dir = base_out
    # initialize a run-local logger (console + run.log)
    logger = create_logger(out_dir)
    logger.info(f"[transfer_attack] logger initialized, log_path={os.path.join(out_dir, 'run.log')}")

    results_predictions_path = os.path.join(out_dir, "results_predictions.csv")
    results_accuracy_path = os.path.join(out_dir, "robust.csv")
    adv_dir = os.path.join(out_dir, "adv_samples")
    os.makedirs(adv_dir, exist_ok=True)
    return out_dir, results_predictions_path, results_accuracy_path, adv_dir, logger


def select_modality_cfg(args, logger: logging.Logger = None):
    """Resolve args.modality into modality enum and dataset config.

    Returns: modality_key, modality_enum, cfg, val_json, dataset_root, centre_path, ds_name
    """
    modality_key = args.modality.lower()
    modality_map = {k.value.lower(): k for k in MODALITY_DATASETS.keys()}
    if logger:
        logger.info(f"[transfer_attack] modality_map keys: {list(modality_map.keys())}")
    if modality_key not in modality_map:
        raise ValueError(f"Unsupported modality: {args.modality}. Supported: {list(modality_map.keys())}")
    modality = modality_map[modality_key]
    cfg = MODALITY_DATASETS[modality]
    if logger:
        logger.info(f"[transfer_attack] selected cfg for modality {modality.value}: {cfg}")
    # Resolve defaults; may be overridden later by explicit args
    val_json = args.val_json or cfg.get("val_json")
    dataset_root = args.dataset_root or cfg.get("dataset_root")
    centre_path = args.centre_embeddings or cfg.get("centre_embeddings_path")
    ds_name = cfg.get('dataset_name')
    return modality_key, modality, cfg, val_json, dataset_root, centre_path, ds_name


def _make_loader_with_uniform_sampling(modality, dataset_root, val_json, lbl_to_idx, batch_size, num_workers, run_max_samples):
    """Build a DataLoader that uniformly samples the validation set when run_max_samples is not None.

    If run_max_samples is None or >= dataset size, falls back to the existing build_loader for full dataset.
    """
    # If no sampling requested, use the shared build_loader
    if not run_max_samples:
        return build_loader(modality, dataset_root, val_json, lbl_to_idx, batch_size=batch_size, num_workers=num_workers)

    # Build full JsonDataset (do not use its internal random sampling) then pick uniformly spaced indices
    dataset = JsonDataset(dataset_root, val_json, lbl_to_idx, max_samples=None, debug=False)
    N = len(dataset)
    if N == 0:
        return build_loader(modality, dataset_root, val_json, lbl_to_idx, batch_size=batch_size, num_workers=num_workers)
    if run_max_samples >= N:
        return build_loader(modality, dataset_root, val_json, lbl_to_idx, batch_size=batch_size, num_workers=num_workers)

    # Uniformly spaced indices across [0, N-1]
    inds = np.linspace(0, N - 1, num=int(run_max_samples), dtype=int).tolist()
    subset = data_utils.Subset(dataset, inds)
    collate = CollateFn(modality, True, BindModelType.UNIBIND)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate,
    )


def _run_attack_loop(logger, src_model, target_model, loader, mean, std, device, ns, results_csv_path, adv_dir, label_names=None):
    """Shared attack loop: iterate loader, create attacks, save adv samples and write CSV.

    Args:
        logger: Logger instance
        src_model: source model (on device)
        target_model: target model (on device)
        loader: DataLoader providing batches
        mean, std: normalization tensors
        device: torch.device
        ns: namespace with fields steps, eps, run_max_samples
        results_csv_path: path to write per-combo CSV
        adv_dir: directory to save adversarial samples

    Returns:
        dict with totals and correct counts
    """
    attack_model = AttackModel(src_model, mean=mean, std=std)
    stage1 = APGDAttack(logger, attack_model, norm="linf", n_restarts=1, n_iter=ns.steps, eps=ns.eps / 255.0, loss_type="ce", device=device)
    stage2 = APGDAttack(logger, attack_model, norm="linf", n_restarts=1, n_iter=ns.steps, eps=ns.eps / 255.0, loss_type="ce", device=device)

    total = 0
    processed = 0
    correct_src_clean = 0
    correct_target_clean = 0
    correct_src_adv = 0
    correct_target_adv = 0

    os.makedirs(adv_dir, exist_ok=True)
    csv_file = open(results_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["path", "label", "label_name", "src_clean", "target_clean", "src_adv", "target_adv", "cosine_sim", "l2_dist", "adv_path"])
    # Place per-run CSVs outside the adv_samples directory so adv_samples contains only images
    run_out_dir = os.path.dirname(results_csv_path) or adv_dir
    # CSV for samples where src model under attack is wrong but target (RobustBind^4) is correct
    fail_csv_path = os.path.join(run_out_dir, "src_attack_wrong.csv")
    fail_csv = open(fail_csv_path, "w", newline="")
    fail_writer = csv.writer(fail_csv)
    fail_writer.writerow(["clean_path", "adv_path", "label", "label_name", "src_clean", "target_clean", "src_adv", "target_adv", "cosine_sim", "l2_dist"])
    n_src_attack_wrong = 0

    # CSV for samples where both models were correct on the clean input but after attack
    # the source model becomes wrong while the target remains correct.
    clean_then_fail_csv_path = os.path.join(run_out_dir, "target_robust_correct.csv")
    clean_then_fail_csv = open(clean_then_fail_csv_path, "w", newline="")
    clean_then_fail_writer = csv.writer(clean_then_fail_csv)
    clean_then_fail_writer.writerow(["clean_path", "adv_path", "label", "label_name", "src_clean", "target_clean", "src_adv", "target_adv", "cosine_sim", "l2_dist"])
    n_target_robust_correct = 0

    

    for batch in tqdm(loader, desc="Batches", leave=False):
        x = batch['inputs']  # tensor of shape [B, C, H, W]
        y = batch['labels'].to(device)
        paths = batch.get('paths', [None] * x.size(0))
        x = x.to(device)

        B = x.size(0)
        if getattr(ns, 'run_max_samples', None) and processed >= ns.run_max_samples:
            break
        if getattr(ns, 'run_max_samples', None) and processed + B > ns.run_max_samples:
            # trim
            keep = ns.run_max_samples - processed
            x = x[:keep]
            y = y[:keep]
            paths = paths[:keep]
            B = x.size(0)

        # Use dataset-specific temperature for logits when available (passed via ns)
        T = getattr(ns, 'temperature', None)
        if T is None:
            T = 1000.0
        with torch.no_grad():
            # call protected _logits to pass-through temperature where supported
            logits_src_clean, _ = src_model._logits(x, temperature=T)
            preds_src_clean = logits_src_clean.argmax(dim=1)
            correct_src_clean += (preds_src_clean == y).sum().item()

            logits_target_clean, _ = target_model._logits(x, temperature=T)
            preds_target_clean = logits_target_clean.argmax(dim=1)
            correct_target_clean += (preds_target_clean == y).sum().item()

        # Compute original embeddings for L2 loss
        with torch.no_grad():
            emb_orig = src_model(x, mode=ForwardMode.EMBEDDINGS)

        adv = two_stage_attack(logger, src_model, x, y, stage1, stage2, mean, std)

        with torch.no_grad():
            logits_src_adv, _ = src_model._logits(adv, temperature=T)
            preds_src_adv = logits_src_adv.argmax(dim=1)
            correct_src_adv += (preds_src_adv == y).sum().item()

            logits_target_adv, _ = target_model._logits(adv, temperature=T)
            preds_target_adv = logits_target_adv.argmax(dim=1)
            correct_target_adv += (preds_target_adv == y).sum().item()

        # Per-sample metrics and save adv samples
        with torch.no_grad():
            emb_adv = src_model(adv, mode=ForwardMode.EMBEDDINGS)
            cos_per = torch.nn.functional.cosine_similarity(emb_adv, emb_orig, dim=1).cpu().tolist()
            l2_per = torch.norm(emb_adv - emb_orig, dim=1).cpu().tolist()

        # adv is normalized; unnormalize for saving
        adv_un = adv.detach().clone()
        unnormalize_inplace(adv_un, mean, std)

        for i in range(x.size(0)):
            pth = paths[i] if i < len(paths) else None
            label_i = int(y[i].item())
            label_name_i = None
            if label_names is not None and 0 <= label_i < len(label_names):
                label_name_i = label_names[label_i]
            src_clean_i = int(preds_src_clean[i].item())
            target_clean_i = int(preds_target_clean[i].item())
            src_adv_i = int(preds_src_adv[i].item())
            target_adv_i = int(preds_target_adv[i].item())
            cosine_i = float(cos_per[i])
            l2_i = float(l2_per[i])

            adv_filename_base = f"adv_total{total}_proc{processed}_idx{i}"
            if adv_un.dim() == 4 and adv_un.size(1) == 3:
                arr = (adv_un[i].cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                adv_path = os.path.join(adv_dir, adv_filename_base + ("_" + os.path.basename(pth) if pth else "") + ".png")
                Image.fromarray(arr).save(adv_path)
            else:
                adv_path = os.path.join(adv_dir, adv_filename_base + ".npy")
                np.save(adv_path, adv_un[i].cpu().numpy())

            csv_writer.writerow([pth or "", label_i, label_name_i or "", src_clean_i, target_clean_i, src_adv_i, target_adv_i, cosine_i, l2_i, adv_path])
            # If src model under attack is wrong but target model predicts correctly, record to fail CSV
            if src_adv_i != label_i and target_adv_i == label_i:
                fail_writer.writerow([pth or "", adv_path, label_i, label_name_i or "", src_clean_i, target_clean_i, src_adv_i, target_adv_i, cosine_i, l2_i])
                n_src_attack_wrong += 1
                try:
                    logger.info(f"[run] Recorded src_attack_wrong: path={pth or ''} adv={adv_path} label={label_i}")
                except Exception:
                    pass

            # If the source model was correct on the clean input but after attack the source is wrong
            # while the target remains correct, record to the "clean_then_fail" CSV. We only require
            # the source to be correct on the clean input (target clean correctness is not required).
            if src_clean_i == label_i and src_adv_i != label_i and target_adv_i == label_i:
                clean_then_fail_writer.writerow([pth or "", adv_path, label_i, label_name_i or "", src_clean_i, target_clean_i, src_adv_i, target_adv_i, cosine_i, l2_i])
                n_target_robust_correct += 1
                try:
                    logger.info(f"[run] Recorded target_robust_correct: path={pth or ''} adv={adv_path} label={label_i}")
                except Exception:
                    pass

        processed += B
        total += B
        if getattr(ns, 'run_max_samples', None) and processed >= ns.run_max_samples:
            break

    csv_file.close()
    fail_csv.close()
    clean_then_fail_csv.close()
    try:
        logger.info(f"[run] src_attack_wrong entries: {n_src_attack_wrong}")
        logger.info(f"[run] target_robust_correct entries: {n_target_robust_correct}")
    except Exception:
        pass
    return {
        'total': total,
        'correct_src_clean': correct_src_clean,
        'correct_target_clean': correct_target_clean,
        'correct_src_adv': correct_src_adv,
        'correct_target_adv': correct_target_adv,
    }


def main(args):
    LOGGER.info(f"[transfer_attack] main start, args.modality={getattr(args,'modality',None)}, src_model={getattr(args,'src_model',None)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    gpu_indices_all = list(range(torch.cuda.device_count()))
    LOGGER.info(f"[transfer_attack] detected gpu_indices_all={gpu_indices_all}")
    if len(gpu_indices_all) == 0:
        raise RuntimeError("No CUDA devices available.")

    # Prepare output directories and initialize a run-local logger (console + run.log)
    out_dir, results_predictions_path, results_accuracy_path, adv_dir, logger = prepare_output_and_logger(args)

    # Output CSVs and adv samples directory
    # - results_predictions.csv: per-sample prediction rows (aggregated across workers)
    # - robust.csv: aggregated accuracy metrics (one row summarizing counts and accuracies)
    results_predictions_path = os.path.join(out_dir, "results_predictions.csv")
    results_accuracy_path = os.path.join(out_dir, "robust.csv")
    adv_dir = os.path.join(out_dir, "adv_samples")
    os.makedirs(adv_dir, exist_ok=True)

    # Select dataset config using MODALITY_DATASETS mapping (supports a few modalities)
    modality_key = args.modality.lower()
    # Only support image modality via mapping keys names
    modality_map = {k.value.lower(): k for k in MODALITY_DATASETS.keys()}
    logging.getLogger("transfer_unibind").info(f"[transfer_attack] modality_map keys: {list(modality_map.keys())}")
    if modality_key not in modality_map:
        raise ValueError(f"Unsupported modality: {args.modality}. Supported: {list(modality_map.keys())}")
    modality = modality_map[modality_key]
    cfg = MODALITY_DATASETS[modality]
    logging.getLogger("transfer_unibind").info(f"[transfer_attack] selected cfg for modality {modality.value}: {cfg}")

    # Support running multiple modalities in sequence when user passes --modality all or comma-separated list
    import sys
    if modality_key == 'all' or ',' in args.modality:
        if modality_key == 'all':
            run_list = ['image', 'audio', 'thermal']
        else:
            run_list = [s.strip() for s in args.modality.split(',') if s.strip()]

        for m in run_list:
            mod_out = os.path.join(out_dir, m)
            os.makedirs(mod_out, exist_ok=True)
            # We'll fork a worker process and pass arguments via a Namespace below.
            logging.getLogger("transfer_unibind").info(f"Forking worker for modality {m} -> {mod_out}")
            # Previously we launched a subprocess; keep behavior but call directly in a forked process
            ctx = get_context('spawn')
            worker_ns = argparse.Namespace(**vars(args))
            worker_ns.modality = m
            worker_ns.output = mod_out
            # ensure the child does not re-enter run_all orchestration
            worker_ns.run_all = False
            # pass through overrides
            if args.val_json:
                worker_ns.val_json = args.val_json
            if args.dataset_root:
                worker_ns.dataset_root = args.dataset_root
            if args.centre_embeddings:
                worker_ns.centre_embeddings = args.centre_embeddings
            if args.unibind_weights:
                worker_ns.unibind_weights = args.unibind_weights
            if args.robust_lora_weights:
                worker_ns.robust_lora_weights = args.robust_lora_weights
            if getattr(args, 'src_model', None):
                worker_ns.src_model = args.src_model
            worker_ns.eps = args.eps
            worker_ns.steps = args.steps
            worker_ns.batch_size = args.batch_size
            worker_ns.run_max_samples = args.run_max_samples

            def _run_worker(ns):
                try:
                    main(ns)
                except SystemExit:
                    return
                except Exception:
                    import traceback
                    tb = traceback.format_exc()
                    logger.exception(f"Exception in worker for modality {ns.modality}:\n{tb}")

            p = ctx.Process(target=_run_worker, args=(worker_ns,))
            p.start()
            p.join()
        sys.exit(0)

    # Override with explicit json/root if provided
    val_json = args.val_json or cfg["val_json"]
    dataset_root = args.dataset_root or cfg["dataset_root"]
    centre_path = args.centre_embeddings or cfg["centre_embeddings_path"]

    # If batch_size not provided, use per-dataset default from CLEAN_VAL_BATCH_SIZE_MAP
    ds_name = cfg.get('dataset_name')
    if args.batch_size is None:
        # Prefer attack-specific batch size when running adversarial generation / transfer attacks
        args.batch_size = ATTACK_VAL_BATCH_SIZE_MAP.get(ds_name, CLEAN_VAL_BATCH_SIZE_MAP.get(ds_name, DEFAULT_BATCH_SIZE_FALLBACK))
        logger.info(f"Using default batch_size={args.batch_size} for dataset {ds_name} (attack batch size)")

    logger.info(f"Dataset: {cfg['dataset_name']} val_json={val_json} centres={centre_path}")

    # If multiple GPUs available, spawn one worker per GPU and aggregate results
    if len(gpu_indices_all) > 1:
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        workers = []
        # Interpret args.run_max_samples as TOTAL samples across all workers. Compute per-worker cap.
        if getattr(args, 'run_max_samples', None):
            per_worker_max = math.ceil(args.run_max_samples / len(gpu_indices_all))
        else:
            per_worker_max = None

        for wi, gi in enumerate(gpu_indices_all):
            # pass a copy of args with adjusted max_samples so each worker processes at most per_worker_max
            worker_args = argparse.Namespace(**vars(args))
            worker_args.run_max_samples = per_worker_max
            # per-worker use same dataset temperature when available
            worker_args.temperature = DATASET_TEMPERATURES.get(ds_name, 1000.0)
            # per-worker output paths
            worker_args._worker_adv_dir = os.path.join(adv_dir, f"gpu{gi}")
            worker_args._worker_csv = os.path.join(out_dir, f"results_gpu{gi}.csv")
            p = ctx.Process(target=_attack_worker, args=(gi, wi, len(gpu_indices_all), worker_args, result_queue, out_dir))
            p.start()
            workers.append(p)
            logger.info(f"Launched attack worker on gpu{gi} (per-worker run_max_samples={per_worker_max})")

        # Collect results
        total = 0
        correct_src_clean = 0
        correct_target_clean = 0
        correct_src_adv = 0
        correct_target_adv = 0
        done = 0
        while done < len(workers):
            res = result_queue.get()
            if res is None:
                done += 1
                continue
            total += res.get('total', 0)
            correct_src_clean += res.get('correct_src_clean', 0)
            correct_target_clean += res.get('correct_target_clean', 0)
            correct_src_adv += res.get('correct_src_adv', 0)
            correct_target_adv += res.get('correct_target_adv', 0)

        for p in workers:
            p.join()
        # Aggregate per-worker CSVs (if any) into a single sorted CSV
        try:
            import glob
            csv_paths = glob.glob(os.path.join(out_dir, "results_gpu*.csv"))
            if csv_paths:
                agg_rows = []
                for cp in csv_paths:
                    with open(cp, "r") as f:
                        lines = f.read().splitlines()
                    if not lines:
                        continue
                    header = lines[0]
                    for ln in lines[1:]:
                        if ln.strip():
                            agg_rows.append(ln)
                # sort by path then label (csv: path,label,...)
                def sort_key(row: str):
                    parts = row.split(",")
                    path = parts[0]
                    try:
                        label = int(parts[1])
                    except Exception:
                        label = 0
                    return (path, label)

                agg_rows.sort(key=sort_key)
                # Write aggregated per-sample predictions
                agg_path = os.path.join(out_dir, "results_predictions.csv")
                with open(agg_path, "w") as outf:
                    outf.write(header + "\n")
                    outf.write("\n".join(agg_rows) + "\n")
                logger.info(f"Aggregated per-sample predictions saved to {agg_path}")

                # Write aggregated accuracy summary CSV (one row)
                try:
                    acc_path = os.path.join(out_dir, "robust.csv")
                    def pct(a, b):
                        return 0.0 if b == 0 else float(a) / float(b) * 100.0
                    with open(acc_path, "w", newline="") as af:
                        writer = csv.writer(af)
                        writer.writerow(["run_name", "src_model", "target_model", "modality", "dataset", "eps", "run_max_samples", "batch", "total", "correct_src_clean", "correct_src_adv", "correct_target_adv", "src_clean_acc_pct", "src_adv_acc_pct", "target_adv_acc_pct"]) 
                        run_name = os.path.basename(out_dir)
                        writer.writerow([run_name, getattr(args, 'src_model', ''), getattr(args, 'robust_lora_weights', ''), getattr(args, 'modality', ''), ds_name or '', getattr(args, 'eps', ''), getattr(args, 'run_max_samples', ''), getattr(args, 'batch_size', ''), total, correct_src_clean, correct_src_adv, correct_target_adv, f"{pct(correct_src_clean, total):.2f}", f"{pct(correct_src_adv, total):.2f}", f"{pct(correct_target_adv, total):.2f}"])
                    logger.info(f"Aggregated accuracy CSV saved to {acc_path}")
                except Exception as e:
                    logger.warning(f"Failed to write accuracy CSV: {e}")
        except Exception as e:
            logger.warning(f"Failed to aggregate CSVs: {e}")
    else:
        # Single-GPU path (fall back to existing behaviour)
        device = torch.device(f"cuda:0")
        torch.cuda.set_device(device)

        # Load centre embeddings and mapping
        logging.getLogger("transfer_unibind").info(f"[transfer_attack] loading centre embeddings from {centre_path} on device {device}")
        centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
        logging.getLogger("transfer_unibind").info(f"[transfer_attack] loaded centres: {len(centre_labels)} labels, lbl_to_idx size={len(lbl_to_idx)}")

        # Build source and target models
    # Source can be UniBind, clip-vit-14 (CLIP-like), or ImageBind
        if args.src_model == 'unibind':
            src_kwargs = argparse.Namespace(
                use_flash_attention=True,
                use_lora=False,
                lora_rank=4,
                lora_alpha=8,
                use_modality_head_mlp=False,
                lora_weights=None,
                modality_head_mlp_weights=None,
            )
            src_unibind_kwargs = get_unibind_kwargs(src_kwargs)
            src_models = build_models_on_devices([0], centre_emb, centre_labels, lbl_to_idx, modality, args.unibind_weights, src_unibind_kwargs, logger)
            src_model = src_models[0]
        elif args.src_model == 'clip':
            # Use a true Hugging Face CLIP model as the source encoder
            # Pass label_to_index and let the classifier build its class strings
            src_model = CLIPClassifier(device, modality, None, logger=logger, label_to_index=lbl_to_idx)
            src_model = src_model.to(device)
            src_model.eval()
        elif args.src_model == 'imagebind':
            # Pass label_to_index and let the classifier build its class strings
            src_model = ImageBindClassifier(device, modality, None, logger=logger, label_to_index=lbl_to_idx)
            src_model = src_model.to(device)
            src_model.eval()
        else:
            raise ValueError(f"Unknown src_model: {args.src_model}")

        # Target: RobustBind^4 (apply LoRA weights)
        # Choose LoRA weights: user-provided or modality-specific default
        robust_w = args.robust_lora_weights or MODALITY_ROBUST_LORA_WEIGHTS.get(modality_key)
        target_kwargs = argparse.Namespace(
            use_flash_attention=True,
            use_lora=True,
            lora_rank=4,
            lora_alpha=8,
            use_modality_head_mlp=False,
            lora_weights=robust_w,
            modality_head_mlp_weights=None,
        )
        target_unibind_kwargs = get_unibind_kwargs(target_kwargs)
        target_models = build_models_on_devices([0], centre_emb, centre_labels, lbl_to_idx, modality, args.unibind_weights, target_unibind_kwargs, logger)
        target_model = target_models[0]

        # Build loader and get normalization tensors
        loader = _make_loader_with_uniform_sampling(modality, dataset_root, val_json, lbl_to_idx, batch_size=args.batch_size, num_workers=0, run_max_samples=args.run_max_samples)
        mean, std = get_normalization_tensors(modality, device)

        # Use shared attack loop to run the single-GPU combo
        combo_ns = argparse.Namespace(**vars(args))
        combo_ns.run_max_samples = args.run_max_samples
        combo_ns.steps = args.steps
        combo_ns.eps = args.eps
        # per-dataset temperature
        combo_ns.temperature = DATASET_TEMPERATURES.get(ds_name, 1000.0)
        results = _run_attack_loop(logger, src_model, target_model, loader, mean, std, device, combo_ns, results_predictions_path, adv_dir, label_names=centre_labels)
        total = results.get('total', 0)
        correct_src_clean = results.get('correct_src_clean', 0)
        correct_target_clean = results.get('correct_target_clean', 0)
        correct_src_adv = results.get('correct_src_adv', 0)
        correct_target_adv = results.get('correct_target_adv', 0)

        # Write accuracy CSV for this single-run
        try:
            def pct(a, b):
                return 0.0 if b == 0 else float(a) / float(b) * 100.0
            with open(results_accuracy_path, "w", newline="") as af:
                writer = csv.writer(af)
                writer.writerow(["run_name", "src_model", "target_model", "modality", "dataset", "eps", "run_max_samples", "batch", "total", "correct_src_clean", "correct_src_adv", "correct_target_adv", "src_clean_acc_pct", "src_adv_acc_pct", "target_adv_acc_pct"]) 
                run_name = os.path.basename(out_dir)
                # determine target_model string (LoRA weights path or modality default)
                target_model_name = getattr(args, 'robust_lora_weights', None) or MODALITY_ROBUST_LORA_WEIGHTS.get(modality_key)
                writer.writerow([run_name, getattr(args, 'src_model', ''), target_model_name, getattr(args, 'modality', ''), ds_name or '', getattr(args, 'eps', ''), getattr(args, 'run_max_samples', ''), getattr(args, 'batch_size', ''), total, correct_src_clean, correct_src_adv, correct_target_adv, f"{pct(correct_src_clean, total):.2f}", f"{pct(correct_src_adv, total):.2f}", f"{pct(correct_target_adv, total):.2f}"])
            logger.info(f"Wrote accuracy CSV to {results_accuracy_path}")
        except Exception as e:
            logger.warning(f"Failed to write accuracy CSV: {e}")

    # (Multi-GPU path aggregated results are now in total/correct_*; single-GPU handled in the else branch above.)

    # Summarize
    def pct(a, b):
        return 0.0 if b == 0 else float(a) / float(b) * 100.0

    logger.info("=== Summary ===")
    logger.info(f"Samples evaluated: {total}")
    logger.info(f"UniBind clean accuracy: {pct(correct_src_clean, total):.2f}% ({correct_src_clean}/{total})")
    logger.info(f"RobustBind^4 clean accuracy: {pct(correct_target_clean, total):.2f}% ({correct_target_clean}/{total})")
    logger.info(f"UniBind adv accuracy: {pct(correct_src_adv, total):.2f}% ({correct_src_adv}/{total})")
    logger.info(f"RobustBind^4 adv accuracy: {pct(correct_target_adv, total):.2f}% ({correct_target_adv}/{total})")
    # Also print summary to console for visibility when subprocesses are run


def _attack_worker(gpu_index: int, worker_id: int, num_workers: int, args, result_queue, out_dir: str):
    """Worker process: runs the attack on a single GPU, processing only batches where (batch_idx % num_workers) == worker_id.

    Reports a dict with counts via result_queue, then sends None to signal completion.
    """
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)

    # Per-worker logger
    log_path = os.path.join(out_dir, f"attack_gpu{gpu_index}.log")
    logger = setup_logger(log_path)
    logger.info(f"[gpu{gpu_index}] Worker start (id={worker_id}/{num_workers})")

    # Load centre embeddings and mapping on this device
    centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(args.centre_embeddings or "./centre_embs/image_in_center_embeddings.pkl", device)

    # Build source model on this GPU (support unibind, clip-vit-14, imagebind)
    if getattr(args, 'src_model', 'unibind') == 'unibind':
        src_kwargs = argparse.Namespace(
            use_flash_attention=True,
            use_lora=False,
            lora_rank=4,
            lora_alpha=8,
            use_modality_head_mlp=False,
            lora_weights=None,
            modality_head_mlp_weights=None,
        )
        src_unibind_kwargs = get_unibind_kwargs(src_kwargs)
        src_models = build_models_on_devices([gpu_index], centre_emb, centre_labels, lbl_to_idx, Modality[args.modality.upper()], args.unibind_weights, src_unibind_kwargs, logger)
        src_model = src_models[gpu_index]
    elif getattr(args, 'src_model', 'unibind') == 'clip':
        # CLIP-like encoder: use true CLIP wrapper
        src_model = CLIPClassifier(device, Modality[args.modality.upper()], None, logger=logger, label_to_index=lbl_to_idx)
        src_model = src_model.to(device)
        src_model.eval()
    elif getattr(args, 'src_model', 'unibind') == 'imagebind':
        src_model = ImageBindClassifier(device, Modality[args.modality.upper()], None, logger=logger, label_to_index=lbl_to_idx)
        src_model = src_model.to(device)
        src_model.eval()
    else:
        raise ValueError(f"Unknown src_model: {getattr(args, 'src_model')}")

    target_kwargs = argparse.Namespace(
        use_flash_attention=True,
        use_lora=True,
        lora_rank=4,
        lora_alpha=8,
        use_modality_head_mlp=False,
        lora_weights=args.robust_lora_weights,
        modality_head_mlp_weights=None,
    )
    target_unibind_kwargs = get_unibind_kwargs(target_kwargs)
    target_models = build_models_on_devices([gpu_index], centre_emb, centre_labels, lbl_to_idx, Modality[args.modality.upper()], args.unibind_weights, target_unibind_kwargs, logger)
    
    loader = _make_loader_with_uniform_sampling(Modality[args.modality.upper()], args.dataset_root, args.val_json, lbl_to_idx, batch_size=args.batch_size, num_workers=0, run_max_samples=args.run_max_samples)

    # Get dataset normalization tensors for this worker
    mean, std = get_normalization_tensors(Modality[args.modality.upper()], device)

    target_model = target_models[gpu_index]
    attack_model = AttackModel(src_model, mean=mean, std=std)
    stage1 = APGDAttack(logger, attack_model, norm="linf", n_restarts=1, n_iter=args.steps, eps=args.eps / 255.0, loss_type="ce", device=device)
    stage2 = APGDAttack(logger, attack_model, norm="linf", n_restarts=1, n_iter=args.steps, eps=args.eps / 255.0, loss_type="ce", device=device)

    total = 0
    correct_src_clean = 0
    correct_target_clean = 0
    correct_src_adv = 0
    correct_target_adv = 0
    # Prepare per-worker output paths
    worker_adv_dir = getattr(args, '_worker_adv_dir', os.path.join(out_dir, f"gpu{gpu_index}_adv"))
    worker_csv = getattr(args, '_worker_csv', os.path.join(out_dir, f"results_gpu{gpu_index}.csv"))
    os.makedirs(worker_adv_dir, exist_ok=True)
    csv_file = open(worker_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["path", "label", "src_clean", "target_clean", "src_adv", "target_adv", "cosine_sim", "l2_dist", "adv_path"])
    # Place per-worker fail CSVs outside the worker adv dir so adv dir contains only images
    worker_csv_dir = os.path.dirname(worker_csv) or out_dir
    # fail CSV for worker
    fail_csv_path = os.path.join(worker_csv_dir, f"src_attack_wrong_gpu{gpu_index}.csv")
    fail_csv = open(fail_csv_path, "w", newline="")
    fail_writer = csv.writer(fail_csv)
    fail_writer.writerow(["clean_path", "adv_path", "label", "label_name", "src_clean", "target_clean", "src_adv", "target_adv", "cosine_sim", "l2_dist"])
    n_src_attack_wrong_worker = 0
    # CSV for samples where the source was correct on the clean input but after attack
    # the source model becomes wrong while the target remains correct.
    clean_then_fail_csv_path = os.path.join(worker_csv_dir, f"target_robust_correct_gpu{gpu_index}.csv")
    clean_then_fail_csv = open(clean_then_fail_csv_path, "w", newline="")
    clean_then_fail_writer = csv.writer(clean_then_fail_csv)
    clean_then_fail_writer.writerow(["clean_path", "adv_path", "label", "label_name", "src_clean", "target_clean", "src_adv", "target_adv", "cosine_sim", "l2_dist"])
    n_target_robust_correct_worker = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"gpu{gpu_index} batches")):
        if (batch_idx % num_workers) != worker_id:
            continue
        x = batch['inputs'].to(device)
        y = batch['labels'].to(device)
        paths = batch.get('paths', [None] * x.size(0))

        B = x.size(0)
        if getattr(args, 'run_max_samples', None) and total >= args.run_max_samples:
            break
        if getattr(args, 'run_max_samples', None) and total + B > args.run_max_samples:
            keep = args.run_max_samples - total
            x = x[:keep]
            y = y[:keep]
            B = x.size(0)

        # use dataset-specific temperature if provided on args
        T = getattr(args, 'temperature', 1000.0)
        with torch.no_grad():
            logits_src_clean, _ = src_model._logits(x, temperature=T)
            preds_src_clean = logits_src_clean.argmax(dim=1)
            correct_src_clean += (preds_src_clean == y).sum().item()

            logits_target_clean, _ = target_model._logits(x, temperature=T)
            preds_target_clean = logits_target_clean.argmax(dim=1)
            correct_target_clean += (preds_target_clean == y).sum().item()

        with torch.no_grad():
            emb_orig = src_model(x, mode=ForwardMode.EMBEDDINGS)

        adv = two_stage_attack(logger, src_model, x, y, stage1, stage2, mean, std)

        with torch.no_grad():
            logits_src_adv, _ = src_model._logits(adv, temperature=T)
            preds_src_adv = logits_src_adv.argmax(dim=1)
            correct_src_adv += (preds_src_adv == y).sum().item()

            logits_target_adv, _ = target_model._logits(adv, temperature=T)
            preds_target_adv = logits_target_adv.argmax(dim=1)
            correct_target_adv += (preds_target_adv == y).sum().item()

        # Per-sample metrics and save adv samples for worker
        with torch.no_grad():
            emb_adv = src_model(adv, mode=ForwardMode.EMBEDDINGS)
            cos_per = torch.nn.functional.cosine_similarity(emb_adv, emb_orig, dim=1).cpu().tolist()
            l2_per = torch.norm(emb_adv - emb_orig, dim=1).cpu().tolist()

        adv_un = adv.detach().clone()
        unnormalize_inplace(adv_un, mean, std)

        for i in range(x.size(0)):
            pth = paths[i] if i < len(paths) else None
            label_i = int(y[i].item())
            label_name_i = None
            # We can map label index to human-readable label if centre_labels are available in this scope
            try:
                label_name_i = centre_labels[label_i]
            except Exception:
                label_name_i = None
            src_clean_i = int(preds_src_clean[i].item())
            target_clean_i = int(preds_target_clean[i].item())
            src_adv_i = int(preds_src_adv[i].item())
            target_adv_i = int(preds_target_adv[i].item())
            cosine_i = float(cos_per[i])
            l2_i = float(l2_per[i])

            adv_filename_base = f"adv_gpu{gpu_index}_total{total}_idx{i}"
            if adv_un.dim() == 4 and adv_un.size(1) == 3:
                arr = (adv_un[i].cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                adv_path = os.path.join(worker_adv_dir, adv_filename_base + ("_" + os.path.basename(pth) if pth else "") + ".png")
                Image.fromarray(arr).save(adv_path)
            else:
                adv_path = os.path.join(worker_adv_dir, adv_filename_base + ".npy")
                np.save(adv_path, adv_un[i].cpu().numpy())

            csv_writer.writerow([pth or "", label_i, label_name_i or "", src_clean_i, target_clean_i, src_adv_i, target_adv_i, cosine_i, l2_i, adv_path])
            if src_adv_i != label_i and target_adv_i == label_i:
                fail_writer.writerow([pth or "", adv_path, label_i, label_name_i or "", src_clean_i, target_clean_i, src_adv_i, target_adv_i, cosine_i, l2_i])
                n_src_attack_wrong_worker += 1
                try:
                    logger.info(f"[gpu{gpu_index}] Recorded src_attack_wrong: path={pth or ''} adv={adv_path} label={label_i}")
                except Exception:
                    pass
            # Record cases where the source was correct on the clean input and after attack
            # the source becomes wrong while the target stays correct.
            if src_clean_i == label_i and src_adv_i != label_i and target_adv_i == label_i:
                clean_then_fail_writer.writerow([pth or "", adv_path, label_i, label_name_i or "", src_clean_i, target_clean_i, src_adv_i, target_adv_i, cosine_i, l2_i])
                n_target_robust_correct_worker += 1
                try:
                    logger.info(f"[gpu{gpu_index}] Recorded target_robust_correct: path={pth or ''} adv={adv_path} label={label_i}")
                except Exception:
                    pass

        total += B

    csv_file.close()
    try:
        fail_csv.close()
    except Exception:
        pass
    try:
        clean_then_fail_csv.close()
    except Exception:
        pass
    try:
        logger.info(f"[gpu{gpu_index}] src_attack_wrong entries: {n_src_attack_wrong_worker}")
        logger.info(f"[gpu{gpu_index}] target_robust_correct entries: {n_target_robust_correct_worker}")
    except Exception:
        pass

    result_queue.put({'total': total, 'correct_src_clean': correct_src_clean, 'correct_target_clean': correct_target_clean, 'correct_src_adv': correct_src_adv, 'correct_target_adv': correct_target_adv})
    result_queue.put(None)


def _worker_run_combos(gpu_index, combo_ns_list):
    """Run assigned combos on a single physical GPU, reusing loaded models when possible.

    The child process sets CUDA_VISIBLE_DEVICES so the selected physical GPU appears as cuda:0.
    Combos are grouped by (src_model, modality, unibind_weights, robust_lora_weights) and
    models are loaded once per group and reused for all combos in the group.
    """
    import os
    from collections import defaultdict
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    # After remapping visible devices, the GPU will be cuda:0 inside this process
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # Group combos that can share loaded models
    groups = defaultdict(list)
    for ns in combo_ns_list:
        key = (
            getattr(ns, 'src_model', None),
            getattr(ns, 'modality', None),
            getattr(ns, 'unibind_weights', None),
            getattr(ns, 'robust_lora_weights', None),
        )
        groups[key].append(ns)

    for (src_model_name, modality_str, unibind_w, robust_w), ns_list in groups.items():
        # Use first ns to resolve dataset paths and common settings
        ns0 = ns_list[0]
        try:
            # Resolve modality enum from mapping
            mod_enum = next(k for k in MODALITY_DATASETS.keys() if k.value.lower() == modality_str)
        except StopIteration:
            logging.getLogger("transfer_unibind").warning(f"[gpu{gpu_index}] Unknown modality: {modality_str}; skipping group")
            continue

        cfg = MODALITY_DATASETS[mod_enum]
        val_json = getattr(ns0, 'val_json', None) or cfg.get('val_json')
        dataset_root = getattr(ns0, 'dataset_root', None) or cfg.get('dataset_root')
        centre_path = getattr(ns0, 'centre_embeddings', None) or cfg.get('centre_embeddings_path')

        # Prepare per-group logger
        try:
            log_dir = ns0.output
        except Exception:
            log_dir = '.'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"worker_gpu{gpu_index}.log")
        logger = setup_logger(log_path)
        logger.info(f"[gpu{gpu_index}] Starting group: src={src_model_name} mod={modality_str} unibind={unibind_w} robust={robust_w}")

        # Load centre embeddings on this device
        try:
            centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
        except Exception as e:
            logger.error(f"[gpu{gpu_index}] Failed to load centre embeddings {centre_path}: {e}")
            continue

        # Build reusable models depending on src_model_name
        src_model_obj = None
        try:
            if src_model_name == 'unibind':
                src_kwargs = argparse.Namespace(
                    use_flash_attention=True,
                    use_lora=False,
                    lora_rank=4,
                    lora_alpha=8,
                    use_modality_head_mlp=False,
                    lora_weights=None,
                    modality_head_mlp_weights=None,
                )
                src_unibind_kwargs = get_unibind_kwargs(src_kwargs)
                src_models = build_models_on_devices([0], centre_emb, centre_labels, lbl_to_idx, mod_enum, unibind_w, src_unibind_kwargs, logger)
                src_model_obj = src_models[0]
            elif src_model_name == 'clip':
                unique_labels = sorted(set(centre_labels))
                src_model_obj = CLIPClassifier(device, mod_enum, centre_labels, logger=logger, label_to_index=lbl_to_idx).to(device)
                src_model_obj.eval()
            elif src_model_name == 'imagebind':
                unique_labels = sorted(set(centre_labels))
                src_model_obj = ImageBindClassifier(device, mod_enum, centre_labels, logger=logger, label_to_index=lbl_to_idx).to(device)
                src_model_obj.eval()
            else:
                logger.error(f"[gpu{gpu_index}] Unknown src_model: {src_model_name}; skipping group")
                continue
        except Exception as e:
            # Log full traceback to help diagnose failures (e.g., tokenizer download, missing files)
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[gpu{gpu_index}] Failed to build source model (src={src_model_name}): {e}\n{tb}")
            logger.error("Possible causes: missing tokenizer/model files, network disabled for downloading tokenizer, or incompatible library versions.")
            continue

        # Build target model once per group (RobustBind^4 LoRA applied)
        try:
            target_kwargs = argparse.Namespace(
                use_flash_attention=True,
                use_lora=True,
                lora_rank=4,
                lora_alpha=8,
                use_modality_head_mlp=False,
                lora_weights=robust_w,
                modality_head_mlp_weights=None,
            )
            target_unibind_kwargs = get_unibind_kwargs(target_kwargs)
            target_models = build_models_on_devices([0], centre_emb, centre_labels, lbl_to_idx, mod_enum, unibind_w, target_unibind_kwargs, logger)
            target_model_obj = target_models[0]
        except Exception as e:
            logger.error(f"[gpu{gpu_index}] Failed to build target model: {e}")
            continue

        # Precompute normalization tensors for this modality
        mean, std = get_normalization_tensors(mod_enum, device)

        # Now run each combo in this group (different eps/steps/batch_size)
        for ns in ns_list:
            try:
                logger.info(f"[gpu{gpu_index}] Running combo: src={ns.src_model} eps={ns.eps} mod={ns.modality}")
                loader = _make_loader_with_uniform_sampling(mod_enum, dataset_root, val_json, lbl_to_idx, batch_size=ns.batch_size, num_workers=0, run_max_samples=getattr(ns, 'run_max_samples', None))
                results_csv_path = os.path.join(ns.output, "results_predictions.csv")
                adv_dir = os.path.join(ns.output, "adv_samples")
                start_ts = datetime.now()
                try:
                    res = _run_attack_loop(logger, src_model_obj, target_model_obj, loader, mean, std, device, ns, results_csv_path, adv_dir, label_names=centre_labels)
                    exit_code = 0
                except Exception:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"[gpu{gpu_index}] Exception while running combo {ns.output}:\n{tb}")
                    res = {'total': 0, 'correct_src_clean': 0, 'correct_target_clean': 0, 'correct_src_adv': 0, 'correct_target_adv': 0}
                    exit_code = 1
                end_ts = datetime.now()
                elapsed = (end_ts - start_ts).total_seconds()

                logger.info(f"[gpu{gpu_index}] Combo finished: output={ns.output} samples={res.get('total',0)} elapsed_s={elapsed:.2f} exit_code={exit_code}")
            except Exception:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"[gpu{gpu_index}] Unexpected exception running combo {ns.output}:\n{tb}")

        # Done with this group; free models
        try:
            del src_model_obj
            del target_model_obj
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transfer attack: UniBind (source) -> RobustBind^4 (target) for classification datasets")
    ap.add_argument("--modality", type=str, default="image", help="modality to run (image|audio|...)")
    ap.add_argument("--val_json", type=str, default=VAL_JSONS.get('image'), help="Path to validation JSON for dataset (overrides default)")
    ap.add_argument("--dataset_root", type=str, default="/data/datasets/ImageNet-1K", help="Dataset root (overrides default)")
    ap.add_argument("--centre_embeddings", type=str, default="./centre_embs/image_in_center_embeddings.pkl", help="Path to centre embeddings pickle (overrides default)")
    ap.add_argument("--unibind_weights", type=str, default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt", help="Path to UniBind pretrain weights")
    ap.add_argument("--robust_lora_weights", type=str, default=None, help="Path to RobustBind^4 LoRA weights (target). If omitted, a modality-specific default is used.")
    ap.add_argument("--src_model", type=str, default="unibind", choices=["unibind", "clip", "imagebind"], help="Which source encoder to attack (unibind|clip|imagebind)")
    # default to running the full grid unless user explicitly disables
    ap.add_argument("--run_all", action='store_true', default=True, help="Run the full grid of experiments (per-modality source models x EPS_LIST x MODALITIES).")
    ap.add_argument("--run_max_samples", type=int, default=DEFAULT_RUN_MAX_SAMPLES, help="When --run_all is used, this sets per-run --run_max_samples (total across GPUs).")
    ap.add_argument("--eps", type=float, default=EPS_LIST[-1], help="ε in pixel values (out of 255)")
    ap.add_argument("--steps", type=int, default=100, help="APGD iterations per stage")
    ap.add_argument("--batch_size", type=int, default=None, help="Batch size (if omitted, a per-dataset default map is used)")
    # Note: --max_samples has been removed; use --run_max_samples instead for all run limits.
    ap.add_argument("--output", type=str, default="/data/output/transfer_attack", help="Output directory to save logs/results")
    args = ap.parse_args()
    # If user requests a single-run (not the grid), validate the chosen src_model is allowed for the modality
    if not args.run_all:
        mod = args.modality
        # If there is no entry for this modality, return an empty allowed list
        allowed = MODALITY_SRC_MODELS.get(mod, [])
        if args.src_model not in allowed:
            logging.getLogger("transfer_unibind").error(f"Error: src_model '{args.src_model}' is not supported for modality '{mod}'. Allowed: {allowed}")
            import sys
            sys.exit(1)
    # If run_all was requested, run the grid of experiments by invoking this script as subprocesses
    if args.run_all:
        # Parallel run-all orchestration: build combo tasks and distribute across GPUs round-robin
        import sys
        # Create a timestamped base_out for the run_all outputs (matches main() behavior)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output:
            base_out = os.path.join(args.output, ts)
        else:
            base_out = f"output/transfer_unibind_to_robustbind_{ts}"
        os.makedirs(base_out, exist_ok=True)
        ctx = get_context('spawn')

        # Build all combo namespaces
        all_worker_ns = []
        for eps in EPS_LIST:
            for mod in MODALITIES:
                src_list = MODALITY_SRC_MODELS.get(mod, [])
                if not src_list:
                    logging.getLogger("transfer_unibind").info(f"Skipping modality {mod}: no source models configured")
                    continue
                for src in src_list:
                    run_name = f"src-{src}_eps{int(eps)}_mod-{mod}"
                    run_out = os.path.join(base_out, run_name)
                    os.makedirs(run_out, exist_ok=True)
                    # determine per-modality/default batch size
                    try:
                        mod_enum = next(k for k in MODALITY_DATASETS.keys() if k.value.lower() == mod)
                        ds_name = MODALITY_DATASETS[mod_enum]["dataset_name"]
                    except StopIteration:
                        ds_name = None
                    # Use attack-specific batch sizes for attack runs when available
                    batch_for_mod = ATTACK_VAL_BATCH_SIZE_MAP.get(ds_name, CLEAN_VAL_BATCH_SIZE_MAP.get(ds_name, DEFAULT_BATCH_SIZE_FALLBACK)) if ds_name else DEFAULT_BATCH_SIZE_FALLBACK

                    worker_ns = argparse.Namespace(**vars(args))
                    worker_ns.modality = mod
                    worker_ns.src_model = src
                    worker_ns.eps = eps
                    worker_ns.batch_size = batch_for_mod
                    worker_ns.run_max_samples = args.run_max_samples
                    worker_ns.output = run_out
                    worker_ns.run_all = False
                    # attach per-run temperature
                    worker_ns.temperature = DATASET_TEMPERATURES.get(ds_name, 1000.0)
                    # pass through optional overrides
                    if args.unibind_weights:
                        worker_ns.unibind_weights = args.unibind_weights
                    # Use user-provided LoRA weights or fall back to modality-specific defaults
                    if args.robust_lora_weights:
                        worker_ns.robust_lora_weights = args.robust_lora_weights
                    else:
                        worker_ns.robust_lora_weights = MODALITY_ROBUST_LORA_WEIGHTS.get(mod)
                    if args.centre_embeddings:
                        worker_ns.centre_embeddings = args.centre_embeddings
                    if args.val_json:
                        worker_ns.val_json = args.val_json
                    else:
                        vj = VAL_JSONS.get(mod)
                        if vj:
                            worker_ns.val_json = vj
                    if args.dataset_root:
                        worker_ns.dataset_root = args.dataset_root

                    all_worker_ns.append(worker_ns)

        # If no GPU found, fall back to sequential runs
        num_gpus = max(1, torch.cuda.device_count())
        if num_gpus <= 1 or len(all_worker_ns) == 0:
            logging.getLogger("transfer_unibind").info("No multiple GPUs detected or no tasks: running combos sequentially")
            run_records = []
            for worker_ns in all_worker_ns:
                logging.getLogger("transfer_unibind").info(f"Forking combo: src={worker_ns.src_model} eps={int(worker_ns.eps)} mod={worker_ns.modality}")
                try:
                    main(worker_ns)
                except SystemExit:
                    pass
                except Exception:
                    import traceback
                    tb = traceback.format_exc()
                    logging.getLogger("transfer_unibind").exception(f"Exception in combo {worker_ns.modality}:\n{tb}")
                run_records.append(worker_ns.output)
        else:
            # Distribute tasks round-robin across physical GPUs
            gpu_indices_all = list(range(torch.cuda.device_count()))
            tasks_per_gpu = [[] for _ in gpu_indices_all]
            for i, ns in enumerate(all_worker_ns):
                tasks_per_gpu[i % len(gpu_indices_all)].append(ns)

            # _worker_run_combos is defined at module level so it is importable/picklable by multiprocessing.spawn

            # Launch one process per GPU
            workers = []
            for gi, gpu_idx in enumerate(gpu_indices_all):
                worker_tasks = tasks_per_gpu[gi]
                if not worker_tasks:
                    continue
                p = ctx.Process(target=_worker_run_combos, args=(gpu_idx, worker_tasks))
                p.start()
                workers.append(p)
                logging.getLogger("transfer_unibind").info(f"Launched worker on gpu{gpu_idx} with {len(worker_tasks)} tasks")

            # Wait for workers to finish
            for p in workers:
                p.join()

            # Collect run directories for aggregation
            run_records = [ns.output for ns in all_worker_ns]

        # After all runs complete, collect per-run `robust.csv` files and merge into a final
        # accuracy CSV directly under the timestamped base_out directory. This final CSV
        # contains one row per run with fields: src_model, target_model, modality, dataset,
        # eps, run_max_samples (data size), batch, total, src_clean_acc_pct, src_robust_acc_pct,
        # robustbind_robust_acc_pct, plus raw counts.
        final_rows = []
        final_header = ["src_model", "target_model", "modality", "dataset", "eps", "run_max_samples", "batch", "total", "correct_src_clean", "correct_src_adv", "correct_target_adv", "src_clean_acc_pct", "src_adv_acc_pct", "target_adv_acc_pct"]
        for rdir in run_records:
            # prefer per-run robust.csv, fall back to earlier aggregated formats if needed
            robust_csv = os.path.join(rdir, 'robust.csv')
            if os.path.exists(robust_csv):
                try:
                    with open(robust_csv, 'r') as f:
                        rdr = csv.reader(f)
                        rows = list(rdr)
                    if len(rows) < 2:
                        continue
                    hdr = rows[0]
                    vals = rows[1]
                    # Build a dict mapping header->value
                    m = {k: v for k, v in zip(hdr, vals)}
                    src_model = m.get('src_model', '')
                    target_model = m.get('target_model', '')
                    modality = m.get('modality', '')
                    dataset = m.get('dataset', '')
                    eps = m.get('eps', '')
                    run_max = m.get('run_max_samples', '')
                    batch = m.get('batch', '')
                    total_v = m.get('total', '')
                    correct_src_clean = m.get('correct_src_clean', '')
                    correct_src_adv = m.get('correct_src_adv', '')
                    correct_target_adv = m.get('correct_target_adv', '')
                    src_clean_pct = m.get('src_clean_acc_pct', '')
                    src_adv_pct = m.get('src_adv_acc_pct', '')
                    target_adv_pct = m.get('target_adv_acc_pct', '')
                    final_rows.append([src_model, target_model, modality, dataset, eps, run_max, batch, total_v, correct_src_clean, correct_src_adv, correct_target_adv, src_clean_pct, src_adv_pct, target_adv_pct])
                except Exception:
                    continue
            else:
                # Fallback: attempt to read previous aggregated per-sample CSV and extract counts
                agg_csv = os.path.join(rdir, 'results_aggregated.csv')
                if not os.path.exists(agg_csv):
                    run_csv = os.path.join(rdir, 'results.csv')
                    if os.path.exists(run_csv):
                        agg_csv = run_csv
                    else:
                        import glob
                        parts = glob.glob(os.path.join(rdir, 'results_gpu*.csv'))
                        if parts:
                            # try to compute totals from per-worker CSVs by counting lines
                            total_v = 0
                            correct_src_clean = 0
                            correct_src_adv = 0
                            correct_target_adv = 0
                            for p in parts:
                                with open(p, 'r') as f:
                                    lines = f.read().splitlines()
                                if not lines:
                                    continue
                                for ln in lines[1:]:
                                    if not ln.strip():
                                        continue
                                    cols = ln.split(',')
                                    # expected order: path,label,src_clean,target_clean,src_adv,target_adv,...
                                    try:
                                        src_clean = int(cols[2])
                                        src_adv = int(cols[4])
                                        tgt_adv = int(cols[5])
                                        label = int(cols[1])
                                    except Exception:
                                        continue
                                    total_v += 1
                                    if src_clean == label:
                                        correct_src_clean += 1
                                    if src_adv == label:
                                        correct_src_adv += 1
                                    if tgt_adv == label:
                                        correct_target_adv += 1
                            # try to infer run metadata from folder name
                            bn = os.path.basename(rdir)
                            parts = bn.split('_')
                            src = parts[0].split('-')[1] if len(parts) > 0 and '-' in parts[0] else ''
                            eps_s = parts[1].replace('eps', '') if len(parts) > 1 else ''
                            mod = parts[2].split('-')[1] if len(parts) > 2 and '-' in parts[2] else ''
                            final_rows.append([src, '', mod, '', eps_s, '', '', str(total_v), str(correct_src_clean), str(correct_src_adv), str(correct_target_adv), '', '', ''])

        # Sort final rows by src_model, target_model, modality, dataset
        final_rows.sort(key=lambda r: (r[0] or '', r[1] or '', r[2] or '', r[3] or ''))
        final_csv = os.path.join(base_out, 'robust.csv')
        if final_rows:
            with open(final_csv, 'w', newline='') as outf:
                writer = csv.writer(outf)
                writer.writerow(final_header)
                for row in final_rows:
                    writer.writerow(row)
            logging.getLogger("transfer_unibind").info(f"Final accuracy CSV written to: {final_csv}")
        else:
            logging.getLogger("transfer_unibind").info("No per-run robust.csv files found; no final accuracy CSV produced.")

        sys.exit(0)
    try:
        main(args)
    except Exception as e:
        # Best-effort logging to console and to run.log
        import traceback
        tb = traceback.format_exc()
        logging.getLogger("transfer_unibind").exception(f"Exception in transfer_attack:\n{tb}")
        # attempt to write to run.log in output dir
        try:
            out_dir = args.output if hasattr(args, 'output') else '/tmp'
            with open(os.path.join(out_dir, 'run.log'), 'a') as f:
                f.write('\nException:\n')
                f.write(tb)
        except Exception:
            pass
        raise
