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
import sys
import traceback
import subprocess
import threading
import queue
import glob
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
    "audio": "./ckpts/audio_eps4_lora_weights_old.pt",
    "thermal": "./ckpts/thermal_eps4_lora_weights_old.pt",
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

# Per-modality allowed source models. This lets each modality use only compatible encoders.
# Keys are modality names (same as MODALITIES entries). Values are lists of src_model strings.
MODALITY_SRC_MODELS = {
    "image": [
        "unibind", 
        "clip", 
        # "imagebind"
    ],
    "audio": [
        "unibind", 
        # "clip-vit-14"
    ],
    "thermal": [
        "unibind", 
        # "imagebind"
    ],
}

# Populate the derived ALL_SRC_MODELS after the authoritative modality mapping is defined
ALL_SRC_MODELS = sorted({s for lst in MODALITY_SRC_MODELS.values() for s in lst})

# -----------------------------------------------------------------------------


def setup_logger(log_path: str, stream_to_console: bool = True) -> logging.Logger:
    # Use a logger name unique to the log_path so multiple concurrent
    # loggers (base run.log, per-worker logs) don't share handlers and
    # cause duplicate entries to be written into each other's files.
    logger_name = f"transfer_attack:{os.path.abspath(log_path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Clear handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    # Write an initial log entry so the file is created and visible immediately
    logger.info(f"Logger initialized: {log_path}")
    return logger


# Create a logger for a given output directory. Returns a fresh logger instance
# configured to write to console and to out_dir/run.log.
def create_logger(out_dir: str) -> logging.Logger:
    log_path = os.path.join(out_dir, "run.log")
    # Top-level run logger should also log to console
    logger = setup_logger(log_path, stream_to_console=True)
    return logger


def _format_ns_summary(ns, max_item_len=200):
    """Return a safe, multi-line string summary of a Namespace's settings.

    Each key will be on its own line as `key: value`. Long values are
    truncated to `max_item_len` chars to keep logs readable.
    """
    try:
        d = dict(vars(ns))
    except Exception:
        try:
            # Fallback: pick public attributes
            d = {k: getattr(ns, k) for k in dir(ns) if not k.startswith('_') and not callable(getattr(ns, k))}
        except Exception:
            return repr(ns)

    lines = []
    for k in sorted(d.keys()):
        try:
            v = d[k]
            s = repr(v)
            if len(s) > max_item_len:
                s = s[:max_item_len] + "..."
        except Exception:
            s = "<unreprable>"
        lines.append(f"{k}: {s}")
    # Join with newline and indent for readability in logs
    return "\n" + "\n".join(["  " + ln for ln in lines])


def prepare_output_and_logger(args):
    """Create timestamped output dir, adv dir, and initialize a run-local logger.

    Returns: out_dir, results_predictions_path, results_accuracy_path, adv_dir, logger
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        base_out = os.path.join(args.output, ts)

    os.makedirs(base_out, exist_ok=True)
    out_dir = base_out
    # initialize a run-local logger (console + run.log)
    logger = create_logger(out_dir)
    logger.info(f"[transfer_attack] logger initialized, log_path={os.path.join(out_dir, 'run.log')}")
    # Log effective task settings so every task run records the exact configuration
    try:
        logger.info(f"[task settings] { _format_ns_summary(args) }")
    except Exception:
        try:
            logger.info(f"[task settings] (failed to format args) {repr(args)}")
        except Exception:
            pass

    results_predictions_path = os.path.join(out_dir, "results_predictions.csv")
    results_accuracy_path = os.path.join(out_dir, "robust.csv")
    adv_dir = os.path.join(out_dir, "adv_samples")
    os.makedirs(adv_dir, exist_ok=True)
    return out_dir, results_predictions_path, results_accuracy_path, adv_dir, logger


def spawn_namespace_subprocess(ns: argparse.Namespace):
    """Spawn the current script as a subprocess with CLI flags derived from ns.

    This avoids calling main(ns) recursively in the same process.
    """
    cmd = [sys.executable, os.path.abspath(__file__)]

    def add_flag(name, flag=None):
        v = getattr(ns, name, None)
        if v is None:
            return
        if flag is None:
            flag = f"--{name}"
        # boolean flags
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    # Common flags used by the script
    add_flag('modality')
    add_flag('src_model')
    add_flag('eps')
    add_flag('steps')
    add_flag('batch_size')
    add_flag('run_max_samples')
    add_flag('output')
    add_flag('val_json')
    add_flag('dataset_root')
    add_flag('centre_embeddings')
    add_flag('unibind_weights')
    add_flag('robust_lora_weights')

    # Instead of launching a subprocess, call main(ns) directly to run in-process.
    # This avoids recursion issues because worker namespaces have run_all=False.
    try:
        main(ns)
    except Exception:
        tb = traceback.format_exc()
        print(f"Exception running worker main(ns): {tb}", file=sys.stderr)


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
    # Resolve defaults; prefer explicit user overrides but avoid accepting the
    # parser-level IMAGE defaults (which would silently override modality
    # specific defaults). Only treat args.* as an override if it is provided
    # and not equal to the image-default values defined above.
    image_default_val_json = VAL_JSONS.get('image')
    image_default_dataset_root = "/data/datasets/ImageNet-1K"
    image_default_centre = "./centre_embs/image_in_center_embeddings.pkl"

    if getattr(args, 'val_json', None) and args.val_json != image_default_val_json:
        val_json = args.val_json
    else:
        val_json = cfg.get("val_json")

    if getattr(args, 'dataset_root', None) and args.dataset_root != image_default_dataset_root:
        dataset_root = args.dataset_root
    else:
        dataset_root = cfg.get("dataset_root")

    # Always use the modality's configured centre embeddings path. Do NOT
    # accept the top-level parser `--centre_embeddings` as an override because
    # that can point to image-specific centres and incorrectly be applied to
    # other modalities.
    centre_path = cfg.get("centre_embeddings_path")
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
    # Write per-combo summary CSV with accuracy metrics (robust.csv)
    try:
        results_dir = os.path.dirname(results_csv_path) or adv_dir
        results_summary_path = os.path.join(results_dir, "robust.csv")
        def _percent(a, b):
            return 0.0 if b == 0 else float(a) / float(b) * 100.0
        run_name = os.path.basename(results_dir) or "run"
        with open(results_summary_path, "w", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerow(["run_name", "src_model", "target_model", "modality", "eps", "run_max_samples", "batch", "total", "correct_src_clean", "correct_src_adv", "correct_target_adv", "src_clean_acc_percent", "src_adv_acc_percent", "target_adv_acc_percent"])
            writer.writerow([
                run_name,
                getattr(ns, 'src_model', ''),
                getattr(ns, 'robust_lora_weights', ''),
                getattr(ns, 'modality', ''),
                getattr(ns, 'eps', ''),
                getattr(ns, 'run_max_samples', ''),
                getattr(ns, 'batch_size', ''),
                total,
                correct_src_clean,
                correct_src_adv,
                correct_target_adv,
                f"{_percent(correct_src_clean, total):.2f}",
                f"{_percent(correct_src_adv, total):.2f}",
                f"{_percent(correct_target_adv, total):.2f}",
            ])
        try:
            logger.info(f"Wrote per-combo summary CSV to: {results_summary_path}")
        except Exception:
            pass
    except Exception:
        try:
            logger.exception("Failed to write per-combo robust.csv")
        except Exception:
            pass
    return {
        'total': total,
        'correct_src_clean': correct_src_clean,
        'correct_target_clean': correct_target_clean,
        'correct_src_adv': correct_src_adv,
        'correct_target_adv': correct_target_adv,
    }


def main(args=None):
    # If args not provided, this is the top-level entry; parse CLI args here.
    if args is None:
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
        ap.add_argument("--run_max_samples", type=int, default=500, help="When --run_all is used, this sets per-run --run_max_samples (total across GPUs).")
        ap.add_argument("--eps", type=float, default=[2.0, 4.0], help="ε in pixel values (out of 255)")
        ap.add_argument("--steps", type=int, default=100, help="APGD iterations per stage")
        ap.add_argument("--batch_size", type=int, default=None, help="Batch size (if omitted, a per-dataset default map is used)")
        # Note: --max_samples has been removed; use --run_max_samples instead for all run limits.
        ap.add_argument("--output", type=str, default="/data/output/transfer_attack", help="Output directory to save logs/results")
        args = ap.parse_args()

    # If top-level run_all is requested, create a timestamped base_out and a run-level logger here
    if getattr(args, 'run_all', False):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output:
            base_out = os.path.join(args.output, ts)
        else:
            base_out = f"output/transfer_unibind_to_robustbind_{ts}"
        os.makedirs(base_out, exist_ok=True)
        logger = create_logger(base_out)
        logger.info(f"[transfer_attack] main start (run_all), args.modality={getattr(args,'modality',None)}, src_model={getattr(args,'src_model',None)} base_out={base_out}")
        # Build all combo namespaces and distribute across GPUs (moved from __main__ block)
        all_worker_ns = []

        # Determine which modalities actually have configured source models.
        # Log skipped modalities once (instead of repeating per-ε in EPS_LIST).
        modality_to_src = {mod: MODALITY_SRC_MODELS.get(mod, []) for mod in MODALITIES}
        eps_str = ",".join(str(e) for e in EPS_LIST)
        for mod_k, src_list_k in modality_to_src.items():
            if not src_list_k:
                logger.info(f"Skipping modality {mod_k}: no source models configured (eps=[{eps_str}])")

        for eps in EPS_LIST:
            for mod in MODALITIES:
                src_list = modality_to_src.get(mod, [])
                if not src_list:
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
                    batch_for_mod = ATTACK_VAL_BATCH_SIZE_MAP.get(ds_name, CLEAN_VAL_BATCH_SIZE_MAP.get(ds_name, DEFAULT_BATCH_SIZE_FALLBACK)) if ds_name else DEFAULT_BATCH_SIZE_FALLBACK

                    worker_ns = argparse.Namespace(**vars(args))
                    worker_ns.modality = mod
                    worker_ns.src_model = src
                    worker_ns.eps = eps
                    worker_ns.batch_size = batch_for_mod
                    worker_ns.run_max_samples = args.run_max_samples
                    worker_ns.output = run_out
                    worker_ns.run_all = False
                    worker_ns.temperature = DATASET_TEMPERATURES.get(ds_name, 1000.0)
                    if args.unibind_weights:
                        worker_ns.unibind_weights = args.unibind_weights
                    if args.robust_lora_weights:
                        worker_ns.robust_lora_weights = args.robust_lora_weights
                    else:
                        worker_ns.robust_lora_weights = MODALITY_ROBUST_LORA_WEIGHTS.get(mod)
                    # Treat centre/val overrides as explicit only if they differ from
                    # the image-mode parser defaults so the parser's image defaults
                    # don't silently replace modality defaults.
                    image_default_val_json = VAL_JSONS.get('image')
                    image_default_dataset_root = "/data/datasets/ImageNet-1K"
                    image_default_centre = "./centre_embs/image_in_center_embeddings.pkl"
                    # Never propagate the top-level `--centre_embeddings` into
                    # per-modality worker namespaces; let workers resolve the
                    # correct centre path from MODALITY_DATASETS.
                    worker_ns.centre_embeddings = None
                    if getattr(args, 'val_json', None) and args.val_json != image_default_val_json:
                        worker_ns.val_json = args.val_json
                    else:
                        vj = VAL_JSONS.get(mod)
                        if vj:
                            worker_ns.val_json = vj
                    # By default use the per-modality dataset_root from MODALITY_DATASETS.
                    # Only override with the top-level --dataset_root if the user explicitly
                    # provided a different path (the argparse default is ImageNet which
                    # should not silently override other modalities).
                    try:
                        worker_ns.dataset_root = MODALITY_DATASETS[mod_enum]["dataset_root"]
                    except Exception:
                        # fallback to args.dataset_root if mapping missing
                        worker_ns.dataset_root = getattr(args, 'dataset_root', None)
                    # Only treat a top-level --dataset_root as an explicit override if it
                    # differs from the image parser default. This prevents the parser's
                    # image default from being applied to other modalities.
                    image_default_dataset_root = "/data/datasets/ImageNet-1K"
                    if getattr(args, 'dataset_root', None) and args.dataset_root != image_default_dataset_root:
                        worker_ns.dataset_root = args.dataset_root

                    all_worker_ns.append(worker_ns)

                # (previously logged planned tasks here per-modality which caused repeated
                # output when building the full grid). We now only record tasks after the
                # whole grid is built (see later) to avoid duplicate logging.

        # After building the full list of worker namespaces, log planned tasks once
        # so users see a single summary instead of repeated incremental summaries.
        if all_worker_ns:
            logger.info(f"[transfer_attack] Planned {len(all_worker_ns)} task(s) for --run_all:")
            for ti, tns in enumerate(all_worker_ns):
                try:
                    logger.info(
                        f"[task {ti}] output={getattr(tns, 'output', '')} src={getattr(tns, 'src_model', None)}"
                        f" mod={getattr(tns, 'modality', None)} eps={getattr(tns, 'eps', None)} run_max_samples={getattr(tns, 'run_max_samples', None)}"
                    )
                except Exception:
                    logger.info(f"[task {ti}] (failed to format task metadata) output={getattr(tns, 'output', '')}")

        # If no GPU found, fall back to sequential runs
        num_gpus = max(1, torch.cuda.device_count())
        run_records = []
        if num_gpus <= 1 or len(all_worker_ns) == 0:
            logger.info("No multiple GPUs detected or no tasks: running combos sequentially")
            for worker_ns in all_worker_ns:
                logger.info(f"Forking combo: src={worker_ns.src_model} eps={int(worker_ns.eps)} mod={worker_ns.modality}")
                try:
                    # Run worker as a subprocess to avoid calling main() recursively
                    spawn_namespace_subprocess(worker_ns)
                except SystemExit:
                    pass
                except Exception:
                    tb = traceback.format_exc()
                    logger.exception(f"Exception in combo {worker_ns.modality}:\n{tb}")
                run_records.append(worker_ns.output)
                try:
                    logger.info(f"[transfer_attack] Task done: output={worker_ns.output} src={getattr(worker_ns,'src_model','')} mod={getattr(worker_ns,'modality','')} eps={getattr(worker_ns,'eps','')}")
                except Exception:
                    logger.info(f"[transfer_attack] Task done: output={worker_ns.output}")
        else:
            # Put all tasks into a multiprocessing queue and let worker processes pick tasks from it.
            # This mirrors the concurrency approach used by tools/ablate_logsumexp.py (spawned
            # processes) which isolates HF/accelerate/device placement and avoids meta-tensor
            # `.cuda()` copy errors that can happen inside threads.
            ctx = get_context("spawn")
            gpu_indices_all = list(range(torch.cuda.device_count()))
            task_queue = ctx.Queue()
            # Queue for workers to send completion notifications back to the parent
            completion_queue = ctx.Queue()
            for ns in all_worker_ns:
                task_queue.put(ns)
            # Put sentinel None for each worker so they terminate when done
            for _ in gpu_indices_all:
                task_queue.put(None)

            workers = []
            for gpu_idx in gpu_indices_all:
                p = ctx.Process(target=_process_worker_from_queue, args=(gpu_idx, task_queue, completion_queue))
                p.start()
                workers.append(p)
                logger.info(f"Launched worker process for gpu{gpu_idx} picking from global queue")
            # Show a progress bar that updates as worker processes report task completion.
            # We'll wait on the completion_queue for exactly len(all_worker_ns) messages.
            done = 0
            total_tasks = len(all_worker_ns)
            from tqdm import tqdm as _tqdm_parent
            pbar = _tqdm_parent(total=total_tasks, desc="Tasks", leave=True)
            try:
                while done < total_tasks:
                    try:
                        msg = completion_queue.get()
                    except Exception:
                        # unexpected queue failure; break and join processes below
                        break
                    done += 1
                    pbar.update(1)
                    try:
                        # msg expected to be a dict with keys: output, src, modality, eps, gpu, status
                        logger.info(f"[transfer_attack] Task done: output={msg.get('output','')} src={msg.get('src','')} mod={msg.get('modality','')} eps={msg.get('eps','')} gpu={msg.get('gpu', '')} status={msg.get('status','')}")
                    except Exception:
                        logger.info(f"[transfer_attack] Task done (unable to format msg): {msg}")
            finally:
                pbar.close()

            # Wait for all processes to finish cleanly
            for p in workers:
                p.join()

            run_records = [ns.output for ns in all_worker_ns]

        # After all runs complete, merge per-run results into a final robust.csv under base_out.
        merge_run_records(base_out, run_records, logger)

        sys.exit(0)

    # Single-run path: initialize per-run output and logger
    out_dir, results_predictions_path, results_accuracy_path, adv_dir, logger = prepare_output_and_logger(args)
    logger.info(f"[transfer_attack] main start, args.modality={getattr(args,'modality',None)}, src_model={getattr(args,'src_model',None)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    gpu_indices_all = list(range(torch.cuda.device_count()))
    logger.info(f"[transfer_attack] detected gpu_indices_all={gpu_indices_all}")
    if len(gpu_indices_all) == 0:
        raise RuntimeError("No CUDA devices available.")

    # Prepare output directories and initialize a run-local logger (console + run.log)

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
    logger.info(f"[transfer_attack] modality_map keys: {list(modality_map.keys())}")
    if modality_key not in modality_map:
        raise ValueError(f"Unsupported modality: {args.modality}. Supported: {list(modality_map.keys())}")
    modality = modality_map[modality_key]
    cfg = MODALITY_DATASETS[modality]
    logger.info(f"[transfer_attack] selected cfg for modality {modality.value}: {cfg}")

    # Support running multiple modalities in sequence when user passes --modality all or comma-separated list
    if modality_key == 'all' or ',' in args.modality:
        if modality_key == 'all':
            run_list = ['image', 'audio', 'thermal']
        else:
            run_list = [s.strip() for s in args.modality.split(',') if s.strip()]

        for m in run_list:
            mod_out = os.path.join(out_dir, m)
            os.makedirs(mod_out, exist_ok=True)
            # We'll fork a worker process and pass arguments via a Namespace below.
            logger.info(f"Forking worker for modality {m} -> {mod_out}")
            # Previously we launched a subprocess; we'll use a thread to run the modality subprocess
            worker_ns = argparse.Namespace(**vars(args))
            worker_ns.modality = m
            worker_ns.output = mod_out
            # ensure the child does not re-enter run_all orchestration
            worker_ns.run_all = False
            # pass through overrides, but avoid using the parser's image defaults
            # as implicit overrides for other modalities.
            image_default_val_json = VAL_JSONS.get('image')
            image_default_dataset_root = "/data/datasets/ImageNet-1K"
            image_default_centre = "./centre_embs/image_in_center_embeddings.pkl"
            # val_json: use only if explicitly provided and not the image default
            if getattr(args, 'val_json', None) and args.val_json != image_default_val_json:
                worker_ns.val_json = args.val_json
            else:
                # fall back to modality-specific suggestion if available
                vj = VAL_JSONS.get(m)
                if vj:
                    worker_ns.val_json = vj
            # dataset_root: override only when user provided a non-image root
            if getattr(args, 'dataset_root', None) and args.dataset_root != image_default_dataset_root:
                worker_ns.dataset_root = args.dataset_root
            else:
                # leave worker_ns.dataset_root as-is; the child will resolve from MODALITY_DATASETS
                pass
            # Do not forward the top-level centre_embeddings to child workers;
            # children should resolve the correct per-modality centre path.
            worker_ns.centre_embeddings = None
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

            # Run the modality worker in a dedicated thread (spawn a subprocess inside it)
            t = threading.Thread(target=spawn_namespace_subprocess, args=(worker_ns,))
            t.start()
            t.join()
        sys.exit(0)

    # Override with explicit json/root if provided. Avoid using the parser-level
    # image defaults as implicit overrides for other modalities: only accept
    # args.* when the user provided a non-image value.
    image_default_val_json = VAL_JSONS.get('image')
    image_default_dataset_root = "/data/datasets/ImageNet-1K"
    image_default_centre = "./centre_embs/image_in_center_embeddings.pkl"

    if getattr(args, 'val_json', None) and args.val_json != image_default_val_json:
        val_json = args.val_json
    else:
        val_json = cfg.get("val_json")

    if getattr(args, 'dataset_root', None) and args.dataset_root != image_default_dataset_root:
        dataset_root = args.dataset_root
    else:
        dataset_root = cfg.get("dataset_root")

    if getattr(args, 'centre_embeddings', None) and args.centre_embeddings != image_default_centre:
        centre_path = args.centre_embeddings
    else:
        centre_path = cfg.get("centre_embeddings_path")

    # If batch_size not provided, use per-dataset default from CLEAN_VAL_BATCH_SIZE_MAP
    ds_name = cfg.get('dataset_name')
    if args.batch_size is None:
        # Prefer attack-specific batch size when running adversarial generation / transfer attacks
        args.batch_size = ATTACK_VAL_BATCH_SIZE_MAP.get(ds_name, CLEAN_VAL_BATCH_SIZE_MAP.get(ds_name, DEFAULT_BATCH_SIZE_FALLBACK))
        logger.info(f"Using default batch_size={args.batch_size} for dataset {ds_name} (attack batch size)")

    logger.info(f"Dataset: {cfg['dataset_name']} val_json={val_json} centres={centre_path}")

    # If multiple GPUs available, spawn one worker per GPU and aggregate results
    if len(gpu_indices_all) > 1:
        # Use threads and a thread-safe queue for per-GPU attack workers
        result_queue = queue.Queue()
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
            t = threading.Thread(target=run_attack_worker_thread, args=(gi, wi, len(gpu_indices_all), worker_args, result_queue, out_dir))
            t.start()
            workers.append(t)
            logger.info(f"Launched attack worker thread on gpu{gi} (per-worker run_max_samples={per_worker_max})")

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

        for t in workers:
            t.join()
        # Aggregate per-worker CSVs (if any) into a single sorted CSV
        try:
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
                    def percent(a, b):
                        return 0.0 if b == 0 else float(a) / float(b) * 100.0
                    with open(acc_path, "w", newline="") as af:
                        writer = csv.writer(af)
                        writer.writerow(["run_name", "src_model", "target_model", "modality", "dataset", "eps", "run_max_samples", "batch", "total", "correct_src_clean", "correct_src_adv", "correct_target_adv", "src_clean_acc_percent", "src_adv_acc_percent", "target_adv_acc_percent"]) 
                        run_name = os.path.basename(out_dir)
                        writer.writerow([run_name, getattr(args, 'src_model', ''), getattr(args, 'robust_lora_weights', ''), getattr(args, 'modality', ''), ds_name or '', getattr(args, 'eps', ''), getattr(args, 'run_max_samples', ''), getattr(args, 'batch_size', ''), total, correct_src_clean, correct_src_adv, correct_target_adv, f"{percent(correct_src_clean, total):.2f}", f"{percent(correct_src_adv, total):.2f}", f"{percent(correct_target_adv, total):.2f}"])
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
        logger.info(f"[transfer_attack] loading centre embeddings from {centre_path} on device {device}")
        centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
        logger.info(f"[transfer_attack] loaded centres: {len(centre_labels)} labels, lbl_to_idx size={len(lbl_to_idx)}")

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


def _worker_run_combos(gpu_index, combo_ns_list):
    """Run assigned combos on a single physical GPU, reusing loaded models when possible.

    This function is thread-compatible: it sets the CUDA device for the current thread
    to the provided physical GPU index and runs combos. It does NOT modify environment
    variables like CUDA_VISIBLE_DEVICES (which are process-global).
    """
    from collections import defaultdict
    # Use the actual GPU index inside this process/thread
    device = torch.device(f'cuda:{gpu_index}')
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
            # Fallback to a temporary per-gpu logger if we haven't created the group logger yet
            try:
                tmp_log_path = os.path.join('.', f"worker_gpu{gpu_index}.log")
                tmp_logger = setup_logger(tmp_log_path, stream_to_console=True)
                tmp_logger.warning(f"[gpu{gpu_index}] Unknown modality: {modality_str}; skipping group")
            except Exception:
                print(f"[gpu{gpu_index}] Unknown modality: {modality_str}; skipping group", file=sys.stderr)
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
        logger = setup_logger(log_path, stream_to_console=True)
        logger.info(f"[gpu{gpu_index}] Starting group: src={src_model_name} mod={modality_str} unibind={unibind_w} robust={robust_w}")

        # Load centre embeddings on this device
        try:
            centre_emb, centre_labels, lbl_to_idx, idx_to_lbl = load_label_mapping(centre_path, device)
        except Exception as e:
            logger.error(f"[gpu{gpu_index}] Failed to load centre embeddings {centre_path}: {e}")
            continue

        # Precompute normalization tensors for this modality (re-used across combos)
        mean, std = get_normalization_tensors(mod_enum, device)

        # Run each combo in this group. For isolation and to avoid sharing model state
        # across combos, build fresh src and target model instances for every combo.
        for ns in ns_list:
            try:
                logger.info(f"[gpu{gpu_index}] Running combo: src={ns.src_model} eps={ns.eps} mod={ns.modality}")
                try:
                    # Include resolved paths (centre/val/dataset) in the logged settings
                    # so users can see both the requested args *and* the final
                    # resolved locations used by the worker.
                    ns_for_log = argparse.Namespace(**vars(ns))
                    setattr(ns_for_log, 'resolved_centre_embeddings', centre_path)
                    setattr(ns_for_log, 'resolved_val_json', val_json)
                    setattr(ns_for_log, 'resolved_dataset_root', dataset_root)
                    logger.info(f"[gpu{gpu_index}] Task settings: {_format_ns_summary(ns_for_log)}")
                except Exception:
                    logger.info(f"[gpu{gpu_index}] Task settings: (failed to format namespace) output={getattr(ns,'output','')}")
                loader = _make_loader_with_uniform_sampling(mod_enum, dataset_root, val_json, lbl_to_idx, batch_size=ns.batch_size, num_workers=0, run_max_samples=getattr(ns, 'run_max_samples', None))
                results_csv_path = os.path.join(ns.output, "results_predictions.csv")
                adv_dir = os.path.join(ns.output, "adv_samples")

                # Build fresh source model for this combo
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
                        src_w = getattr(ns, 'unibind_weights', unibind_w)
                        src_models = build_models_on_devices([gpu_index], centre_emb, centre_labels, lbl_to_idx, mod_enum, src_w, src_unibind_kwargs, logger)
                        src_model_obj = src_models.get(gpu_index)
                        if src_model_obj is None:
                            logger.error(f"[gpu{gpu_index}] build_models_on_devices returned no model for gpu {gpu_index}: keys={list(src_models.keys())}")
                            # skip this combo
                            continue
                    elif src_model_name == 'clip':
                        src_model_obj = CLIPClassifier(device, mod_enum, centre_labels, logger=logger, label_to_index=lbl_to_idx).to(device)
                        src_model_obj.eval()
                    elif src_model_name == 'imagebind':
                        src_model_obj = ImageBindClassifier(device, mod_enum, centre_labels, logger=logger, label_to_index=lbl_to_idx).to(device)
                        src_model_obj.eval()
                    else:
                        logger.error(f"[gpu{gpu_index}] Unknown src_model: {src_model_name}; skipping combo")
                        continue
                except Exception:
                    tb = traceback.format_exc()
                    logger.error(f"[gpu{gpu_index}] Failed to build source model for combo {ns.output}:\n{tb}")
                    continue

                # Build fresh target model for this combo (RobustBind^4 LoRA applied)
                target_model_obj = None
                try:
                    target_kwargs = argparse.Namespace(
                        use_flash_attention=True,
                        use_lora=True,
                        lora_rank=4,
                        lora_alpha=8,
                        use_modality_head_mlp=False,
                        lora_weights=getattr(ns, 'robust_lora_weights', robust_w),
                        modality_head_mlp_weights=None,
                    )
                    target_unibind_kwargs = get_unibind_kwargs(target_kwargs)
                    tgt_unibind_w = getattr(ns, 'unibind_weights', unibind_w)
                    target_models = build_models_on_devices([gpu_index], centre_emb, centre_labels, lbl_to_idx, mod_enum, tgt_unibind_w, target_unibind_kwargs, logger)
                    target_model_obj = target_models.get(gpu_index)
                    if target_model_obj is None:
                        logger.error(f"[gpu{gpu_index}] build_models_on_devices returned no target model for gpu {gpu_index}: keys={list(target_models.keys())}")
                        # cleanup src and skip
                        try:
                            del src_model_obj
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                        continue
                except Exception:
                    tb = traceback.format_exc()
                    logger.error(f"[gpu{gpu_index}] Failed to build target model for combo {ns.output}:\n{tb}")
                    try:
                        logger.error(f"[gpu{gpu_index}] Debug: device={device}, cuda_available={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()}")
                        logger.error(f"[gpu{gpu_index}] Debug: mod_enum={mod_enum}, unibind_weights={tgt_unibind_w}, robust_lora_weights={getattr(ns, 'robust_lora_weights', robust_w)}")
                        logger.error(f"[gpu{gpu_index}] Debug: centre_path={centre_path}, val_json={val_json}, dataset_root={dataset_root}")
                    except Exception:
                        logger.exception(f"[gpu{gpu_index}] Failed while emitting debug info for target build")
                    try:
                        del src_model_obj
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    continue

                # Run the attack loop for this combo
                start_ts = datetime.now()
                try:
                    res = _run_attack_loop(logger, src_model_obj, target_model_obj, loader, mean, std, device, ns, results_csv_path, adv_dir, label_names=centre_labels)
                    exit_code = 0
                except Exception:
                    tb = traceback.format_exc()
                    logger.error(f"[gpu{gpu_index}] Exception while running combo {ns.output}:\n{tb}")
                    res = {'total': 0, 'correct_src_clean': 0, 'correct_target_clean': 0, 'correct_src_adv': 0, 'correct_target_adv': 0}
                    exit_code = 1
                end_ts = datetime.now()
                elapsed = (end_ts - start_ts).total_seconds()

                logger.info(f"[gpu{gpu_index}] Combo finished: output={ns.output} samples={res.get('total',0)} elapsed_s={elapsed:.2f} exit_code={exit_code}")

                # Cleanup models and free GPU memory before next combo
                try:
                    del src_model_obj
                    del target_model_obj
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            except Exception:
                tb = traceback.format_exc()
                logger.error(f"[gpu{gpu_index}] Unexpected exception running combo {ns.output}:\n{tb}")


def merge_run_records(base_out, run_records, logger):
    """Merge per-run results (robust.csv or results_gpu*.csv) into a final robust.csv under base_out.

    This mirrors the aggregation logic used in main and is exposed so tests can call it.
    """
    final_rows = []
    final_header = ["src_model", "target_model", "modality", "dataset", "eps", "run_max_samples", "batch", "total", "correct_src_clean", "correct_src_adv", "correct_target_adv", "src_clean_acc_percent", "src_adv_acc_percent", "target_adv_acc_percent"]
    for rdir in run_records:
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
                m = {k: v for k, v in zip(hdr, vals)}
                final_rows.append([m.get('src_model',''), m.get('target_model',''), m.get('modality',''), m.get('dataset',''), m.get('eps',''), m.get('run_max_samples',''), m.get('batch',''), m.get('total',''), m.get('correct_src_clean',''), m.get('correct_src_adv',''), m.get('correct_target_adv',''), m.get('src_clean_acc_percent',''), m.get('src_adv_acc_percent',''), m.get('target_adv_acc_percent','')])
            except Exception:
                continue
        else:
            parts = glob.glob(os.path.join(rdir, 'results_gpu*.csv'))
            if parts:
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
                bn = os.path.basename(rdir)
                parts_bn = bn.split('_')
                src = parts_bn[0].split('-')[1] if len(parts_bn) > 0 and '-' in parts_bn[0] else ''
                eps_s = parts_bn[1].replace('eps', '') if len(parts_bn) > 1 else ''
                mod = parts_bn[2].split('-')[1] if len(parts_bn) > 2 and '-' in parts_bn[2] else ''
                final_rows.append([src, '', mod, '', eps_s, '', '', str(total_v), str(correct_src_clean), str(correct_src_adv), str(correct_target_adv), '', '', ''])

    final_rows.sort(key=lambda r: (r[0] or '', r[1] or '', r[2] or '', r[3] or ''))
    final_csv = os.path.join(base_out, 'robust.csv')
    if final_rows:
        with open(final_csv, 'w', newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(final_header)
            for row in final_rows:
                writer.writerow(row)
        logger.info(f"Final accuracy CSV written to: {final_csv}")
    else:
        logger.info("No per-run robust.csv files found; no final accuracy CSV produced.")


def _process_worker_from_queue(gpu_idx, q, completion_q=None):
    """Process target: pick tasks from a multiprocessing.Queue and run combos on the given GPU.

    This mirrors the thread-based worker but runs in a separate process (spawned via
    multiprocessing.get_context('spawn')). Using processes isolates library state
    (HF accelerate, meta tensor dispatch) which avoids meta-tensor `.cuda()` issues.
    """
    try:
        device = torch.device(f'cuda:{gpu_idx}')
        try:
            torch.cuda.set_device(device)
        except Exception:
            # set_device failure shouldn't prevent the process from attempting the work;
            # model builders also set their device where needed.
            pass
    except Exception:
        # If torch isn't available or no CUDA, proceed and let inner code handle failures
        device = None

    while True:
        try:
            ns = q.get()
        except Exception:
            break
        if ns is None:
            break

        status = 'done'
        try:
            _worker_run_combos(gpu_idx, [ns])
        except Exception:
            status = 'error'
            tb = traceback.format_exc()
            try:
                log_path = os.path.join(getattr(ns, 'output', '.'), f"worker_gpu{gpu_idx}.log")
                lg = setup_logger(log_path, stream_to_console=True)
                lg.error(f"Exception processing task {getattr(ns, 'output', '')}:\n{tb}")
            except Exception:
                print(tb, file=sys.stderr)
        finally:
            # Notify parent that this task is complete (success or failure)
            try:
                if completion_q is not None:
                    completion_q.put({
                        'output': getattr(ns, 'output', ''),
                        'src': getattr(ns, 'src_model', ''),
                        'modality': getattr(ns, 'modality', ''),
                        'eps': getattr(ns, 'eps', ''),
                        'gpu': gpu_idx,
                        'status': status,
                    })
            except Exception:
                # best-effort: don't crash worker if parent queue fails
                pass


def run_attack_worker_thread(gpu_index, worker_index, num_workers, worker_args, result_queue, out_dir):
    """Wrapper for running _attack_worker inside a thread.

    This sets the correct CUDA device for the thread and then calls the existing
    _attack_worker function (which performs the per-GPU attack work). If
    _attack_worker is missing or raises, we capture and log the traceback.
    """
    try:
        device = torch.device(f'cuda:{gpu_index}')
        torch.cuda.set_device(device)
    except Exception:
        # If device setup fails, still attempt to run the worker and let it handle device selection
        pass
    try:
        # Call the worker function if it exists in this module
        if '_attack_worker' in globals():
            return globals()['_attack_worker'](gpu_index, worker_index, num_workers, worker_args, result_queue, out_dir)
        else:
            raise RuntimeError("_attack_worker not found in module scope")
    except Exception:
        tb = traceback.format_exc()
        # Try to log to out_dir/run.log if possible
        try:
            log_path = os.path.join(out_dir, f"worker_thread_gpu{gpu_index}.log")
            logg = setup_logger(log_path, stream_to_console=True)
            logg.error(f"Exception in run_attack_worker_thread:\n{tb}")
        except Exception:
            print(tb, file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        raise
