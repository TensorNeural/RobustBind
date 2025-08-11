#!/usr/bin/env python3
"""Orchestrates alignment and robust training runs (Python replacement for previous shell script)."""
from __future__ import annotations
import argparse
import subprocess
import shlex
import logging
import os
import sys
from enum import Enum
from typing import List, Tuple, Optional

MODEL_TYPE_TO_MODALITIES = {
    "vision": [
        "image", 
        "video", 
        "event"
    ],
    "audio": ["audio"],
    "thermal": ["thermal"],
    "point": ["point"],
}

class RobustMode(str, Enum):
    LORA = "lora"
    FULL_FINE_TUNE = "full_fine_tune"

ROBUST_EPSILONS = [
    2,
]

ROBUST_LORA_RANKS = [
    2,
    4,
    8,
]

ROBUST_LORA_ALPHAS = [
    4,
    8,
    16,
]

ROBUST_EPOCHS = 2

ROBUST_TRAIN_MODALITY_TO_DATASET = {
    # "image": "ImageNet-1K",
    # "image": "Places365",
    "video": "Kinetics-400",
    # "video": "UCF-101",
    # "video": "MSR-VTT",
    # "event": "N-Caltech-101",
    # "event": "N-ImageNet-1K",
    "audio": "FSD-50K",
    # "audio": "ESC-50",
    # "audio": "UrbanSound8K",
    "thermal": "LLVIP",
    # "thermal": "RGB-T",
    "point": "ModelNet40",
    # "point": "ShapeNet",
}

ROBUST_VAL_MODALITY_TO_DATASET = {
    # "image": "ImageNet-1K",
    # "image": "Places365",
    # "video": "UCF-101",
    "video": "MSR-VTT",
    # "event": "N-Caltech-101",
    # "event": "N-ImageNet-1K",
    "audio": "ESC-50",
    # "audio": "UrbanSound8K",
    "thermal": "LLVIP",
    # "thermal": "RGB-T",
    "point": "ModelNet40",
    # "point": "ShapeNet",
}

ROBUST_DATASET_TO_BATCH_SIZE = {
    "ImageNet-1K": 1,
    "Places365": 70,
    "Kinetics-400": 25,
    "UCF-101": 6,
    "MSR-VTT": 6,
    "N-Caltech-101": 70,
    "N-ImageNet-1K": 70,
    "FSD-50K": 90,
    "ESC-50": 90,
    "UrbanSound8K": 2,
    "LLVIP": 280,
    "RGB-T": 16,
    "ModelNet40": 64,
    "ShapeNet": 64,
}

ROBUST_TRAIN_MAX_SAMPLES_MAP = {
    "ImageNet-1K": 18,
    "Places365": 0,
    "Kinetics-400": 241258,
    "UCF-101": 9537,
    "MSR-VTT": 2990,
    "N-Caltech-101": 3060,
    "N-ImageNet-1K": 1281167,
    "FSD-50K": 36796,
    "ESC-50": 1600,
    "UrbanSound8K": 7079,
    "LLVIP": 67900,
    "RGB-T": 800,
    "ModelNet40": 9843,
    "ShapeNet": 20480,
}

ROBUST_VAL_MAX_SAMPLES_MAP = {
    "ImageNet-1K": 6,
    "Places365": 3000,
    "MSR-VTT": 2990,
    "N-Caltech-101": 3000,
    "N-ImageNet-1K": 3000,
    "ESC-50": 400,
    "UrbanSound8K": 1653,
    "LLVIP": 16974,
    "RGB-T": 500,
    "ModelNet40": 2468,
    "ShapeNet": 2048,
}

ALIGN_EPOCHS = 1
ALIGN_TRAIN_MODALITY_TO_DATASET = {
    "image": "ImageNet-1K",
    "video": "MSR-VTT",
    "audio": "ESC-50",
    "thermal": "LLVIP",
    "event": "N-Caltech-101",
}
ALIGN_VAL_MODALITY_TO_DATASET = ALIGN_TRAIN_MODALITY_TO_DATASET.copy()

ALIGN_DATASET_TO_BATCH_SIZE = {
    "ImageNet-1K": 2000,
    "MSR-VTT": 200,
    "ESC-50": 500,
    "LLVIP": 2000,
    "N-Caltech-101": 500,
}
ALIGN_TRAIN_MAX_SAMPLES_MAP = {
    "ImageNet-1K": 4,
    "MSR-VTT": 4,
    "ESC-50": 4,
    "LLVIP": 4,
    "N-Caltech-101": 4,
}
ALIGN_VAL_MAX_SAMPLES_MAP = ALIGN_TRAIN_MAX_SAMPLES_MAP.copy()

ROBUST_TRAIN_JSON_MAP = {k: f"./datasets/{k}/train_data.json" for k in ROBUST_TRAIN_MAX_SAMPLES_MAP}
ROBUST_VAL_JSON_MAP = {
    "ImageNet-1K": "./datasets/ImageNet-1K/val_data_3000.json",
    **{k: f"./datasets/{k}/val_data.json" for k in ROBUST_VAL_MAX_SAMPLES_MAP if k != "ImageNet-1K"}
}
ALIGN_TRAIN_JSON_MAP = {
    "ImageNet-1K": "./datasets/ImageNet-1K/train_data_align.json",
    "MSR-VTT": "./datasets/MSR-VTT/train_data_align.json",
    "ESC-50": "./datasets/ESC-50/train_data_align.json",
    "LLVIP": "./datasets/LLVIP/train_data_align.json",
    "N-Caltech-101": "./datasets/N-Caltech-101/train_data_align.json",
}
ALIGN_VAL_JSON_MAP = {
    "ImageNet-1K": "./datasets/ImageNet-1K/val_data.json",
    "MSR-VTT": "./datasets/MSR-VTT/val_data.json",
    "ESC-50": "./datasets/ESC-50/val_data.json",
    "LLVIP": "./datasets/LLVIP/val_data.json",
    "N-Caltech-101": "./datasets/N-Caltech-101/val_data.json",
}

EMB_SUFFIX_MAP = {
    "ImageNet-1K": "in",
    "Places365": "p365",
    "UCF-101": "ucf",
    "MSR-VTT": "msrvtt",
    "N-Caltech-101": "caltech",
    "N-ImageNet-1K": "nin",
    "ESC-50": "esc",
    "UrbanSound8K": "us",
    "LLVIP": "llvip",
    "RGB-T": "rgbt",
    "ModelNet40": "modelnet40",
    "ShapeNet": "shapenet",
}

def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() or 1
    except Exception:
        return int(os.environ.get("WORLD_SIZE", "1"))


def build_centre_emb_path(modality: str, dataset: str) -> str:
    suffix = EMB_SUFFIX_MAP.get(dataset) or dataset.lower().replace("-", "").replace("_", "")[:8]
    return f"./centre_embs/{modality}_{suffix}_center_embeddings.pkl"


def run_command(cmd: List[str], logger: logging.Logger, dry_run: bool) -> int:
    logger.info(f"[JOB] {' '.join(shlex.quote(c) for c in cmd)}")
    if dry_run:
        return 0
    
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        logger.error(f"[FAIL] Return code {proc.returncode}")
    return proc.returncode


def torchrun_prefix(nproc: int) -> List[str]:
    return ["torchrun", f"--nproc_per_node={nproc}", "train_robust_unibind.py"]


def build_alignment_cmd(nproc: int, args, model_type: str, modality: str, train_dataset: str, val_dataset: str,
                        train_json: str, val_json: str, emb_path: str, train_bs: int, val_bs: int,
                        train_max: int, val_max: int, epochs: int) -> List[str]:
    cmd = torchrun_prefix(nproc) + [
        "--training_mode", "alignment",
        "--model_type", model_type,
        "--train_modality", modality,
        "--val_modality", modality,
        "--train_dataset_name", train_dataset,
        "--val_dataset_name", val_dataset,
        "--train_dataset_root", f"{args.dataset_root}/{train_dataset}",
        "--val_dataset_root", f"{args.dataset_root}/{val_dataset}",
        "--train_json", train_json,
        "--val_json", val_json,
        "--pretrain_weights", args.pretrain_weights,
        "--center_emb", emb_path,
        "--train_batch_size", str(train_bs),
        "--val_batch_size", str(val_bs),
        "--num_workers", str(args.num_workers),
        "--train_max_samples", str(train_max),
        "--val_max_samples", str(val_max),
        "--epochs", str(epochs),
        "--tensorboard_data_dir", args.tensorboard_dir,
        "--output_dir", args.output_dir,
    ]
    if args.use_flash_attention:
        cmd.append("--use_flash_attention")

    return cmd


def build_robust_base_cmd(nproc: int, args, model_type: str, modality: str, train_dataset: str, val_dataset: str,
                          train_json: str, val_json: str, emb_path: str, train_bs: int, val_bs: int,
                          train_max: int, val_max: int, epochs: int, eps: int, mode: RobustMode) -> List[str]:
    cmd = torchrun_prefix(nproc) + [
        "--training_mode", "robust",
        "--model_type", model_type,
        "--train_modality", modality,
        "--val_modality", modality,
        "--train_dataset_name", train_dataset,
        "--val_dataset_name", val_dataset,
        "--train_dataset_root", f"{args.dataset_root}/{train_dataset}",
        "--val_dataset_root", f"{args.dataset_root}/{val_dataset}",
        "--train_json", train_json,
        "--val_json", val_json,
        "--pretrain_weights", args.pretrain_weights,
        "--center_emb", emb_path,
        "--train_batch_size", str(train_bs),
        "--val_batch_size", str(val_bs),
        "--num_workers", str(args.num_workers),
        "--train_max_samples", str(train_max),
        "--val_max_samples", str(val_max),
        "--epochs", str(epochs),
        "--robust_epsilon", str(eps),
        "--robust_training_mode", mode.value,
        "--tensorboard_data_dir", args.tensorboard_dir,
        "--output_dir", args.output_dir,
    ]
    if args.use_flash_attention:
        cmd.append("--use_flash_attention")

    return cmd


def build_full_fine_tune_jobs_for_modality(nproc: int, args, model_type: str, modality: str,
                                           train_dataset: str, val_dataset: str, train_json: str, val_json: str,
                                           emb_path: str, train_bs: int, val_bs: int, train_max: int, val_max: int) -> List[List[str]]:
    jobs: List[List[str]] = []
    for eps in args.robust_epsilons:
        jobs.append(
            build_robust_base_cmd(
                nproc, args, model_type, modality, train_dataset, val_dataset,
                train_json, val_json, emb_path, train_bs, val_bs, train_max, val_max,
                args.robust_epochs, eps, RobustMode.FULL_FINE_TUNE
            )
        )
    return jobs


def build_lora_jobs_for_modality(nproc: int, args, model_type: str, modality: str,
                                 train_dataset: str, val_dataset: str, train_json: str, val_json: str,
                                 emb_path: str, train_bs: int, val_bs: int, train_max: int, val_max: int) -> List[List[str]]:
    jobs: List[List[str]] = []
    for eps in args.robust_epsilons:
        for lora_rank in args.robust_lora_ranks:
            for lora_alpha in args.robust_lora_alphas:
                cmd = build_robust_base_cmd(
                    nproc, args, model_type, modality, train_dataset, val_dataset,
                    train_json, val_json, emb_path, train_bs, val_bs, train_max, val_max,
                    args.robust_epochs, eps, RobustMode.LORA
                )
                cmd.extend([
                    "--robust_lora_rank", str(lora_rank),
                    "--robust_lora_alpha", str(lora_alpha),
                    "--robust_use_modality_head_mlp",
                ])
                jobs.append(cmd)
    return jobs


def fetch_alignment_dataset_config(modality: str) -> Optional[Tuple[str, str]]:
    train_dataset = ALIGN_TRAIN_MODALITY_TO_DATASET.get(modality)
    val_dataset = ALIGN_VAL_MODALITY_TO_DATASET.get(modality)
    if not train_dataset or not val_dataset:
        return None
    
    return train_dataset, val_dataset


def fetch_robust_dataset_config(modality: str) -> Optional[Tuple[str, str]]:
    train_dataset = ROBUST_TRAIN_MODALITY_TO_DATASET.get(modality)
    val_dataset = ROBUST_VAL_MODALITY_TO_DATASET.get(modality)
    if not train_dataset or not val_dataset:
        return None
    
    return train_dataset, val_dataset


def build_alignment_jobs(args, logger) -> List[List[str]]:
    jobs: List[List[str]] = []
    nproc = gpu_count()
    for model_type, modalities in MODEL_TYPE_TO_MODALITIES.items():
        for modality in modalities:
            ds_pair = fetch_alignment_dataset_config(modality)
            if ds_pair is None:
                continue

            train_dataset, val_dataset = ds_pair
            train_bs = ALIGN_DATASET_TO_BATCH_SIZE.get(train_dataset)
            val_bs = ALIGN_DATASET_TO_BATCH_SIZE.get(val_dataset)
            train_max = ALIGN_TRAIN_MAX_SAMPLES_MAP.get(train_dataset)
            val_max = ALIGN_VAL_MAX_SAMPLES_MAP.get(val_dataset)
            train_json = ALIGN_TRAIN_JSON_MAP.get(train_dataset)
            val_json = ALIGN_VAL_JSON_MAP.get(val_dataset)
            if None in (train_bs, val_bs, train_max, val_max, train_json, val_json):
                logger.warning(f"[SKIP][ALIGN] modality={modality} dataset={train_dataset}")
                continue

            emb_path = build_centre_emb_path(modality, val_dataset)
            jobs.append(
                build_alignment_cmd(
                    nproc, args, model_type, modality, train_dataset, val_dataset,
                    train_json, val_json, emb_path, train_bs, val_bs, train_max, val_max, args.align_epochs
                )
            )
    return jobs


def build_robust_jobs(args, logger) -> List[List[str]]:
    jobs: List[List[str]] = []
    nproc = gpu_count()
    for model_type, modalities in MODEL_TYPE_TO_MODALITIES.items():
        for modality in modalities:
            ds_pair = fetch_robust_dataset_config(modality)
            if ds_pair is None:
                continue

            train_dataset, val_dataset = ds_pair
            train_bs = ROBUST_DATASET_TO_BATCH_SIZE.get(train_dataset)
            val_bs = ROBUST_DATASET_TO_BATCH_SIZE.get(val_dataset)
            train_max = ROBUST_TRAIN_MAX_SAMPLES_MAP.get(train_dataset)
            val_max = ROBUST_VAL_MAX_SAMPLES_MAP.get(val_dataset)
            train_json = ROBUST_TRAIN_JSON_MAP.get(train_dataset)
            val_json = ROBUST_VAL_JSON_MAP.get(val_dataset)
            if None in (train_bs, val_bs, train_max, val_max, train_json, val_json):
                logger.warning(f"[SKIP][ROBUST] modality={modality} dataset={train_dataset}")
                continue

            emb_path = build_centre_emb_path(modality, val_dataset)
            for mode in args.robust_modes:
                if mode is RobustMode.FULL_FINE_TUNE and not args.allow_full_fine_tune:
                    continue

                if mode is RobustMode.FULL_FINE_TUNE:
                    jobs.extend(
                        build_full_fine_tune_jobs_for_modality(
                            nproc, args, model_type, modality, train_dataset, val_dataset,
                            train_json, val_json, emb_path, train_bs, val_bs, train_max, val_max
                        )
                    )
                else:
                    jobs.extend(
                        build_lora_jobs_for_modality(
                            nproc, args, model_type, modality, train_dataset, val_dataset,
                            train_json, val_json, emb_path, train_bs, val_bs, train_max, val_max
                        )
                    )
    return jobs


def parse_args():
    p = argparse.ArgumentParser("Trainer")
    p.add_argument("--dataset_root", default="/home/user/datasets")
    p.add_argument("--output_dir", default="output")
    p.add_argument("--tensorboard_dir", default="tensorboard")
    p.add_argument("--pretrain_weights", default="./ckpts/pretrained_weights_flash_atten.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_flash_attention", action="store_true", default=True)
    p.add_argument("--do_alignment", action="store_true", default=False)
    p.add_argument("--align_epochs", type=int, default=ALIGN_EPOCHS)
    p.add_argument("--do_robust", action="store_true", default=False)
    p.add_argument("--robust_epochs", type=int, default=ROBUST_EPOCHS)
    p.add_argument("--robust_epsilons", type=int, nargs="*", default=ROBUST_EPSILONS)
    p.add_argument("--robust_lora_ranks", type=int, nargs="*", default=ROBUST_LORA_RANKS)
    p.add_argument("--robust_lora_alphas", type=int, nargs="*", default=ROBUST_LORA_ALPHAS)
    p.add_argument(
        "--robust_modes",
        nargs="*",
        type=RobustMode,
        choices=list(RobustMode),
        default=list(RobustMode),
        help="Robust training modes to run",
    )
    p.add_argument("--allow_full_fine_tune", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true", default=False)
    p.add_argument("--stop_on_error", action="store_true", default=False)
    p.add_argument("--max_jobs", type=int, default=None)
    return p.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("orchestrator")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    return logger


def main():
    args = parse_args()
    logger = setup_logger()
    if not args.do_alignment and not args.do_robust:
        logger.error("No work: enable --do_alignment and/or --do_robust")
        sys.exit(1)

    jobs: List[List[str]] = []
    if args.do_alignment:
        logger.info("Collecting alignment jobs")
        jobs.extend(build_alignment_jobs(args, logger))

    if args.do_robust:
        logger.info("Collecting robust jobs")
        jobs.extend(build_robust_jobs(args, logger))

    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    logger.info(f"Planned jobs: {len(jobs)} (dry_run={args.dry_run})")
    for idx, cmd in enumerate(jobs, 1):
        logger.info(f"[RUN {idx}/{len(jobs)}]")
        rc = run_command(cmd, logger, args.dry_run)
        if rc != 0 and args.stop_on_error:
            logger.error("Stopping due to failure")
            break

    logger.info("Done")


if __name__ == "__main__":
    main()
