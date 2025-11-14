#!/usr/bin/env python3
"""Shared dataset configuration: modality -> dataset mapping and batch size maps.

This centralizes duplicated constants used by multiple tools.
"""
from typing import Dict
from model import Modality

# Modalities -> dataset configuration
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


# Per-dataset temperature mapping (used by some evaluation scripts)
DATASET_TEMPERATURES: Dict[str, float] = {
    "ESC-50": 200.0,
    "N-Caltech-101": 100.0,
    "ImageNet-1K": 100.0,
    "LLVIP": 1000.0,
    "MSR-VTT": 50.0,
}


# Batch size map (clean val) per dataset
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


# Batch size map (attack val) per dataset — used for adversarial generation / attack runs
ATTACK_VAL_BATCH_SIZE_MAP: Dict[str, int] = {
    "ImageNet-1K": 70,
    "Places365": 70,
    "ModelNet40": 64,
    "ShapeNet": 64,
    "ESC-50": 50,
    "UrbanSound8K": 50,
    "LLVIP": 70,
    "RGB-T": 16,
    "MSR-VTT": 800,
    "UCF-101": 800,
    "N-Caltech-101": 70,
    "N-ImageNet-1K": 70,
}

__all__ = [
    "MODALITY_DATASETS",
    "DATASET_TEMPERATURES",
    "CLEAN_VAL_BATCH_SIZE_MAP",
    "ATTACK_VAL_BATCH_SIZE_MAP",
]
