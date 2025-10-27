# tsne_point_pipeline.py
import os
import argparse
import json
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime
from torch.utils.data import Subset, DataLoader
from multiprocessing import Pool, set_start_method
from functools import partial
import random
from tqdm import tqdm
from matplotlib import colors as mcolors


from model import UniBindClassifier, ForwardMode
from attack import AttackModel, APGDAttack, two_stage_attack
from data_util import JsonDataset, get_transform_fn, get_normalization_tensors, load_label_mapping
from shared_types import Modality

set_start_method("spawn", force=True)

try:
    from cuml.manifold import TSNE as CuMLTSNE
    import cupy as cp
    TSNE_BACKEND = "cuml"
except ImportError:
    from openTSNE import TSNE as OpenTSNE
    TSNE_BACKEND = "openTSNE"

# === Constants
NUM_SAMPLES = 40
PERPLEXITY = 5

class ModelType(str, Enum):
    UNIBIND = "UniBind"
    ROBUST2 = "RobustBind2"
    ROBUST4 = "RobustBind4"

class EpsLevel(str, Enum):
    CLEAN = "clean"
    EPS2 = "eps2"
    EPS4 = "eps4"

EPSILON_LEVELS = {
    EpsLevel.CLEAN: None,
    EpsLevel.EPS2: 2 / 255.0,
    EpsLevel.EPS4: 4 / 255.0
}

LORA_WEIGHTS_LIST_MAP = {
    Modality.IMAGE: ["./ckpts/vision_eps2_lora_weights.pt", "./ckpts/vision_eps4_lora_weights.pt"],
    Modality.AUDIO: ["./ckpts/audio_eps2_lora_weights.pt", "./ckpts/audio_eps4_lora_weights.pt"],
}

MODALITY_COLOR = {
    Modality.IMAGE: "blue",
    Modality.AUDIO: "red",
    Modality.POINT: "purple",
}

CLASS_NAMES_PER_MODALITY = {
    Modality.IMAGE: [
        "sports car, sport car", 
        # "convertible", "car wheel", "grille, radiator grille",
        # "bullet train, bullet", "limousine, limo", "racer, race car, racing car", "go-kart",
        # "model t", "pickup, pickup truck", "beach wagon, station wagon", "jeep, landrover",
        # "cab, hack, taxi, taxicab"
    ],
    Modality.AUDIO: ["car_horn"],
    Modality.POINT: ["car"]
}

MODALITIES = [Modality.IMAGE, Modality.AUDIO, Modality.POINT]

# === Helpers
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "tsne.log"))
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def fit_tsne(X, perplexity):
    if TSNE_BACKEND == "cuml":
        coords = CuMLTSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=400,
            n_iter=2500,
            random_state=42,
            early_exaggeration=2.0,
            init="random",
        ).fit_transform(cp.asarray(X))
        coords = cp.asnumpy(coords)
    else:
        tsne = OpenTSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=400,
            n_iter=2500,
            early_exaggeration=2.0,
            init="random",
            random_state=42,
            n_jobs=-1
        )
        coords = tsne.fit(X)
    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0))
    return coords

def build_model(device, pretrain_weights, modality, label_to_index, centre_embeddings, centre_labels, use_flash_attention, lora_weights=None):
    model = UniBindClassifier(
        device=device,
        pretrain_weights=pretrain_weights,
        modality=modality,
        centre_embeddings=centre_embeddings,
        centre_labels=centre_labels,
        label_to_index=label_to_index,
        use_flash_attention=use_flash_attention,
        use_modality_head_mlp=False,
        use_lora=(lora_weights is not None),
        lora_weights=lora_weights
    )
    return model.to(device)

def run_attack(model, x, eps, mean, std, device, logger):
    labels = torch.zeros(x.size(0), dtype=torch.long, device=device)
    attack_model = AttackModel(model, mean, std)
    stage1 = APGDAttack(model=attack_model, norm="linf", n_iter=10, n_restarts=1, eps=eps, loss_type="ce", device=device, logger=logger)
    stage2 = APGDAttack(model=attack_model, norm="linf", n_iter=10, n_restarts=1, eps=eps, loss_type="ce", device=device, logger=logger)
    return two_stage_attack(logger, model, x, labels, stage1, stage2, mean, std)

def extract_embeddings(model, x, device):
    with torch.no_grad():
        return model(x, ForwardMode.EMBEDDINGS).cpu().numpy()

def embeddings_already_saved(name):
    return os.path.exists(f"output/embeddings/{name}.npy")

def save_embeddings(name, embeddings):
    os.makedirs("output/embeddings", exist_ok=True)
    np.save(f"output/embeddings/{name}.npy", embeddings)

def get_class_samples(modality, dataset_name, val_json, train_json, dataset_root, device, label_to_index, logger):
    target_classes = CLASS_NAMES_PER_MODALITY[modality]
    samples_per_class = NUM_SAMPLES // len(target_classes)
    val_data = json.load(open(val_json)) if os.path.exists(val_json) else []
    train_data = json.load(open(train_json)) if os.path.exists(train_json) else []
    combined_data = val_data + train_data
    combined_matches = []

    for cls in tqdm(target_classes, desc=f"[{modality.name}] Sampling", ncols=80):
        val_cls = [i for i, e in enumerate(val_data) if cls in e["label"].lower()]
        train_cls = [i + len(val_data) for i, e in enumerate(train_data) if cls in e["label"].lower()]
        matches = val_cls + train_cls
        if len(matches) < samples_per_class:
            raise ValueError(f"Too few samples for class {cls}")
        combined_matches.extend(random.sample(matches, samples_per_class))

    tmp_json = f"./output/tmp_combined_{modality.name}.json"
    json.dump(combined_data, open(tmp_json, "w"))
    dataset = JsonDataset(os.path.join(dataset_root, dataset_name), tmp_json, get_transform_fn(modality), label_to_index)
    loader = DataLoader(Subset(dataset, combined_matches), batch_size=len(combined_matches), shuffle=False)
    return next(iter(loader))[0].to(device)

# === t-SNE Plotting
def plot_tsne_per_combo(name_prefix, save_dir):
    folder = "output/embeddings"
    all_embs, all_colors = [], []

    for modality in MODALITIES:
        fpath = os.path.join(folder, f"{name_prefix}_{modality.name}.npy")
        if os.path.exists(fpath):
            all_embs.append(np.load(fpath))
            all_colors += [MODALITY_COLOR[modality]] * len(np.load(fpath))

    if not all_embs:
        return
    X = np.concatenate(all_embs, axis=0)
    coords = fit_tsne(X, PERPLEXITY)
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=all_colors, s=60, alpha=0.6)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"tsne_{name_prefix}.png"), dpi=300)
    plt.close()

def generate_all_tsne_combo_charts(output_dir, only_clean, only_unibind):
    eps_levels = [EpsLevel.CLEAN] if only_clean else list(EpsLevel)
    model_types = [ModelType.UNIBIND] if only_unibind else list(ModelType)
    prefixes = [f"{m.name}_{e.name}" for m in model_types for e in eps_levels]
    plot_fn = partial(plot_tsne_per_combo, save_dir=output_dir)
    with Pool() as pool:
        pool.map(plot_fn, prefixes)

# === Main
def main(args):
    tsne_dir = os.path.join("output", "tsne_charts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    logger = setup_logger(tsne_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        Modality.IMAGE: ("ImageNet-1K", args.val_json_image, args.train_json_image, args.centre_emb_image),
        Modality.AUDIO: ("ESC-50", args.val_json_audio, args.train_json_audio, args.centre_emb_audio),
        Modality.POINT: ("ModelNet40", args.val_json_point, args.train_json_point, args.centre_emb_point),
    }

    if not args.skip_embed:
        for model in ([ModelType.UNIBIND] if args.only_unibind else list(ModelType)):
            for eps in ([EpsLevel.CLEAN] if args.only_clean else list(EpsLevel)):
                for modality in MODALITIES:
                    logger.info(f"== {model.name} @ {eps.name} - {modality.name} ==")
                    save_name = f"{model.name}_{eps.name}_{modality.name}"
                    if embeddings_already_saved(save_name) and not args.force_recompute:
                        logger.info(f"[SKIP] {save_name}")
                        continue
                    dataset, val_json, train_json, centre_path = config[modality]
                    centre_emb, centre_labels, label_to_index, _ = load_label_mapping(centre_path, device)
                    x = get_class_samples(modality, dataset, val_json, train_json, args.dataset_root, device, label_to_index, logger)
                    mean, std = get_normalization_tensors(modality, device)
                    lora_list = LORA_WEIGHTS_LIST_MAP.get(modality, [None, None])
                    lora_path = lora_list[1 if eps == EpsLevel.EPS4 else 0] if eps != EpsLevel.CLEAN else None
                    model_obj = build_model(device, args.pretrain_weights, modality, label_to_index, centre_emb, centre_labels, args.use_flash_attention, lora_path)
                    if EPSILON_LEVELS[eps]:
                        x = run_attack(model_obj, x, EPSILON_LEVELS[eps], mean, std, device, logger)
                    emb = extract_embeddings(model_obj, x, device)
                    save_embeddings(save_name, emb)
                    logger.info(f"[{modality.name}] Saved embedding: {save_name}")

    logger.info("== t-SNE Visualization ==")
    generate_all_tsne_combo_charts(tsne_dir, args.only_clean, args.only_unibind)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/data/datasets")
    parser.add_argument("--val_json_image", default="./datasets/ImageNet-1K/val_data.json")
    parser.add_argument("--train_json_image", default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--val_json_audio", default="./datasets/ESC-50/val_data.json")
    parser.add_argument("--train_json_audio", default="./datasets/ESC-50/train_data.json")
    parser.add_argument("--val_json_point", default="./datasets/ModelNet40/val_data.json")
    parser.add_argument("--train_json_point", default="./datasets/ModelNet40/train_data.json")
    parser.add_argument("--centre_emb_image", default="./centre_embs/image_in_center_embeddings.pkl")
    parser.add_argument("--centre_emb_audio", default="./centre_embs/audio_esc_center_embeddings.pkl")
    parser.add_argument("--centre_emb_point", default="./centre_embs/point_modelnet40_center_embeddings.pkl")
    parser.add_argument("--pretrain_weights", default="./ckpts/pretrained_weights_flash_atten.pt")
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--skip_embed", action="store_true", default=False)
    parser.add_argument("--only_clean", action="store_true", default=True)
    parser.add_argument("--only_unibind", action="store_true", default=True)
    parser.add_argument("--force_recompute", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
