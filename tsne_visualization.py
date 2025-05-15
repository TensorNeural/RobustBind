import os
import argparse
import json
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime
from sklearn.manifold import TSNE
from torch.utils.data import Subset, DataLoader
from multiprocessing import Pool
from functools import partial

from model import UniBindModel, ForwardMode
from attack import AttackModel, APGDAttack, two_stage_attack
from data_util import JsonDataset, get_transform_fn, get_normalization_tensors, load_label_mapping
from transform import unnormalize_inplace
from shared_types import Modality

# === Thread config
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

# === Enums
class ModelType(str, Enum):
    UNIBIND = "UniBind"
    ROBUST2 = "RobustBind2"
    ROBUST4 = "RobustBind4"

class EpsLevel(str, Enum):
    CLEAN = "clean"
    EPS2 = "eps2"
    EPS4 = "eps4"

# === Constants
NUM_SAMPLES = 38
PERPLEXITY = 5
EPSILON_LEVELS = {
    EpsLevel.CLEAN: None,
    EpsLevel.EPS2: 2 / 255.0,
    EpsLevel.EPS4: 4 / 255.0
}
LORA_WEIGHTS_LIST_MAP = {
    Modality.IMAGE: ["./ckpts/vision_eps2_lora_weights.pt", "./ckpts/vision_eps4_lora_weights.pt"],
    Modality.AUDIO: ["./ckpts/audio_eps2_lora_weights.pt", "./ckpts/audio_eps4_lora_weights.pt"],
    Modality.EVENT: ["./ckpts/vision_eps2_lora_weights.pt", "./ckpts/vision_eps4_lora_weights.pt"],
}
MODALITY_COLOR = {
    Modality.IMAGE: "blue",
    Modality.AUDIO: "red",
    Modality.EVENT: "green"
}
CLASS_NAME_PER_MODALITY = {
    Modality.IMAGE: "church",
    Modality.AUDIO: "church",
    Modality.EVENT: "church"
}
MODALITIES = [Modality.IMAGE, Modality.AUDIO, Modality.EVENT]

# === Logger
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "tsne.log"))
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# === Multi-GPU wrapper
class DataParallelWithReplication(torch.nn.DataParallel):
    def replicate(self, module, device_ids):
        replicas = super().replicate(module, device_ids)
        for i, replica in enumerate(replicas):
            device = torch.device(f"cuda:{device_ids[i]}")
            if hasattr(module, "centre_embeddings"):
                replica.centre_embeddings = module.centre_embeddings.to(device)
            if hasattr(module, "centre_label_indices"):
                replica.centre_label_indices = module.centre_label_indices.to(device)
            if hasattr(module, "centre_class_mask"):
                replica.centre_class_mask = module.centre_class_mask.to(device)
        return replicas

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# === Data/Model Helpers
def get_class_samples(modality, dataset_name, val_json, train_json, dataset_root, device, label_to_index):
    target_class = CLASS_NAME_PER_MODALITY[modality]

    def load_json(path):
        if not os.path.exists(path): return []
        with open(path) as f: return json.load(f)

    val_data = load_json(val_json)
    val_matches = [i for i, e in enumerate(val_data) if target_class in e["label"].lower()]
    if len(val_matches) >= NUM_SAMPLES:
        selected = val_matches[:NUM_SAMPLES]
        dataset = JsonDataset(
            dataset_root=os.path.join(dataset_root, dataset_name),
            data_json_path=val_json,
            transform=get_transform_fn(modality),
            label_to_index=label_to_index
        )
    else:
        train_data = load_json(train_json)
        combined_data = val_data + train_data
        combined_matches = val_matches + [i + len(val_data) for i, e in enumerate(train_data) if target_class in e["label"].lower()]
        if not combined_matches:
            raise ValueError(f"[{modality.name}] No examples found for class '{target_class}'.")

        selected = combined_matches[:NUM_SAMPLES]
        combined_path = f"./output/tmp_combined_{modality.name.lower()}.json"
        with open(combined_path, "w") as f:
            json.dump(combined_data, f)

        dataset = JsonDataset(
            dataset_root=os.path.join(dataset_root, dataset_name),
            data_json_path=combined_path,
            transform=get_transform_fn(modality),
            label_to_index=label_to_index
        )

    subset = Subset(dataset, selected)
    loader = DataLoader(subset, batch_size=len(selected), shuffle=False, num_workers=os.cpu_count())
    for batch in loader:
        x, _ = batch
        return x.to(device)

def build_model(device, pretrain_weights, modality, label_to_index, centre_embeddings, centre_labels, use_flash_attention, lora_weights=None):
    model = UniBindModel(
        device=device,
        pretrain_weights=pretrain_weights,
        modality=modality,
        centre_embeddings=centre_embeddings,
        centre_labels=centre_labels,
        label_to_index=label_to_index,
        logger=None,
        use_flash_attention=use_flash_attention,
        use_fine_tune=True,
        use_lora=(lora_weights is not None),
        lora_weights=lora_weights
    )
    if torch.cuda.device_count() > 1:
        model = DataParallelWithReplication(model)
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

def save_embeddings(name, embeddings):
    os.makedirs("output/embeddings", exist_ok=True)
    np.save(f"output/embeddings/{name}.npy", embeddings)

def embeddings_already_saved(name):
    return os.path.exists(f"output/embeddings/{name}.npy")

def plot_tsne_per_combo(name_prefix, save_dir):
    folder = "output/embeddings"
    all_embs, all_colors = [], []

    for modality in MODALITIES:
        fname = f"{name_prefix}_{modality.name}.npy"
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            continue

        emb = np.load(fpath)
        all_embs.append(emb)
        all_colors.extend([MODALITY_COLOR[modality]] * len(emb))

    if not all_embs:
        return

    X = np.concatenate(all_embs, axis=0)
    coords = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42).fit_transform(X)
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=all_colors, s=60)
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tsne_{name_prefix}.png"), dpi=300)
    plt.close()

def plot_center_embeddings_tsne(modality_to_center_emb, device, save_dir):
    all_embs, all_colors = [], []

    for modality, (emb_path, color) in modality_to_center_emb.items():
        if not os.path.exists(emb_path):
            continue

        try:
            centre_emb, centre_labels, _, _ = load_label_mapping(emb_path, device)
            if isinstance(centre_emb, torch.Tensor):
                centre_emb = centre_emb.cpu().numpy()

            target_class = CLASS_NAME_PER_MODALITY[modality]
            for emb, label in zip(centre_emb, centre_labels):
                if target_class in label.lower():
                    all_embs.append(emb)
                    all_colors.append(color)

        except Exception:
            continue

    if not all_embs:
        return

    X = np.stack(all_embs, axis=0)
    coords = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42).fit_transform(X)
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=all_colors, s=60)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_center_embeddings.png"), dpi=200)
    plt.close()

def generate_all_tsne_combo_charts(tsne_output_dir, only_clean=False, only_unibind=False):
    eps_levels = [EpsLevel.CLEAN] if only_clean else list(EpsLevel)
    model_types = [ModelType.UNIBIND] if only_unibind else list(ModelType)
    prefixes = [f"{model.name}_{eps.name}" for model in model_types for eps in eps_levels]
    plot_fn = partial(plot_tsne_per_combo, save_dir=tsne_output_dir)
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(plot_fn, prefixes)

# === Main
def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tsne_output_dir = os.path.join("output", "tsne_charts", timestamp)
    logger = setup_logger(tsne_output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modality_config = {
        Modality.IMAGE: ("Places365", args.val_json_image, args.train_json_image, args.centre_emb_image),
        Modality.EVENT: ("N-ImageNet-1K", args.val_json_event, args.train_json_event, args.centre_emb_event),
        Modality.AUDIO: ("ESC-50", args.val_json_audio, args.train_json_audio, args.centre_emb_audio),
    }

    eps_levels = [EpsLevel.CLEAN] if args.only_clean else list(EpsLevel)
    model_types = [ModelType.UNIBIND] if args.only_unibind else list(ModelType)

    for model_enum in model_types:
        for eps_enum in eps_levels:
            eps = EPSILON_LEVELS[eps_enum]
            for modality, (dataset, val_json, train_json, center_emb_path) in modality_config.items():
                save_name = f"{model_enum.name}_{eps_enum.name}_{modality.name}"
                if embeddings_already_saved(save_name) and not args.force_recompute:
                    logger.info(f"[SKIP] {save_name} already saved.")
                    continue

                logger.info(f"== {modality.name} - {model_enum.value} @ {eps_enum.name} ==")
                centre_emb, centre_labels, label_to_index, _ = load_label_mapping(center_emb_path, device)
                x = get_class_samples(modality, dataset, val_json, train_json, args.dataset_root, device, label_to_index)
                mean, std = get_normalization_tensors(modality, device)

                lora_list = LORA_WEIGHTS_LIST_MAP[modality]
                lora_path = lora_list[0] if eps_enum == EpsLevel.EPS2 else lora_list[1] if eps_enum == EpsLevel.EPS4 else None

                model = build_model(
                    device=device,
                    pretrain_weights=args.pretrain_weights,
                    modality=modality,
                    label_to_index=label_to_index,
                    centre_embeddings=centre_emb,
                    centre_labels=centre_labels,
                    use_flash_attention=args.use_flash_attention,
                    lora_weights=lora_path
                )

                if eps is not None:
                    logger.info("Running adversarial attack...")
                    x = run_attack(model, x, eps, mean, std, device, logger)
                else:
                    logger.info("Skipping attack for CLEAN.")

                logger.info("Extracting embeddings...")
                embeddings = extract_embeddings(model, x, device)
                save_embeddings(save_name, embeddings)

    logger.info("Generating t-SNE visualizations...")
    generate_all_tsne_combo_charts(tsne_output_dir, args.only_clean, args.only_unibind)
    plot_center_embeddings_tsne({
        Modality.IMAGE: (args.centre_emb_image, MODALITY_COLOR[Modality.IMAGE]),
        Modality.AUDIO: (args.centre_emb_audio, MODALITY_COLOR[Modality.AUDIO]),
        Modality.EVENT: (args.centre_emb_event, MODALITY_COLOR[Modality.EVENT]),
    }, device=device, save_dir=tsne_output_dir)

# === CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/user/datasets")
    parser.add_argument("--val_json_image", default="./datasets/Places365/val_data.json")
    parser.add_argument("--val_json_event", default="./datasets/N-ImageNet-1K/val_data.json")
    parser.add_argument("--val_json_audio", default="./datasets/ESC-50/val_data.json")
    parser.add_argument("--train_json_image", default="./datasets/Places365/train_data.json")
    parser.add_argument("--train_json_event", default="./datasets/N-ImageNet-1K/train_data.json")
    parser.add_argument("--train_json_audio", default="./datasets/ESC-50/train_data.json")
    parser.add_argument("--centre_emb_image", default="./centre_embs/image_p365_center_embeddings.pkl")
    parser.add_argument("--centre_emb_event", default="./centre_embs/event_nin_center_embeddings.pkl")
    parser.add_argument("--centre_emb_audio", default="./centre_embs/audio_esc_center_embeddings.pkl")
    parser.add_argument("--pretrain_weights", default="./ckpts/pretrained_weights_flash_atten.pt")
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--force_recompute", action="store_true", default=False)
    parser.add_argument("--only_clean", action="store_true", default=False)
    parser.add_argument("--only_unibind", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
