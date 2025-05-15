import os
import argparse
import json
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from enum import Enum
from sklearn.manifold import TSNE
from torch.utils.data import Subset, DataLoader
from multiprocessing import Pool

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

MODALITIES = [Modality.IMAGE, Modality.AUDIO, Modality.EVENT]

# === Logger
def setup_logger():
    os.makedirs("output", exist_ok=True)
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler("output/tsne.log")
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
def get_church_samples(modality, dataset_name, val_json, train_json, dataset_root, device, label_to_index):
    def load_json(path):
        if not os.path.exists(path): return []
        with open(path) as f: return json.load(f)

    val_data = load_json(val_json)
    val_matches = [i for i, e in enumerate(val_data) if "church" in e["label"].lower()]
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
        combined_matches = val_matches + [i + len(val_data) for i, e in enumerate(train_data) if "church" in e["label"].lower()]
        if not combined_matches:
            raise ValueError(f"[{modality.name}] No 'church' examples in val or train.")

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
    print(f"Building model for {modality.name}, {lora_weights}...")
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

def plot_tsne_per_combo(name_prefix):
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
        print(f"[SKIP] No embeddings found for {name_prefix}")
        return

    X = np.concatenate(all_embs, axis=0)
    coords = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42).fit_transform(X)
    xs, ys = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c=all_colors, s=60)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=m.name.lower(),
                   markerfacecolor=MODALITY_COLOR[m],
                   markeredgecolor='k', markersize=10)
        for m in MODALITIES
    ]
    # plt.legend(handles=handles, title="Modality", fontsize=10, title_fontsize=11)
    # plt.title(f"t-SNE: {name_prefix}", fontsize=13)
    # plt.xlabel("t-SNE 1", fontsize=12)
    # plt.ylabel("t-SNE 2", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs("output/tsne_charts", exist_ok=True)
    plt.savefig(f"output/tsne_charts/tsne_{name_prefix}.png", dpi=300)
    plt.close()
    print(f"✅ Saved: output/tsne_charts/tsne_{name_prefix}.png")

def plot_center_embeddings_tsne(modality_to_center_emb, device):
    os.makedirs("output/tsne_charts", exist_ok=True)
    all_embs, all_colors = [], []

    for modality, (emb_path, color) in modality_to_center_emb.items():
        if not os.path.exists(emb_path):
            print(f"[WARN] Center embedding file missing for {modality.name}: {emb_path}")
            continue

        try:
            centre_emb, centre_labels, _, _ = load_label_mapping(emb_path, device)
            if isinstance(centre_emb, torch.Tensor):
                centre_emb = centre_emb.cpu().numpy()

            for emb, label in zip(centre_emb, centre_labels):
                if "church" in label.lower():
                    all_embs.append(emb)
                    all_colors.append(color)

        except Exception as e:
            print(f"[ERROR] Failed to load {emb_path} for {modality.name}: {e}")
            continue

    if not all_embs:
        print("No 'church' class centers found.")
        return

    X = np.stack(all_embs, axis=0)
    coords = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X)
    xs, ys = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c=all_colors, s=60)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=mod.name.lower(),
                   markerfacecolor=MODALITY_COLOR[mod],
                   markeredgecolor='k', markersize=10)
        for mod in modality_to_center_emb
    ]
    # plt.legend(handles=handles, title="Modality")
    # plt.title("t-SNE of Center Embeddings (church only)")
    # plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("output/tsne_charts/tsne_center_embeddings_church.png", dpi=200)
    plt.close()
    print("✅ Saved t-SNE chart: tsne_center_embeddings_church.png")

# === Dispatcher for all (model + eps) combos
def generate_all_tsne_combo_charts(only_clean=False, only_unibind=False):
    print("Generating t-SNE charts for all (model, eps) combos...")
    eps_levels = [EpsLevel.CLEAN] if only_clean else list(EpsLevel)
    model_types = [ModelType.UNIBIND] if only_unibind else list(ModelType)
    prefixes = [f"{model.name}_{eps.name}" for model in model_types for eps in eps_levels]

    with Pool(processes=os.cpu_count()) as pool:
        pool.map(plot_tsne_per_combo, prefixes)

# === Main
def main(args):
    logger = setup_logger()
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
                x = get_church_samples(modality, dataset, val_json, train_json, args.dataset_root, device, label_to_index)
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

    logger.info("Generating per-(model, eps) t-SNE charts...")
    generate_all_tsne_combo_charts(args.only_clean, args.only_unibind)
    plot_center_embeddings_tsne({
        Modality.IMAGE: (args.centre_emb_image, MODALITY_COLOR[Modality.IMAGE]),
        Modality.AUDIO: (args.centre_emb_audio, MODALITY_COLOR[Modality.AUDIO]),
        Modality.EVENT: (args.centre_emb_event, MODALITY_COLOR[Modality.EVENT]),
    }, device=device)


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
    parser.add_argument("--force_recompute", action="store_true", default=False, help="Force regenerate .npy embeddings")
    parser.add_argument("--only_clean", action="store_true", default=False, help="Only generate clean embeddings")
    parser.add_argument("--only_unibind", action="store_true", default=False, help="Only run UniBind model")
    args = parser.parse_args()
    main(args)
