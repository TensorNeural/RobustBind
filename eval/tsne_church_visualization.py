import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from model import UniBindModel
from attacks import apgd_attack
from data_util import val_data_loader, load_label_mapping, get_normalization_tensors

# === Constants ===

NUM_SAMPLES = 30
EPSILON_LEVELS = {
    "clean": None,
    "eps2": 2 / 255.0,
    "eps4": 4 / 255.0
}
MODELS = {
    "UniBind": {"use_lora": False, "lora_path": None},
    "RobustBind2": {"use_lora": True, "lora_path": "ckpts/vision_eps2_lora_weights.pt"},
    "RobustBind4": {"use_lora": True, "lora_path": "ckpts/vision_eps4_lora_weights.pt"},
}
MODALITY_COLOR = {"image": "blue", "event": "green", "audio": "red"}
MODALITY_TO_DATASET = {"image": "Places365", "event": "N-ImageNet-1K", "audio": "ESC-50"}
THUMB_POSITIONS = {"image": (0.05, 0.8), "event": (0.05, 0.6), "audio": (0.05, 0.4)}
PERPLEXITY = 5

# === Functions ===

def get_church_samples(modality, dataset_name, val_json, center_emb, dataset_root, device):
    raw_emb, raw_lbls, lbl_to_idx, idx_to_lbl = load_label_mapping(center_emb, device)
    loader = val_data_loader(
        modality=modality,
        dataset_root=os.path.join(dataset_root, dataset_name),
        val_json=val_json,
        label_to_index=lbl_to_idx,
        batch_size=64,
        num_workers=2,
        max_samples=5000,
    )
    xs = []
    for batch in loader:
        x, y = batch["data"], batch["label"]
        for i in range(len(y)):
            if "church" in idx_to_lbl[y[i].item()].lower():
                xs.append(x[i])
                if len(xs) == NUM_SAMPLES:
                    return torch.stack(xs)
    raise ValueError("Not enough 'church' samples found.")

def run_embedding_eval(x, model, eps=None, device="cuda", mean=None, std=None):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        emb_clean = model(x, mode="embed").cpu().numpy()
    if eps is None:
        return emb_clean
    x_adv = apgd_attack(model, x, eps=eps, loss_type="ce", mean=mean, std=std)
    with torch.no_grad():
        emb_adv = model(x_adv, mode="embed").cpu().numpy()
    return emb_adv

def plot_tsne(coords, labels, colors, image, modality, tag, out_dir):
    plt.figure(figsize=(10, 8))
    plt.title(f"{modality.upper()} – {tag.upper()}")

    for i, (x, y) in enumerate(coords):
        plt.scatter(x, y, c=colors[i], label=labels[i] if i % NUM_SAMPLES == 0 else "", edgecolors="k", s=60)
        if i % NUM_SAMPLES == 0:
            plt.text(x + 0.5, y, labels[i], fontsize=8)

    thumb_x, thumb_y = THUMB_POSITIONS[modality]
    ax = plt.gca()
    inset_ax = ax.inset_axes([thumb_x, thumb_y, 0.1, 0.1])
    inset_ax.imshow(image)
    inset_ax.axis("off")

    for j in range(3):
        idx = j * NUM_SAMPLES
        x1, y1 = coords[idx]
        bbox = inset_ax.get_position()
        cx = bbox.x0 + bbox.width / 2
        cy = bbox.y0 + bbox.height / 2
        trans = ax.figure.transFigure.inverted().transform
        ax.annotate("",
                    xy=(x1, y1), xycoords='data',
                    xytext=(cx, cy), textcoords='figure fraction',
                    arrowprops=dict(arrowstyle="->", color=MODALITY_COLOR[modality], lw=1))

    plt.axis("off")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tsne_{modality}_{tag}.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

# === Pipeline ===

def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_jsons = {
        "image": args.val_json_image,
        "event": args.val_json_event,
        "audio": args.val_json_audio
    }
    center_embs = {
        "image": args.center_emb_image,
        "event": args.center_emb_event,
        "audio": args.center_emb_audio
    }

    for modality in ["image", "event", "audio"]:
        dataset = MODALITY_TO_DATASET[modality]
        val_json = val_jsons[modality]
        center_emb = center_embs[modality]

        print(f"[{modality.upper()}] Loading samples...")
        x = get_church_samples(modality, dataset, val_json, center_emb, args.dataset_root, device)
        mean, std = get_normalization_tensors(modality, device)
        visuals = x[0:1].cpu()  # for thumbnail

        all_embeddings = {}
        for model_name, config in MODELS.items():
            model = UniBindModel(
                device=device,
                pretrain_weights=args.pretrain_weights,
                modality=modality,
                centre_embeddings=None,
                centre_labels=None,
                label_to_index=None,
                logger=None,
                use_flash_attention=args.use_flash_attention,
                use_lora=config["use_lora"]
            ).to(device).eval()

            if config["lora_path"]:
                model.load_lora_weights(config["lora_path"])

            for tag, eps in EPSILON_LEVELS.items():
                key = f"{tag}_{model_name}"
                print(f"Running {modality}/{key}...")
                all_embeddings[key] = run_embedding_eval(x, model, eps, device, mean, std)

        # t-SNE and render
        for tag in EPSILON_LEVELS:
            points = []
            labels = []
            colors = []

            for model_name in MODELS:
                key = f"{tag}_{model_name}"
                emb = all_embeddings[key]
                points.append(emb)
                labels += [model_name] * NUM_SAMPLES
                colors += [MODALITY_COLOR[modality]] * NUM_SAMPLES

            X = np.vstack(points)
            coords = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42).fit_transform(X)
            image = visuals[0].permute(1, 2, 0).numpy()
            plot_tsne(coords, labels, colors, image, modality, tag, out_dir=os.path.join(args.output_dir, "tsne_charts"))

# === CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--val_json_image", required=True)
    parser.add_argument("--val_json_event", required=True)
    parser.add_argument("--val_json_audio", required=True)
    parser.add_argument("--center_emb_image", required=True)
    parser.add_argument("--center_emb_event", required=True)
    parser.add_argument("--center_emb_audio", required=True)
    parser.add_argument("--pretrain_weights", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_flash_attention", action="store_true")
    args = parser.parse_args()

    run_pipeline(args)
