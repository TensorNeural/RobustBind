import os
import json
import torch
import shutil
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from shared_types import Modality
from model import UniBindModel, ForwardMode
from attack import AttackModel, APGDAttack, two_stage_attack
from data_util import get_transform_fn, get_normalization_tensors, load_label_mapping
import logging

TOP_K = 5
NUM_QUERY_PER_CLASS = 5
NUM_TARGET_PER_CLASS = 20

LORA_VARIANTS = {
    "clean": None,
    "eps2": "./ckpts/audio_eps2_lora_weights.pt",
    "eps4": "./ckpts/audio_eps4_lora_weights.pt"
}

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

def load_label_map(path):
    with open(path) as f:
        return json.load(f)


def load_samples(modality, dataset_name, val_json, dataset_root, center_emb_path, max_per_class, device, class_map):
    print(f"[INFO] Loading samples for modality: {modality}, dataset: {dataset_name}")
    print(f"[INFO] Using val_json: {val_json}")
    print(f"[INFO] Class map contains {len(class_map)}")

    _, _, label_to_index, _ = load_label_mapping(center_emb_path, device)
    transform = get_transform_fn(modality)

    with open(val_json, "r") as f:
        entries = json.load(f)

    print(f"[INFO] Loaded {len(entries)} entries from {val_json}")

    buckets, labels, metas = defaultdict(list), defaultdict(list), defaultdict(list)
    failed, skipped = 0, 0

    for entry in tqdm(entries, desc=f"[{modality}] Loading samples", unit="sample"):
        label_str = entry["label"].strip().lower()
        if label_str not in class_map:
            skipped += 1
            continue
        if len(buckets[label_str]) >= max_per_class:
            continue

        rel_path = entry["data"]
        full_path = os.path.join(dataset_root, dataset_name, rel_path)
        try:
            tensor = transform([full_path], device=device)[0]
            buckets[label_str].append(tensor)
            labels[label_str].append(label_str)
            metas[label_str].append(full_path)
        except Exception as e:
            print(f"[WARN] Failed to process {full_path}: {e}")
            failed += 1

    print(f"[INFO] Skipped entries with unknown label: {skipped}")
    print(f"[INFO] Failed transformations: {failed}")
    print(f"[INFO] Found {sum(len(v) for v in buckets.values())} usable samples for {modality}")

    results = []
    for label in class_map:
        for i in range(min(max_per_class, len(buckets[label]))):
            results.append({
                "data": buckets[label][i],
                "label": labels[label][i],
                "path": metas[label][i],
                "mapped_class": class_map[label]
            })

    if not results:
        print(f"[ERROR] No usable samples found for {modality}")
        raise RuntimeError(f"No samples found for modality {modality}")
    return results

def extract_embeddings(model, x, device):
    with torch.no_grad():
        return torch.nn.functional.normalize(model(x.to(device), ForwardMode.EMBEDDINGS), dim=-1).cpu()

def extract_embeddings_with_two_stage_attack(model, x, eps, mean, std, device, logger):
    labels = torch.zeros(x.size(0), dtype=torch.long, device=device)
    attack_model = AttackModel(model, mean, std)
    stage1 = APGDAttack(logger, attack_model, "linf", 10, 1, eps, "ce", device)
    stage2 = APGDAttack(logger, attack_model, "linf", 10, 1, eps, "ce", device)
    adv_x = two_stage_attack(logger, model, x, labels, stage1, stage2, mean, std)
    with torch.no_grad():
        return torch.nn.functional.normalize(model(adv_x, ForwardMode.EMBEDDINGS), dim=-1).cpu()

def build_model(args, logger, device, modality, lora_path=None):
    model = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=modality,
        centre_embeddings=None,
        centre_labels=None,
        label_to_index=None,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=(lora_path is not None)
    )
    if torch.cuda.device_count() > 1:
        model = DataParallelWithReplication(model)
    model = model.to(device).eval()
    if lora_path:
        model.load_lora_weights(lora_path)
    return model

def precompute_targets(args, logger, device):
    logger.info("[INFO] Precomputing targets for all modalities")
    all_targets = []
    for modality in [Modality.EVENT, Modality.IMAGE, Modality.POINT]:
        logger.info(f"[INFO] Precomputing embeddings for {modality} modality")

        model = build_model(args, logger, device, modality)

        # Load label_to_index from center embedding
        _, _, label_to_index, _ = load_label_mapping(args.center_embs[modality], device)
        class_names = list(label_to_index.keys())

        # Use identity map: label → label
        class_map = {label.strip().lower(): label for label in class_names}

        samples = load_samples(
            modality=modality,
            dataset_name=args.datasets[modality],
            val_json=args.val_jsons[modality],
            dataset_root=args.dataset_root,
            center_emb_path=args.center_embs[modality],
            max_per_class=NUM_TARGET_PER_CLASS,
            device=device,
            class_map=class_map
        )

        # === Batched embedding extraction with tqdm ===
        batch_size = 500
        embeddings = []
        data = [s["data"] for s in samples]

        for i in tqdm(range(0, len(data), batch_size), desc=f"[{modality}] Extracting embeddings"):
            x_batch = torch.stack(data[i:i + batch_size]).to(device)
            emb_batch = extract_embeddings(model, x_batch, device)
            embeddings.append(emb_batch)

        emb = torch.cat(embeddings, dim=0)

        for i, s in enumerate(samples):
            all_targets.append({
                "embedding": emb[i],
                "label": s["label"],
                "modality": modality,
                "esc_class": s["mapped_class"],  # same as label since class_map is identity
                "path": s["path"]
            })

        print(f"[INFO] Finished {modality}, total targets so far: {len(all_targets)}")

    return all_targets

def evaluate_and_save_results(logger, samples, embeddings, target_matrix, all_targets, label_map, out_dir):
    logger.info(f"[INFO] Evaluating results for {len(samples)} samples, {len(all_targets)} targets")
    results, hits = [], 0
    for i, s in enumerate(tqdm(samples)):
        q_emb, q_class, q_path = embeddings[i], s["mapped_class"], s["path"]
        sims = torch.matmul(q_emb.unsqueeze(0), torch.nn.functional.normalize(target_matrix, dim=1).T).squeeze(0)
        topk = torch.topk(sims, k=TOP_K).indices
        q_out = os.path.join(out_dir, str(i))
        os.makedirs(q_out, exist_ok=True)
        shutil.copyfile(q_path, os.path.join(q_out, "query_audio.wav"))

        hit = False
        for rank, idx in enumerate(topk.tolist()):
            t = all_targets[idx]
            expected = label_map[q_class][t["modality"].value]
            match = (t["label"] == expected)
            if match: hit = True
            dst_path = os.path.join(q_out, f"{rank+1}_{t['modality'].value}_{t['label']}.ext")
            if os.path.exists(t["path"]):
                shutil.copyfile(t["path"], dst_path)
            results.append({
                "query_index": i,
                "query_class": q_class,
                "retrieved_label": t["label"],
                "retrieved_modality": t["modality"].value,
                "retrieved_path": t["path"],
                "similarity": round(sims[idx].item(), 4),
                "match": match,
                "copy_path": dst_path
            })
        if hit: hits += 1

    with open(os.path.join(out_dir, "top5_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[RESULT] Recall@5 = {hits}/{len(samples)} = {hits / len(samples):.3f}")

def run_single_experiment(logger, variant_name, lora_path, args, device, label_map, audio_class_map, all_targets, target_matrix, base_out):
    out_dir = os.path.join(base_out, variant_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[INFO] Running variant: {variant_name}")

    model = build_model(args, logger, device, Modality.AUDIO, lora_path=lora_path)
    samples = load_samples(Modality.AUDIO, "ESC-50", args.val_json_audio, args.dataset_root,
                           args.center_emb_audio, NUM_QUERY_PER_CLASS, device, audio_class_map)
    mean, std = get_normalization_tensors(Modality.AUDIO, device)
    x = torch.stack([s["data"] for s in samples]).to(device)

    eps = None
    if "eps2" in variant_name:
        eps = 2 / 255.
    elif "eps4" in variant_name:
        eps = 4 / 255.

    emb
    if eps:
        emb = extract_embeddings_with_two_stage_attack(model, x, eps, mean, std, device, logger)
    else:
        emb = extract_embeddings(model, x, device)
    evaluate_and_save_results(logger, samples, emb, target_matrix, all_targets, label_map, out_dir)

def run(args):
    print(f"[INFO] Running cross-modality search with args: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_out = os.path.join(args.output_dir, "cross_modal_search", timestamp)
    logger = setup_logger(base_out)
    logger.info(f"[INFO] Using dataset root: {args.dataset_root}")
    logger.info("Loading label map")
    label_map = load_label_map(args.label_map)
    esc_classes = list(label_map.keys())
    audio_class_map = {cls: cls for cls in esc_classes}
    

    all_targets = precompute_targets(args, logger, device)
    target_matrix = torch.stack([t["embedding"] for t in all_targets])

    for variant_name, lora_path in LORA_VARIANTS.items():
        run_single_experiment(logger, variant_name, lora_path, args, device, label_map, audio_class_map, all_targets, target_matrix, base_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/user/datasets")
    parser.add_argument("--val_json_audio", default="./datasets/ESC-50/val_data.json")
    parser.add_argument("--val_json_image", default="./datasets/Places365/val_data.json")
    parser.add_argument("--val_json_event", default="./datasets/N-ImageNet-1K/val_data.json")
    parser.add_argument("--val_json_point", default="./datasets/ModelNet40/val_data.json")
    parser.add_argument("--center_emb_audio", default="./centre_embs/audio_esc_center_embeddings.pkl")
    parser.add_argument("--center_emb_image", default="./centre_embs/image_p365_center_embeddings.pkl")
    parser.add_argument("--center_emb_event", default="./centre_embs/event_nin_center_embeddings.pkl")
    parser.add_argument("--center_emb_point", default="./centre_embs/point_modelnet40_center_embeddings.pkl")
    parser.add_argument("--label_map", default="./datasets/esc50_label_map.json")
    parser.add_argument("--pretrain_weights", default="./ckpts/pretrained_weights_flash_atten.pt")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    args = parser.parse_args()

    args.datasets = {
        Modality.IMAGE: "Places365",
        Modality.EVENT: "N-ImageNet-1K",
        Modality.POINT: "ModelNet40"
    }
    args.val_jsons = {
        Modality.IMAGE: args.val_json_image,
        Modality.EVENT: args.val_json_event,
        Modality.POINT: args.val_json_point
    }
    args.center_embs = {
        Modality.IMAGE: args.center_emb_image,
        Modality.EVENT: args.center_emb_event,
        Modality.POINT: args.center_emb_point
    }

    run(args)
