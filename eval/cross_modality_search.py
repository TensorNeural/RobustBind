import os
import json
import shutil
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from model import UniBindModel
from attacks import apgd_attack
from data_util import val_data_loader, load_label_mapping, get_normalization_tensors

# === Constants ===
QUERY_CLASS = "airplane"
NUM_QUERIES = 20
TOP_K = 5
EPSILONS = {"clean": None, "eps2": 2/255, "eps4": 4/255}
MODELS = {
    "UniBind": {"use_lora": False, "lora_path": None},
    "RobustBind2": {"use_lora": True, "lora_path": "ckpts/vision_eps2_lora_weights.pt"},
    "RobustBind4": {"use_lora": True, "lora_path": "ckpts/vision_eps4_lora_weights.pt"},
}
TARGET_MODALITIES = ["image", "event", "point"]

# === Helpers ===
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_class_samples(modality, dataset_name, val_json, dataset_root, center_emb, target_class, max_needed, device):
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
    x_list, meta_list = [], []
    for batch in loader:
        x, y, path = batch["data"], batch["label"], batch["path"]
        for i in range(len(y)):
            label_name = idx_to_lbl[y[i].item()].lower()
            if target_class in label_name:
                x_list.append(x[i])
                meta_list.append(path[i])
                if len(x_list) >= max_needed:
                    return torch.stack(x_list), meta_list
    raise ValueError(f"Not enough '{target_class}' samples in {modality}/{dataset_name}")

def build_model(args, modality, device, use_lora=False, lora_path=None):
    model = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=modality,
        centre_embeddings=None,
        centre_labels=None,
        label_to_index=None,
        logger=None,
        use_flash_attention=args.use_flash_attention,
        use_lora=use_lora,
    ).to(device).eval()
    if lora_path:
        model.load_lora_weights(lora_path)
    return model

def extract_embeddings(model, x, eps, mean, std, device):
    x = x.to(device)
    if eps is not None:
        x = apgd_attack(model, x, eps=eps, loss_type="ce", mean=mean, std=std)
    with torch.no_grad():
        emb = model(x, mode="embed")
    return F.normalize(emb, dim=-1)

def compute_topk(query_emb, target_embs, k=TOP_K):
    sims = F.cosine_similarity(query_emb.unsqueeze(0), target_embs)
    topk = torch.topk(sims, k)
    return topk.indices.cpu().tolist(), topk.values.cpu().tolist()

def copy_and_log_result(dst_dir, model_name, eps_tag, query_idx, tgt_mod, rank, src_path, score, tgt_path, query_meta, log):
    name = f"{model_name}_{eps_tag}_q{query_idx}_{tgt_mod}_top{rank+1}.jpg"
    dst_path = os.path.join(dst_dir, name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copyfile(src_path, dst_path)
    log.append({
        "query_index": query_idx,
        "query_model": model_name,
        "query_eps": eps_tag,
        "query_audio": query_meta,
        "target_modality": tgt_mod,
        "target_path": tgt_path,
        "similarity": round(score, 5),
        "copy_path": dst_path
    })

def search_one_query(query_vec, query_idx, query_meta, model_name, eps_tag, clean_targets, target_meta, args, log):
    for tgt_mod in TARGET_MODALITIES:
        tgt_vecs = clean_targets[tgt_mod]
        tgt_paths = target_meta[tgt_mod]
        top_ids, top_scores = compute_topk(query_vec, tgt_vecs)
        for rank, (tid, score) in enumerate(zip(top_ids, top_scores)):
            src_path = os.path.join(args.dataset_root, args.datasets[tgt_mod], tgt_paths[tid])
            copy_and_log_result(
                os.path.join(args.output_dir, "cross_modality_search", "copied"),
                model_name, eps_tag, query_idx, tgt_mod, rank,
                src_path, score, tgt_paths[tid], query_meta, log
            )

# === Main Logic ===
def run_audio_to_modal_search(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(os.path.join(args.output_dir, "cross_modality_search"))
    copied_dir = ensure_dir(os.path.join(out_dir, "copied"))

    # Load query audio
    audio_x, audio_meta = load_class_samples(
        "audio", "ESC-50", args.val_json_audio,
        args.dataset_root, args.center_emb_audio,
        target_class=QUERY_CLASS, max_needed=NUM_QUERIES, device=device
    )

    # Load clean target sets
    clean_targets, target_meta = {}, {}
    for modality in TARGET_MODALITIES:
        x, paths = load_class_samples(
            modality, args.datasets[modality], args.val_jsons[modality],
            args.dataset_root, args.center_embs[modality],
            target_class=QUERY_CLASS, max_needed=1000, device=device
        )
        model = build_model(args, modality, device)
        mean, std = get_normalization_tensors(modality, device)
        emb = extract_embeddings(model, x, None, mean, std, device)
        clean_targets[modality] = emb.cpu()
        target_meta[modality] = paths

    # Run search across models and eps
    results = []
    for model_name, cfg in MODELS.items():
        model = build_model(args, "audio", device, use_lora=cfg["use_lora"], lora_path=cfg["lora_path"])
        mean, std = get_normalization_tensors("audio", device)

        for eps_tag, eps in EPSILONS.items():
            print(f"[SEARCH] Model={model_name} | Perturb={eps_tag}")
            query_embs = extract_embeddings(model, audio_x, eps, mean, std, device)
            for i in tqdm(range(NUM_QUERIES)):
                search_one_query(
                    query_embs[i], i, audio_meta[i], model_name, eps_tag,
                    clean_targets, target_meta, args, results
                )

    with open(os.path.join(out_dir, "top5_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved results to {out_dir}/top5_results.json")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--pretrain_weights", required=True)
    parser.add_argument("--val_json_audio", required=True)
    parser.add_argument("--val_json_image", required=True)
    parser.add_argument("--val_json_event", required=True)
    parser.add_argument("--val_json_point", required=True)
    parser.add_argument("--center_emb_audio", required=True)
    parser.add_argument("--center_emb_image", required=True)
    parser.add_argument("--center_emb_event", required=True)
    parser.add_argument("--center_emb_point", required=True)
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    args.datasets = {
        "image": "ImageNet-1K",
        "event": "N-ImageNet-1K",
        "point": "ModelNet40"
    }
    args.val_jsons = {
        "image": args.val_json_image,
        "event": args.val_json_event,
        "point": args.val_json_point
    }
    args.center_embs = {
        "image": args.center_emb_image,
        "event": args.center_emb_event,
        "point": args.center_emb_point
    }

    run_audio_to_modal_search(args)
