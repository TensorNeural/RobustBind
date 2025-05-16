import os
import json
import torch
import shutil
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from shared_types import Modality
from model import UniBindModel, ForwardMode
from attack import AttackModel, APGDAttack, two_stage_attack
from data_util import get_transform_fn, get_normalization_tensors, load_label_mapping

TOP_K = 5
NUM_QUERY_PER_CLASS = 5
NUM_TARGET_PER_CLASS = 20
LORA_VARIANTS = {
    "clean": None,
    "eps2": "./ckpts/audio_eps2_lora_weights.pt",
    "eps4": "./ckpts/audio_eps4_lora_weights.pt"
}
QUERY_BATCH_SIZE = 100
MODALITY_EXTENSIONS = {
    Modality.AUDIO: ".wav",
    Modality.IMAGE: ".jpg",
    Modality.EVENT: ".png",
    Modality.POINT: ".png",
    Modality.THERMAL: ".png",
    Modality.VIDEO: ".mp4"
}
DATASET_NAMES = {
    Modality.AUDIO: "ESC-50",
    Modality.IMAGE: "Places365",
    Modality.EVENT: "N-ImageNet-1K",
    Modality.POINT: "ModelNet40",
}

def setup_ddp():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}"), rank, dist.get_world_size()

def setup_logger(log_dir, rank):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"Logger{rank}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"[%(asctime)s] [Rank {rank}] [%(filename)s:%(lineno)d] - %(message)s"
    )
    fh = logging.FileHandler(os.path.join(log_dir, f"log_rank{rank}.txt"))
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def render_bin_to_png(src_path, dst_path, width=240, height=180):
    try:
        with open(src_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 5)
        if raw.size == 0:
            return
        x, y, p = raw[:, 0], raw[:, 1], (raw[:, 2] >> 7) & 1
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.scatter(x[p == 0], y[p == 0], c='b', s=0.2)
        ax.scatter(x[p == 1], y[p == 1], c='r', s=0.2)
        plt.savefig(dst_path, dpi=100, transparent=True)
        plt.close(fig)
    except Exception as e:
        print(f"Error rendering {src_path}: {e}")

def render_npz_to_png(src_path, dst_path, width=224, height=224):
    try:
        ev = np.load(src_path)["event_data"]
        x, y, p = ev["x"], ev["y"], ev["p"]
        x_norm = (x - x.min()) / (x.ptp() + 1e-5) * (width - 1)
        y_norm = (y - y.min()) / (y.ptp() + 1e-5) * (height - 1)
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.scatter(x_norm[p == 0], y_norm[p == 0], c='b', s=0.2)
        ax.scatter(x_norm[p == 1], y_norm[p == 1], c='r', s=0.2)
        plt.savefig(dst_path, dpi=100, transparent=True)
        plt.close(fig)
    except Exception as e:
        print(f"Error rendering {src_path}: {e}")

def build_model(args, device, modality, centre_emb, centre_labels, label_to_index, lora_path=None):
    logger = None
    model = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=modality,
        centre_embeddings=centre_emb,
        centre_labels=centre_labels,
        label_to_index=label_to_index,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=(lora_path is not None)
    ).to(device).eval()
    if lora_path:
        model.load_lora_weights(lora_path)
    return model

def extract_embeddings(model, x, device):
    with torch.no_grad():
        return model(x.to(device), ForwardMode.EMBEDDINGS)

def extract_with_attack(model, x, eps, mean, std, device, logger):
    labels = torch.zeros(x.size(0), dtype=torch.long, device=device)
    atk_model = AttackModel(model, mean, std)
    atk1 = APGDAttack(logger, atk_model, 100, "linf", 1, eps, "ce", 1, False, device)
    atk2 = APGDAttack(logger, atk_model, 100, "linf", 1, eps, "ce", 1, False, device)
    adv = two_stage_attack(logger, model, x, labels, atk1, atk2, mean, std)
    return extract_embeddings(model, adv, device)

def extract_and_gather_targets(args, device, rank, world_size, logger):
    all_targets_local = []

    for modality in [Modality.IMAGE, Modality.EVENT, Modality.POINT]:
        logger.info(f"[{modality}] Rank {rank} preparing target entries...")

        if rank == 0:
            raw_entries = json.load(open(args.val_jsons[modality]))
            class_counter = defaultdict(int)
            filtered = []
            for e in raw_entries:
                if class_counter[e["label"]] < NUM_TARGET_PER_CLASS:
                    filtered.append(e)
                    class_counter[e["label"]] += 1
            partitions = [filtered[i::world_size] for i in range(world_size)]
        else:
            partitions = None

        my_entries = [None]
        dist.scatter_object_list(my_entries, partitions if rank == 0 else my_entries, src=0)
        entries = my_entries[0]
        logger.info(f"[{modality}] Rank {rank} received {len(entries)} entries.")

        emb, labels, label_to_index, _ = load_label_mapping(args.center_embs[modality], device)
        model = build_model(args, device, modality, emb, labels, label_to_index)
        transform = get_transform_fn(modality)

        for e in tqdm(entries, desc=f"[{modality}] Extracting", disable=(rank != 0)):
            path = os.path.join(args.dataset_root, DATASET_NAMES[modality], e["data"])
            try:
                x = transform([path], device=device)[0]
                emb = extract_embeddings(model, x.unsqueeze(0), device)[0]
                all_targets_local.append({
                    "embedding": emb.cpu().numpy(),
                    "label": e["label"],
                    "modality": modality,
                    "path": path
                })
            except Exception as ex:
                logger.warning(f"Failed to process {path}: {ex}")

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, all_targets_local)

    all_targets = [x for sublist in gathered for x in sublist]
    for t in all_targets:
        t["embedding"] = torch.tensor(t["embedding"], device=device)
    if rank == 0:
        logger.info(f"[Global] Gathered {len(all_targets)} total targets.")
    return all_targets

def extract_audio_queries(args, device, rank, world_size, logger):
    logger.info(f"[Audio] Rank {rank} preparing query entries...")

    if rank == 0:
        raw_entries = json.load(open(args.val_jsons[Modality.AUDIO]))
        class_counter = defaultdict(int)
        filtered = []

        for e in raw_entries:
            label = e["label"]
            if label not in args.label_map:
                continue  # ❌ skip if not in label map
            if not all(mod.value in args.label_map[label] for mod in [Modality.IMAGE, Modality.EVENT, Modality.POINT]):
                continue  # ❌ skip if missing modality mapping

            if class_counter[label] < NUM_QUERY_PER_CLASS:
                filtered.append(e)
                class_counter[label] += 1

        logger.info(f"[Audio] Rank 0 kept {len(filtered)} filtered queries after label map validation.")
        partitions = [filtered[i::world_size] for i in range(world_size)]

        query_dir = os.path.join(args.output_dir, "cross_modality_search", "queries_by_class")
        os.makedirs(query_dir, exist_ok=True)
        per_class = defaultdict(list)
        for e in filtered:
            per_class[e["label"]].append(e)
        for label, group in per_class.items():
            with open(os.path.join(query_dir, f"{label}_queries.json"), "w") as f:
                json.dump(group, f, indent=2)
    else:
        partitions = None

    my_entries = [None]
    dist.scatter_object_list(my_entries, partitions if rank == 0 else my_entries, src=0)
    entries = my_entries[0]
    logger.info(f"[Audio] Rank {rank} received {len(entries)} queries.")

    emb, labels, label_to_index, _ = load_label_mapping(args.center_embs[Modality.AUDIO], device)
    transform = get_transform_fn(Modality.AUDIO)
    queries = []
    for e in entries:
        path = os.path.join(args.dataset_root, DATASET_NAMES[Modality.AUDIO], e["data"])
        try:
            x = transform([path], device=device)[0]
            queries.append({"tensor": x, "label": e["label"], "path": path})
        except Exception as ex:
            logger.warning(f"Failed to load query {path}: {ex}")
    return emb, labels, label_to_index, queries

def evaluate_lora_variant(variant, lora_path, args, queries, all_targets, target_matrix, out_base, logger, device, rank, world_size):
    logger.info(f"[{variant}] Rank {rank} starting evaluation with {len(queries)} queries")
    model = build_model(args, device, Modality.AUDIO, *args.audio_model_info, lora_path)
    eps = 2 / 255. if "eps2" in variant else 4 / 255. if "eps4" in variant else None
    mean, std = get_normalization_tensors(Modality.AUDIO, device)

    emb_x = []
    for i in range(0, len(queries), QUERY_BATCH_SIZE):
        x_batch = torch.stack([q["tensor"] for q in queries[i:i + QUERY_BATCH_SIZE]])
        logger.info(f"[{variant}] Rank {rank} attacking batch {i} → {i+len(x_batch)}")
        emb_batch = extract_with_attack(model, x_batch, eps, mean, std, device, logger) if eps else extract_embeddings(model, x_batch, device)
        emb_x.append(emb_batch)
    emb_x = torch.cat(emb_x, dim=0)

    hits, results = 0, []
    class_hits = defaultdict(int)
    class_total = defaultdict(int)

    out_dir = os.path.join(out_base, variant, f"rank{rank}")
    os.makedirs(out_dir, exist_ok=True)

    for i, s in enumerate(queries):
        sims = torch.matmul(emb_x[i].unsqueeze(0), torch.nn.functional.normalize(target_matrix, dim=1).T).squeeze(0)
        topk = torch.topk(sims, k=TOP_K).indices
        q_dir = os.path.join(out_dir, str(i))
        os.makedirs(q_dir, exist_ok=True)
        shutil.copyfile(s["path"], os.path.join(q_dir, "query_audio.wav"))

        class_total[s["label"]] += 1
        matched = False
        for r, idx in enumerate(topk.tolist()):
            t = all_targets[idx]
            expected = args.label_map[s["label"]][t["modality"].value]
            match = (t["label"] == expected)
            if match:
                matched = True
            ext = MODALITY_EXTENSIONS.get(t["modality"], ".bin")
            dst = os.path.join(q_dir, f"{r+1}_{t['modality'].value}_{t['label']}{ext}")
            if t["path"].endswith(".bin"):
                render_bin_to_png(t["path"], dst)
            elif t["path"].endswith(".npz"):
                render_npz_to_png(t["path"], dst)
            elif os.path.exists(t["path"]):
                shutil.copyfile(t["path"], dst)
            results.append({
                "query_index": i,
                "query_class": s["label"],
                "retrieved_label": t["label"],
                "retrieved_modality": t["modality"].value,
                "similarity": round(sims[idx].item(), 4),
                "match": match
            })
        if matched:
            hits += 1
            class_hits[s["label"]] += 1

    with open(os.path.join(out_dir, "top5_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"[{variant}] Rank {rank} Recall@5 summary:")
    for cls in sorted(class_total.keys()):
        h, t = class_hits[cls], class_total[cls]
        logger.info(f"  - {cls}: {h}/{t} = {h / t:.3f}")
    logger.info(f"[{variant}] Rank {rank} Total Recall@5: {hits}/{len(queries)} = {hits / len(queries):.3f}")

    # Gather to rank 0
    recall_data = {
        "rank": rank,
        "variant": variant,
        "hits": hits,
        "total": len(queries),
        "class_hits": dict(class_hits),
        "class_total": dict(class_total)
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, recall_data)

    if rank == 0:
        total_hits = sum(d["hits"] for d in gathered)
        total_queries = sum(d["total"] for d in gathered)
        logger.info(f"[{variant}] Global Recall@5: {total_hits}/{total_queries} = {total_hits / total_queries:.3f}")

        agg_hits = defaultdict(int)
        agg_total = defaultdict(int)
        for d in gathered:
            for cls, h in d["class_hits"].items():
                agg_hits[cls] += h
            for cls, t in d["class_total"].items():
                agg_total[cls] += t

        logger.info(f"[{variant}] Global Recall@5 per class:")
        for cls in sorted(agg_total.keys()):
            h, t = agg_hits[cls], agg_total[cls]
            logger.info(f"  - {cls}: {h}/{t} = {h / t:.3f}")

        with open(os.path.join(out_base, f"{variant}_recall_summary.json"), "w") as f:
            json.dump(gathered, f, indent=2)

def run(args, device, rank, world_size):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_base = os.path.join(args.output_dir, "cross_modality_search", timestamp)
    logger = setup_logger(out_base, rank)
    args.label_map = json.load(open(args.label_map))

    logger.info(f"[Setup] Rank {rank} starting data prep...")

    all_targets = extract_and_gather_targets(args, device, rank, world_size, logger)
    if rank == 0:
        target_matrix = torch.stack([t["embedding"] for t in all_targets]).to(device)
    else:
        emb_dim = all_targets[0]["embedding"].shape[0]
        target_matrix = torch.empty((len(all_targets), emb_dim), device=device)

    dist.broadcast(target_matrix, src=0)
    # target_matrix = target_matrix.to(device)


    emb, labels, label_to_index, queries = extract_audio_queries(args, device, rank, world_size, logger)
    args.audio_model_info = (emb, labels, label_to_index)

    for name, lora_path in LORA_VARIANTS.items():
        evaluate_lora_variant(name, lora_path, args, queries, all_targets, target_matrix, out_base, logger, device, rank, world_size)
    
    if rank == 0:
        # Count total examples per modality
        modality_counts = defaultdict(int)
        for t in all_targets:
            modality_counts[t["modality"].value] += 1

        logger.info("\n=== Total Target Examples Per Modality ===")
        for mod, count in sorted(modality_counts.items()):
            logger.info(f"{mod}: {count} examples")

        logger.info("\n=== Final Global Recall@5 Summary Across Variants ===")
        summary = {}
        for name in LORA_VARIANTS.keys():
            path = os.path.join(out_base, f"{name}_recall_summary.json")
            if not os.path.exists(path):
                logger.warning(f"Missing summary file: {path}")
                continue

            data = json.load(open(path))
            total_hits = sum(d["hits"] for d in data)
            total_queries = sum(d["total"] for d in data)
            summary[name] = {
                "total_hits": total_hits,
                "total_queries": total_queries,
                "recall_at_5": round(total_hits / total_queries, 4) if total_queries else 0.0
            }

        for name, stats in summary.items():
            logger.info(f"[{name}] Recall@5 = {stats['total_hits']} / {stats['total_queries']} = {stats['recall_at_5']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--val_json_audio", required=True)
    parser.add_argument("--val_json_image", required=True)
    parser.add_argument("--val_json_event", required=True)
    parser.add_argument("--val_json_point", required=True)
    parser.add_argument("--center_emb_audio", required=True)
    parser.add_argument("--center_emb_image", required=True)
    parser.add_argument("--center_emb_event", required=True)
    parser.add_argument("--center_emb_point", required=True)
    parser.add_argument("--label_map", required=True)
    parser.add_argument("--pretrain_weights", required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    args = parser.parse_args()

    args.val_jsons = {
        Modality.IMAGE: args.val_json_image,
        Modality.EVENT: args.val_json_event,
        Modality.POINT: args.val_json_point,
        Modality.AUDIO: args.val_json_audio
    }
    args.center_embs = {
        Modality.IMAGE: args.center_emb_image,
        Modality.EVENT: args.center_emb_event,
        Modality.POINT: args.center_emb_point,
        Modality.AUDIO: args.center_emb_audio
    }

    device, rank, world_size = setup_ddp()
    run(args, device, rank, world_size)
    dist.barrier()
    dist.destroy_process_group()
