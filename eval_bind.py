import argparse
import os
import torch
import logging
import torch.distributed as dist
from datetime import datetime
import gc
import json

from model import UniBindClassifier, LanguageBindClassifier, ImageBindClassifier, Modality
from eval import evaluate_clean, evaluate_two_stage
from data_util import (
    load_label_mapping,
    val_data_loader,
    get_normalization_tensors
)
from shared_types import BindModelType

def setup_logger(rank, output_path):
    logger = logging.getLogger(f"EvalLogger-Rank{rank}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[RANK {rank}] %(asctime)s - %(message)s")

    file_handler = logging.FileHandler(output_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def normalize_label(label: str) -> str:
    return (
        label.replace("_", " ")
             .replace("-", " ")
             .replace("/", " or ")
             .strip()
    )


def build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, use_lora=False):
    logger.info(f"Building {args.model_type.value.upper()} model ...")
    if args.model_type == BindModelType.UNIBIND:
        return UniBindClassifier(
            device=device,
            pretrain_weights=args.pretrain_weights,
            modality=args.modality,
            centre_embeddings=raw_emb,
            centre_labels=raw_lbls,
            label_to_index=lbl_to_idx,
            logger=logger,
            use_flash_attention=args.use_flash_attention,
            use_lora=use_lora
        ).to(device)
    
    with open(args.classes_json, "r") as f:
        class_strings = json.load(f)

    class_strings = [normalize_label(lbl) for lbl in class_strings]
    if args.model_type == BindModelType.LANGUAGEBIND:
        return LanguageBindClassifier(
            device=device,
            modality=args.modality,
            class_strings=class_strings,
            logger=logger
        ).to(device)

    if args.model_type == BindModelType.IMAGEBIND:
        return ImageBindClassifier(
            device=device,
            modality=args.modality,
            class_strings=class_strings,
            logger=logger
        ).to(device)

    raise ValueError(f"Unsupported model type: {args.model_type}")

def evaluate_all_models(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()

    args.output_dir = os.path.join(args.output_dir, "eval", args.modality.value, args.dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.output_dir, f"rank{rank}_{timestamp}.log")

    logger = setup_logger(rank, log_path)
    logger.info(f"Evaluating {args.model_type.value.upper()} on {args.modality.value.upper()} / {args.dataset_name.upper()} epsilons={args.epsilons}")

    raw_emb, raw_lbls, lbl_to_idx, _ = load_label_mapping(args.center_emb, device)

    clean_loader = None
    model = build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx)

    if args.run_clean_eval:
        clean_loader = val_data_loader(
            modality=args.modality,
            dataset_root=args.dataset_root,
            val_json=args.clean_val_json,
            label_to_index=lbl_to_idx,
            batch_size=args.clean_val_batch_size,
            num_workers=args.num_workers,
            max_samples=args.clean_val_max_samples,
            model_type=args.model_type
        )

    final_results = []

    logger.info("Evaluating original model ...")

    if args.run_clean_eval:
        acc = evaluate_clean(logger, device, model, clean_loader)
        entry = f"[ORIGINAL] Clean acc = {acc:.4f}"
        if rank == 0:
            final_results.append(entry)
        
        logger.info(entry)

    eps_list = [float(e.strip()) / 255.0 for e in args.epsilons.split(",")]
    attack_loader = val_data_loader(
        modality=args.modality,
        dataset_root=args.dataset_root,
        val_json=args.attack_val_json,
        label_to_index=lbl_to_idx,
        batch_size=args.attack_val_batch_size,
        num_workers=args.num_workers,
        max_samples=args.attack_val_max_samples,
        model_type=args.model_type
    )
    mean, std = get_normalization_tensors(args.modality, device, model_type=args.model_type)
    for eps in eps_list:
        acc = evaluate_two_stage(
            logger, device, model, attack_loader,
            attack_loss_type=args.val_attack_loss,
            iteration_count=args.two_stage_iters,
            epsilon=eps,
            mean=mean,
            std=std,
        )
        entry = f"[ORIGINAL] Robust acc @ eps={eps*255:.0f}/255 = {acc:.4f}"
        logger.info(entry)
        if rank == 0:
            final_results.append(entry)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if args.model_type == BindModelType.UNIBIND:
        model_attack = build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, use_lora=True)
        for lora_path in args.lora_weights_list:
            logger.info(f"Evaluating robust model: {lora_path}")
            model_attack.load_lora_weights(lora_path)

            if args.run_clean_eval:
                acc = evaluate_clean(logger, device, model_attack, clean_loader)
                entry = f"[{os.path.basename(lora_path)}] Clean acc = {acc:.4f}"
                logger.info(entry)
                if rank == 0:
                    final_results.append(entry)

            for eps in eps_list:
                acc = evaluate_two_stage(
                    logger, device, model_attack, attack_loader,
                    attack_loss_type=args.val_attack_loss,
                    iteration_count=args.two_stage_iters,
                    epsilon=eps,
                    mean=mean,
                    std=std
                )
                entry = f"[{os.path.basename(lora_path)}] Robust acc @ eps={eps*255:.0f}/255 = {acc:.4f}"
                logger.info(entry)
                if rank == 0:
                    final_results.append(entry)

    all_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_results, final_results)

    if rank == 0:
        logger.info("Final results:")
        for rank_results in all_results:
            for line in rank_results:
                logger.info(line)

        result_path = os.path.join(args.output_dir, f"results_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved results to {result_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation for UniBind, LanguageBind, and ImageBind")
    parser.add_argument("--model_type", required=True, choices=[e.value for e in BindModelType])
    parser.add_argument("--modality", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--clean_val_json", required=True)
    parser.add_argument("--attack_val_json", required=True)
    parser.add_argument("--clean_val_batch_size", type=int, default=64)
    parser.add_argument("--attack_val_batch_size", type=int, default=64)
    parser.add_argument("--clean_val_max_samples", type=int, default=3000)
    parser.add_argument("--attack_val_max_samples", type=int, default=3000)

    parser.add_argument("--classes_json", required=True)
    parser.add_argument("--pretrain_weights", required=True)
    parser.add_argument("--center_emb", required=True)
    parser.add_argument("--lora_weights_list", nargs="+", default=[])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--val_attack_loss", type=str, default="ce")
    parser.add_argument("--epsilons", type=str, default="2,4")
    parser.add_argument("--run_clean_eval", action="store_true", default=False)
    parser.add_argument("--two_stage_iters", type=int, default=100)
    args = parser.parse_args()
    
    args.model_type = BindModelType(args.model_type)
    args.modality = Modality(args.modality)
    evaluate_all_models(args)