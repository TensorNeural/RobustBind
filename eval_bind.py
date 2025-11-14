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


def build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, use_lora=False, align_enabled=False):
    """Factory that builds the requested model.

    align_enabled controls the UniBind modality head MLP usage for this specific instantiation,
    allowing us to evaluate BOTH aligned (+ALIGN) and non-aligned variants in a single run.
    """
    logger.info(f"Building {args.model_type.value.upper()} model (align={align_enabled}) ...")
    if args.model_type == BindModelType.UNIBIND:
        # Best-effort: only load modality head MLP weights if provided and exists
        mlp_weights = args.modality_head_mlp_weights if align_enabled else None
        if mlp_weights is not None and not os.path.isfile(mlp_weights):
            logger.warning(f"Modality head MLP weights not found at '{mlp_weights}'. Proceeding without loading.")
            mlp_weights = None

        return UniBindClassifier(
            device=device,
            pretrain_weights=args.pretrain_weights,
            modality=args.modality,
            centre_embeddings=raw_emb,
            centre_labels=raw_lbls,
            label_to_index=lbl_to_idx,
            logger=logger,
            use_flash_attention=args.use_flash_attention,
            use_lora=use_lora,
            use_modality_head_mlp=align_enabled,
            modality_head_mlp_weights=mlp_weights
        ).to(device)

    # Non-UniBind models ignore alignment completely.
    if args.model_type == BindModelType.LANGUAGEBIND:
        return LanguageBindClassifier(
            device=device,
            modality=args.modality,
            class_strings=None,
            logger=logger,
            label_to_index=lbl_to_idx
        ).to(device)
    if args.model_type == BindModelType.IMAGEBIND:
        return ImageBindClassifier(
            device=device,
            modality=args.modality,
            class_strings=None,
            logger=logger,
            label_to_index=lbl_to_idx
        ).to(device)
    raise ValueError(f"Unsupported model type: {args.model_type}")

def evaluate_all_models(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()

    # Determine session timestamp: prefer value passed from the shell script for consistent session-wide pathing
    session_ts = args.session_timestamp if getattr(args, "session_timestamp", None) else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Output directory structure: <output_dir>/eval/<YYYY-MM-DD_HH-MM-SS>/<modality>/<dataset>/<model>
    args.output_dir = os.path.join(
        args.output_dir,
        "eval",
        session_ts,
        args.modality.value,
        args.dataset_name,
        args.model_type.value,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # Log file name without timestamp (timestamp captured in directory path)
    log_path = os.path.join(args.output_dir, f"rank{rank}.log")

    logger = setup_logger(rank, log_path)
    logger.info(f"Evaluating {args.model_type.value.upper()} on {args.modality.value.upper()} / {args.dataset_name.upper()} epsilons={args.epsilons}")

    raw_emb, raw_lbls, lbl_to_idx, _ = load_label_mapping(args.center_emb, device)

    clean_loader = None
    # Prepare data loaders (shared across alignment settings)
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
    # Parse epsilons, skipping empty tokens (e.g., trailing commas or empty string)
    _eps_tokens = [e.strip() for e in str(args.epsilons or "").split(",") if e.strip() != ""]
    eps_list = [float(e) / 255.0 for e in _eps_tokens]
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

    # Alignment is controlled by CLI (--use_modality_head_mlp) for UniBind; ignored for others
    align_enabled = bool(args.use_modality_head_mlp) if args.model_type == BindModelType.UNIBIND else False
    align_suffix = " +ALIGN" if (args.model_type == BindModelType.UNIBIND and align_enabled) else ""

    if not args.skip_original:
        logger.info(f"Evaluating original model (align={align_enabled}) ...")
        model = build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, align_enabled=align_enabled)
        if args.run_clean_eval:
            acc = evaluate_clean(logger, device, model, clean_loader)
            entry = f"[ORIGINAL{align_suffix}] Clean acc = {acc:.4f}"
            if rank == 0:
                final_results.append(entry)
            logger.info(entry)
        for eps in eps_list:
            acc = evaluate_two_stage(
                logger, device, model, attack_loader,
                attack_loss_type=args.val_attack_loss,
                iteration_count=args.two_stage_iters,
                epsilon=eps,
                mean=mean,
                std=std,
            )
            entry = f"[ORIGINAL{align_suffix}] Robust acc @ eps={eps*255:.0f}/255 = {acc:.4f}"
            logger.info(entry)
            if rank == 0:
                final_results.append(entry)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # LoRA / robust variants (only for UniBind)
    if args.model_type == BindModelType.UNIBIND and len(args.lora_weights_list) > 0:
        for lora_path in args.lora_weights_list:
            logger.info(f"Evaluating robust LoRA model: {lora_path} (align={align_enabled})")
            model_attack = build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, use_lora=True, align_enabled=align_enabled)
            model_attack.load_lora_weights(lora_path)
            tag = os.path.basename(lora_path)
            if args.run_clean_eval:
                acc = evaluate_clean(logger, device, model_attack, clean_loader)
                entry = f"[{tag}{align_suffix}] Clean acc = {acc:.4f}"
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
                entry = f"[{tag}{align_suffix}] Robust acc @ eps={eps*255:.0f}/255 = {acc:.4f}"
                logger.info(entry)
                if rank == 0:
                    final_results.append(entry)
            del model_attack
            torch.cuda.empty_cache()
            gc.collect()

    all_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_results, final_results)

    if rank == 0:
        logger.info("Final results:")
        for rank_results in all_results:
            for line in rank_results:
                logger.info(line)

        # Save results without timestamp in filename (timestamp in directory)
        result_path = os.path.join(args.output_dir, "results.json")
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved results to {result_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation for UniBind, LanguageBind, and ImageBind")
    parser.add_argument("--model_type", required=True, choices=[e.value for e in BindModelType])
    parser.add_argument("--modality", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", default="/data/output")
    parser.add_argument("--session_timestamp", default=None, help="Session-wide timestamp (YYYY-MM-DD_HH-MM-SS) to group results. If omitted, a new timestamp is generated.")
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
    parser.add_argument("--use_modality_head_mlp", action="store_true", default=False)
    parser.add_argument("--modality_head_mlp_weights", default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--val_attack_loss", type=str, default="ce")
    parser.add_argument("--epsilons", type=str, default="2,4")
    parser.add_argument("--run_clean_eval", action="store_true", default=False)
    parser.add_argument("--two_stage_iters", type=int, default=100)
    parser.add_argument("--skip_original", action="store_true", default=False, help="If set, skip evaluating the original (no-LoRA) model. Useful when benchmarking specific robust LoRA weights only.")
    args = parser.parse_args()
    
    args.model_type = BindModelType(args.model_type)
    args.modality = Modality(args.modality)
    evaluate_all_models(args)