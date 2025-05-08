import argparse
import os
import torch
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from model import UniBindModel
from eval import evaluate_clean, evaluate_two_stage
from data_util import (
    load_label_mapping,
    val_data_loader,
    get_normalization_tensors
)


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


def build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, idx_to_lbl, lora_weights=None):
    model = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=(lora_weights is not None),
        lora_weights=lora_weights
    )
    model.to(device)
    return model


def evaluate_all_models(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    args.output_dir = os.path.join(args.output_dir, args.modality, args.dataset_name, "eval")
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.output_dir, f"rank{rank}_{timestamp}.log")

    logger = setup_logger(rank, log_path)
    logger.info(f"Evaluating {args.modality.upper()} on {args.dataset_name.upper()}")

    raw_emb, raw_lbls, lbl_to_idx, idx_to_lbl = load_label_mapping(args.center_emb, device)

    val_loader = val_data_loader(
        modality=args.modality,
        dataset_root=args.dataset_root,
        val_json=args.val_json,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_samples=args.val_max_samples
    )

    mean, std = get_normalization_tensors(args.modality, device)
    eps_list = [float(e.strip()) / 255.0 for e in args.epsilons.split(",")]

    # Original model (no LoRA)
    logger.info("Evaluating original (pretrained-only) model ...")
    model = build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, idx_to_lbl)

    if args.run_clean_eval:
        acc = evaluate_clean(logger, device, model, val_loader)
        logger.info(f"[ORIGINAL] Clean acc = {acc:.4f}")

    for eps in eps_list:
        acc = evaluate_two_stage(
            logger, device, model, val_loader,
            attack_loss_type=args.val_attack_loss,
            iteration_count=args.two_stage_iters,
            epsilon=eps,
            mean=mean,
            std=std
        )
        logger.info(f"[ORIGINAL] Robust acc @ eps={eps*255:.0f}/255 = {acc:.4f}")

    # # Robust models (LoRA)
    # for lora_path in args.lora_weights_list:
    #     logger.info(f"Evaluating robust model: {lora_path}")
    #     model = build_model(args, device, logger, raw_emb, raw_lbls, lbl_to_idx, idx_to_lbl, lora_weights=lora_path)

    #     if args.run_clean_eval:
    #         acc = evaluate_clean(logger, device, model, val_loader)
    #         logger.info(f"[{os.path.basename(lora_path)}] Clean acc = {acc:.4f}")

    #     for eps in eps_list:
    #         acc = evaluate_two_stage(
    #             logger, device, model, val_loader,
    #             attack_loss_type=args.val_attack_loss,
    #             iteration_count=args.two_stage_iters,
    #             epsilon=eps,
    #             mean=mean,
    #             std=std
    #         )
    #         logger.info(f"[{os.path.basename(lora_path)}] Robust acc @ eps={eps*255:.0f}/255 = {acc:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation script for UniBind models")
    parser.add_argument("--modality", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--pretrain_weights", required=True)
    parser.add_argument("--center_emb", required=True)
    parser.add_argument("--lora_weights_list", nargs="+", default=[])
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_max_samples", type=int, default=3000)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--val_attack_loss", type=str, default="ce")
    parser.add_argument("--epsilons", type=str, default="2,4", help="Comma-separated epsilon values (e.g. 2,4)")
    parser.add_argument("--run_clean_eval", action="store_true", default=False)
    parser.add_argument("--two_stage_iters", type=int, default=100)
    args = parser.parse_args()

    evaluate_all_models(args)
