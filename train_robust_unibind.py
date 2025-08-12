#!/usr/bin/env python3
from math import e
import argparse
import logging
import os
import shutil
from datetime import datetime
import time
import json
from urllib import request as urlrequest, error as urlerror
# Discord webhook will be read from environment in main() and passed into helper.
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from shared_types import Modality
from model import UniBind, UniBindClassifier, ForwardMode
from training import train_alignment_epoch, train_robust_epoch
from eval import evaluate_alignment_acc, evaluate_robust_one_stage
from attack import PGDAttack, APGDAttack, AttackModel
from meter import AverageMeter
from data_util import (
    load_label_mapping,
    train_data_loader,
    val_data_loader,
    get_normalization_tensors
)
from ddp import ProxyDDP

class RelativePathFormatter(logging.Formatter):
    def __init__(self, rank, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.rank = rank

    def format(self, record):
        run_dir = os.getcwd()
        record.rank = self.rank
        record.relativepath = os.path.relpath(os.path.abspath(record.pathname), run_dir)
        return super().format(record)

def run_alignment_training(args, device, logger, writer, train_loader, val_loader, raw_emb, raw_lbls, lbl_to_idx, epochs, output_base):
    run_start_time = time.time()
    logger.info(f"[Align] Running alignment training with train modality: {args.train_modality}")
    unibind_train = UniBind(
        args=argparse.Namespace(pretrain_weights=args.pretrain_weights, modality=args.train_modality),
        use_flash_attention=args.use_flash_attention,
        use_lora=False,
        use_modality_head_mlp=True,
        modality_head_mlp_weights=args.align_modality_head_mlp_weights,
        logger=logger
    ).to(device)

    unibind_train.enable_modality_head_mlp()
    unibind_train = ProxyDDP(unibind_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    model_val = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.val_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=False,
        use_modality_head_mlp=True,
        modality_head_mlp_weights=args.align_modality_head_mlp_weights
    ).to(device)
    model_val.eval()

    # Log parameter counts before optimizer creation
    total_params = sum(p.numel() for p in unibind_train.parameters())
    trainable_params_count = sum(p.numel() for p in unibind_train.parameters() if p.requires_grad)
    pct_trainable = (trainable_params_count / total_params * 100.0) if total_params > 0 else 0.0
    logger.info(f"[Align] Params | trainable={trainable_params_count:,} / total={total_params:,} ({pct_trainable:.2f}% trainable)")

    trainable_params = [p for p in unibind_train.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=3e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4
    )

    best_acc = -1.0
    modality = Modality(args.train_modality)

    for epoch in range(epochs):
        logger.info(f"[Align][Epoch {epoch + 1}/{epochs}] Starting training")
        train_loader.sampler.set_epoch(epoch)

        train_alignment_epoch(
            logger=logger,
            device=device,
            model=unibind_train,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            total_epochs=epochs,
            writer=writer,
            modality=modality
        )

        ckpt_path = os.path.join(output_base, f"epoch_{epoch + 1}_mlp_{args.model_type}.pt")
        if dist.get_rank() == 0:
            unibind_train.save_modality_head_mlp_weights(ckpt_path)

        torch.cuda.empty_cache()
        dist.barrier()

        model_val.load_modality_head_mlp_weights(ckpt_path)
        acc = evaluate_alignment_acc(logger, device, model_val, val_loader)
        logger.info(f"[Align][Epoch {epoch + 1}] Acc on {args.val_modality} = {acc:.4f}")
        if dist.get_rank() == 0:
            # Emit per-epoch alignment accuracy to TensorBoard
            writer.add_scalar("val/acc", acc, epoch + 1)
            writer.flush()

        if dist.get_rank() == 0 and acc > best_acc:
            best_acc = acc
            best_ckpt_path = os.path.join(output_base, f"best_mlp_weights_{args.model_type}.pt")
            shutil.copyfile(ckpt_path, best_ckpt_path)
            logger.info(f"[Align] Best checkpoint saved: {best_ckpt_path}")

    writer.close()

    if dist.get_rank() == 0:
        logger.info(f"[Align] Best alignment accuracy: {best_acc:.4f}")
    return best_acc, time.time() - run_start_time

def run_robust_training(
    args,
    device,
    logger,
    writer,
    train_loader,
    val_loader,
    train_emb,
    train_lbls,
    train_lbl_to_idx,
    val_emb,
    val_lbls,
    val_lbl_to_idx,
    epochs,
    output_base
):
    run_start_time = time.time()
    logger.info(f"[Robust] Running robust training with train modality: {args.train_modality}")
    if args.robust_use_modality_head_mlp:
        logger.info(f"[Robust] Modality head MLP requested. Weights path: {args.robust_modality_head_mlp_weights}")
        
    train_mean, train_std = get_normalization_tensors(args.train_modality, device)
    val_mean, val_std = get_normalization_tensors(args.val_modality, device)

    use_lora = (args.robust_training_mode == "lora")

    model_original = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.train_modality,
        centre_embeddings=train_emb,
        centre_labels=train_lbls,
        label_to_index=train_lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=False,
        use_modality_head_mlp=args.robust_use_modality_head_mlp,
        modality_head_mlp_weights=args.robust_modality_head_mlp_weights
    ).to(device)
    model_original.eval()

    model_train = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.train_modality,
        centre_embeddings=train_emb,
        centre_labels=train_lbls,
        label_to_index=train_lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=use_lora,
        lora_rank=args.robust_lora_rank,
        lora_alpha=args.robust_lora_alpha,
        use_modality_head_mlp=args.robust_use_modality_head_mlp,
        modality_head_mlp_weights=args.robust_modality_head_mlp_weights
    ).to(device)

    if args.robust_training_mode == "full_fine_tune":
        logger.info("[Robust][Mode] Full fine-tuning: unfreezing backbone (MLPs remain frozen)")
        model_train.enable_full_fine_tune()
    else:
        logger.info("[Robust][Mode] LoRA training: enabling LoRA parameters only")
        model_train.enable_lora()

    model_val = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.val_modality,
        centre_embeddings=val_emb,
        centre_labels=val_lbls,
        label_to_index=val_lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=use_lora,
        lora_rank=args.robust_lora_rank,
        lora_alpha=args.robust_lora_alpha,
        use_modality_head_mlp=args.robust_use_modality_head_mlp,
        modality_head_mlp_weights=args.robust_modality_head_mlp_weights
    ).to(device)
    model_val.eval()

    train_attack = PGDAttack(
        logger=logger,
        model=AttackModel(model_train, train_mean, train_std),
        epsilon=args.robust_epsilon,
        alpha=1 / 255,
        steps=10,
        norm='linf',
        random_start=True,
        clamp_min=0.0,
        clamp_max=1.0,
        loss_type=args.robust_train_attack_loss
    )
    eval_attack = APGDAttack(
        logger=logger,
        model=AttackModel(model_val, val_mean, val_std),
        norm='linf',
        n_restarts=1,
        n_iter=50,
        eps=args.robust_epsilon,
        loss_type=args.robust_val_attack_loss,
        device=device,
    )

    model_train = ProxyDDP(model_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    # Log parameter counts before optimizer creation
    total_params = sum(p.numel() for p in model_train.parameters())
    trainable_params_count = sum(p.numel() for p in model_train.parameters() if p.requires_grad)
    pct_trainable = (trainable_params_count / total_params * 100.0) if total_params > 0 else 0.0
    logger.info(f"[Robust] Params | trainable={trainable_params_count:,} / total={total_params:,} ({pct_trainable:.2f}% trainable)")

    params = [p for p in model_train.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=3e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4
    )

    best_acc = -1.0
    meters = {k: AverageMeter() for k in ["loss", "cos_sim", "rcos_sim", "acc", "racc"]}

    if dist.get_rank() == 0 and use_lora:
        writer.add_text("config/lora_rank", str(args.robust_lora_rank))
        writer.add_text("config/lora_alpha", str(args.robust_lora_alpha))
        writer.add_scalar("config/lora_rank", args.robust_lora_rank, 0)
        writer.add_scalar("config/lora_alpha", args.robust_lora_alpha, 0)

    for epoch in range(epochs):
        logger.info(f"[Robust][Epoch {epoch + 1}/{epochs}] Starting training")
        train_loader.sampler.set_epoch(epoch)

        train_robust_epoch(
            logger=logger,
            device=device,
            model_train=model_train,
            model_original=model_original,
            mean=train_mean,
            std=val_std if val_std is not None and False else train_std,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            attack=train_attack,
            train_loss_type=args.robust_train_loss,
            epoch=epoch,
            total_epochs=epochs,
            loss_meter=meters["loss"],
            cos_sim_meter=meters["cos_sim"],
            rcos_sim_meter=meters["rcos_sim"],
            acc_meter=meters["acc"],
            racc_meter=meters["racc"],
            writer=writer
        )

        if args.robust_training_mode == "full_fine_tune":
            ckpt_path = os.path.join(output_base, f"epoch_{epoch + 1}_backbone_{args.model_type}.pt")
            if dist.get_rank() == 0:
                model_train.save_backbone(ckpt_path)
        else:
            ckpt_path = os.path.join(output_base, f"epoch_{epoch + 1}_lora_{args.model_type}.pt")
            if dist.get_rank() == 0:
                model_train.save_lora_weights(ckpt_path)

        torch.cuda.empty_cache()
        dist.barrier()

        if args.robust_training_mode == "full_fine_tune":
            logger.info(f"[Robust][Epoch {epoch + 1}] Loading backbone weights for validation: {args.model_type}")
            model_val.load_backbone(ckpt_path)
        else:
            logger.info(f"[Robust][Epoch {epoch + 1}] Loading LoRA weights for validation: {args.model_type}")
            model_val.load_lora_weights(ckpt_path)

        acc = evaluate_robust_one_stage(logger, device, model_val, val_loader, eval_attack, val_mean, val_std)
        logger.info(f"[Robust][Epoch {epoch + 1}] Robust Acc on {args.val_modality} = {acc:.4f}")
        if dist.get_rank() == 0:
            # Emit per-epoch robust accuracy to TensorBoard
            writer.add_scalar("val/robust_acc", acc, epoch + 1)
            writer.flush()

        if dist.get_rank() == 0 and acc > best_acc:
            best_acc = acc
            if args.robust_training_mode == "full_fine_tune":
                best_ckpt_path = os.path.join(output_base, f"best_backbone_weights_{args.model_type}.pt")
                shutil.copyfile(ckpt_path, best_ckpt_path)
            else:
                best_ckpt_path = os.path.join(output_base, f"best_lora_weights_{args.model_type}.pt")
                shutil.copyfile(ckpt_path, best_ckpt_path)

    writer.close()
    logger.info(f"[Robust] Best robust accuracy: {best_acc:.4f}")
    return best_acc, time.time() - run_start_time


def _post_discord(logger: logging.Logger, webhook_url: str, content: str) -> None:
    """Post a simple message to Discord via provided webhook URL.
    Retries with exponential backoff until success. No-ops if webhook is empty/None.
    """
    if not webhook_url:
        return

    webhook_url = webhook_url.strip()
    backoff = 1.0
    attempt = 0
    while True:
        attempt += 1
        try:
            data = json.dumps({"content": content}).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "RobustBind/1.0"
            }
            req = urlrequest.Request(webhook_url, data=data, headers=headers)
            opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
            with opener.open(req, timeout=10) as _:
                return
        except Exception as e:
            logger.warning(f"[Notify] Discord post failed (attempt {attempt}): {e}. Retrying in {backoff:.1f}s…")
            time.sleep(backoff)
            backoff *= 2.0


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def auto_discover_alignment_mlp(args, session_output_dir: Path, logger: logging.Logger) -> bool:
    if args.training_mode != "robust":
        return False
    
    if args.robust_modality_head_mlp_weights:
        logger.info(f"[AutoLoad][Robust] Using user-provided modality head MLP weights: {args.robust_modality_head_mlp_weights}")
        return True
    
    candidate = session_output_dir / "align" / args.train_modality.value / f"best_mlp_weights_{args.model_type}.pt"
    if candidate.exists():
        args.robust_modality_head_mlp_weights = str(candidate)
        if not args.robust_use_modality_head_mlp:
            args.robust_use_modality_head_mlp = True
            logger.info(f"[AutoLoad][Robust] Loaded and enabled modality head MLP weights: {candidate}")
        else:
            logger.info(f"[AutoLoad][Robust] Loaded modality head MLP weights: {candidate}")
        return True
    
    logger.info("[AutoLoad][Robust] No alignment MLP weights loaded (none present).")
    return False


def main():
    parser = argparse.ArgumentParser("UniBind Training")
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--training_mode", choices=["alignment", "robust"], default="robust")

    parser.add_argument("--pretrain_weights", required=True)
    parser.add_argument("--train_center_emb", default=None)
    parser.add_argument("--val_center_emb", default=None)

    parser.add_argument("--train_modality", required=True)
    parser.add_argument("--val_modality", required=True)
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--val_dataset_name", required=True)
    parser.add_argument("--train_dataset_root", required=True)
    parser.add_argument("--val_dataset_root", required=True)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", required=True)

    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_max_samples", type=int, default=10)
    parser.add_argument("--val_max_samples", type=int, default=10)

    parser.add_argument("--use_flash_attention", action="store_true", default=False)

    parser.add_argument("--align_modality_head_mlp_weights", default=None)

    parser.add_argument("--robust_train_attack_loss", default="l2")
    parser.add_argument("--robust_val_attack_loss", default="ce")
    parser.add_argument("--robust_train_loss", default="l2")

    parser.add_argument("--robust_lora_rank", type=int, default=4)
    parser.add_argument("--robust_lora_alpha", type=float, default=8)
    parser.add_argument("--robust_epsilon", type=int, default=4)

    parser.add_argument("--robust_training_mode", choices=["lora", "full_fine_tune"], default="lora", help="Mode for robust training")

    parser.add_argument("--robust_use_modality_head_mlp", action="store_true", default=False)
    parser.add_argument("--robust_modality_head_mlp_weights", default=None)

    parser.add_argument("--tensorboard_data_dir", default="tensorboard")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--session_output_dir", default=None)
    parser.add_argument("--session_timestamp", default=None)
    parser.add_argument("--tensorboard_root", default=None)
    parser.add_argument("--epochs", type=int, default=2, help="Total number of training epochs")
    args = parser.parse_args()

    if args.val_center_emb is None:
        raise ValueError("Must provide --val_center_emb")

    args.train_modality = Modality(args.train_modality)
    args.val_modality = Modality(args.val_modality)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    if args.training_mode == "robust":
        # Keep integer epsilon for display and normalize for computation
        args.robust_epsilon_int = int(args.robust_epsilon)
        args.robust_epsilon = args.robust_epsilon_int / 255.0

    session_output_dir = Path(args.session_output_dir)

    # Build run-specific subdirectory under session_output_dir
    if args.training_mode == "alignment":
        subdir = f"align/{args.train_modality.value}"
    else:
        if args.robust_training_mode == "lora":
            subdir = f"robust/{args.train_modality.value}/lora_r{args.robust_lora_rank}_a{args.robust_lora_alpha}_{args.robust_epsilon}"
        else:
            subdir = f"robust/{args.train_modality.value}/full_fine_tune_{args.robust_epsilon}"

    run_output_base = session_output_dir / subdir
    run_output_base.mkdir(parents=True, exist_ok=True)

    # Logs go under run_output_base
    log_path = os.path.join(run_output_base, f"rank{rank}.log")
    formatter = RelativePathFormatter(rank, '[RANK %(rank)d] %(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers = [ch, fh]

    # Read Discord webhook once and reuse
    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")

    if args.training_mode == "robust":
        # Attempt autoload now that logger is configured
        # Auto discover alignment MLP (if any) after logger setup
        found_mlp = auto_discover_alignment_mlp(args, session_output_dir, logger)
        if args.robust_use_modality_head_mlp and not found_mlp and not args.robust_modality_head_mlp_weights:
            logger.warning("[Robust] Modality head MLP requested but no weights found. Proceeding with randomly initialized head.")

    # Load val centers (required)
    train_emb, train_lbls, train_lbl_to_idx = None, None, None
    val_emb, val_lbls, val_lbl_to_idx, _ = load_label_mapping(args.val_center_emb, device)

    # Load train centers if provided; else reuse val centers
    if args.train_center_emb:
        train_emb, train_lbls, train_lbl_to_idx, _ = load_label_mapping(args.train_center_emb, device)
        logger.info(f"[Centers] Train labels: {len(train_lbls)} | Val labels: {len(val_lbls)}")
    else:
        logger.info(f"[Centers] Val labels: {len(val_lbls)}")

    prefix = "[Align]" if args.training_mode == "alignment" else "[Robust]"
    logger.info(f"{prefix} [Train: {args.train_modality} | {args.train_dataset_name}] => [Val: {args.val_modality} | {args.val_dataset_name}]")
    if args.training_mode == "robust":
        logger.info(f"{prefix} Mode selected: {args.robust_training_mode}")
        logger.info(f"{prefix} epsilon={args.robust_epsilon_int}/255")
        if args.robust_training_mode == "lora":
            logger.info(f"{prefix} lora_rank={args.robust_lora_rank} lora_alpha={args.robust_lora_alpha}")
    else:
        logger.info(f"{prefix} Mode selected")

    tb_root = Path(args.tensorboard_root)
    tb_path = tb_root / str(args.session_timestamp) / subdir / f"rank{rank}"
    logger.info(f"[TensorBoard] Log directory: {tb_path}")

    tb_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_path))

    # Build loaders with separate label mappings
    train_loader = train_data_loader(
        modality=args.train_modality,
        dataset_root=args.train_dataset_root,
        train_json=args.train_json,
        label_to_index=train_lbl_to_idx,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        max_samples=args.train_max_samples
    )
    val_loader = val_data_loader(
        modality=args.val_modality,
        dataset_root=args.val_dataset_root,
        val_json=args.val_json,
        label_to_index=val_lbl_to_idx,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_samples=args.val_max_samples
    )

    # Notify start (rank 0 only)
    if dist.get_rank() == 0:
        if args.training_mode == "alignment":
            content = (
                "🚀 Training started\n"
                f"• **Mode:** `alignment`\n"
                f"• **Session:** `{args.session_timestamp}`\n"
                f"• **Modality:** `{args.train_modality.value}`\n"
                f"• **Train dataset:** `{args.train_dataset_name}`\n"
                f"• **Val dataset:** `{args.val_dataset_name}`\n"
                f"• **Epochs:** `{args.epochs}`\n"
            )
        else:
            lora_lines = (
                f"• **LoRA rank:** `{args.robust_lora_rank}`\n"
                f"• **LoRA alpha:** `{args.robust_lora_alpha}`\n"
            ) if args.robust_training_mode == "lora" else ""

            content = (
                "🚀 Training started\n"
                f"• **Mode:** `robust/{args.robust_training_mode}`\n"
                f"• **Session:** `{args.session_timestamp}`\n"
                f"• **Modality:** `{args.train_modality.value}`\n"
                f"• **Train dataset:** `{args.train_dataset_name}`\n"
                f"• **Val dataset:** `{args.val_dataset_name}`\n"
                f"• **Epochs:** `{args.epochs}`\n"
                f"• **Epsilon:** `{args.robust_epsilon_int}/255`\n"
                f"{lora_lines}"
            )
        _post_discord(logger, discord_webhook, content)

    run_wall_start = time.time()

    if args.training_mode == "alignment":
        logger.info(f"[Align] {args.train_modality} | Train dataset: {args.train_dataset_name} | Val dataset: {args.val_dataset_name} | Epochs: {args.epochs}")
        best_acc, elapsed = run_alignment_training(
            args,
            device,
            logger,
            writer,
            train_loader,
            val_loader,
            val_emb,
            val_lbls,
            val_lbl_to_idx,
            args.epochs,
            run_output_base
        )
    else:
        logger.info(f"[Robust] ({args.robust_training_mode}) {args.train_modality} | Train dataset: {args.train_dataset_name} | Val dataset: {args.val_dataset_name} | Epsilon: {args.robust_epsilon_int}/255 | Epochs: {args.epochs}")
        if args.robust_training_mode == "lora":
            logger.info(f"[Robust] (LoRA) rank: {args.robust_lora_rank}, alpha: {args.robust_lora_alpha}")

        best_acc, elapsed = run_robust_training(
            args,
            device,
            logger,
            writer,
            train_loader,
            val_loader,
            train_emb,
            train_lbls,
            train_lbl_to_idx,
            val_emb,
            val_lbls,
            val_lbl_to_idx,
            args.epochs,
            run_output_base
        )

    # Notify finish (rank 0 only)
    if dist.get_rank() == 0:
        wall_elapsed = time.time() - run_wall_start
        dur = _fmt_duration(wall_elapsed)
        acc_str = f"{best_acc:.4f}" if isinstance(best_acc, (int, float)) else str(best_acc)
        if args.training_mode == "alignment":
            content = (
                "✅ Training finished\n"
                f"• **Mode:** `alignment`\n"
                f"• **Session:** `{args.session_timestamp}`\n"
                f"• **Modality:** `{args.train_modality.value}`\n"
                f"• **Train dataset:** `{args.train_dataset_name}`\n"
                f"• **Val dataset:** `{args.val_dataset_name}`\n"
                f"• **Epochs:** `{args.epochs}`\n"
                f"• **Accuracy:** `{acc_str}`\n"
                f"• **Duration:** `{dur}`\n"
            )
        else:
            lora_lines = (
                f"• **LoRA rank:** `{args.robust_lora_rank}`\n"
                f"• **LoRA alpha:** `{args.robust_lora_alpha}`\n"
            ) if args.robust_training_mode == "lora" else ""
            content = (
                "✅ Training finished\n"
                f"• **Mode:** `robust/{args.robust_training_mode}`\n"
                f"• **Session:** `{args.session_timestamp}`\n"
                f"• **Modality:** `{args.train_modality.value}`\n"
                f"• **Train dataset:** `{args.train_dataset_name}`\n"
                f"• **Val dataset:** `{args.val_dataset_name}`\n"
                f"• **Epochs:** `{args.epochs}`\n"
                f"• **Epsilon:** `{args.robust_epsilon_int}/255`\n"
                f"• **Accuracy:** `{acc_str}`\n"
                f"• **Duration:** `{dur}`\n"
                f"{lora_lines}"
            )
        _post_discord(logger, discord_webhook, content)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
