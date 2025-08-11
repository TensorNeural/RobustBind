#!/usr/bin/env python3
from math import e
import argparse
import logging
import os
import shutil
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

class RelativePathFormatter(logging.Formatter):
    def __init__(self, rank, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.rank = rank

    def format(self, record):
        run_dir = os.getcwd()
        record.rank = self.rank
        record.relativepath = os.path.relpath(os.path.abspath(record.pathname), run_dir)
        return super().format(record)


def run_alignment_training(args, device, logger, writer, train_loader, val_loader, raw_emb, raw_lbls, lbl_to_idx, epochs):
    unibind_train = UniBind(
        args=argparse.Namespace(pretrain_weights=args.pretrain_weights, modality=args.train_modality),
        use_flash_attention=args.use_flash_attention,
        use_lora=False,
        use_modality_head_mlp=True,
        modality_head_mlp_weights=args.align_modality_head_mlp_weights,
        logger=logger
    ).to(device)

    unibind_train.enable_modality_head_mlp()
    unibind_train = DDP(unibind_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

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

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_mlp_{args.model_type}.pt")
        if dist.get_rank() == 0:
            unibind_train.module.save_modality_head_mlp_weights(ckpt_path)

        torch.cuda.empty_cache()
        dist.barrier()

        model_val.load_modality_head_mlp_weights(ckpt_path)
        acc = evaluate_alignment_acc(logger, device, model_val, val_loader)
        logger.info(f"[Align][Epoch {epoch + 1}] Acc on {args.val_modality} = {acc:.4f}")

        if dist.get_rank() == 0 and acc > best_acc:
            best_acc = acc
            best_ckpt_path = os.path.join(args.output_dir, f"best_mlp_weights_{args.model_type}.pt")
            shutil.copyfile(ckpt_path, best_ckpt_path)
            logger.info(f"[Align] Best checkpoint saved: {best_ckpt_path}")

    writer.close()

    if dist.get_rank() == 0:
        logger.info(f"[Align] Best alignment accuracy: {best_acc:.4f}")

def run_robust_training(args, device, logger, writer, train_loader, val_loader, raw_emb, raw_lbls, lbl_to_idx, epochs):
    train_mean, train_std = get_normalization_tensors(args.train_modality, device)
    val_mean, val_std = get_normalization_tensors(args.val_modality, device)

    model_original = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.train_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
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
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=args.robust_use_lora,
        lora_rank=args.robust_lora_rank,
        lora_alpha=args.robust_lora_alpha,
        use_modality_head_mlp=args.robust_use_modality_head_mlp,
        modality_head_mlp_weights=args.robust_modality_head_mlp_weights
    ).to(device)
    model_train = DDP(model_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    if args.robust_training_mode == "full_fine_tune":
        logger.info("[Robust][Mode] Full fine-tuning: unfreezing backbone (MLPs remain frozen)")
        model_train.module.enable_full_fine_tune()
    elif args.robust_training_mode == "lora":
        logger.info("[Robust][Mode] LoRA training: enabling LoRA parameters only")
        model_train.module.enable_lora()

    model_val = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.val_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=args.robust_use_lora,
        lora_rank=args.robust_lora_rank,
        lora_alpha=args.robust_lora_alpha,
        use_modality_head_mlp=args.robust_use_modality_head_mlp,
        modality_head_mlp_weights=args.robust_modality_head_mlp_weights
    ).to(device)
    model_val.eval()

    train_attack = PGDAttack(
        logger,
        AttackModel(model_train, train_mean, train_std),
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
        AttackModel(model_val, val_mean, val_std),
        norm='linf',
        n_restarts=1,
        n_iter=50,
        eps=args.robust_epsilon,
        loss_type=args.robust_val_attack_loss,
        device=device,
        logger=logger
    )

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

    if dist.get_rank() == 0:
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
            std=val_std if val_std is not None and False else train_std,  # keep original if you intended train_std
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
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_backbone_{args.model_type}.pt")
            if dist.get_rank() == 0:
                model_train.module.save_backbone(ckpt_path)
        else:
            ckpt_path = None
            if args.robust_use_lora:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_lora_{args.model_type}.pt")
                if dist.get_rank() == 0:
                    model_train.module.save_lora_weights(ckpt_path)

        torch.cuda.empty_cache()
        dist.barrier()

        if args.robust_training_mode == "full_fine_tune":
            logger.info(f"[Robust][Epoch {epoch + 1}] Loading backbone weights for validation: {args.model_type}")
            model_val.load_backbone(ckpt_path)
        else:
            if args.robust_use_lora:
                logger.info(f"[Robust][Epoch {epoch + 1}] Loading LoRA weights for validation: {args.model_type}")
                model_val.load_lora_weights(ckpt_path)

        acc = evaluate_robust_one_stage(logger, device, model_val, val_loader, eval_attack, val_mean, val_std)
        logger.info(f"[Robust][Epoch {epoch + 1}] Robust Acc on {args.val_modality.upper()} = {acc:.4f}")

        if dist.get_rank() == 0 and acc > best_acc:
            best_acc = acc
            if args.robust_training_mode == "full_fine_tune":
                best_ckpt_path = os.path.join(args.output_dir, f"best_backbone_weights_{args.model_type}.pt")
                shutil.copyfile(ckpt_path, best_ckpt_path)
            else:
                if args.robust_use_lora:
                    best_ckpt_path = os.path.join(args.output_dir, f"best_lora_weights_{args.model_type}.pt")
                    shutil.copyfile(ckpt_path, best_ckpt_path)

    writer.close()
    logger.info(f"[Robust] Best robust accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser("UniBind Training")
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--training_mode", choices=["alignment", "robust"], default="robust")

    parser.add_argument("--train_modality", required=True)
    parser.add_argument("--val_modality", required=True)
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--val_dataset_name", required=True)
    parser.add_argument("--train_dataset_root", required=True)
    parser.add_argument("--val_dataset_root", required=True)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", required=True)

    parser.add_argument("--pretrain_weights", required=True)
    parser.add_argument("--center_emb", required=True)

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
    parser.add_argument("--epochs", type=int, default=2, help="Total number of training epochs")
    args = parser.parse_args()

    args.train_modality = Modality(args.train_modality)
    args.val_modality = Modality(args.val_modality)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    if args.training_mode == "robust":
        args.robust_epsilon = args.robust_epsilon / 255.0

    if args.training_mode == "alignment":
        mode_token = "align"
        train_mode_token = "mlp_align"
    else:
        eps_int = int(round(args.robust_epsilon * 255))
        mode_token = f"eps{eps_int}"
        train_mode_token = "lora_r{args.robust_lora_rank}_a{args.robust_lora_alpha}" if args.robust_training_mode == "lora" else "full_fine_tune"

    args.output_dir = os.path.join(
        args.output_dir,
        "train",
        args.model_type,
        f"{args.train_dataset_name}__{args.val_dataset_name}",
        mode_token,
        train_mode_token
    )
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.output_dir, f"rank{rank}_{timestamp}.log")
    formatter = RelativePathFormatter(rank, '[RANK %(rank)d] %(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers = [ch, fh]

    prefix = "[Align]" if args.training_mode == "alignment" else "[Robust]"
    logger.info(f"{prefix} [Train: {args.train_modality} | {args.train_dataset_name}] => [Val: {args.val_modality} | {args.val_dataset_name}]")
    if args.training_mode == "robust":
        logger.info(f"{prefix} Mode selected")
        logger.info(f"{prefix} use_lora={args.robust_use_lora} | use_full_fine_tune={args.robust_use_full_fine_tune} | epsilon={args.robust_epsilon:.5f}")
    else:
        logger.info(f"{prefix} Mode selected")

    tb_path = os.path.join(
        args.output_dir,
        args.tensorboard_data_dir,
        f"rank{rank}",
        train_mode_token,
        timestamp
    )
    writer = SummaryWriter(log_dir=tb_path)

    # Load centre embeddings and label mapping BEFORE creating data loaders so labels are indexed correctly
    raw_emb, raw_lbls, lbl_to_idx, _ = load_label_mapping(args.center_emb, device)

    train_loader = train_data_loader(
        modality=args.train_modality,
        dataset_root=args.train_dataset_root,
        train_json=args.train_json,
        label_to_index=lbl_to_idx,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        max_samples=args.train_max_samples
    )
    val_loader = val_data_loader(
        modality=args.val_modality,
        dataset_root=args.val_dataset_root,
        val_json=args.val_json,
        label_to_index=lbl_to_idx,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_samples=args.val_max_samples
    )

    if args.training_mode == "alignment":
        run_alignment_training(args, device, logger, writer, train_loader, val_loader, raw_emb, raw_lbls, lbl_to_idx, args.epochs)
    else:
        run_robust_training(args, device, logger, writer, train_loader, val_loader, raw_emb, raw_lbls, lbl_to_idx, args.epochs)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
