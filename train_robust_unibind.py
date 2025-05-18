import argparse, os, logging, shutil
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from model import UniBindModel
from training import train_epoch
from eval import evaluate_robust_one_stage
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


def train_and_evaluate(args, logger, writer, device, raw_emb, raw_lbls, lbl_to_idx, train_loader, val_loader):
    train_mean, train_std = get_normalization_tensors(args.train_modality, device)
    val_mean, val_std = get_normalization_tensors(args.val_modality, device)

    logger.info(f"Loading model for training: {args.train_modality.upper()}")
    model_original = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.train_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
    ).to(device)
    model_original.eval()

    model_train = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.train_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_fine_tune=False,
    ).to(device)

    model_train = DDP(model_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    model_val = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.val_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_fine_tune=False
    ).to(device)
    model_val.eval()

    train_attack = PGDAttack(
        logger,
        AttackModel(model_train, train_mean, train_std),
        epsilon=args.epsilon,
        alpha=1 / 255,
        steps=10,
        norm='linf',
        random_start=True,
        clamp_min=0.0,
        clamp_max=1.0,
        loss_type=args.train_attack_loss
    )

    eval_attack = APGDAttack(
        AttackModel(model_val, val_mean, val_std),
        norm='linf',
        n_restarts=1,
        n_iter=50,
        eps=args.epsilon,
        loss_type=args.val_attack_loss,
        device=device,
        logger=logger
    )

    optimizer = AdamW([p for p in model_train.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-4)
    logger.info("Steps per epoch: %d", len(train_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=2,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4
    )

    best_acc = -1.0
    meters = {k: AverageMeter() for k in ["loss", "cos_sim", "rcos_sim", "acc", "racc"]}

    if dist.get_rank() == 0:
        writer.add_text("config/lora_rank", str(args.lora_rank))
        writer.add_text("config/lora_alpha", str(args.lora_alpha))
        writer.add_scalar("config/lora_rank", args.lora_rank, 0)
        writer.add_scalar("config/lora_alpha", args.lora_alpha, 0)

    for epoch in range(2):
        logger.info(f"Epoch {epoch + 1}/2")
        train_loader.sampler.set_epoch(epoch)

        train_epoch(
            logger=logger,
            device=device,
            model_train=model_train,
            model_original=model_original,
            mean=train_mean,
            std=train_std,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            attack=train_attack,
            train_loss_type=args.train_loss,
            epoch=epoch,
            total_epochs=2,
            loss_meter=meters["loss"],
            cos_sim_meter=meters["cos_sim"],
            rcos_sim_meter=meters["rcos_sim"],
            acc_meter=meters["acc"],
            racc_meter=meters["racc"],
            writer=writer
        )

        # === Save LoRA weights (rank 0 only)
        lora_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_lora.pt")
        if dist.get_rank() == 0:
            model_train.module.save_lora_weights(lora_path)

        torch.cuda.empty_cache()
        dist.barrier()

        # === Reinitialize model for validation modality and load weights
        logger.info(f"[EPOCH {epoch + 1}] Loading LoRA weights for validation: {args.val_modality.upper()}")
        model_val.load_lora_weights(lora_path)

        acc = evaluate_robust_one_stage(logger, device, model_val, val_loader, eval_attack, val_mean, val_std)
        logger.info(f"[Epoch {epoch + 1}] robust acc on {args.val_modality.upper()} = {acc:.4f}")

        if dist.get_rank() == 0 and acc > best_acc:
            best_acc = acc
            best_lora_path = os.path.join(args.output_dir, "best_lora_weights.pt")
            shutil.copyfile(lora_path, best_lora_path)

    writer.close()
    logger.info(f"Best robust accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser("UniBind Training")
    parser.add_argument("--model_type", required=True)
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
    parser.add_argument("--train_max_samples", type=int)
    parser.add_argument("--val_max_samples", type=int, default=3000)
    parser.add_argument("--train_attack_loss", default="l2")
    parser.add_argument("--val_attack_loss", default="ce")
    parser.add_argument("--train_loss", default="l2")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--epsilon", type=int, default=4)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--tensorboard_data_dir", default="tensorboard")
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    args.output_dir = os.path.join(
        args.output_dir,
        "train",
        args.model_type,
        f"{args.train_dataset_name}__{args.val_dataset_name}",
        f"eps{args.epsilon}",
        f"lora_r{args.lora_rank}_a{args.lora_alpha}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    args.epsilon = args.epsilon / 255.0

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

    logger.info(f"[Train: {args.train_modality.upper()} | {args.train_dataset_name}] => [Val: {args.val_modality.upper()} | {args.val_dataset_name}]")
    logger.info(f"LoRA config → rank: {args.lora_rank}, alpha: {args.lora_alpha}")

    raw_emb, raw_lbls, lbl_to_idx, _ = load_label_mapping(args.center_emb, device)
    train_lbl_to_idx = None
    if args.train_loss == "ce":
        train_lbl_to_idx = lbl_to_idx

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
        label_to_index=lbl_to_idx,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_samples=args.val_max_samples
    )

    tb_path = os.path.join(
        args.output_dir,
        args.tensorboard_data_dir,
        f"rank{rank}",
        f"lora_r{args.lora_rank}_a{args.lora_alpha}",
        timestamp
    )
    writer = SummaryWriter(log_dir=tb_path)

    train_and_evaluate(args, logger, writer, device, raw_emb, raw_lbls, lbl_to_idx, train_loader, val_loader)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
