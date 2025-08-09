import argparse, os, logging, shutil
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from model import UniBindClassifier, ForwardMode
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

@torch.no_grad()
def evaluate_one_stage(logger, device, model_val, val_loader):
    """Clean evaluation without adversarial perturbations."""
    model_val.eval()
    total_correct = 0
    total = 0
    for inp, lbl in val_loader:
        inp = inp.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        logits, _ = model_val(inp, mode=ForwardMode.LOGITS)
        preds = logits.argmax(dim=1)
        total_correct += (preds == lbl).sum().item()
        total += lbl.size(0)
    acc = 100.0 * total_correct / max(1, total)
    logger.info(f"[EVAL] Accuracy = {acc:.4f}")
    return acc

def run_alignment_training(args, logger, writer, device, raw_emb, raw_lbls, lbl_to_idx, train_loader, val_loader):
    """
    Non-adversarial alignment training.
    Trains only the modality head MLP against center embeddings via CE.
    """
    _train_mean, _train_std = get_normalization_tensors(args.train_modality, device)
    _val_mean, _val_std = get_normalization_tensors(args.val_modality, device)

    logger.info(f"[ALIGN] Models: train={args.train_modality.upper()} | val={args.val_modality.upper()}")

    # Trainable model (train modality)
    model_train = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.train_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=False,
        use_modality_head_mlp=True,
        modality_head_mlp_weights=args.modality_head_mlp_weights
    ).to(device)
    model_train = DDP(model_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    logger.info("[ALIGN] Enabling MLP-only parameter training")
    model_train.module.unibind.enable_modality_head_mlp()

    # Validation model (val modality)
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
        modality_head_mlp_weights=args.modality_head_mlp_weights
    ).to(device)
    model_val.eval()

    params = [p for p in model_train.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=3e-3, weight_decay=1e-4)
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
    for epoch in range(2):
        logger.info(f"[ALIGN] Epoch {epoch + 1}/2")
        train_loader.sampler.set_epoch(epoch)

        model_train.train()
        running_loss = 0.0
        for step, (inp, lbl) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model_train(inp, mode=ForwardMode.LOGITS)
            loss = torch.nn.functional.cross_entropy(logits, lbl)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if dist.get_rank() == 0:
                writer.add_scalar("alignment/loss", loss.item(), epoch * len(train_loader) + step)

            if step % 20 == 0:
                logger.info(f"[ALIGN] Epoch {epoch+1} Step {step}/{len(train_loader)} "
                            f"Loss={loss.item():.6f} AvgLoss={(running_loss/(step+1)):.6f}")

        if dist.get_rank() == 0:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_mlp_{args.model_type}.pt")
            model_train.module.save_modality_head_mlp_weights(ckpt_path)

        torch.cuda.empty_cache()
        dist.barrier()

        model_val.load_modality_head_mlp_weights(ckpt_path)
        acc = evaluate_one_stage(logger, device, model_val, val_loader)
        logger.info(f"[ALIGN][Epoch {epoch + 1}] acc on {args.val_modality.upper()} = {acc:.4f}")

        if dist.get_rank() == 0 and acc > best_acc:
            best_acc = acc
            best_ckpt_path = os.path.join(args.output_dir, f"best_mlp_weights_{args.model_type}.pt")
            shutil.copyfile(ckpt_path, best_ckpt_path)

    writer.close()
    logger.info(f"[ALIGN] Best accuracy: {best_acc:.4f}")

def run_robust_training(args, logger, writer, device, raw_emb, raw_lbls, lbl_to_idx, train_loader, val_loader):
    train_mean, train_std = get_normalization_tensors(args.train_modality, device)
    val_mean, val_std     = get_normalization_tensors(args.val_modality, device)

    logger.info(f"Loading model for training: {args.train_modality.upper()}")
    model_original = UniBindClassifier(
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

    model_train = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.train_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_modality_head_mlp=False,
    ).to(device)

    model_train = DDP(model_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    # === Control training mode ===
    if args.use_full_finetune:
        logger.info("[Mode] Full fine-tuning: unfreezing backbone, keeping MLP frozen")
        model_train.module.unibind.enable_full_fine_tune()
    elif args.use_lora:
        logger.info("[Mode] LoRA training: enabling LoRA params only")
        model_train.module.unibind.enable_lora()
    else:
        logger.info("[Mode] Frozen model — no parameters will be trained")

    model_val = UniBindClassifier(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality=args.val_modality,
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        logger=logger,
        use_flash_attention=args.use_flash_attention,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_modality_head_mlp=False
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

        if dist.get_rank() == 0:
            if args.use_full_finetune:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_backbone_{args.model_type}.pt")
                model_train.module.save_backbone(ckpt_path)
            elif args.use_lora:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}_lora_{args.model_type}.pt")
                model_train.module.save_lora_weights(ckpt_path)

        torch.cuda.empty_cache()
        dist.barrier()

        if args.use_full_finetune:
            logger.info(f"[EPOCH {epoch + 1}] Loading backbone weights for validation: {args.model_type}")
            model_val.load_backbone(ckpt_path)
        elif args.use_lora:
            logger.info(f"[EPOCH {epoch + 1}] Loading LoRA weights for validation: {args.model_type}")
            model_val.load_lora_weights(ckpt_path)

        acc = evaluate_robust_one_stage(logger, device, model_val, val_loader, eval_attack, val_mean, val_std)
        logger.info(f"[Epoch {epoch + 1}] robust acc on {args.val_modality.upper()} = {acc:.4f}")

        if dist.get_rank() == 0 and acc > best_acc:
            best_acc = acc
            if args.use_full_finetune:
                best_ckpt_path = os.path.join(args.output_dir, f"best_backbone_weights_{args.model_type}.pt")
            elif args.use_lora:
                best_ckpt_path = os.path.join(args.output_dir, f"best_lora_weights_{args.model_type}.pt")
            shutil.copyfile(ckpt_path, best_ckpt_path)

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
    parser.add_argument("--val_max_samples", type=int, default=200)
    parser.add_argument("--train_attack_loss", default="l2")
    parser.add_argument("--val_attack_loss", default="ce")
    parser.add_argument("--train_loss", default="l2")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--epsilon", type=int, default=4)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--training_mode", choices=["alignment", "robust"], default="robust",
                        help="alignment: modality-head MLP training (non-adversarial); robust: training with attacks")
    parser.add_argument('--use_full_finetune', action='store_true', default=True,
                        help='Enable full fine-tuning (backbone)')
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Enable LoRA adapter training instead of full fine-tuning")
    parser.add_argument("--use_modality_head_mlp", action="store_true", default=False,
                        help="Apply/train the modality-specific head MLP (alignment mode)")
    parser.add_argument("--modality_head_mlp_weights", default=None,
                        help="Optional path to load pre-trained modality head MLP weights")
    parser.add_argument("--tensorboard_data_dir", default="tensorboard")
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # interpret epsilon
    args.epsilon = args.epsilon / 255.0

    # outputs directory token (no inline ternaries)
    if args.training_mode == "alignment":
        mode_token = "align"
    else:
        mode_token = f"eps{int(round(args.epsilon * 255))}"

    if args.use_lora:
        train_mode_token = f"lora_r{args.lora_rank}_a{args.lora_alpha}"
    elif args.use_full_finetune:
        train_mode_token = "ft"
    else:
        train_mode_token = "frozen"

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

    logger.info(f"[Train: {args.train_modality.upper()} | {args.train_dataset_name}] "
                f"=> [Val: {args.val_modality.upper()} | {args.val_dataset_name}]")
    logger.info(f"Mode: {args.training_mode.upper()} | LoRA rank/alpha: {args.lora_rank}/{args.lora_alpha} | epsilon={args.epsilon:.5f}")

    if args.training_mode == "alignment" and not args.use_modality_head_mlp:
        logger.warning("[ALIGN] --use_modality_head_mlp is False; enabling it for alignment mode.")
        args.use_modality_head_mlp = True

    raw_emb, raw_lbls, lbl_to_idx, _ = load_label_mapping(args.center_emb, device)

    # For alignment, use CE over centers
    if args.training_mode == "alignment":
        args.train_loss = "ce"
        logger.info("[ALIGN] Forcing train_loss='ce'")

    train_loader = train_data_loader(
        modality=args.train_modality,
        dataset_root=args.train_dataset_root,
        train_json=args.train_json,
        label_to_index=(lbl_to_idx if args.train_loss == "ce" else None),
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
        train_mode_token,
        timestamp
    )
    writer = SummaryWriter(log_dir=tb_path)

    if args.training_mode == "alignment":
        run_alignment_training(args, logger, writer, device, raw_emb, raw_lbls, lbl_to_idx, train_loader, val_loader)
    else:
        run_robust_training(args, logger, writer, device, raw_emb, raw_lbls, lbl_to_idx, train_loader, val_loader)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
