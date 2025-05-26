import argparse, os, logging, shutil, math
import time
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW

from downstream.llava.dataset_coco import COCOCaptionDataset
from downstream.llava.dataset_vqa import VQADataset
from downstream.llava.model.builder import load_pretrained_model
from downstream.llava.constants import DEFAULT_IMAGE_TOKEN


class RelativePathFormatter(logging.Formatter):
    def __init__(self, rank, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.rank = rank

    def format(self, record):
        run_dir = os.getcwd()
        record.rank = self.rank
        record.relativepath = os.path.relpath(os.path.abspath(record.pathname), run_dir)
        return super().format(record)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.35 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(logger, writer, model, dataloader, optimizer, scheduler, scaler, device, epoch, total_epochs, stage, rank):
    model.train()
    start_time = time.time()
    step_base = epoch * len(dataloader)
    total_samples = 0
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        step_global = step_base + step
        batch_start = time.time()

        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            out = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                images=batch["images"]
            )
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        batch_size = batch["input_ids"].shape[0]
        total_samples += batch_size

        # Log for all ranks
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[{stage.upper()}] Epoch {epoch+1}/{total_epochs}, Step {step+1}/{len(dataloader)}, "
            f"BatchSize={batch_size}, Loss={loss_val:.4f}, LR={current_lr:.6f}"
        )
        writer.add_scalar(f"{stage}/loss_step", loss_val, step_global)
        writer.add_scalar(f"{stage}/lr_step", current_lr, step_global)

        batch_end = time.time()
        logger.info(f"[{stage.upper()}] Batch {step+1} time: {batch_end - batch_start:.2f}s, Total samples: {total_samples}")

    avg_loss = total_loss / len(dataloader)

    logger.info(f"[{stage.upper()}] Epoch {epoch+1} completed in {time.time() - start_time:.2f}s")
    logger.info(f"[{stage.upper()}] Avg Loss: {avg_loss:.4f}, Samples: {total_samples}")
    writer.add_scalar(f"{stage}/loss_epoch", avg_loss, epoch)
    writer.add_scalar(f"{stage}/lr_epoch", optimizer.param_groups[0]["lr"], epoch)

    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    logger.info(f"[{stage.upper()}] Epoch {epoch+1} total global samples: {int(sample_tensor.item())}")

    return avg_loss


def train(args, logger, writer, device, rank):
    logger.info("Loading LLaVA model with UniBind encoder...")
    local_model_dir = os.path.join(".cache", args.pretrained_model.replace("/", "--"))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=local_model_dir,
        model_name=os.path.basename(args.pretrained_model),
        model_base=None,
        torch_dtype=torch.float16,
        device=device,
        device_map=None,
        use_unibind=True,
        unibind_pretrain_weights=args.unibind_weights,
        unibind_use_lora=False,
        unibind_lora_weights=None,
        freeze_projector=False,
        freeze_unibind=True,
    )

    tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 2048

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = GradScaler()

    # === COCO Captioning Stage ===
    logger.info("Loading COCO Caption dataset...")
    coco_ds = COCOCaptionDataset(
        args.coco_json,
        os.path.join(args.dataset_root, "COCO", "caption"),
        tokenizer,
        image_processor,
        max_samples=args.coco_max_samples,
    )
    coco_loader = DataLoader(
        coco_ds,
        batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(coco_ds),
        collate_fn=coco_ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    num_steps_coco = args.coco_epochs * len(coco_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=int(0.1 * num_steps_coco), total_steps=num_steps_coco)

    logger.info(f"Training on COCO for {args.coco_epochs} epoch(s)...")
    for epoch in range(args.coco_epochs):
        train_one_epoch(
            logger=logger,
            writer=writer,
            model=ddp_model,
            dataloader=coco_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.coco_epochs,
            stage="coco",
            rank=rank
        )

    if rank == 0:
        torch.save(model.get_model().mm_projector.state_dict(), os.path.join(args.output_dir, "coco_projector.pt"))

    dist.barrier()

    # === VQA Stage ===
    logger.info("Loading VQA v2.0 dataset...")
    vqa_ds = VQADataset(
        args.vqa_json,
        os.path.join(args.dataset_root, "VQA2"),
        tokenizer,
        image_processor,
        max_samples=args.vqa_max_samples,
    )
    vqa_loader = DataLoader(
        vqa_ds,
        batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(vqa_ds),
        collate_fn=vqa_ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    num_steps_vqa = args.vqa_epochs * len(vqa_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=int(0.1 * num_steps_vqa), total_steps=num_steps_vqa)

    logger.info(f"Fine-tuning on VQA for {args.vqa_epochs} epoch(s)...")
    for epoch in range(args.vqa_epochs):
        train_one_epoch(
            logger=logger,
            writer=writer,
            model=ddp_model,
            dataloader=vqa_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.vqa_epochs,
            stage="vqa",
            rank=rank
        )

    if rank == 0:
        torch.save(model.get_model().mm_projector.state_dict(), os.path.join(args.output_dir, "projector.pt"))

    writer.close()
    logger.info("Training completed.")


def main():
    parser = argparse.ArgumentParser("LLaVA-UniBind Trainer")
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--vqa_json", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--unibind_weights", required=True)
    parser.add_argument("--output_dir", default="output/llava")
    parser.add_argument("--coco_epochs", type=int, default=1)
    parser.add_argument("--vqa_epochs", type=int, default=2)
    parser.add_argument("--coco_max_samples", type=int, default=None)
    parser.add_argument("--vqa_max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    args.output_dir = os.path.join(args.output_dir, f"{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, f"rank{rank}.log")
    formatter = RelativePathFormatter(rank, '[RANK %(rank)d] %(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers = [ch, fh]

    tb_path = os.path.join(args.output_dir, "tensorboard", f"rank{rank}")
    writer = SummaryWriter(log_dir=tb_path)

    logger.info("Starting LLaVA-UniBind training pipeline")
    logger.info(f"COCO epochs: {args.coco_epochs}, VQA epochs: {args.vqa_epochs}, LR: {args.lr}, Batch size: {args.batch_size}")

    train(args, logger, writer, device, rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
