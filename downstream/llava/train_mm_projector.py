import os, argparse, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

from downstream.llava.dataset_coco import COCOCaptionDataset
from downstream.llava.dataset_vqa import VQADataset
from downstream.llava.model.builder import load_pretrained_model
from downstream.llava.constants import DEFAULT_IMAGE_TOKEN

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return torch.device("cuda", local_rank), local_rank

def train_one_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
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
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--vqa_json", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--unibind_weights", required=True)
    parser.add_argument("--output_dir", default="output/llava")
    parser.add_argument("--coco_epochs", type=int, default=1)
    parser.add_argument("--vqa_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device, rank = setup_ddp()
    os.makedirs(args.output_dir, exist_ok=True)

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
        freeze_projector=False,   # ✅ enable mm_projector training
        freeze_unibind=True,      # ✅ keep vision encoder frozen
    )

    tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 2048

    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[DEBUG] Trainable parameter: {name} - {param.shape}")
            trainable_params.append(param)

    optimizer = AdamW(trainable_params, lr=args.lr)
    scaler = GradScaler()

    # Stage 1: COCO Caption Pretraining
    coco_root = os.path.join(args.dataset_root, "COCO", "caption")
    coco_ds = COCOCaptionDataset(args.coco_json, coco_root, tokenizer, image_processor)
    coco_loader = DataLoader(
        coco_ds,
        batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(coco_ds),
        collate_fn=coco_ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True)

    for epoch in range(args.coco_epochs):
        loss = train_one_epoch(ddp_model, coco_loader, optimizer, scaler, device)
        if rank == 0:
            print(f"[COCO EPOCH {epoch}] loss: {loss:.4f}")
    if rank == 0:
        torch.save(model.get_model().mm_projector.state_dict(), os.path.join(args.output_dir, "coco_projector.pt"))
    dist.barrier()

    # Stage 2: VQA Finetuning
    vqa_root = os.path.join(args.dataset_root, "VQA2")
    vqa_ds = VQADataset(args.vqa_json, vqa_root, tokenizer, image_processor)
    vqa_loader = DataLoader(
        vqa_ds,
        batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(vqa_ds),
        collate_fn=vqa_ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    for epoch in range(args.vqa_epochs):
        loss = train_one_epoch(ddp_model, vqa_loader, optimizer, scaler, device)
        if rank == 0:
            print(f"[VQA EPOCH {epoch}] loss: {loss:.4f}")
    if rank == 0:
        torch.save(model.get_model().mm_projector.state_dict(), os.path.join(args.output_dir, "projector.pt"))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
