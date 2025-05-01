import argparse
import logging
import os
import time
from datetime import datetime
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from attack import PGDAttack, APGDAttack, AttackModel
from datasets.datasets import ImageNetDataset
from utils.data_transform import IMAGE_TRANSFORM, IMAGE_MEAN, IMAGE_STD
from utils.utils import load_centre_embeddings
from model import UniBindModel
from meter import AverageMeter
from training import train_epoch
from eval import evaluate_robust_one_stage, evaluate_two_stage, evaluate_clean
from params import find_lr
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class RelativePathFormatter(logging.Formatter):
    def __init__(self, rank, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.rank = rank

    def format(self, record):
        run_dir = os.getcwd()
        abs_path = os.path.abspath(record.pathname)
        try:
            record.relativepath = os.path.relpath(abs_path, run_dir)
            record.rank = self.rank
        except ValueError:
            record.relativepath = record.pathname  # fallback
        return super().format(record)

def train_and_evaluate(
    logger,
    writer,
    device,
    raw_emb,
    raw_lbls,
    lbl_to_idx,
    idx_to_lbl,
    pretrain_weights,
    use_flash_attention,
    train_mean,
    train_std,
    train_loader,
    val_mean,
    val_std,
    val_loader,
    train_attack_loss_type,
    val_attack_loss_type,
    train_loss_type,
    out_dir,
    epsilon,
):
    logger.info("Starting training and evaluation ...")
    is_main = not dist.is_initialized() or dist.get_rank() == 0

    logger.info("Initializing original model ...")
    model_original = UniBindModel(
        device=device,
        pretrain_weights=pretrain_weights,
        modality="image",
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger,
        use_flash_attention=use_flash_attention,
        fine_tuned_weights=None
    )
    model_original.to(device)

    logger.info("Initializing training model ...")
    model_train = UniBindModel(
        device=device,
        pretrain_weights=pretrain_weights,
        modality="image",
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger,
        use_flash_attention=use_flash_attention,
        use_lora=True,
        use_fine_tune=False,
        fine_tuned_weights=None,
    )
    model_train.to(device)
    model_train = DDP(model_train, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    logger.info(f"Starting training for 2 epochs with epsilon={(epsilon * 255):.0f}/255")

    epochs = 2
    steps_per_epoch = len(train_loader)

    trainable_params = [p for p in model_train.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=3e-3, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=3e-3,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    train_attack = PGDAttack(
        logger=logger,
        model=AttackModel(model_train, train_mean, train_std),
        epsilon=epsilon,
        alpha=1/255,
        steps=10,
        norm='linf',
        random_start=True,
        clamp_min=0.0,
        clamp_max=1.0,
        loss_type=train_attack_loss_type
    )
    eval_attack = APGDAttack(
        model=AttackModel(model_train, train_mean, train_std),
        norm='Linf',
        n_restarts=1,
        n_iter=50,
        eps=epsilon,
        loss=val_attack_loss_type,
        device=device,
        logger=logger
    )

    loss_meter = AverageMeter()
    cos_sim_meter = AverageMeter()
    rcos_sim_meter = AverageMeter()
    acc_meter = AverageMeter()
    racc_meter = AverageMeter()

    model_original.eval()
    best_acc = -1.0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs} -----------------------------------------")
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
            train_loss_type=train_loss_type,
            epoch=epoch,
            total_epochs=epochs,
            loss_meter=loss_meter,
            cos_sim_meter=cos_sim_meter,
            rcos_sim_meter=rcos_sim_meter,
            acc_meter=acc_meter,
            racc_meter=racc_meter,
            writer=writer
        )

        if is_main:
            logger.info(f"Saving lora weights for epoch {epoch+1} ...")
            model_train.module.save_lora_weights(os.path.join(out_dir, f"epoch_{epoch+1}_lora_weights.pt"))
        
        logger.info(f"Evaluating robust accuracy with 50-iter one-stage attack, epoch {epoch+1}")
        robust_acc = evaluate_robust_one_stage(
            logger,
            device,
            model_train, 
            val_loader, 
            eval_attack,
            val_mean, 
            val_std
        )
        logger.info(f"[Epoch {epoch+1}] robust acc (one-stage 50 iter) = {robust_acc:.4f}")
        logger.info(f"Epoch {epoch+1} total time (training+eval): {time.time() - time.time():.2f} seconds")

        if robust_acc > best_acc:
            best_acc = robust_acc
            
            if is_main:
                logger.info(f"New best checkpoint: robust acc={best_acc:.4f}. Saving lora weights ...")
                model_train.module.save_lora_weights(os.path.join(out_dir, f"best_lora_weights.pt"))
    
    
    writer.close()
    logger.info(f"Training complete! Best robust (one-stage) accuracy was {best_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset_root", type=str, default="/home/user/datasets/ImageNet-1K")
    parser.add_argument("--train_json", type=str, default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--val_json", type=str, default="./datasets/ImageNet-1K/val_data.json")
    parser.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten.pt")
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--center_emb", type=str, default="./centre_embs/image_in_center_embeddings.pkl")
    parser.add_argument("--train_batch_size", type=int, default=70)
    parser.add_argument("--val_batch_size", type=int, default=70)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--val_max_samples", type=int, default=3000)
    parser.add_argument("--train_attack_loss", type=str, default="l2")
    parser.add_argument("--val_attack_loss", type=str, default="ce")
    parser.add_argument("--train_loss", type=str, default="l2")
    parser.add_argument("--epsilon", type=float, default=2/255)
    parser.add_argument("--tensorboard_data_dir", type=str, default="tensorboard")
    args = parser.parse_args()

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

        os.makedirs(args.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"training_{timestamp}_rank{rank}.log"

        formatter = RelativePathFormatter(rank=rank, fmt='[RANK %(rank)d] %(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s')

        file_handler = logging.FileHandler(os.path.join(args.output_dir, log_filename), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers = [console_handler, file_handler]

        
        logger.info(f"Rank: {rank}, Local rank: {local_rank}")
        logger.info(f"Using device: {device}")

        logger.info("Loading center embeddings ...")
        raw_emb, raw_lbls = load_centre_embeddings(args.center_emb, device)
        raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
        unique_lbls = sorted(list(set(raw_lbls)))
        lbl_to_idx = {l: i for i, l in enumerate(unique_lbls)}
        idx_to_lbl = {v: k for k, v in lbl_to_idx.items()}

        mean_t = torch.tensor(IMAGE_MEAN, device=device).view(1, -1, 1, 1)
        std_t = torch.tensor(IMAGE_STD, device=device).view(1, -1, 1, 1)

        logger.info("Loading train dataset ...")
        train_ds = ImageNetDataset(
            dataset_root=args.dataset_root,
            data_json_path=args.train_json,
            transform=IMAGE_TRANSFORM,
            max_samples=args.train_max_samples,
            debug=False,
            label_to_index=lbl_to_idx,
            index_to_label=idx_to_lbl
        )
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        logger.info("Loading val dataset ...")
        val_ds = ImageNetDataset(
            dataset_root=args.dataset_root,
            data_json_path=args.val_json,
            transform=IMAGE_TRANSFORM,
            max_samples=args.val_max_samples,
            debug=False,
            label_to_index=lbl_to_idx,
            index_to_label=idx_to_lbl
        )
        val_sampler = DistributedSampler(val_ds, shuffle=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.tensorboard_data_dir, f"rank{rank}", timestamp))
        train_and_evaluate(
            logger=logger,
            writer=writer,
            device=device,
            raw_emb=raw_emb,
            raw_lbls=raw_lbls,
            lbl_to_idx=lbl_to_idx,
            idx_to_lbl=idx_to_lbl,
            pretrain_weights=args.pretrain_weights,
            use_flash_attention=args.use_flash_attention,
            train_mean=mean_t,
            train_std=std_t,
            train_loader=train_loader,
            val_mean=mean_t,
            val_std=std_t,
            val_loader=val_loader,
            train_attack_loss_type=args.train_attack_loss,
            val_attack_loss_type=args.val_attack_loss,
            train_loss_type=args.train_loss,
            out_dir=args.output_dir,
            epsilon=args.epsilon
        )

        best_lora_weights_path = os.path.join(args.output_dir, "best_lora_weights.pt")
        if os.path.exists(best_lora_weights_path):
            logger.info("Loading best lora weights for final two-stage & clean evaluations ...")
            final_model = UniBindModel(
                device=device,
                pretrain_weights=args.pretrain_weights,
                modality="image",
                centre_embeddings=raw_emb,
                centre_labels=raw_lbls,
                label_to_index=lbl_to_idx,
                index_to_label=idx_to_lbl,
                logger=logger,
                use_flash_attention=args.use_flash_attention,
                use_lora=True,
                lora_weights=best_lora_weights_path
            )
            final_model.to(device)
            final_model = DDP(final_model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

            final_robust_acc = evaluate_two_stage(
                logger,
                device,
                final_model,
                val_loader,
                attack_loss_type=args.val_attack_loss,
                iteration_count=100,
                epsilon=args.epsilon,
                mean=mean_t,
                std=std_t
            )
            logger.info(f"Final two-stage robust accuracy = {final_robust_acc:.4f}")

            final_clean_acc = evaluate_clean(logger, device, final_model, val_loader)
            logger.info(f"Final clean accuracy = {final_clean_acc:.4f}")
        else:
            logger.warning("No best_lora_weights.pt found. Skipping final evaluations.")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
