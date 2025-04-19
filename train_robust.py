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

class RelativePathFormatter(logging.Formatter):
    def format(self, record):
        run_dir = os.getcwd()
        abs_path = os.path.abspath(record.pathname)
        try:
            record.relativepath = os.path.relpath(abs_path, run_dir)
        except ValueError:
            record.relativepath = record.pathname  # fallback
        return super().format(record)

def train_and_evaluate(
    logger,
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
    attack_loss_type,
    train_loss_type,
    device,
    out_dir,
    epsilon,
):
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tensorBoard2"))
    logger.info("Initializing original + training models ...")
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
        fine_tuned_weights=None,       
    )
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
        fine_tuned_weights=None
    )

    logger.info(f"Starting training for 2 epochs with epsilon={(epsilon * 255):.0f}/255")

    # 2 epochs, OneCycleLR
    epochs = 2
    steps_per_epoch = len(train_loader)

    trainable_params = [p for p in model_train.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=3e-3,               # Sweet spot from LR finder
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,             # 10% warmup (default and ideal)
        anneal_strategy='cos',     # Cosine decay after warmup
        div_factor=25.0,           # Start at max_lr / 25 = 1.2e-4
        final_div_factor=1e4       # End at ~1.2e-8
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
        loss_type=attack_loss_type
    )
    eval_attack = APGDAttack(
        predict=AttackModel(model_train, train_mean, train_std).logits,
        norm='Linf',
        n_restarts=1,
        n_iter=50,
        eps=epsilon,
        loss=attack_loss_type,
        device=device,
        logger=logger
    )

    loss_meter = AverageMeter()
    cos_sim_meter = AverageMeter()
    acc_meter = AverageMeter()
    racc_meter = AverageMeter()

    model_original.eval()
    best_acc = -1.0
    for ep in range(epochs):
        logger.info(f"Epoch {ep+1}/{epochs} -----------------------------------------")
        epoch_start_time = time.time()

        # Train
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
            epoch=ep,
            total_epochs=epochs,
            loss_meter=loss_meter,
            cos_sim_meter=cos_sim_meter,
            acc_meter=acc_meter,
            racc_meter=racc_meter,
            writer=writer
        )

        # Evaluate robust accuracy (one-stage, 50 steps)
        logger.info(f"Evaluating robust accuracy with 50-iter one-stage attack, epoch {ep+1}")
        robust_acc = evaluate_robust_one_stage(
            logger,
            device,
            model_train, 
            val_loader, 
            eval_attack,
            val_mean, 
            val_std
        )
        logger.info(f"[Epoch {ep+1}] robust acc (one-stage 50 iter) = {robust_acc:.4f}")

        epoch_end_time = time.time()
        logger.info(f"Epoch {ep+1} total time (training+eval): {epoch_end_time - epoch_start_time:.2f} seconds")

        # Best checkpoint
        if robust_acc > best_acc:
            best_acc = robust_acc
            logger.info(f"New best checkpoint: robust acc={best_acc:.4f}. Saving fine-tuned weights ...")
            model_train.save_fine_tuned_weights(os.path.join(out_dir, "best_fine_tuned_weights.pt"))
    
    writer.close()
    logger.info(f"Training complete! Best robust (one-stage) accuracy was {best_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset_root", type=str, default="/home/user/datasets/ImageNet-1K")
    parser.add_argument("--train_json", type=str, default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--val_json", type=str, default="./datasets/ImageNet-1K/val_data.json")
    parser.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten.pt")
    parser.add_argument("--use_flash_attention", action="store_true", default=True, 
                        help="Use flash attention for training")
    parser.add_argument("--center_emb", type=str, default="./centre_embs/image_in_center_embeddings.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--attack_loss", type=str, default="l2")
    parser.add_argument("--train_loss", type=str, default="l2")
    parser.add_argument("--epsilon", type=float, default=2/255)
    parser.add_argument("--lr_finder", action='store_true', default=False,
                        help="runs the LR Finder instead of the main training")
    parser.add_argument("--lr_finder_steps", type=int, default=200,
                        help="Max steps for LR finder")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_{timestamp}.log"

    formatter = RelativePathFormatter('%(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s')

    file_handler = logging.FileHandler(os.path.join(args.output_dir, log_filename), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = [console_handler, file_handler]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1) Load center embeddings
    logger.info("Loading center embeddings ...")
    raw_emb, raw_lbls = load_centre_embeddings(args.center_emb, device)
    raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
    unique_lbls = sorted(list(set(raw_lbls)))
    lbl_to_idx = {l: i for i, l in enumerate(unique_lbls)}
    idx_to_lbl = {v: k for k, v in lbl_to_idx.items()}

    mean_t = torch.tensor(IMAGE_MEAN, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(IMAGE_STD, device=device).view(1, -1, 1, 1)

    # 2) Datasets
    logger.info("Loading train dataset ...")
    train_ds = ImageNetDataset(
        dataset_root=args.dataset_root,
        data_json_path=args.train_json,
        transform=IMAGE_TRANSFORM,
        max_samples=10,
        debug=False,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    if args.lr_finder:
        logger.info("Running LR finder ...")
        lrs, losses, smoothed_losses = find_lr(
            logger=logger,
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
            attack_loss_type=args.attack_loss,
            train_loss_type=args.train_loss,
            epsilon=args.epsilon,
            steps=args.lr_finder_steps
        )
        with open(os.path.join(args.output_dir, "lr_finder_results.json"), "w") as f:
            json.dump({
                "lrs": lrs, 
                "losses": losses, 
                "smoothed_losses": smoothed_losses
            }, f)
        return

    logger.info("Loading val dataset ...")
    val_ds = ImageNetDataset(
        dataset_root=args.dataset_root,
        data_json_path=args.val_json,
        transform=IMAGE_TRANSFORM,
        max_samples=2,
        debug=False,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    train_and_evaluate(
        logger=logger,
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
        attack_loss_type=args.attack_loss,
        train_loss_type=args.train_loss,
        device=device,
        out_dir=args.output_dir,
        epsilon=args.epsilon
    )

    # 5) Final two-stage + clean checks (optional)
    best_fine_tuned_ckpt_path = os.path.join(args.output_dir, "best_fine_tuned_weights.pt")
    if os.path.exists(best_fine_tuned_ckpt_path):
        logger.info("Loading best fine tuned weights for final two-stage & clean evaluations ...")
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
            fine_tuned_weights=best_fine_tuned_ckpt_path
        )

        final_robust_acc = evaluate_two_stage(
            logger,
            device,
            final_model, 
            val_loader,
            attack_loss_type=args.attack_loss,
            iteration_count=100, 
            epsilon=args.epsilon,
            mean=mean_t, 
            std=std_t
        )
        logger.info(f"Final two-stage robust accuracy = {final_robust_acc:.4f}")

        final_clean_acc = evaluate_clean(logger, device, final_model, val_loader)
        logger.info(f"Final clean accuracy = {final_clean_acc:.4f}")
    else:
        logger.warning("No best_fine_tuned_weights.pt found. Skipping final evaluations.")

if __name__ == "__main__":
    main()