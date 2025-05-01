import argparse
import logging
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from datasets.datasets import ImageNetDataset
from utils.data_transform import IMAGE_TRANSFORM, IMAGE_MEAN, IMAGE_STD
from utils.utils import load_centre_embeddings
from params import find_lr
import json

class RelativePathFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)

    def format(self, record):
        run_dir = os.getcwd()
        abs_path = os.path.abspath(record.pathname)
        try:
            record.relativepath = os.path.relpath(abs_path, run_dir)
        except ValueError:
            record.relativepath = record.pathname  # fallback
        return super().format(record)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset_root", type=str, default="/home/user/datasets/ImageNet-1K")
    parser.add_argument("--train_json", type=str, default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten.pt")
    parser.add_argument("--use_flash_attention", action="store_true", default=True)
    parser.add_argument("--center_emb", type=str, default="./centre_embs/image_in_center_embeddings.pkl")
    parser.add_argument("--train_batch_size", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--train_attack_loss", type=str, default="l2")
    parser.add_argument("--val_attack_loss", type=str, default="ce")
    parser.add_argument("--train_loss", type=str, default="l2")
    parser.add_argument("--lr_finder_steps", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=2/255)
    args = parser.parse_args()
    device = torch.device("cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"lr_finder_{timestamp}.log"

    formatter = RelativePathFormatter(fmt='%(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s')

    file_handler = logging.FileHandler(os.path.join(args.output_dir, log_filename), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = [console_handler, file_handler]

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
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

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
        attack_loss_type=args.train_attack_loss,
        train_loss_type=args.train_loss,
        epsilon=args.epsilon,
        steps=args.lr_finder_steps
    )
    with open(os.path.join(args.output_dir, "lr_finder_results.json"), "w") as f:
        json.dump({"lrs": lrs, "losses": losses, "smoothed_losses": smoothed_losses}, f)

if __name__ == "__main__":
    main()
