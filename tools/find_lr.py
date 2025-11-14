import argparse
import logging
import os
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_transform import IMAGE_TRANSFORM, IMAGE_MEAN, IMAGE_STD
from utils.utils import load_centre_embeddings
from params import find_lr
import json

class ImageNetDataset(Dataset):
    """
    Modified ImageNet dataset class that accepts external label-to-index
    and index-to-label mappings or callables. This way, you can define a
    static, consistent mapping outside of this dataset class and simply
    pass them in here.

    Expects data_json_path to have entries like:
        [
            {
                "data": "val/n02123045/ILSVRC2012_val_00033837.JPEG",
                "label": "tabby, tabby cat"
            },
            ...
        ]

    Args:
        dataset_root: The directory where ImageNet data is stored
        data_json_path: JSON with 'data' (relative path) and 'label' (string)
        label_to_index: A function or dict to convert label string -> integer ID
        index_to_label: A function or dict to convert integer ID -> label string
        transform: Optional transforms for the image
        max_samples: Optionally limit dataset size for debugging
        debug: If True, does not randomize when picking `max_samples`
    """

    def __init__(
        self,
        dataset_root,
        data_json_path,
        transform=None,
        max_samples=None,
        debug=False,
        # Optionally pass these in if you already have them:
        label_to_index=None,
        index_to_label=None,
    ):
        super().__init__()
        self.transform = transform
        self.root_dir = dataset_root

        # Save references to label mappers (callables or dict)
        self.label_to_index_fn = label_to_index
        self.index_to_label_fn = index_to_label

        # Load the JSON file describing data samples
        with open(data_json_path, "r") as f:
            data_entries = json.load(f)

        # Build self.samples as a list of (relative_path, label_str)
        self.samples = []
        for item in data_entries:
            rel_path = item["data"]   # e.g., "val/n02123045/ILSVRC2012_val_00033837.JPEG"
            label_str = item["label"] # e.g., "tabby, tabby cat"
            self.samples.append((rel_path, label_str))

        # Optionally limit dataset size for debugging
        if max_samples is not None and max_samples < len(self.samples):
            if debug:
                indices = torch.arange(max_samples)[:max_samples]
            else:
                indices = torch.randperm(len(self.samples))[:max_samples]
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label_str = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert label string to integer if label_to_index_fn is provided
        if self.label_to_index_fn is not None:
            if callable(self.label_to_index_fn):
                final_idx = self.label_to_index_fn(label_str)
            else:
                # If it's a dictionary, do a dict lookup
                final_idx = self.label_to_index_fn.get(label_str, 0)
        else:
            # Fallback: no label mapping → could default to 0
            final_idx = 0

        return image, final_idx

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
    parser.add_argument("--dataset_root", type=str, default="/data/datasets/ImageNet-1K")
    parser.add_argument("--train_json", type=str, default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten_image_patchs.pt")
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
