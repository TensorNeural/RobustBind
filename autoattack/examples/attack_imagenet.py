import argparse
import json
import os

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import Dataset

from autoattack.attack_bind import Attack, AutoAttackRunner
from utils.data_transform import IMAGE_TRANSFORM, IMAGE_MEAN, IMAGE_STD


class ImageNetDataset(Dataset):
    """
    Custom ImageNet dataset class (formerly ImageNetValDataset).
    Reads a JSON file with entries of the form:
        [
            {
                "data": "val/n02123045/ILSVRC2012_val_00033837.JPEG",
                "label": "tabby, tabby cat"
            },
            ...
        ]
    and uses a separate JSON mapping center labels to WordNet IDs (or direct string labels) to
    derive final integer indices.
    """
    def __init__(self, dataset_root, center_label_to_wordnet_path, data_json_path, transform=None):
        super().__init__()
        self.transform = transform
        self.root_dir = dataset_root

        # Load center_label_to_wordnet map. Example structure (simplified):
        # { "tabby, tabby cat": "n02123045", "cash machine, ATM": "n02977058", ... }
        with open(center_label_to_wordnet_path, "r") as f:
            raw_map = json.load(f)

        # Build label_to_final_index and index_to_label lookups
        self.label_to_final_index = {}
        self.index_to_label = {}

        idx_counter = 0
        for label_str, _wordnet_id in raw_map.items():
            self.label_to_final_index[label_str] = idx_counter
            self.index_to_label[idx_counter] = label_str
            idx_counter += 1

        # Load the JSON file describing data samples
        self.samples = []
        with open(data_json_path, "r") as f:
            data_entries = json.load(f)

        for item in data_entries:
            rel_path = item["data"]   # e.g., "val/n02123045/ILSVRC2012_val_00033837.JPEG"
            label_str = item["label"] # e.g., "tabby, tabby cat"
            final_idx = self.label_to_final_index.get(label_str, 0)
            self.samples.append((rel_path, final_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, final_idx = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, final_idx


class ImageNetAttack(Attack):
    """
    Attack class for ImageNet, mirroring the style used for Places365Attack.
    """

    def __init__(
        self,
        dataset_root,
        center_label_to_wordnet_path,
        data_json_path,
        batch_size=25,
        max_samples=50000,
        epsilons=[2 / 255],
        norm="Linf",
        version="custom",
        log_root="./logs",
        modality="image",
    ):
        ds = ImageNetDataset(
            dataset_root=dataset_root,
            center_label_to_wordnet_path=center_label_to_wordnet_path,
            data_json_path=data_json_path,
            transform=IMAGE_TRANSFORM
        )

        dataset_name = "ImageNet_1K"
        centre_embeddings_path = "./centre_embs/image_in_center_embeddings.pkl"

        super().__init__(
            dataset=ds,
            dataset_name=dataset_name,
            centre_embeddings_path=centre_embeddings_path,
            batch_size=batch_size,
            max_samples=max_samples,
            epsilons=epsilons,
            norm=norm,
            version=version,
            log_root=log_root,
            modality=modality,
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        )

        self.center_label_to_wordnet_path = center_label_to_wordnet_path

    def get_indices_from_labels(self, centre_labels, device) -> torch.Tensor:
        # If the dataset is a Subset, fetch the parent's maps
        if isinstance(self.dataset, torch.utils.data.Subset):
            label_to_final_index = self.dataset.dataset.label_to_final_index
        else:
            label_to_final_index = self.dataset.label_to_final_index

        mapped_indices = []
        for lbl in centre_labels:
            idx = label_to_final_index.get(lbl, 0)
            mapped_indices.append(idx)

        return torch.tensor(mapped_indices, dtype=torch.int64, device=device)

    def get_labels_from_indices(self, indices) -> list:
        # If the dataset is a Subset, fetch the parent's maps
        if isinstance(self.dataset, torch.utils.data.Subset):
            index_to_label = self.dataset.dataset.index_to_label
        else:
            index_to_label = self.dataset.index_to_label

        labels = []
        for idx in indices:
            actual_idx = idx.item() if torch.is_tensor(idx) else idx
            label_str = index_to_label.get(actual_idx, "Unknown")
            labels.append(label_str)

        return labels


def main():
    parser = argparse.ArgumentParser()

    # Group dataset arguments
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--dataset_root",
        type=str,
        default="/home/user/datasets/ImageNet-1K",
        help="Root path to the ImageNet dataset",
    )
    dataset_group.add_argument(
        "--data_json_path",
        type=str,
        default="./datasets/ImageNet-1K/val_data_5000.json",
        help="Path to the JSON file describing the dataset (with 'data' and 'label' fields)",
    )
    dataset_group.add_argument(
        "--center_label_to_wordnet_path",
        type=str,
        default="./datasets/ImageNet-1K/center_to_wordnet.json",
        help="Path to the JSON mapping center labels to WordNet IDs",
    )

    # Group output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--dataset_adversary_root",
        type=str,
        default="/home/user/datasets/ImageNet-1K/val_5000_adv",
        help="Output directory for adversarial samples",
    )
    output_group.add_argument(
        "--metadata_output_root",
        type=str,
        default="./datasets/ImageNet-1K/",
        help="Directory for metadata files",
    )
    output_group.add_argument(
        "--metadata_prefix",
        type=str,
        default="val_5000_adv",
        help="Prefix for metadata files",
    )

    # Group attack arguments
    attack_group = parser.add_argument_group("Attack")
    attack_group.add_argument(
        "--batch_size",
        type=int,
        default=160,
        help="Attack batch size",
    )
    attack_group.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Max samples to attack",
    )
    attack_group.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[2 / 255, 4 / 255],
        help="Attack epsilons",
    )
    attack_group.add_argument(
        "--norm",
        type=str,
        default="Linf",
        help="Norm type (e.g., Linf)",
    )
    attack_group.add_argument(
        "--version",
        type=str,
        default="custom",
        help="Attack version string",
    )
    attack_group.add_argument(
        "--modality",
        type=str,
        default="image",
        help="Which modality to use (e.g., 'image', 'audio', etc.)",
    )
    attack_group.add_argument(
        "--log_root",
        type=str,
        default="./logs",
        help="Directory for logs (auto-named log file)",
    )
    attack_group.add_argument(
        "--centre_embeddings_path",
        type=str,
        default="./centre_embs/image_in_center_embeddings.pkl",
        help="Path to center embeddings file",
    )

    args = parser.parse_args()
    cudnn.benchmark = True

    attack = ImageNetAttack(
        dataset_root=args.dataset_root,
        center_label_to_wordnet_path=args.center_label_to_wordnet_path,
        data_json_path=args.data_json_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        epsilons=args.epsilons,
        norm=args.norm,
        version=args.version,
        log_root=args.log_root,
        modality=args.modality,
    )
    # Update if user provides different path for center embeddings
    attack.centre_embeddings_path = args.centre_embeddings_path

    # Run the AutoAttack pipeline
    runner = AutoAttackRunner(
        dataset_adversary_root=args.dataset_adversary_root,
        metadata_output_root=args.metadata_output_root,
        metadata_prefix=args.metadata_prefix,
    )
    runner.run(attack)


if __name__ == "__main__":
    main()
