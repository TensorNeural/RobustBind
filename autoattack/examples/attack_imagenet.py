import argparse
import json

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from autoattack.attack_bind import Attack, AutoAttackRunner


class ImageNetAttack(Attack):
    def __init__(
        self,
        dataset_root,
        save_dir="./results",
        batch_size=25,
        max_samples=50000,
        epsilons=[2 / 255, 4 / 255],
        norm="Linf",
        version="custom",
        log_root="./logs",
        modality="image",
    ):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        ds = datasets.ImageFolder(root=dataset_root, transform=transform)
        dataset_name = "ImageNet_1K"
        centre_embeddings_path = "./centre_embs/image_in_center_embeddings.pkl"
        self.center_label_to_wordnet_path = (
            "./datasets/ImageNet-1K/center_to_wordnet.json"
        )

        super().__init__(
            dataset=ds,
            dataset_name=dataset_name,
            centre_embeddings_path=centre_embeddings_path,
            save_dir=save_dir,
            batch_size=batch_size,
            max_samples=max_samples,
            epsilons=epsilons,
            norm=norm,
            version=version,
            log_root=log_root,
            modality=modality,
            mean=mean,
            std=std,
        )

        print(f"Loading WordNet mapping from: {self.center_label_to_wordnet_path}")
        with open(self.center_label_to_wordnet_path, "r") as f:
            self.center_label_to_wordnet = json.load(f)
        print("Mapping loaded successfully.")

    def get_label_indices(self, centre_labels, device) -> torch.Tensor:
        if isinstance(self.dataset, torch.utils.data.Subset):
            class_to_index = self.dataset.dataset.class_to_idx
        else:
            class_to_index = self.dataset.class_to_idx

        mapped_indices = []
        for lbl in centre_labels:
            wn_cls = self.center_label_to_wordnet.get(lbl, "")
            idx = class_to_index.get(wn_cls, 0)
            mapped_indices.append(idx)

        return torch.tensor(mapped_indices, dtype=torch.int64, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Root path to ImageNet dataset"
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[2 / 255, 4 / 255])
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--version", type=str, default="custom")
    parser.add_argument(
        "--log_root",
        type=str,
        default="./logs",
        help="Directory for logs (auto-named log file)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        help="Which modality to use: 'image', 'audio', etc.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Output directory for .pth results",
    )
    parser.add_argument(
        "--centre_embeddings_path",
        type=str,
        default="./centre_embs/image_in_center_embeddings.pkl",
        help="Path to center embeddings file",
    )
    parser.add_argument(
        "--center_label_to_wordnet_path",
        type=str,
        default="./datasets/ImageNet-1K/center_to_wordnet.json",
        help="Path to center-label-to-WordNet mapping",
    )

    args = parser.parse_args()

    attack = ImageNetAttack(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        epsilons=args.epsilons,
        norm=args.norm,
        version=args.version,
        log_root=args.log_root,
        modality=args.modality,
    )

    attack.save_dir = args.save_dir
    attack.centre_embeddings_path = args.centre_embeddings_path

    if (
        args.center_label_to_wordnet_path
        != "./datasets/ImageNet-1K/center_to_wordnet.json"
    ):
        attack.center_label_to_wordnet_path = args.center_label_to_wordnet_path

    runner = AutoAttackRunner()
    runner.run(attack)


if __name__ == "__main__":
    main()
