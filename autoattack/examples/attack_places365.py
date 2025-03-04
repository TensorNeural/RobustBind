import os
import json
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from autoattack.attack_bind import Attack, AutoAttackRunner

class Places365ValDataset(Dataset):
    def __init__(self, dataset_root, center_to_places_path, transform=None):
        super().__init__()

        self.transform = transform
        self.val_dir = os.path.join(dataset_root, "val_large")
        val_txt_path = os.path.join(dataset_root, "places365_val.txt")

        with open(center_to_places_path, "r") as f:
            raw_map = json.load(f)

        self.label_to_final_index = {}
        self.index_to_label = {}

        for label_str, indices in raw_map.items():
            if indices:
                final_idx = indices[0]
                self.label_to_final_index[label_str] = final_idx

                for idx in indices:
                    self.index_to_label[idx] = label_str

        self.samples = []

        with open(val_txt_path, "r") as f:
            for line in f:
                line = line.strip()

                if line:
                    filename, str_label = line.split()
                    original_int_label = int(str_label)
                    label_str = self.index_to_label.get(original_int_label, "unknown_label")
                    final_idx = self.label_to_final_index.get(label_str, 0)
                    self.samples.append((filename, final_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, final_label_idx = self.samples[idx]
        img_path = os.path.join(self.val_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, final_label_idx

class Places365Attack(Attack):
    def __init__(
        self,
        dataset_root,
        center_to_places_path,
        save_dir="./results",
        batch_size=25,
        max_samples=50000,
        epsilons=[2/255, 4/255],
        norm='Linf',
        version='custom',
        log_root="./logs",
        modality="image"
    ):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        ds = Places365ValDataset(dataset_root, center_to_places_path, transform=transform)
        dataset_name = "Places365"
        centre_embeddings_path = "./centre_embs/image_p365_center_embeddings.pkl"

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
            std=std
        )

    def get_label_indices(self, centre_labels, device) -> torch.Tensor:
        if isinstance(self.dataset, torch.utils.data.Subset):
            label_to_final_index = self.dataset.dataset.label_to_final_index
        else:
            label_to_final_index = self.dataset.label_to_final_index

        mapped_indices = []
        for lbl in centre_labels:
            idx = label_to_final_index.get(lbl, 0)
            mapped_indices.append(idx)

        return torch.tensor(mapped_indices, dtype=torch.int64, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--center_to_places_path', type=str, default="./datasets/Places365/center_to_places365.json")
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--epsilons', nargs='+', type=float, default=[2/255, 4/255])
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--version', type=str, default='custom')
    parser.add_argument('--log_root', type=str, default="./logs")
    parser.add_argument('--modality', type=str, default="image")
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--centre_embeddings_path', type=str, default="./centre_embs/image_p365_center_embeddings.pkl")

    args = parser.parse_args()
    cudnn.benchmark = True

    attack = Places365Attack(
        dataset_root=args.dataset_root,
        center_to_places_path=args.center_to_places_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        epsilons=args.epsilons,
        norm=args.norm,
        version=args.version,
        log_root=args.log_root,
        modality=args.modality
    )

    attack.save_dir = args.save_dir
    attack.centre_embeddings_path = args.centre_embeddings_path

    runner = AutoAttackRunner()
    runner.run(attack)


if __name__ == "__main__":
    main()
