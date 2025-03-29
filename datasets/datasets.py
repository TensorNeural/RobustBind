from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torch


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
