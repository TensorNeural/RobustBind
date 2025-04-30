# import argparse
# import json
# import os

# import torch
# import torch.backends.cudnn as cudnn
# from PIL import Image
# from torch.utils.data import Dataset

# from autoattack.attack_bind import Attack, AutoAttackRunner
# from utils.data_transform import IMAGE_TRANSFORM, IMAGE_MEAN, IMAGE_STD


# class Places365ValDataset(Dataset):
#     """
#     A Dataset class that reads from a JSON file whose entries look like:
#         [
#           {
#             "data": "val_large/Places365_val_00024649.jpg",
#             "label": "bridge"
#           },
#           {
#             "data": "val_large/Places365_val_00014659.jpg",
#             "label": "bow_window"
#           },
#           ...
#         ]
#     'data' is the path (relative to dataset_root) to the image,
#     and 'label' is a string label corresponding to a place category.
#     """

#     def __init__(self, dataset_root, center_to_places_path, transform=None):
#         super().__init__()
#         self.transform = transform

#         # Here, we assume the images are in a folder named "val_large" under `dataset_root`
#         # and that there's a JSON file (e.g., "places365_val_large.json") listing images + labels.
#         self.val_dir = os.path.join(dataset_root, "val_large")
#         val_json_path = os.path.join(dataset_root, "places365_val_large.json")

#         # Load the mapping from labels to final indices (center_to_places_path).
#         # Example of raw_map structure:
#         # {
#         #   "bridge": [127],
#         #   "bow_window": [235],
#         #   "balcony": [19],
#         #   ...
#         # }
#         with open(center_to_places_path, "r") as f:
#             raw_map = json.load(f)

#         # Build label <-> index mappings:
#         self.label_to_final_index = {}
#         self.index_to_label = {}

#         for label_str, idx_list in raw_map.items():
#             if idx_list:
#                 final_idx = idx_list[0]
#                 self.label_to_final_index[label_str] = final_idx
#                 for idx in idx_list:
#                     self.index_to_label[idx] = label_str

#         # Read the JSON file that contains data + label entries
#         self.samples = []
#         with open(val_json_path, "r") as f:
#             data_entries = json.load(f)

#         for item in data_entries:
#             rel_path = item["data"]   # e.g., "val_large/Places365_val_00024649.jpg"
#             label_str = item["label"] # e.g., "bridge"
#             # Look up the final integer index from our label -> index map; default to 0 if not found
#             final_idx = self.label_to_final_index.get(label_str, 0)
#             # The filename in "data" might already be prefixed by 'val_large/', so we handle carefully
#             # For safety, we'll ensure we only append the base filename if the path includes 'val_large/'
#             image_filename = os.path.basename(rel_path)
#             self.samples.append((image_filename, final_idx))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         filename, final_label_idx = self.samples[idx]
#         img_path = os.path.join(self.val_dir, filename)
#         image = Image.open(img_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image, final_label_idx


# class Places365Attack(Attack):
#     def __init__(
#         self,
#         dataset_root,
#         center_to_places_path,
#         batch_size=25,
#         max_samples=50000,
#         epsilons=[2 / 255, 4 / 255],
#         norm="Linf",
#         version="custom",
#         log_root="./logs",
#         modality="image",
#     ):
#         ds = Places365ValDataset(
#             dataset_root=dataset_root,
#             center_to_places_path=center_to_places_path,
#             transform=IMAGE_TRANSFORM,
#         )

#         dataset_name = "Places365"
#         centre_embeddings_path = "./centre_embs/image_p365_center_embeddings.pkl"

#         super().__init__(
#             dataset=ds,
#             dataset_name=dataset_name,
#             centre_embeddings_path=centre_embeddings_path,
#             batch_size=batch_size,
#             max_samples=max_samples,
#             epsilons=epsilons,
#             norm=norm,
#             version=version,
#             log_root=log_root,
#             modality=modality,
#             mean=IMAGE_MEAN,
#             std=IMAGE_STD,
#         )

#         self.center_to_places_path = center_to_places_path

#     def get_indices_from_labels(self, centre_labels, device) -> torch.Tensor:
#         if isinstance(self.dataset, torch.utils.data.Subset):
#             label_to_final_index = self.dataset.dataset.label_to_final_index
#         else:
#             label_to_final_index = self.dataset.label_to_final_index

#         mapped_indices = []
#         for lbl in centre_labels:
#             idx = label_to_final_index.get(lbl, 0)
#             mapped_indices.append(idx)

#         return torch.tensor(mapped_indices, dtype=torch.int64, device=device)

#     def get_labels_from_indices(self, indices) -> list:
#         if isinstance(self.dataset, torch.utils.data.Subset):
#             index_to_label = self.dataset.dataset.index_to_label
#         else:
#             index_to_label = self.dataset.index_to_label

#         labels = []
#         for idx in indices:
#             actual_idx = idx.item() if torch.is_tensor(idx) else idx
#             label_str = index_to_label.get(actual_idx, "Unknown")
#             labels.append(label_str)

#         return labels


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dataset_root",
#         type=str,
#         default="/home/user/datasets/Places365",
#         help="Root path to the Places365 dataset (containing 'val_large' and JSON).",
#     )
#     parser.add_argument(
#         "--dataset_adversary_root",
#         type=str,
#         default="/home/user/datasets/Places365/val_adv",
#         help="Output directory for adversarial samples",
#     )
#     parser.add_argument(
#         "--center_to_places_path",
#         type=str,
#         default="./datasets/Places365/center_to_places365.json",
#         help="Path to the center-to-Places365 mapping JSON file",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=200,
#         help="Batch size for adversarial attacks",
#     )
#     parser.add_argument(
#         "--max_samples",
#         type=int,
#         default=5000,
#         help="Maximum number of samples to attack",
#     )
#     parser.add_argument(
#         "--epsilons",
#         nargs="+",
#         type=float,
#         default=[2 / 255, 4 / 255],
#         help="Attack epsilon(s)",
#     )
#     parser.add_argument(
#         "--norm",
#         type=str,
#         default="Linf",
#         help="Norm type for the attack (e.g., Linf)",
#     )
#     parser.add_argument(
#         "--version",
#         type=str,
#         default="custom",
#         help="Version string for the attack",
#     )
#     parser.add_argument(
#         "--log_root",
#         type=str,
#         default="./logs",
#         help="Directory for logs (auto-named log file)",
#     )
#     parser.add_argument(
#         "--modality",
#         type=str,
#         default="image",
#         help="Data modality (e.g., image or audio)",
#     )
#     parser.add_argument(
#         "--centre_embeddings_path",
#         type=str,
#         default="./centre_embs/image_p365_center_embeddings.pkl",
#         help="Path to center embeddings file",
#     )
#     parser.add_argument(
#         "--metadata_output_root",
#         type=str,
#         default="./datasets/Places365/",
#         help="Directory for metadata files",
#     )
#     parser.add_argument(
#         "--metadata_prefix",
#         type=str,
#         default="val_adv",
#         help="Prefix for metadata files",
#     )

#     args = parser.parse_args()
#     cudnn.benchmark = True

#     attack = Places365Attack(
#         dataset_root=args.dataset_root,
#         center_to_places_path=args.center_to_places_path,
#         batch_size=args.batch_size,
#         max_samples=args.max_samples,
#         epsilons=args.epsilons,
#         norm=args.norm,
#         version=args.version,
#         log_root=args.log_root,
#         modality=args.modality,
#     )

#     # Update if user provides different paths
#     attack.centre_embeddings_path = args.centre_embeddings_path
#     if args.center_to_places_path != "./datasets/Places365/center_to_places365.json":
#         attack.center_to_places_path = args.center_to_places_path

#     runner = AutoAttackRunner(
#         dataset_adversary_root=args.dataset_adversary_root,
#         metadata_output_root=args.metadata_output_root,
#         metadata_prefix=args.metadata_prefix,
#     )
#     runner.run(attack)


# if __name__ == "__main__":
#     main()
