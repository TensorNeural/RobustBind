import os
import json
from collections import defaultdict

# Root directory containing dataset folders
DATASETS_ROOT = './'

# List of dataset directories
DATASET_DIRS = [
    "ESC-50", "FSD-50K", "ImageNet-1K", "Kinetics-400", "LLVIP", "ModelNet40",
    "MSR-VTT", "N-Caltech-101", "N-ImageNet-1K", "Places365", "UCF-101", "UrbanSound8K"
]

# Expected metadata files per dataset
METADATA_FILES = ["train_data.json", "val_data.json", "test_data.json"]

# Output dictionary
label_distribution = {}

for dataset in DATASET_DIRS:
    dataset_path = os.path.join(DATASETS_ROOT, dataset)
    label_counts = defaultdict(int)
    
    for file in METADATA_FILES:
        json_path = os.path.join(dataset_path, file)
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    label = item.get("label") or item.get("class") or item.get("category")
                    if label is not None:
                        label_counts[label] += 1
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    if label_counts:
        label_distribution[dataset] = dict(sorted(label_counts.items(), key=lambda x: x[0]))

# Print results
for dataset, counts in label_distribution.items():
    print(f"\n=== {dataset} ===")
    for label, count in counts.items():
        print(f"{label}: {count}")

# Optionally save to JSON
with open("label_distribution.json", "w") as f:
    json.dump(label_distribution, f, indent=2)
