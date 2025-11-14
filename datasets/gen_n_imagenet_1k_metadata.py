#!/usr/bin/env python3

import os
import json
import argparse

SYNSET_MAPPING_FILE = "LOC_synset_mapping.txt"
OUTPUT_DIR = "N-ImageNet-1K"

# Special case handling for ambiguous classes
CRANE_LABELS = {
    "n02012849": "crane",   # Bird
    "n03126707": "crane2"   # Machine
}

def load_synset_mapping(dataset_root):
    mapping_file = os.path.join(dataset_root, SYNSET_MAPPING_FILE)
    synset_to_human = {}
    human_to_synset = {}

    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file not found: {mapping_file}")
        return synset_to_human, human_to_synset

    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                synset_id, human_readable = parts
                if synset_id in CRANE_LABELS:
                    human_readable = CRANE_LABELS[synset_id]
                synset_to_human[synset_id] = human_readable
                human_to_synset[human_readable] = synset_id

    print(f"✅ Loaded {len(synset_to_human)} synsets.")
    return synset_to_human, human_to_synset

def generate_val_metadata(dataset_root, synset_mapping, class_names):
    val_dir = os.path.join(dataset_root, "val")
    dataset_root_abs = os.path.abspath(dataset_root)
    metadata = []

    if not os.path.isdir(val_dir):
        print(f"❌ val/ directory not found: {val_dir}")
        return

    for class_folder in sorted(os.listdir(val_dir)):
        class_path = os.path.join(val_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Fallback to synset ID if mapping is missing
        label = synset_mapping.get(class_folder, class_folder)
        if label not in class_names:
            class_names.append(label)

        for file in sorted(os.listdir(class_path)):
            if file.endswith(".npz"):
                data_path = os.path.join(class_path, file)
                metadata.append({
                    "data": os.path.relpath(data_path, dataset_root_abs),
                    "label": label
                })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "val_data.json")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📄 Saved val_data.json with {len(metadata)} entries.")

def save_center_to_wordnet(output_file, human_to_synset):
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, "w") as f:
        json.dump(human_to_synset, f, indent=2)
    print(f"📄 Saved center_to_wordnet.json to {output_path}")

def save_class_names(class_names):
    class_names = sorted(set(class_names))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    output_path = os.path.join(OUTPUT_DIR, "classes.json")
    with open(output_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"📄 Saved classes.json with {len(class_to_idx)} classes.")

def main():
    parser = argparse.ArgumentParser(description="Generate val_data.json for N-ImageNet-1K (NPZ format)")
    parser.add_argument("DATASET_ROOT", type=str, help="Path to dataset root containing val/")
    args = parser.parse_args()

    synset_to_human, human_to_synset = load_synset_mapping(args.DATASET_ROOT)
    class_names = []
    generate_val_metadata(args.DATASET_ROOT, synset_to_human, class_names)
    save_center_to_wordnet("center_to_wordnet.json", human_to_synset)
    save_class_names(class_names)

    print("✅ All done.")

if __name__ == "__main__":
    main()
