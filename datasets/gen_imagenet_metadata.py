import os
import json
import argparse

# Hardcoded path for synset mapping file
SYNSET_MAPPING_FILE = "LOC_synset_mapping.txt"
OUTPUT_DIR = "ImageNet-1K"  # JSON output directory relative to script

# Special case handling: Disambiguating "crane"
CRANE_LABELS = {
    "n02012849": "crane",   # Bird crane (bird)
    "n03126707": "crane2"   # Construction crane (machine)
}

def load_synset_mapping(dataset_root):
    """Loads ImageNet class names and generates mappings for synset-to-human and human-to-synset."""
    mapping_file = os.path.join(dataset_root, SYNSET_MAPPING_FILE)
    synset_to_human = {}
    human_to_synset = {}

    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file {mapping_file} not found.")
        return synset_to_human, human_to_synset

    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)  # Format: synset_id description
            if len(parts) == 2:
                synset_id = parts[0]
                human_readable_name = parts[1]

                # Handle special cases for "crane"
                if synset_id in CRANE_LABELS:
                    human_readable_name = CRANE_LABELS[synset_id]

                synset_to_human[synset_id] = human_readable_name
                human_to_synset[human_readable_name] = synset_id  # Ensure unique mapping

    print(f"Loaded {len(synset_to_human)} synsets from WordNet mapping")
    print(f"Loaded {len(human_to_synset)} unique human-readable labels for center-to-WordNet mapping")

    return synset_to_human, human_to_synset

def generate_metadata(dataset_root, dataset_name, output_filename, synset_mapping, labeled=True):
    """Generates metadata JSON file with paths relative to `DATASET_ROOT`."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found. Skipping...")
        return

    metadata = []

    # Convert dataset_root to absolute path to ensure proper relative calculations
    dataset_root_abs = os.path.abspath(dataset_root)

    # Handle test set separately (since it does not have class subdirectories)
    if not labeled:
        for img_file in sorted(os.listdir(dataset_dir)):  # Directly iterate over test images
            if img_file.lower().endswith((".jpeg", ".jpg", ".png")):
                img_path = os.path.join(dataset_dir, img_file)

                # Ensure the path exists
                if not os.path.exists(img_path):
                    print(f"Warning: Skipping missing or unreadable file {img_path}")
                    continue

                metadata.append({
                    "data": os.path.relpath(img_path, dataset_root_abs),  # Relative to DATASET_ROOT
                    "label": "unknown"  # No labels in test set
                })

    else:
        for class_folder in sorted(os.listdir(dataset_dir)):  # Sort for consistency
            class_path = os.path.join(dataset_dir, class_folder)

            # Ensure it's a valid class directory
            if not os.path.isdir(class_path):
                continue

            class_label = synset_mapping.get(class_folder, "unknown")

            for img_file in sorted(os.listdir(class_path)):
                if img_file.lower().endswith((".jpeg", ".jpg", ".png")):
                    img_path = os.path.join(class_path, img_file)

                    # Ensure file exists and is readable
                    if not os.path.exists(img_path):
                        print(f"Warning: Skipping missing or unreadable file {img_path}")
                        continue

                    metadata.append({
                        "data": os.path.relpath(img_path, dataset_root_abs),  # Relative to DATASET_ROOT
                        "label": class_label
                    })

    # Save JSON file
    with open(output_path, "w") as json_file:
        json.dump(metadata, json_file, indent=2)

    print(f"Metadata saved to {output_path} ({len(metadata)} entries).")

def save_center_to_wordnet_mapping(output_file, human_to_synset):
    """Saves the human-readable name to WordNet ID mapping (center to WordNet)."""
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, "w") as json_file:
        json.dump(human_to_synset, json_file, indent=2)
    
    print(f"Center to WordNet mapping saved to {output_path} ({len(human_to_synset)} entries).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ImageNet metadata and class mappings.")
    parser.add_argument("DATASET_ROOT", type=str, help="Path to the ImageNet dataset root directory.")

    args = parser.parse_args()

    # Ensure the output directory exists in the current working directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading WordNet synset mapping")
    synset_to_human, human_to_synset = load_synset_mapping(args.DATASET_ROOT)

    print("Generating metadata for training set")
    generate_metadata(args.DATASET_ROOT, "train", "train_data.json", synset_to_human, labeled=True)

    print("Generating metadata for validation set")
    generate_metadata(args.DATASET_ROOT, "val", "val_data.json", synset_to_human, labeled=True)

    print("Generating metadata for test set")
    generate_metadata(args.DATASET_ROOT, "test", "test_data.json", synset_to_human, labeled=False)

    print("Saving center to WordNet mapping")
    save_center_to_wordnet_mapping("center_to_wordnet.json", human_to_synset)

    print("Metadata generation complete. All files saved in ImageNet-1K directory relative to the script.")
