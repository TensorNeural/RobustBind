import os
import json
import argparse

OUTPUT_DIR = "UCF-101"

def generate_metadata(dataset_root, split_name, output_name):
    split_dir = os.path.join(dataset_root, split_name)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    metadata = []
    dataset_root_abs = os.path.abspath(dataset_root)

    if not os.path.exists(split_dir):
        print(f"Error: {split_dir} not found")
        return

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(".avi"):
                fpath = os.path.join(class_dir, fname)
                metadata.append({
                    "data": os.path.relpath(fpath, dataset_root_abs),
                    "label": class_name
                })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"{split_name.upper()} metadata saved to {output_path} ({len(metadata)} entries).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UCF101 metadata.")
    parser.add_argument("DATASET_ROOT", help="Root of the UCF101 dataset (with train/ and test/)")
    args = parser.parse_args()

    generate_metadata(args.DATASET_ROOT, "train", "train_data.json")
    generate_metadata(args.DATASET_ROOT, "test", "val_data.json")

    print("Metadata generation complete.")
