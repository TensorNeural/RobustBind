import os
import json
import argparse

OUTPUT_DIR = "ESC-50"

def generate_metadata(dataset_root, dataset_name, output_filename):
    dataset_dir = os.path.join(dataset_root, dataset_name)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found. Skipping...")
        return

    metadata = []
    dataset_root_abs = os.path.abspath(dataset_root)

    for class_folder in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(class_path, fname)
                if os.path.exists(fpath):
                    metadata.append({
                        "data": os.path.relpath(fpath, dataset_root_abs),
                        "label": class_folder
                    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"{dataset_name.upper()} metadata saved to {output_path} ({len(metadata)} entries).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESC-50 train/test metadata.")
    parser.add_argument("DATASET_ROOT", type=str, help="Path to the ESC-50 dataset root")

    args = parser.parse_args()

    generate_metadata(args.DATASET_ROOT, "train", "train_data.json")
    generate_metadata(args.DATASET_ROOT, "test", "val_data.json")

    print("Metadata generation complete.")
