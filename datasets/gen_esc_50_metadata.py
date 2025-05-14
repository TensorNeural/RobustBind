import os
import json
import argparse

OUTPUT_DIR = "ESC-50"

def generate_metadata(dataset_root, dataset_name, output_filename, class_names):
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

        if class_folder not in class_names:
            class_names.append(class_folder)

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

def save_class_names(class_names):
    class_names.sort()
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    path = os.path.join(OUTPUT_DIR, "classes.json")
    with open(path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Class index mapping saved to {path} ({len(class_to_idx)} classes).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESC-50 train/test metadata.")
    parser.add_argument("DATASET_ROOT", type=str, help="Path to the ESC-50 dataset root")

    args = parser.parse_args()

    class_names = []
    generate_metadata(args.DATASET_ROOT, "train", "train_data.json", class_names)
    generate_metadata(args.DATASET_ROOT, "test", "val_data.json", class_names)
    save_class_names(class_names)

    print("Metadata generation complete.")
