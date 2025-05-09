import os
import json
import argparse

OUTPUT_DIR = "UrbanSound8K"

def generate_metadata(dataset_root, split, output_name):
    split_dir = os.path.join(dataset_root, split)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    if not os.path.exists(split_dir):
        print(f"Missing {split}/ folder.")
        return

    metadata = []
    dataset_root_abs = os.path.abspath(dataset_root)

    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(class_path, fname)
                if os.path.exists(fpath):
                    metadata.append({
                        "data": os.path.relpath(fpath, dataset_root_abs),
                        "label": class_name
                    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"{split.upper()} metadata saved to {output_path} ({len(metadata)} entries).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata JSONs for UrbanSound8K.")
    parser.add_argument("DATASET_ROOT", help="Root of UrbanSound8K (with train/ and test/)")

    args = parser.parse_args()

    generate_metadata(args.DATASET_ROOT, "train", "train_data.json")
    generate_metadata(args.DATASET_ROOT, "test", "val_data.json")

    print("Metadata generation complete.")
