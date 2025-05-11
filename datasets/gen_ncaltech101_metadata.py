import os
import json
import argparse

def generate_metadata(dataset_root, split_dir, output_name):
    split_path = os.path.join(dataset_root, split_dir)
    if not os.path.isdir(split_path):
        print(f"❌ Error: {split_path} not found.")
        return

    output_dir = os.path.join(os.getcwd(), "N-Caltech-101")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    entries = []
    for class_name in sorted(os.listdir(split_path)):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(class_dir, fname)
                if os.path.isfile(full_path):
                    entries.append({
                        "data": os.path.relpath(full_path, dataset_root),
                        "label": class_name
                    })

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"✅ {split_dir}: {len(entries)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for N-Caltech-101 (val only → JSON).")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root with val/")
    args = parser.parse_args()

    generate_metadata(args.dataset_root, "val", "val_data.json")
    print("🎉 Metadata generation complete.")
