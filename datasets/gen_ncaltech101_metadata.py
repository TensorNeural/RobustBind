import os
import json
import argparse


def generate_bin_json(dataset_root, dataset_name):
    val_path = os.path.join(dataset_root, "val")
    if not os.path.isdir(val_path):
        print(f"❌ Error: {val_path} not found.")
        return

    output_dir = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "val_data.json")

    entries = []
    for class_name in sorted(os.listdir(val_path)):
        class_dir = os.path.join(val_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(".bin"):
                full_path = os.path.join(class_dir, fname)
                if os.path.isfile(full_path):
                    entries.append({
                        "data": os.path.relpath(full_path, dataset_root),
                        "label": class_name
                    })

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"✅ .bin metadata → {output_path} ({len(entries)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .bin metadata JSON for N-Caltech-101.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to root directory with val/")
    parser.add_argument("--dataset_name", type=str, default="N-Caltech-101", help="Output folder name under current working dir")
    args = parser.parse_args()

    generate_bin_json(args.dataset_root, args.dataset_name)

    print("🎉 Metadata generation complete.")
