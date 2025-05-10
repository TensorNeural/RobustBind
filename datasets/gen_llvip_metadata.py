import os
import json
import argparse

def generate_llvip_metadata(dataset_root, split_name, output_base_dir):
    split_dir = os.path.join(dataset_root, split_name)
    output_path = os.path.join(output_base_dir, f"{split_name}_data.json")

    if not os.path.isdir(split_dir):
        print(f"❌ Error: {split_dir} does not exist. Skipping.")
        return

    metadata = []
    for class_name in ["person", "background"]:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"⚠️ Warning: missing class dir {class_dir}, skipping...")
            continue

        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(class_dir, fname)
                if os.path.exists(full_path):
                    metadata.append({
                        "data": os.path.relpath(full_path, dataset_root),
                        "label": class_name
                    })

    os.makedirs(output_base_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved {len(metadata)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLVIP JSONs with {data, label} entries.")
    parser.add_argument("dataset_root", type=str, help="Path to LLVIP dataset root (contains train/ and val/)")
    args = parser.parse_args()

    output_dir = os.path.join(os.getcwd(), "LLVIP")

    generate_llvip_metadata(args.dataset_root, "train", output_dir)
    generate_llvip_metadata(args.dataset_root, "val", output_dir)

    print("🎉 Done. JSONs saved to ./LLVIP/train_data.json and val_data.json")
