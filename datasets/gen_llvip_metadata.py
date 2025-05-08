import os
import json

def collect_split_metadata(split_name, dataset_root):
    entries = []
    for label in ["person", "background"]:
        class_dir = os.path.join(dataset_root, split_name, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rel_path = os.path.relpath(os.path.join(class_dir, fname), dataset_root)
            entries.append({
                "file_name": rel_path,
                "label": label
            })
    return entries

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of prepared LLVIP dataset")
    parser.add_argument("--output_dir", type=str, default="LLVIP", help="Where to write JSON files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_meta = collect_split_metadata("train", args.dataset_root)
    val_meta = collect_split_metadata("val", args.dataset_root)

    train_json = os.path.join(args.output_dir, "train_data.json")
    val_json = os.path.join(args.output_dir, "val_data.json")

    with open(train_json, "w") as f:
        json.dump(train_meta, f, indent=2)
    with open(val_json, "w") as f:
        json.dump(val_meta, f, indent=2)

    print(f"✅ Wrote {len(train_meta)} train entries → {train_json}")
    print(f"✅ Wrote {len(val_meta)} val entries   → {val_json}")
