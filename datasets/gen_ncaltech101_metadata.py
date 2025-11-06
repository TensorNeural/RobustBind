import os
import json
import argparse


def _collect_bin_entries(split_path: str, dataset_root: str):
    entries = []
    if not os.path.isdir(split_path):
        print(f"ℹ️  Split not found, skipping: {split_path}")
        return entries
    for class_name in sorted(os.listdir(split_path)):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(".bin"):
                full_path = os.path.join(class_dir, fname)
                if os.path.isfile(full_path):
                    entries.append({
                        "data": os.path.relpath(full_path, dataset_root),
                        "label": class_name,
                    })
    return entries


def generate_bin_json_both_splits(dataset_root, dataset_name):
    output_dir = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Train
    train_entries = _collect_bin_entries(os.path.join(dataset_root, "train"), dataset_root)
    if train_entries:
        out_train = os.path.join(output_dir, "train_data.json")
        with open(out_train, "w") as f:
            json.dump(train_entries, f, indent=2)
        print(f"✅ .bin metadata → {out_train} ({len(train_entries)} entries)")
    else:
        print("ℹ️  No train entries found; train_data.json not written.")

    # Val
    val_entries = _collect_bin_entries(os.path.join(dataset_root, "val"), dataset_root)
    if val_entries:
        out_val = os.path.join(output_dir, "val_data.json")
        with open(out_val, "w") as f:
            json.dump(val_entries, f, indent=2)
        print(f"✅ .bin metadata → {out_val} ({len(val_entries)} entries)")
    else:
        print("ℹ️  No val entries found; val_data.json not written.")

    # Classes
    all_entries = (train_entries or []) + (val_entries or [])
    if all_entries:
        class_names = sorted({e["label"] for e in all_entries})
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        classes_path = os.path.join(output_dir, "classes.json")
        with open(classes_path, "w") as f:
            json.dump(class_to_idx, f, indent=2)
        print(f"📄 Saved classes.json with {len(class_to_idx)} classes → {classes_path}")
    else:
        print("ℹ️  No entries found; classes.json not written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .bin metadata JSON for N-Caltech-101 (both train and val).")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to root directory with train/ and/or val/")
    parser.add_argument("--dataset_name", type=str, default="N-Caltech-101", help="Output folder name under current working dir")
    args = parser.parse_args()

    generate_bin_json_both_splits(args.dataset_root, args.dataset_name)

    print("🎉 Metadata generation complete.")
