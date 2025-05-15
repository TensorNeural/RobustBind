import os
import json
import argparse

def generate_metadata(dataset_root, split, output_dir, all_classes):
    split_dir = os.path.join(dataset_root, split)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}_data.json")

    metadata = []

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        readable_label = class_name.replace("_", " ")
        all_classes.add(readable_label)

        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(".pt"):
                rel_path = os.path.join(split, class_name, fname)
                metadata.append({
                    "data": rel_path,
                    "label": readable_label
                })

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Wrote {len(metadata)} entries to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset root containing train/ and val/")
    args = parser.parse_args()

    output_dir = os.path.join(os.getcwd(), "ModelNet40")  # Save JSONs to ./ModelNet40
    all_classes = set()

    generate_metadata(args.dataset_root, "train", output_dir, all_classes)
    generate_metadata(args.dataset_root, "val", output_dir, all_classes)

    # Save classes.json
    classes_path = os.path.join(output_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(sorted(all_classes), f, indent=2)
    print(f"✅ Saved {len(all_classes)} classes to {classes_path}")
