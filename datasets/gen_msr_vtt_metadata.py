import os
import json
import argparse

def load_category_map(category_file):
    category_map = {}
    with open(category_file, "r") as f:
        for line in f:
            if "\t" in line:
                label, idx = line.strip().split("\t")
                category_map[int(idx)] = label
    return category_map

def generate_metadata(info_json_path, split_dir, dataset_root, category_map, split_name, output_base_dir):
    with open(info_json_path, "r") as f:
        data = json.load(f)

    metadata = []
    for item in data["videos"]:
        video_id = item["video_id"]
        category_id = item["category"]
        label = category_map.get(category_id, "unknown")

        video_file = f"{video_id}.mp4"
        full_path = os.path.join(dataset_root, split_dir, video_file)

        if not os.path.isfile(full_path):
            print(f"⚠️ Skipping missing file: {full_path}")
            continue

        metadata.append({
            "data": os.path.relpath(full_path, dataset_root),
            "label": label
        })

    os.makedirs(output_base_dir, exist_ok=True)
    output_path = os.path.join(output_base_dir, f"{split_name}_data.json")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved {len(metadata)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSR-VTT metadata JSONs (data, label) from train/val dirs.")
    parser.add_argument("dataset_root", type=str, help="Path to MSR-VTT dataset root containing train/ and val/")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    category_file = os.path.join(dataset_root, "category.txt")
    output_dir = os.path.join(os.getcwd(), "MSR-VTT")

    category_map = load_category_map(category_file)

    generate_metadata(
        info_json_path=os.path.join(dataset_root, "train_val_videodatainfo.json"),
        split_dir="train",
        dataset_root=dataset_root,
        category_map=category_map,
        split_name="train",
        output_base_dir=output_dir
    )

    generate_metadata(
        info_json_path=os.path.join(dataset_root, "test_videodatainfo.json"),
        split_dir="val",
        dataset_root=dataset_root,
        category_map=category_map,
        split_name="val",
        output_base_dir=output_dir
    )

    print("🎉 Done. Metadata saved to ./MSR-VTT/train_data.json and val_data.json")
