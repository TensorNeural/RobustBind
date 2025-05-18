#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def generate_caption_metadata(coco_subdir, split, output_dir):
    ann_file = os.path.join(coco_subdir, "annotations", f"captions_{split}2017.json")
    if not os.path.exists(ann_file):
        print(f"[!] Skipping {split} — missing: {ann_file}")
        return

    data = load_json(ann_file)
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    image_id_to_captions = defaultdict(list)

    for ann in tqdm(data["annotations"], desc=f"Grouping captions ({os.path.basename(coco_subdir)} {split})"):
        image_id_to_captions[ann["image_id"]].append(ann["caption"])

    entries = []
    for image_id, captions in image_id_to_captions.items():
        if image_id not in image_id_to_filename:
            continue
        entries.append({
            "image": os.path.join(split, image_id_to_filename[image_id]),
            "image_id": image_id,
            "captions": captions
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train_data.json" if split == "train" else "val_data.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"[✓] {os.path.basename(coco_subdir)}-{split} metadata → {out_path} ({len(entries)} entries)")

def main():
    parser = argparse.ArgumentParser(description="Generate metadata for all COCO-style caption datasets")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to COCO root folder (containing caption, densecap, etc.)")
    args = parser.parse_args()

    for subdir in sorted(os.listdir(args.dataset_root)):
        coco_subdir = os.path.join(args.dataset_root, subdir)
        if not os.path.isdir(coco_subdir):
            continue
        ann_dir = os.path.join(coco_subdir, "annotations")
        if not os.path.exists(ann_dir):
            continue
        print(f"\n📁 Processing: {subdir}")
        output_path = os.path.join("./COCO", subdir)
        generate_caption_metadata(coco_subdir, "train", output_path)
        generate_caption_metadata(coco_subdir, "val", output_path)

if __name__ == "__main__":
    main()
