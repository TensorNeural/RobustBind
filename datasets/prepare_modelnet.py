#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
import argparse


def parse_off(path):
    with open(path, "r") as f:
        lines = f.readlines()
    if lines[0].strip() == "OFF":
        lines = lines[1:]
    else:
        lines[0] = lines[0][3:]

    try:
        n_vertices = int(lines[0].split()[0])
        vertices = [list(map(float, line.strip().split())) for line in lines[1:1 + n_vertices]]
        return torch.tensor(vertices, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Error parsing {path}: {e}")


def normalize_pointcloud(pc: torch.Tensor, num_points: int = 8192):
    n = pc.shape[0]
    if n == num_points:
        return pc
    elif n > num_points:
        indices = torch.randperm(n)[:num_points]
        return pc[indices]
    else:
        padding = torch.zeros((num_points - n, 3), dtype=pc.dtype)
        return torch.cat([pc, padding], dim=0)


def convert_off_to_pt(dataset_root: str, num_points: int = 8192):
    source_root = os.path.join(dataset_root, "ModelNet40")
    output_train = os.path.join(dataset_root, "train")
    output_val = os.path.join(dataset_root, "val")
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)

    print(f"📦 Converting .off → .pt (shape: [{num_points}, 3])")

    for class_name in sorted(os.listdir(source_root)):
        class_path = os.path.join(source_root, class_name)
        if not os.path.isdir(class_path):
            continue

        for split in ["train", "test"]:
            input_split = os.path.join(class_path, split)
            output_split = os.path.join(output_train if split == "train" else output_val, class_name)
            os.makedirs(output_split, exist_ok=True)

            files = sorted(f for f in os.listdir(input_split) if f.endswith(".off"))
            count = 0
            for file in tqdm(files, desc=f"{class_name}/{split}", leave=False):
                input_path = os.path.join(input_split, file)
                try:
                    pc_tensor = parse_off(input_path)
                    pc_tensor = normalize_pointcloud(pc_tensor, num_points=num_points)

                    out_name = f"{class_name}_{count:04d}.pt"
                    torch.save(pc_tensor, os.path.join(output_split, out_name))
                    count += 1
                except Exception as e:
                    print(f"⚠️ Skipped {input_path}: {e}")

            print(f"✅ {split} - {class_name}: saved {count} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to directory containing ModelNet40/")
    parser.add_argument("--num_points", type=int, default=8192,
                        help="Number of points to normalize to")
    args = parser.parse_args()

    convert_off_to_pt(args.dataset_root, args.num_points)
