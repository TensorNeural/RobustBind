#!/usr/bin/env python3

import os
import shutil
import argparse
import zipfile
import concurrent.futures
from tqdm import tqdm

def unzip_with_progress(zip_path, extract_to):
    """
    Unzip a .zip file to a specified directory using a progress bar.
    """
    print(f"Unzipping '{zip_path}' into '{extract_to}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, path=extract_to)
    print("Extraction complete.\n")

def move_files_in_parallel(src_dst_pairs, max_workers=None):
    """
    Moves files in parallel given a list of (src, dst) pairs.
    If the destination is on the same filesystem, this will be a simple rename.
    Otherwise, it will copy and then delete the original.
    """
    if not src_dst_pairs:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(shutil.move, src, dst) for src, dst in src_dst_pairs]
        concurrent.futures.wait(futures)

def move_train(dataset_root, max_workers=None):
    """
    Moves training images into a top-level 'train/' folder
    (preserving class folders) using multiple workers.
    """
    train_src_dir = os.path.join(dataset_root, "ILSVRC", "Data", "CLS-LOC", "train")
    train_dest_dir = os.path.join(dataset_root, "train")

    if not os.path.exists(train_src_dir):
        print(f"Training data not found at: {train_src_dir}. Skipping TRAIN.")
        return

    os.makedirs(train_dest_dir, exist_ok=True)

    print("Starting processing of TRAIN dataset...")

    class_count = 0
    for class_folder in sorted(os.listdir(train_src_dir)):
        src_class_dir = os.path.join(train_src_dir, class_folder)
        if not os.path.isdir(src_class_dir):
            continue

        dest_class_dir = os.path.join(train_dest_dir, class_folder)
        os.makedirs(dest_class_dir, exist_ok=True)
        print(f"  Processing class: {class_folder}...")

        file_pairs = []
        for img_name in sorted(os.listdir(src_class_dir)):
            src = os.path.join(src_class_dir, img_name)
            dst = os.path.join(dest_class_dir, img_name)
            file_pairs.append((src, dst))

        move_files_in_parallel(file_pairs, max_workers=max_workers)
        print(f"  Moved {len(file_pairs)} images for class '{class_folder}'.")
        class_count += 1

    print(f"TRAIN dataset moved successfully: {class_count} classes.\n")

def move_test(dataset_root, max_workers=None):
    """
    Moves test images into a top-level 'test/' folder using multiple workers.
    """
    test_src_dir = os.path.join(dataset_root, "ILSVRC", "Data", "CLS-LOC", "test")
    test_dest_dir = os.path.join(dataset_root, "test")

    if not os.path.exists(test_src_dir):
        print(f"Test data not found at: {test_src_dir}. Skipping TEST.")
        return

    os.makedirs(test_dest_dir, exist_ok=True)
    print("Starting processing of TEST dataset...")

    file_pairs = []
    for img_name in sorted(os.listdir(test_src_dir)):
        src = os.path.join(test_src_dir, img_name)
        dst = os.path.join(test_dest_dir, img_name)
        if os.path.isfile(src):
            file_pairs.append((src, dst))

    move_files_in_parallel(file_pairs, max_workers=max_workers)
    print(f"TEST dataset moved successfully: {len(file_pairs)} images.\n")

def move_val(dataset_root, max_workers=None):
    """
    Moves validation images into class directories based on LOC_val_solution.csv
    (which should be in the dataset_root).
    """
    val_src_dir = os.path.join(dataset_root, "ILSVRC", "Data", "CLS-LOC", "val")
    val_dest_dir = os.path.join(dataset_root, "val")
    val_labels_file = os.path.join(dataset_root, "LOC_val_solution.csv")

    if not os.path.exists(val_src_dir):
        print(f"Validation data not found at: {val_src_dir}. Skipping VAL.")
        return
    if not os.path.exists(val_labels_file):
        print(f"LOC_val_solution.csv not found at: {val_labels_file}. Skipping VAL.")
        return

    os.makedirs(val_dest_dir, exist_ok=True)
    print("Starting processing of VALIDATION dataset...")

    val_labels = []
    with open(val_labels_file, "r") as f:
        next(f, None)  # skip header if it exists
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                image_name = parts[0]
                # For multi-label lines (e.g., "class1 class2"), take only the first
                class_id = parts[1].split(" ")[0]
                val_labels.append((image_name, class_id))

    total_images = len(val_labels)
    print(f"  Organizing {total_images} validation images into class directories...")

    file_pairs = []
    for img_name, class_id in val_labels:
        img_name_jpeg = img_name + ".JPEG"
        class_dir = os.path.join(val_dest_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(val_src_dir, img_name_jpeg)
        dst = os.path.join(class_dir, img_name_jpeg)
        if os.path.exists(src):
            file_pairs.append((src, dst))

    move_files_in_parallel(file_pairs, max_workers=max_workers)

    unique_classes = len({cls for _, cls in val_labels})
    print(f"VALIDATION dataset moved successfully into {unique_classes} classes.\n")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the ImageNet dataset.\n\n"
            "STEPS:\n"
            "  1) Automatically unzip (if --zip-file is given).\n"
            "  2) Move train images.\n"
            "  3) Move test images.\n"
            "  4) Organize validation images by class.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "DATASET_ROOT",
        nargs="?",
        default=".",
        help="Path to the dataset root directory (where ILSVRC folder will reside). Defaults to current directory."
    )
    parser.add_argument(
        "--zip-file",
        type=str,
        default=None,
        help="Optional path to the .zip file. If provided, the script will unzip it into DATASET_ROOT first."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker threads for file moving (default: system default)."
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.DATASET_ROOT)

    # 1. If --zip-file is specified, unzip it to dataset_root
    if args.zip_file:
        zip_file_path = os.path.abspath(args.zip_file)
        if not os.path.isfile(zip_file_path):
            print(f"ERROR: Zip file not found: {zip_file_path}")
            return
        unzip_with_progress(zip_file_path, dataset_root)

    # 2. Move TRAIN data
    move_train(dataset_root, max_workers=args.max_workers)

    # 3. Move TEST data
    move_test(dataset_root, max_workers=args.max_workers)

    # 4. Move VAL data
    move_val(dataset_root, max_workers=args.max_workers)

    print("ImageNet dataset preparation complete.")

if __name__ == "__main__":
    main()
