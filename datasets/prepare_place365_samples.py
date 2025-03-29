#!/usr/bin/env python3
import argparse
import os
import json
import shutil
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Copy sampled files (from a JSON) to val_[sample_size] folder and create a label list using places365_val.txt"
    )
    parser.add_argument(
        "--json_file",
        default="./Places365/val_data_5000.json",
        help="Path to the JSON file with sampled items (default: ./Places365/val_data_5000.json)"
    )
    parser.add_argument(
        "--dataset_root",
        default="/home/user/datasets/Places365",
        help="Root directory where the files and places365_val.txt are stored (default: /home/user/datasets/Places365)"
    )
    parser.add_argument(
        "--label_file",
        default="places365_val.txt",
        help="Name of the label file inside dataset_root that maps filenames to numeric labels (default: places365_val.txt)"
    )
    parser.add_argument(
        "--output_root",
        default="/home/user/datasets/Places365",
        help="Directory to store the output folder (default: /home/user/datasets/Places365)"
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Load the JSON file (list of objects with "data")
    # --------------------------------------------------
    try:
        with open(args.json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found: {args.json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        sys.exit(1)

    sample_size = len(data)
    if sample_size == 0:
        print("ERROR: No entries found in JSON. Exiting.")
        sys.exit(1)

    # --------------------------------------------------
    # 2. Parse places365_val.txt to build filename->label
    # --------------------------------------------------
    label_path = os.path.join(args.dataset_root, args.label_file)
    label_map = {}
    try:
        with open(label_path, "r") as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                # Each line in places365_val.txt looks like: "Places365_val_00000001.jpg 165"
                parts = line.split()
                if len(parts) != 2:
                    continue
                filename, numeric_label = parts
                label_map[filename] = numeric_label
    except FileNotFoundError:
        print(f"ERROR: Label file not found: {label_path}")
        sys.exit(1)

    # --------------------------------------------------
    # 3. Create the output folder: val_[sample_size]
    # --------------------------------------------------
    output_folder_name = f"val_{sample_size}"
    output_folder_path = os.path.join(args.output_root, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # --------------------------------------------------
    # 4. Copy each file & build lines for the new .txt
    # --------------------------------------------------
    text_file_entries = []
    not_found_count = 0

    for item in data:
        rel_path = item["data"]  # e.g. "val_large/Places365_val_00024649.jpg"

        # Source file
        src_path = os.path.join(args.dataset_root, rel_path)

        # Filename alone (flatten subdirectories)
        filename = os.path.basename(rel_path)

        # Attempt to look up numeric label from label_map
        if filename not in label_map:
            not_found_count += 1
            print(f"WARNING: '{filename}' not found in {args.label_file}. Label set to -1.")
            numeric_label = -1  # or some placeholder
        else:
            numeric_label = label_map[filename]

        # Copy
        if os.path.exists(src_path):
            dest_path = os.path.join(output_folder_path, filename)
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} -> {dest_path}")

            # IMPORTANT: Only the filename, plus the numeric label
            text_file_entries.append(f"{filename} {numeric_label}")
        else:
            print(f"WARNING: Source file does not exist: {src_path}")

    # --------------------------------------------------
    # 5. Write the new places365_val_[sample_size].txt
    # --------------------------------------------------
    txt_filename = f"places365_val_{sample_size}.txt"
    txt_path = os.path.join(args.dataset_root, txt_filename)
    with open(txt_path, "w") as txt_file:
        for entry in text_file_entries:
            txt_file.write(entry + "\n")

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------
    print(f"\nDone. Copied {len(text_file_entries)} files into '{output_folder_path}'.")
    print(f"Created the label file: {txt_path}")

    if not_found_count > 0:
        print(f"WARNING: {not_found_count} file(s) not found in label map.")

if __name__ == "__main__":
    main()
