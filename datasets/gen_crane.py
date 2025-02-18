import os
import json
import argparse

# Mapping WordNet IDs to human-readable labels
CRANE_LABELS = {
    "n02012849": "crane",   # Bird crane (bird)
    "n03126707": "crane2"   # Construction crane (machine)
}

# Output directory is `ImageNet-1K/` under the current working directory
OUTPUT_DIR = os.path.join(os.getcwd(), "ImageNet-1K")

def generate_crane_metadata(dataset_root, dataset_name, output_filename):
    """Generates metadata JSON file for 'crane' and 'crane2' using relative paths."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    output_path = os.path.join(OUTPUT_DIR, output_filename)  # Save inside ImageNet-1K under current working dir

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory {dataset_dir} not found. Skipping metadata generation.")
        return

    metadata = []
    num_images = 0

    for class_folder in sorted(os.listdir(dataset_dir)):  # Sort for consistency
        if class_folder not in CRANE_LABELS:
            continue  # Skip non-crane classes

        class_path = os.path.join(dataset_dir, class_folder)

        # Ensure it's a valid class directory
        if not os.path.isdir(class_path):
            continue

        for img_file in sorted(os.listdir(class_path)):
            if img_file.lower().endswith((".jpeg", ".jpg", ".png")):
                # Compute relative path with respect to dataset_root
                relative_path = os.path.relpath(os.path.join(class_path, img_file), dataset_root)
                metadata.append({
                    "data": relative_path,  # Use correct relative path
                    "label": CRANE_LABELS[class_folder]  # Convert WordNet ID to label
                })
                num_images += 1

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save JSON file
    with open(output_path, "w") as json_file:
        json.dump(metadata, json_file, indent=2)

    print(f"[INFO] Crane metadata saved to {output_path} ({num_images} entries).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for 'crane' images in ImageNet.")
    parser.add_argument("DATASET_ROOT", type=str, help="Path to the ImageNet dataset root directory.")

    args = parser.parse_args()

    print(f"[INFO] Generating metadata for crane images from {args.DATASET_ROOT}")
    generate_crane_metadata(args.DATASET_ROOT, "val", "crane_test_data.json")

    print(f"[INFO] Crane metadata generation complete. Metadata saved in {OUTPUT_DIR}.")
