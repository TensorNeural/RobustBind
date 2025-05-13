import os
import json
import argparse

# File paths for labels and category mapping
VAL_LABELS_FILE = "places365_val.txt"
TRAIN_LABELS_FILE = "places365_train_standard.txt"
CATEGORY_MAPPING_FILE = "categories_places365.txt"
OUTPUT_DIR = "Places365"

def load_category_mapping(dataset_root):
    """Loads category-to-label mappings from categories_places365.txt, ensuring the third part is used when available."""
    mapping_file = os.path.join(dataset_root, CATEGORY_MAPPING_FILE)
    category_to_label = {}
    label_to_categories = {}

    if not os.path.exists(mapping_file):
        print(f"Error: {mapping_file} not found.")
        return category_to_label, label_to_categories

    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                full_category_name, category_index = parts[0], int(parts[1])
                category_parts = full_category_name.split("/")
                if len(category_parts) >= 3:
                    category_name = category_parts[2]
                else:
                    category_name = category_parts[-1]

                category_to_label[category_index] = category_name

                if category_name in label_to_categories:
                    label_to_categories[category_name].append(category_index)
                else:
                    label_to_categories[category_name] = [category_index]

    print(f"Loaded {len(category_to_label)} scene categories.")
    return category_to_label, label_to_categories

def load_image_labels(dataset_root, label_file):
    """Loads image-to-category mappings from label text files."""
    labels_path = os.path.join(dataset_root, label_file)
    image_to_label = {}

    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found.")
        return image_to_label

    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_path = parts[0]
                image_to_label[img_path] = int(parts[1])

    print(f"Loaded {len(image_to_label)} image labels from {label_file}.")
    return image_to_label

def generate_metadata(dataset_root, dataset_name, output_filename, label_file, category_mapping):
    """Loads labels and generates metadata JSON file for Places365 dataset."""
    dataset_dir = os.path.join(dataset_root, dataset_name)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.exists(dataset_dir):
        print(f"Error: {dataset_dir} not found. Skipping...")
        return

    image_labels = load_image_labels(dataset_root, label_file)
    metadata = []

    for img_relative_path, category_index in image_labels.items():
        img_path = os.path.join(dataset_dir, img_relative_path)

        if not os.path.exists(img_path):
            print(f"Warning: Skipping missing file {img_path}")
            continue

        category_name = category_mapping.get(category_index, "unknown")

        metadata.append({
            "data": os.path.join(dataset_name, img_relative_path),
            "label": category_name
        })

    with open(output_path, "w") as json_file:
        json.dump(metadata, json_file, indent=2)

    print(f"Metadata saved to {output_path} ({len(metadata)} entries).")

def save_label_mapping(output_file, label_to_categories):
    """Saves the label-to-category mapping, supporting multiple indices per category."""
    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, "w") as json_file:
        json.dump(label_to_categories, json_file, indent=2)

    print(f"Label mapping saved to {output_path} ({len(label_to_categories)} entries).")

def generate_class_name_list(dataset_root, output_file):
    """Extracts class names from categories_places365.txt and saves as a list of strings."""
    mapping_file = os.path.join(dataset_root, CATEGORY_MAPPING_FILE)

    if not os.path.exists(mapping_file):
        print(f"Error: {mapping_file} not found.")
        return

    class_names = set()

    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                category_path = parts[0]
                path_parts = category_path.split("/")
                if len(path_parts) >= 3:
                    class_name = path_parts[2]
                else:
                    class_name = path_parts[-1]
                class_names.add(class_name)

    sorted_classes = sorted(class_names)

    output_path = os.path.join(OUTPUT_DIR, output_file)
    with open(output_path, "w") as f:
        json.dump(sorted_classes, f, indent=2)

    print(f"Class name list saved to {output_path} ({len(sorted_classes)} classes).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Places365 metadata.")
    parser.add_argument("DATASET_ROOT", type=str, help="Path to the Places365 dataset root directory.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading category mappings...")
    category_to_label, label_to_categories = load_category_mapping(args.DATASET_ROOT)

    # print("Processing training dataset...")
    # generate_metadata(args.DATASET_ROOT, "train", "train_data.json", TRAIN_LABELS_FILE, category_to_label)

    print("Processing validation dataset...")
    generate_metadata(args.DATASET_ROOT, "val_large", "val_data.json", VAL_LABELS_FILE, category_to_label)

    print("Saving label-to-category mapping...")
    save_label_mapping("center_to_places365.json", label_to_categories)

    print("Saving class name list...")
    generate_class_name_list(args.DATASET_ROOT, "classes_places365.json")

    print("Metadata generation complete. Files saved in Places365 directory.")
