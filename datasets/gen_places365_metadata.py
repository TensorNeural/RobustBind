import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

OUTPUT_DIR = "Places365"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_immediate_subdirs(p: Path) -> List[Path]:
    if not p.exists() or not p.is_dir():
        return []
    return sorted([d for d in p.iterdir() if d.is_dir()])


def discover_classes(dataset_root: Path) -> List[str]:
    """Discover class names from the directory structure (prefer train/, else val/ or val_large/)."""
    for split in ("train", "val", "val_large"):
        split_dir = dataset_root / split
        subdirs = list_immediate_subdirs(split_dir)
        if subdirs:
            return sorted([d.name for d in subdirs])
    # If no subdirs found, return empty (cannot infer classes without mapping files)
    return []


def build_label_maps(classes: List[str]) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    """Build mapping from class name to integer id and reverse mapping for compatibility."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    label_to_categories = {c: [i] for c, i in cls_to_idx.items()}
    return cls_to_idx, label_to_categories


def iter_images_in_class_dir(class_dir: Path) -> List[Path]:
    files = []
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return files


def generate_split_metadata(dataset_root: Path, split: str, classes: List[str], cls_to_idx: Dict[str, int]) -> List[Dict]:
    split_dir = dataset_root / split
    if not split_dir.exists():
        print(f"Info: {split_dir} not found; skipping {split}.")
        return []

    # If split is not organized by class subdirectories, we cannot label without mapping files
    subdirs = list_immediate_subdirs(split_dir)
    if not subdirs:
        print(f"Warning: {split_dir} has no class subdirectories; skipping labeled metadata for {split}.")
        return []

    # Only include classes that actually have directories in this split (robust to partial downloads)
    present_classes = {d.name for d in subdirs if d.name in cls_to_idx}

    records: List[Dict] = []
    for cls in sorted(present_classes):
        class_dir = split_dir / cls
        for img_path in iter_images_in_class_dir(class_dir):
            rel_path = str(Path(split) / cls / img_path.name)
            records.append({
                "data": rel_path,
                "label": cls
            })
    print(f"Collected {len(records)} images for split '{split}'.")
    return records


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote {path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Places365 metadata from directory structure.")
    parser.add_argument("DATASET_ROOT", type=str, help="Path to the Places365 dataset root directory (containing train/ and/or val/).")
    args = parser.parse_args()

    dataset_root = Path(args.DATASET_ROOT)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover classes from the folder structure
    classes = discover_classes(dataset_root)
    if not classes:
        print("Error: Could not discover classes from folder structure (train/ or val/ with subdirectories). Aborting.")
        raise SystemExit(1)

    cls_to_idx, label_to_categories = build_label_maps(classes)

    # Generate metadata for available splits
    all_records = {}
    for split in ("train", "val"):
        recs = generate_split_metadata(dataset_root, split, classes, cls_to_idx)
        if recs:
            filename = f"{split}_data.json" if split != "val_large" else "val_data.json"
            save_json(out_dir / filename, recs)
            all_records[split] = len(recs)

    # Save label mappings and class list
    save_json(out_dir / "center_to_places365.json", label_to_categories)
    save_json(out_dir / "classes_places365.json", classes)

    print("Summary:")
    for split, n in all_records.items():
        print(f" - {split}: {n} images")
    print("Done.")
