import os
import zipfile
import tarfile
import shutil
from tqdm import tqdm

def unzip_file(zip_path, extract_to):
    print(f"📦 Unzipping {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def untar_class_archives(part_dir):
    print(f"📂 Untarring .tar.gz files in {os.path.basename(part_dir)}...")
    tar_files = [f for f in os.listdir(part_dir) if f.endswith(".tar.gz")]
    for tar_name in tqdm(tar_files, desc=f"Untarring {os.path.basename(part_dir)}"):
        tar_path = os.path.join(part_dir, tar_name)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=part_dir)
        os.remove(tar_path)

def move_class_folders_to_train(part_dir, train_dir):
    print(f"🚚 Moving class folders from {os.path.basename(part_dir)} to train/...")
    for item in os.listdir(part_dir):
        full_path = os.path.join(part_dir, item)
        if os.path.isdir(full_path) and item.startswith("n"):  # e.g., n01440764
            shutil.move(full_path, os.path.join(train_dir, item))

def copy_validation_variants(val_root, val_dir):
    print(f"📂 Copying validation variants...")
    for fname in tqdm(os.listdir(val_root), desc="Copying val"):
        src = os.path.join(val_root, fname)
        if os.path.isdir(src):  # misnamed .zip folders
            for class_name in os.listdir(src):
                src_class = os.path.join(src, class_name)
                dst_class = os.path.join(val_dir, class_name)
                os.makedirs(dst_class, exist_ok=True)
                for img in os.listdir(src_class):
                    shutil.copy2(os.path.join(src_class, img), os.path.join(dst_class, img))

def prepare_n_imagenet_1k(dataset_root):
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    training_zip_dir = os.path.join(dataset_root, "training")
    validation_variants_dir = os.path.join(dataset_root, "validation")

    # Extract and untar Part_*.zip
    for zip_name in sorted(os.listdir(training_zip_dir)):
        if not zip_name.endswith(".zip"): continue
        part_name = zip_name.replace(".zip", "")  # e.g., Part_1
        zip_path = os.path.join(training_zip_dir, zip_name)
        part_dir = os.path.join(training_zip_dir, part_name)

        if not os.path.exists(part_dir):
            unzip_file(zip_path, training_zip_dir)

        untar_class_archives(part_dir)
        move_class_folders_to_train(part_dir, train_dir)

    # Copy validation folders (e.g., val_mode_7, val_brightness_5, etc.)
    copy_validation_variants(validation_variants_dir, val_dir)

    print("✅ Finished preparing N-ImageNet-1K")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    args = parser.parse_args()

    prepare_n_imagenet_1k(args.dataset_root)
