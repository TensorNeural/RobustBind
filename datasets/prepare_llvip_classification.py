import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image, ImageStat
from tqdm import tqdm


def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        if obj.find("name").text != "person":
            continue
        b = obj.find("bndbox")
        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def expand_box(box, image_size, padding=0.3):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    return (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(image_size[0], x2 + pad_w),
        min(image_size[1], y2 + pad_h),
    )


def is_dark_crop(crop, threshold=15):
    gray = crop.convert("L")
    stat = ImageStat.Stat(gray)
    return stat.mean[0] < threshold


def get_centered_224_crop(image, person_box, image_size, crop_size=224, padding=0.3):
    img_w, img_h = image_size
    x1, y1, x2, y2 = person_box
    box_w, box_h = x2 - x1, y2 - y1

    if box_w >= crop_size or box_h >= crop_size:
        padded = expand_box(person_box, image_size, padding)
        crop = image.crop(padded)
        return crop.resize((crop_size, crop_size), Image.BICUBIC)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = crop_size // 2
    crop_x1 = max(0, cx - half)
    crop_y1 = max(0, cy - half)
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    if crop_x2 > img_w:
        crop_x1 = img_w - crop_size
        crop_x2 = img_w
    if crop_y2 > img_h:
        crop_y1 = img_h - crop_size
        crop_y2 = img_h

    if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > img_w or crop_y2 > img_h:
        return None
    if not (crop_x1 <= x1 and crop_x2 >= x2 and crop_y1 <= y1 and crop_y2 >= y2):
        return None

    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def fast_background_crop(image, person_boxes, image_size, crop_size=224, max_tries=100):
    img_w, img_h = image_size
    for _ in range(max_tries):
        x1 = random.randint(0, img_w - crop_size)
        y1 = random.randint(0, img_h - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        overlaps = any(not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2)
                       for bx1, by1, bx2, by2 in person_boxes)
        if not overlaps:
            crop = image.crop((x1, y1, x2, y2))
            if not is_dark_crop(crop):
                return crop
    return None


def generate_crops(dataset_root, crop_size=224):
    ir_root = os.path.join(dataset_root, "infrared")
    anno_dir = os.path.join(dataset_root, "Annotations")
    person_out = os.path.join(dataset_root, "person")
    background_out = os.path.join(dataset_root, "background")
    os.makedirs(person_out, exist_ok=True)
    os.makedirs(background_out, exist_ok=True)

    subdirs = ["train", "test"]
    all_images = []
    for sub in subdirs:
        sub_dir = os.path.join(ir_root, sub)
        for fname in os.listdir(sub_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append((sub_dir, fname))

    random.shuffle(all_images)
    person_count = 0
    bg_count = 0

    print("🔹 Generating person and background crops from infrared/train and infrared/test...")
    for img_dir, fname in tqdm(all_images, desc="Images"):
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)
        anno_path = os.path.join(dataset_root, "Annotations", base + ".xml")
        if not os.path.exists(anno_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        boxes = parse_voc_annotation(anno_path)
        if not boxes:
            continue
        img_w, img_h = img.size

        for i, box in enumerate(boxes):
            crop = get_centered_224_crop(img, box, (img_w, img_h), crop_size)
            if crop:
                crop.save(os.path.join(person_out, f"{base}_p{i}_{person_count}.jpg"))
                person_count += 1

        for _ in range(6):
            crop = fast_background_crop(img, boxes, (img_w, img_h), crop_size)
            if crop:
                crop.save(os.path.join(background_out, f"{base}_bg{bg_count}.jpg"))
                bg_count += 1

    print(f"\n✅ Done: {person_count} person crops, {bg_count} background crops.")
    return person_count, bg_count


def balance_and_split(dataset_root, val_ratio=0.2, seed=42):
    random.seed(seed)

    person_dir = os.path.join(dataset_root, "person")
    background_dir = os.path.join(dataset_root, "background")

    person_files = sorted([f for f in os.listdir(person_dir) if f.endswith((".jpg", ".png"))])
    background_files = sorted([f for f in os.listdir(background_dir) if f.endswith((".jpg", ".png"))])

    n = min(len(person_files), len(background_files))
    print(f"📊 Using {n} balanced samples per class")

    person_selected = random.sample(person_files, n)
    background_selected = random.sample(background_files, n)

    val_n = int(n * val_ratio)
    train_n = n - val_n

    splits = {
        "train": {
            "person": person_selected[val_n:],
            "background": background_selected[val_n:]
        },
        "val": {
            "person": person_selected[:val_n],
            "background": background_selected[:val_n]
        }
    }

    for split_name in ["train", "val"]:
        for cls in ["person", "background"]:
            out_dir = os.path.join(dataset_root, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            src_dir = person_dir if cls == "person" else background_dir
            selected = splits[split_name][cls]

            print(f"🔹 Copying {len(selected)} {cls} → {split_name}/...")
            for fname in tqdm(selected, desc=f"{split_name}/{cls}"):
                shutil.copy(os.path.join(src_dir, fname), os.path.join(out_dir, fname))

    print("\n✅ Train/val split complete and balanced.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    # generate_crops(args.dataset_root)
    balance_and_split(args.dataset_root, val_ratio=args.val_ratio)

    print("🎉 All crops generated, balanced, and split into train/val successfully.")
