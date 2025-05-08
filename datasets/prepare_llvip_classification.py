import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageStat
from tqdm import tqdm

MIN_CROP_SIZE = 64

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

def expand_box(box, image_size, padding=0.2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    new_x1 = max(0, x1 - pad_w)
    new_y1 = max(0, y1 - pad_h)
    new_x2 = min(image_size[0], x2 + pad_w)
    new_y2 = min(image_size[1], y2 + pad_h)
    return (new_x1, new_y1, new_x2, new_y2)

def is_dark_crop(crop, threshold=15):
    gray = crop.convert("L")
    stat = ImageStat.Stat(gray)
    return stat.mean[0] < threshold

def crop_background(image, boxes, image_size, box_size, max_tries=50):
    box_w, box_h = box_size
    if box_w < MIN_CROP_SIZE or box_h < MIN_CROP_SIZE:
        return None
    for _ in range(max_tries):
        x1 = random.randint(0, image_size[0] - box_w)
        y1 = random.randint(0, image_size[1] - box_h)
        x2 = x1 + box_w
        y2 = y1 + box_h
        overlaps = any(not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2) for bx1, by1, bx2, by2 in boxes)
        if not overlaps:
            crop = image.crop((x1, y1, x2, y2))
            if not is_dark_crop(crop) and crop.size[0] >= MIN_CROP_SIZE and crop.size[1] >= MIN_CROP_SIZE:
                return crop
    return None

def collect_all_person_crops(image_dir, anno_dir, output_dir, split_name, max_crops):
    out_dir = os.path.join(output_dir, split_name, "person")
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    image_list = sorted(os.listdir(image_dir))
    random.shuffle(image_list)

    print(f"\n🔹 Generating {max_crops} person crops for [{split_name}]...")
    for fname in tqdm(image_list, desc=f"[{split_name}] person"):
        if count >= max_crops:
            break
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        anno_path = os.path.join(anno_dir, base + ".xml")
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
            if count >= max_crops:
                break
            padded = expand_box(box, (img_w, img_h), padding=0.2)
            crop = img.crop(padded)
            if crop.size[0] >= MIN_CROP_SIZE and crop.size[1] >= MIN_CROP_SIZE:
                crop.save(os.path.join(out_dir, f"{base}_p{i}.jpg"))
                count += 1
    print(f"[{split_name}] ✓ Saved {count} person crops.")
    return count

def collect_all_background_crops(image_dir, anno_dir, output_dir, split_name, max_crops):
    out_dir = os.path.join(output_dir, split_name, "background")
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    image_list = sorted(os.listdir(image_dir))
    random.shuffle(image_list)

    print(f"\n🔹 Generating {max_crops} background crops for [{split_name}]...")
    loop = tqdm(total=max_crops, desc=f"[{split_name}] background")
    while count < max_crops:
        for fname in image_list:
            if count >= max_crops:
                break
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(image_dir, fname)
            anno_path = os.path.join(anno_dir, base + ".xml")
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
            for box in boxes:
                if count >= max_crops:
                    break
                box_w, box_h = box[2] - box[0], box[3] - box[1]
                crop = crop_background(img, boxes, (img_w, img_h), (box_w, box_h))
                if crop:
                    crop.save(os.path.join(out_dir, f"{base}_bg{count}.jpg"))
                    count += 1
                    loop.update(1)
    loop.close()
    print(f"[{split_name}] ✓ Saved {count} background crops.")
    return count

def process_split(image_dir, anno_dir, output_dir, split_name, max_crops):
    p = collect_all_person_crops(image_dir, anno_dir, output_dir, split_name, max_crops)
    b = collect_all_background_crops(image_dir, anno_dir, output_dir, split_name, max_crops)
    print(f"\n✅ [{split_name}] Done. Total: {p} person, {b} background.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--max_per_class", type=int, default=7500)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    ir_dir = os.path.join(dataset_root, "infrared")
    anno_dir = os.path.join(dataset_root, "Annotations")

    process_split(
        image_dir=os.path.join(ir_dir, "train"),
        anno_dir=anno_dir,
        output_dir=dataset_root,
        split_name="train",
        max_crops=args.max_per_class
    )

    process_split(
        image_dir=os.path.join(ir_dir, "test"),
        anno_dir=anno_dir,
        output_dir=dataset_root,
        split_name="val",
        max_crops=args.max_per_class
    )

    print("🎉 All splits processed. LLVIP classification dataset is ready.")
