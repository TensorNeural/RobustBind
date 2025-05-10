#!/usr/bin/env python3
import os
import zipfile
import urllib.request
import http.client
import argparse
from urllib.parse import urlparse
from tqdm import tqdm

# === VQA v2.0 URLs (QA + COCO 2014/2015 images) ===
VQA_URLS = {
    "questions_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "questions_val":   "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "questions_test":  "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    "annotations_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "annotations_val":   "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "images_train2014":  "http://images.cocodataset.org/zips/train2014.zip",
    "images_val2014":    "http://images.cocodataset.org/zips/val2014.zip",
    "images_test2015":   "http://images.cocodataset.org/zips/test2015.zip"
}

# === COCO Captioning 2017 ===
COCO_URLS = {
    "annotations":       "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "images_train2017":  "http://images.cocodataset.org/zips/train2017.zip",
    "images_val2017":    "http://images.cocodataset.org/zips/val2017.zip"
}

def check_url_exists(url):
    try:
        parsed = urlparse(url)
        conn = http.client.HTTPSConnection(parsed.netloc) if parsed.scheme == "https" else http.client.HTTPConnection(parsed.netloc)
        conn.request("HEAD", parsed.path)
        res = conn.getresponse()
        return res.status == 200
    except Exception as e:
        print(f"[ERROR] Failed to verify URL: {url} ({e})")
        return False

def verify_all_urls():
    all_ok = True
    print("🔍 Verifying all VQA URLs...")
    for name, url in VQA_URLS.items():
        if check_url_exists(url):
            print(f"[✓] {name}")
        else:
            print(f"[✗] {name} - {url}")
            all_ok = False

    print("\n🔍 Verifying all COCO Caption URLs...")
    for name, url in COCO_URLS.items():
        if check_url_exists(url):
            print(f"[✓] {name}")
        else:
            print(f"[✗] {name} - {url}")
            all_ok = False
    return all_ok

def download(url, dest_path):
    if os.path.exists(dest_path):
        print(f"[✓] Already downloaded: {dest_path}")
        return
    print(f"[↓] Downloading: {url}")
    with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
        total = int(response.getheader('Content-Length', 0))
        with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

def extract(zip_path, target_dir, rename_dir=None):
    print(f"[⇪] Extracting {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    if rename_dir:
        old = os.path.join(target_dir, os.path.splitext(os.path.basename(zip_path))[0])
        new = os.path.join(target_dir, rename_dir)
        if os.path.exists(old) and not os.path.exists(new):
            os.rename(old, new)
    print(f"[✓] Extracted to {target_dir}")

def prepare_vqa2(vqa_root):
    os.makedirs(vqa_root, exist_ok=True)
    for key, url in VQA_URLS.items():
        file = os.path.basename(url)
        path = os.path.join(vqa_root, file)
        download(url, path)
        if "train2014" in key:
            extract(path, vqa_root, rename_dir="train")
        elif "val2014" in key:
            extract(path, vqa_root, rename_dir="val")
        elif "test2015" in key:
            extract(path, vqa_root, rename_dir="test")
        else:
            extract(path, vqa_root)

def prepare_coco_caption(coco_caption_root):
    os.makedirs(coco_caption_root, exist_ok=True)
    for key, url in COCO_URLS.items():
        file = os.path.basename(url)
        path = os.path.join(coco_caption_root, file)
        download(url, path)
        if "train2017" in key:
            extract(path, coco_caption_root, rename_dir="train")
        elif "val2017" in key:
            extract(path, coco_caption_root, rename_dir="val")
        else:
            extract(path, os.path.join(coco_caption_root, "annotations"))

def main():
    parser = argparse.ArgumentParser(description="Prepare VQA v2.0 and COCO 2017 Captioning datasets")
    parser.add_argument("--base_dir", type=str, default="/home/user/datasets", help="Root directory to store all datasets")
    args = parser.parse_args()

    base_dir = args.base_dir
    vqa_dir = os.path.join(base_dir, "VQA2")
    coco_caption_dir = os.path.join(base_dir, "COCO", "caption")

    print(f"✅ Starting dataset preparation under: {base_dir}")
    if not verify_all_urls():
        print("\n❌ One or more dataset URLs are invalid or unavailable. Aborting.")
        return

    prepare_vqa2(vqa_dir)
    prepare_coco_caption(coco_caption_dir)

    print(f"\n✅ Datasets ready in:\n- {vqa_dir}\n- {coco_caption_dir}")

if __name__ == "__main__":
    main()
