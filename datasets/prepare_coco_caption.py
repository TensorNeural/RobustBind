#!/usr/bin/env python3
import os
import zipfile
import urllib.request
import http.client
import argparse
from urllib.parse import urlparse
from tqdm import tqdm

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
        return conn.getresponse().status == 200
    except Exception as e:
        print(f"[ERROR] Failed to verify URL: {url} ({e})")
        return False

def verify_coco_urls():
    all_ok = True
    print("🔍 Verifying COCO URLs...")
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
    # If the renamed target already exists, skip extraction
    if rename_dir:
        new = os.path.join(target_dir, rename_dir)
        if os.path.isdir(new) and os.listdir(new):
            print(f"[✓] Found existing '{rename_dir}' at {new}. Skipping extract.")
            return
    print(f"[⇪] Extracting {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    if rename_dir:
        extracted = os.path.splitext(os.path.basename(zip_path))[0]
        old = os.path.join(target_dir, extracted)
        new = os.path.join(target_dir, rename_dir)
        if os.path.exists(old) and not os.path.exists(new):
            os.rename(old, new)
    print(f"[✓] Extracted to {target_dir}")


def _dir_has_files(d):
    try:
        return any(os.path.isfile(os.path.join(d, f)) for f in os.listdir(d))
    except Exception:
        return False


def is_coco_prepared(coco_root):
    train = os.path.join(coco_root, "train")
    val = os.path.join(coco_root, "val")
    ann = os.path.join(coco_root, "annotations")
    return os.path.isdir(train) and os.path.isdir(val) and os.path.isdir(ann)

def prepare_coco(coco_root):
    os.makedirs(coco_root, exist_ok=True)
    for key, url in COCO_URLS.items():
        filename = os.path.basename(url)
        path = os.path.join(coco_root, filename)
        download(url, path)
        if "train2017" in key:
            extract(path, coco_root, rename_dir="train")
        elif "val2017" in key:
            extract(path, coco_root, rename_dir="val")
        else:
            extract(path, coco_root, rename_dir="annotations")

def main():
    parser = argparse.ArgumentParser(description="Download and extract COCO Captioning 2017")
    parser.add_argument("--output_dir", type=str, default="/data/datasets/COCO/caption")
    args = parser.parse_args()

    print(f"✅ Preparing COCO Caption in: {args.output_dir}")
    # Early exit if already prepared
    if os.path.isdir(args.output_dir) and is_coco_prepared(args.output_dir):
        print("⚠️ Output directory already prepared. Skipping prepare.")
        return

    if not verify_coco_urls():
        print("❌ Some COCO URLs are unreachable. Aborting.")
        return

    prepare_coco(args.output_dir)
    print("✅ COCO Caption download and extraction complete.")

if __name__ == "__main__":
    main()
