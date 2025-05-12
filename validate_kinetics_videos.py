import os
import argparse
import subprocess
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def is_valid_video(full_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1", full_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0 and b"codec_name" in result.stdout
    except Exception:
        return False


def scan_train_split(video_root, num_threads=12):
    train_dir = os.path.join(video_root, "train")
    invalid_dir = os.path.join(video_root, "invalid_train")
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)

    all_videos = sorted([f for f in os.listdir(train_dir) if f.endswith(".mp4")])
    total = len(all_videos)
    print(f"🔍 Validating {total} training videos using {num_threads} threads...\n")

    valid_files = []
    invalid_files = []
    log_valid_path = os.path.join(output_dir, "valid.log")
    log_invalid_path = os.path.join(output_dir, "invalid.log")
    log_lock = threading.Lock()

    def validate(fname):
        rel_path = os.path.join("train", fname)
        full_path = os.path.join(video_root, rel_path)
        is_valid = is_valid_video(full_path)

        with log_lock:
            log_path = log_valid_path if is_valid else log_invalid_path
            with open(log_path, "a") as logf:
                logf.write(f"{rel_path}\n")

        if not is_valid:
            try:
                shutil.move(full_path, os.path.join(invalid_dir, fname))
            except Exception as e:
                with log_lock:
                    with open(log_invalid_path, "a") as logf:
                        logf.write(f"⚠️ Failed to move {rel_path}: {e}\n")

        return rel_path, is_valid

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(validate, fname) for fname in all_videos]
        for future in tqdm(as_completed(futures), total=total, desc="Scanning", unit="video"):
            rel_path, is_valid = future.result()
            (valid_files if is_valid else invalid_files).append(rel_path)

    with open(os.path.join(output_dir, "valid_videos.txt"), "w") as f:
        f.write("\n".join(valid_files))
    with open(os.path.join(output_dir, "invalid_videos.txt"), "w") as f:
        f.write("\n".join(invalid_files))

    print(f"\n📦 Train videos checked: {total}")
    print(f"✅ Valid:   {len(valid_files)} → output/valid_videos.txt")
    print(f"❌ Invalid: {len(invalid_files)} → output/invalid_videos.txt")
    print(f"📂 Moved invalid videos to: {invalid_dir}")
    print(f"📝 Logs saved to: output/valid.log, output/invalid.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Kinetics-400 training videos and move invalid ones.")
    parser.add_argument("--video_root", type=str, required=True,
                        help="Path to Kinetics-400 root directory containing train/")
    args = parser.parse_args()

    scan_train_split(args.video_root, num_threads=12)
