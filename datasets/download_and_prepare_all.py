#!/usr/bin/env python3

import os
import shutil
import sys
import time
import tarfile
import zipfile
import subprocess
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List

import requests
import urllib3
import io
import atexit
from datetime import datetime

# Repo-relative imports for prepare helpers (added later when needed)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(__file__).resolve().parent

ROOT = Path("/data/datasets")
ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO_ROOT / "output"

# ---------- CONFIG: datasets ----------
# You can add more datasets here following the same structure.
DATASETS: List[Dict[str, Any]] = [
    {
        "name": "ImageNet-1K",
        # https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
        "url": "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/6799/4225553/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1763112681&Signature=hObVutiZ0JvE40UAPbGuAIRmZv3mkTPfFRjdC5TWtIQib5pnOMJwCrYD%2FbgjodyZSJwrCdEC4k8TBsgVN6xr1MjD2jVOX5ND%2BZ8jV4uitXy8yYBdWgAW%2BSecd%2Fq8mspbpiHFWxXMJ%2Bn5EA61F3HGBivFiJjeSrVTTUj6fVKMJkwZhtS4eO5B1i4Aijjy0ZSjCdRS9UZ8kW%2FJdd6JwmxYi0v%2B9cn9RaHqO14%2FzTgb7NEu4lcxgDCETHxCACX1Gz3EJ0avfTpgkZGzxWG8hP48i2qdxtQCfaTETRcxU%2Btte5u%2BYBcGrUf5H6S7Q6dv6Rw02FEFYz7a1WfTOykKhs1gIw%3D%3D&response-content-disposition=attachment%3B+filename%3Dimagenet-object-localization-challenge.zip",
        "dest_file": ROOT / "ImageNet-1K.zip",
        "extract_to": ROOT / "ImageNet-1K",
        "type": "zip",
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_imagenet.py",
            "args": lambda extracted: [str(extracted)],
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_imagenet_metadata.py",
            "args": lambda extracted: [str(extracted)],
        },
    },
    # COCO Caption (self-managed download inside the prepare script)
    {
        "name": "COCO-Caption",
        "extract_to": ROOT / "COCO",  # container for subdir 'caption'
        "type": None,  # no external download; prepare handles it
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_coco_caption.py",
            "args": ["--output_dir", str(ROOT / "COCO" / "caption")],
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_coco_metadata.py",
            "args": ["--dataset_root", str(ROOT / "COCO")],
        },
    },
    # VQA v2 (self-managed download inside the prepare script)
    {
        "name": "VQA2",
        "extract_to": ROOT / "VQA2",
        "type": None,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_vqa2.py",
            "args": ["--output_dir", str(ROOT / "VQA2")],
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_vqa2_metadata.py",
            "args": ["--dataset_root", str(ROOT / "VQA2")],
        },
    },
    # Kinetics-400 (self-managed download)
    {
        "name": "Kinetics-400",
        "extract_to": ROOT / "Kinetics-400",
        "type": None,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_kinetics_400.py",
            "args": ["--dataset_root", str(ROOT / "Kinetics-400")],
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_kinetics400_metadata.py",
            "args": ["--dataset_root", str(ROOT / "Kinetics-400"), "--output_dir", "Kinetics-400"],
        },
    },
    {
        "name": "Places365",
        "url": "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar",
        "dest_file": ROOT / "Places365.tar",
        "extract_to": ROOT / "Places365",
        "type": "tar",
        "unwrap_single_dir": True,
        # No explicit prepare script available; extraction provides ready folder structure
        "prepare": None,
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_places365_metadata.py",
            "args": [str(ROOT / "Places365")],
            "optional": True,
        },
    },
    {
        "name": "MSR-VTT",
        # https://www.kaggle.com/datasets/khoahunhtngng/msrvtt
        "url": "https://storage.googleapis.com/kaggle-data-sets/5556037/9190888/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251111%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251111T093512Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=6641ef59b45b9e3e5f10d7e64bb81e7a5b80dcb16d524789ad4eddaea04be9e06e778121828931bff3e5c6f9a2151f8f82bc9c7e31a05d290fd5309ad2e0665ba9a25aa0e80bb61d475d8bb641fa179a02f3efdd177a36be03300ddfa018daee9076684f1099aeb1a7d785e1f59630bf01a5922c76d77e1d1efac8bf5d8a37661ab08a7fff417101bf04160b38fffd2fdbd435a34497c51fd85b374bd533e5a90d9cd5e80bf90023ef47d1eaecd54bc07f00e07ce71099297fa430db7839f373b81c066899876574b23b77a6a3454925cf956dc04c5a641b9f48009972f14c4ca81a68edb5d053c189157189b9a3e1d8c951a3fb633f473006ee83fbda14210c",
        "dest_file": ROOT / "MSR-VTT.zip",
        "extract_to": ROOT / "MSR-VTT",
        "type": "zip",
        "unwrap_single_dir": True,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_msr_vtt.py",
            "args": lambda extracted: [str(extracted)],
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_msr_vtt_metadata.py",
            "args": lambda extracted: [str(extracted)],
        },
    },
    {
        "name": "LLVIP",
        "url": "https://drive.usercontent.google.com/download?id=13v46TKUmhExoTQox3TpK5lyF00D0GVvL&export=download&authuser=0&confirm=t&uuid=8675921f-58e1-4c3d-9e94-1eba63ae5b30",
        "dest_file": ROOT / "LLVIP.zip",
        "extract_to": ROOT / "LLVIP",
        "type": "zip",
        "unwrap_single_dir": True,
        "extra_requests": {"verify": False},
        # We'll import and call functions directly for LLVIP.
        "prepare": {
            "kind": "python",
            "module": "prepare_llvip_classification",
            "callable": "auto_pipeline",  # resolved below
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_llvip_metadata.py",
            "args": lambda extracted: ["--dataset_root", str(extracted)],
        },
    },
    {
        "name": "ESC-50",
        "url": "https://github.com/karolpiczak/ESC-50/archive/master.zip",
        "dest_file": ROOT / "ESC-50.zip",
        "extract_to": ROOT / "ESC-50",
        "type": "zip",
        "unwrap_single_dir": True,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_esc_50.py",
            # auto-detect root containing audio/ and meta/ inside extracted folder
            "args": "detect:esc50",
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_esc_50_metadata.py",
            # ensure we pass the true root with audio/ and meta/
            "args": "detect:esc50",
        },
    },
    {
        "name": "UCF101",
        # Two archives: main videos and train/test split lists
        "archives": [
            {
                "url": "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
                "dest_file": ROOT / "UCF101.rar",
                "type": "rar",
            },
            {
                "url": "http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip",
                "dest_file": ROOT / "UCF101TrainTestSplits-RecognitionTask.zip",
                "type": "zip",
            },
        ],
        "extract_to": ROOT / "UCF101",
        "unwrap_single_dir": True,
        # Requires separate split files (ucfTrainTestlist). We'll skip prepare if not present.
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_ucf101.py",
            "args": "detect:ucf101",
            "optional": True,
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_ucf101_metadata.py",
            "args": lambda extracted: [str(extracted)],
            "optional": True,
        },
    },
    {
        "name": "UrbanSound8K",
        "url": "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz",
        "dest_file": ROOT / "UrbanSound8K.tar.gz",
        "extract_to": ROOT / "UrbanSound8K",
        "type": "targz",
        "unwrap_single_dir": True,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_urbansound8k.py",
            "args": "detect:urbansound8k",
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_urbansound8k_metadata.py",
            "args": lambda extracted: [str(extracted)],
        },
    },
    {
        "name": "FSD-50K",
        "url": "https://zenodo.org/api/records/4060432/files-archive",
        "dest_file": ROOT / "FSD-50K.zip",
        "extract_to": ROOT / "FSD-50K",
        "type": "zip",
        "unwrap_single_dir": True,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_fsd_50k.py",
            "args": [str(ROOT / "FSD-50K")],
            "optional": True,
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_fsd_50k_metadata.py",
            "args": [str(ROOT / "FSD-50K")],
            "optional": True,
        },
    },
    {
        "name": "ModelNet40",
        # https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset
        "url": "https://storage.googleapis.com/kaggle-data-sets/943894/1599485/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251111%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251111T093430Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=469bb99acbe5fabbf8b4a38ffc7da06b3e789d7edf57c166a96b54a3a62bb162c0cb521039172f149e3a31bfad3900005c9666a40e5fe1b39faa45ca7419f9c9c1c221985a516e408d7e6b3ff8a64c1c51d89a25ddc8a431fc83bc1e5208b07f3a32baf26e840dd7698389d245f6c633a70cde47e9ce19504a351b5ce84966229baf20a765ccd04e4fb4b35b2c98a392116b55cb7d2968082caeea383b31f5d1d85cf9a212dc5dcfbe014cb0a6a70b8f11ea805000065aa7dfff88dbbbe082fc0742c9391ae46ac905dce5ac6046951ca15aebc98fc295e0e234029d5cb51bf764939208411e0b5e940227bfb99c37f3676a54e5c1ce43800b838b9d0f05740b",
        "dest_file": ROOT / "ModelNet40.zip",
        "extract_to": ROOT / "ModelNet40",  # expected container with ModelNet40/ inside or already placed
        "type": "zip",
        "unwrap_single_dir": True,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_modelnet.py",
            "args": ["--dataset_root", str(ROOT / "ModelNet40")],
            "optional": True,
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_modelnet_metadata.py",
            "args": ["--dataset_root", str(ROOT / "ModelNet40")],
            "optional": True,
        },
    },
    {
        "name": "N-ImageNet-1K",
        # Download from Hugging Face (validation split only by default)
        # Note: Training parts are very large (>400GB). Included here for completeness.
        "archives": [
            # Training parts (10 zips)
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_1.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_1.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_2.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_2.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_3.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_3.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_4.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_4.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_5.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_5.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_6.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_6.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_7.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_7.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_8.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_rain_Part_8.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_9.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_9.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training/Part_10.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_train_Part_10.zip",
                "type": "zip",
            },
            {
                "url": "https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/validation/extracted_val.zip?download=true",
                "dest_file": ROOT / "N-ImageNet-1K_val.zip",
                "type": "zip",
            },
        ],
        "extract_to": ROOT / "N-ImageNet-1K",
        "type": "zip",
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_n_imagenet_1k.py",
            "args": ["--dataset_root", str(ROOT / "N-ImageNet-1K")],
            "optional": True,
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_n_imagenet_1k_metadata.py",
            "args": [str(ROOT / "N-ImageNet-1K")],
            "optional": True,
        },
    },
    {
        "name": "N-Caltech-101",
        "url": "https://drive.usercontent.google.com/download?id=1hr28hw9i9xOR_-KqdTB5aSbj2XkOdZVG&export=download&authuser=0&confirm=t&uuid=9bb4cbab-dd7c-4db1-96ba-60fb7b5b058f&at=AKSUxGOQUZqwBnuTvghvJoX6Sm7X:1762325775914",
        "dest_file": ROOT / "N-Caltech-101.zip",
        "extract_to": ROOT / "N-Caltech-101",
        "type": "zip",
        "unwrap_single_dir": True,
        "prepare": {
            "kind": "script",
            "path": DATASETS_DIR / "prepare_ncaltech101.py",
            "args": ["--dataset_root", str(ROOT / "N-Caltech-101")],
            "optional": True,
        },
        "gen": {
            "kind": "script",
            "path": DATASETS_DIR / "gen_ncaltech101_metadata.py",
            "args": ["--dataset_root", str(ROOT / "N-Caltech-101"), "--dataset_name", "N-Caltech-101"],
            "optional": True,
        },
    },
]

# ---------- Logging to terminal and file ----------
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

def setup_dual_logging(path: Path):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", buffering=1, encoding="utf-8")
    header = f"\n===== Dataset run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n"
    f.write(header)
    f.flush()
    # Tee both stdout and stderr
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, f)
    sys.stderr = _Tee(orig_err, f)
    def _cleanup():
        try:
            f.flush()
            f.close()
        except Exception:
            pass
        # restore not strictly necessary in script context, but safe
        sys.stdout = orig_out
        sys.stderr = orig_err
    atexit.register(_cleanup)

# ---------- Archive preview (two levels) ----------
def preview_archive_for_task(task: dict, force: bool = False) -> None:
    """Log first two levels inside the archive (both files and directories).
    - If force is False, only preview when unwrap_single_dir is set to reduce verbosity.
    - Never mutates the filesystem.
    """
    try:
        # If task defines multiple archives, preview each one separately
        if "archives" in task and isinstance(task.get("archives"), list):
            idx = 1
            for arch in task["archives"]:
                sub = {
                    "name": f"{task.get('name','?')}[{idx}]",
                    "dest_file": arch.get("dest_file"),
                    "type": arch.get("type"),
                    "unwrap_single_dir": task.get("unwrap_single_dir", False),
                }
                preview_archive_for_task(sub, force=force)
                idx += 1
            return
        if not force and not task.get("unwrap_single_dir"):
            return
        archive: Optional[Path] = task.get("dest_file")
        ftype: Optional[str] = task.get("type")
        name = task.get("name", "?")
        if archive is None or ftype is None:
            print(f"[{name}] No archive configured; preview skipped.")
            return
        if not Path(archive).exists():
            print(f"[{name}] Archive not found at {archive}; preview skipped.")
            return
        # Collect raw names for all members (with any trailing slash preserved for dir entries)
        names: List[str] = []
        dir_entries_explicit: set = set()  # paths explicitly marked as directories by the archive format
        if ftype == "zip":
            with zipfile.ZipFile(archive, "r") as zf:
                infos = zf.infolist()
                for info in infos:
                    n = info.filename
                    names.append(n)
                    if n.endswith('/'):
                        dir_entries_explicit.add(n.strip('/'))
        elif ftype in ("tar", "targz"):
            mode = "r" if ftype == "tar" else "r:gz"
            with tarfile.open(archive, mode) as tf:
                members = tf.getmembers()
                for m in members:
                    names.append(m.name)
                    if m.isdir():
                        dir_entries_explicit.add(m.name.strip('/'))
        elif ftype == "rar":
            try:
                import rarfile
                with rarfile.RarFile(archive) as rf:
                    for i in rf.infolist():
                        names.append(i.filename)
                        # rarfile doesn't always provide a dir marker; rely on trailing '/'
                        if i.filename.endswith('/'):
                            dir_entries_explicit.add(i.filename.strip('/'))
            except Exception:
                print(f"[{name}] Archive preview unavailable for RAR (listing not supported).")
                return
        else:
            print(f"[{name}] Unknown archive type '{ftype}'; preview skipped.")
            return

        # Normalize and build helper indexes
        ignore = {".DS_Store", "__MACOSX"}
        norm_names = []
        for raw in names:
            p = raw.replace("\\", "/").strip("/")
            if p:
                norm_names.append(p)

        all_paths = set(norm_names)  # without trailing slash

        # Build sets for top-level files/dirs and second-level entries
        top_prefix_dirs = set()  # names that appear as a prefix for deeper items
        top_one_part = set()     # names that appear as a single-part entry
        second_children: Dict[str, set] = {}
        for raw in names:
            p = raw.replace("\\", "/").strip("/")
            if not p:
                continue
            parts = p.split("/")
            if parts[0] in ignore:
                continue
            if len(parts) == 1:
                if parts[0] not in ignore:
                    top_one_part.add(parts[0])
            else:
                top_prefix_dirs.add(parts[0])
                child = parts[1]
                if child not in ignore:
                    second_children.setdefault(parts[0], set()).add(child)

        # Helpers
        def is_dir_path(path: str) -> bool:
            return (path in dir_entries_explicit) or any(p.startswith(path + "/") for p in all_paths)

        def children_of(path: str) -> List[str]:
            seen = set()
            prefix = path + "/"
            for p in all_paths:
                if p.startswith(prefix):
                    rest = p[len(prefix):]
                    if rest:
                        seg = rest.split("/", 1)[0]
                        if seg and seg not in ignore:
                            seen.add(seg)
            return sorted(seen)

        # Union of top-level names (files and dirs). A name is a dir if it is a prefix for deeper entries or explicitly marked as a dir
        top_all = sorted(top_one_part.union(top_prefix_dirs))
        top_dirs = {n for n in top_all if (n in top_prefix_dirs or n in dir_entries_explicit)}
        top_files = [n for n in top_all if n not in top_dirs]

        # Decide if we will virtually unwrap in preview (match runtime unwrap rule)
        will_unwrap = bool(task.get("unwrap_single_dir")) and len(top_dirs) == 1 and len(top_files) == 0

        header_note = " (unwrapped view)" if will_unwrap else ""
        print(f"[{name}] Archive preview (first 2 levels{header_note}) for {Path(archive).name}:")

        if not top_all:
            print("  (empty archive)")
        elif will_unwrap:
            # Simulate a single-top-level-dir unwrap: show children of that dir as top-level
            root_dir = next(iter(top_dirs))
            first_level = children_of(root_dir)
            for item in first_level:
                path = f"{root_dir}/{item}"
                dir_flag = is_dir_path(path)
                suffix = "/" if dir_flag else ""
                print(f"  - {item}{suffix}")
                if dir_flag:
                    second_level = children_of(path)
                    for child in second_level:
                        child_path = f"{path}/{child}"
                        if is_dir_path(child_path):
                            print(f"    - {item}/{child}/")
        else:
            # Normal view: show top-level entries and their direct children for directories
            for top in top_all:
                is_dir = top in top_prefix_dirs or top in dir_entries_explicit
                suffix = "/" if is_dir else ""
                print(f"  - {top}{suffix}")
                if is_dir:
                    for child in children_of(top):
                        child_path = f"{top}/{child}"
                        if is_dir_path(child_path):
                            print(f"    - {top}/{child}/")
    except Exception as e:
        print(f"[{task.get('name','?')}] Warning: failed to preview archive: {e}")


# ---------- Download with resume ----------

def download_with_resume(url: str, dest: Path, extra_requests: Optional[dict] = None, chunk_size: int = 2**20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    pos = 0
    if dest.exists():
        pos = dest.stat().st_size
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"
            mode = "ab"

    kwargs: Dict[str, Any] = {"stream": True, "headers": headers, "timeout": 60}
    if extra_requests:
        kwargs.update(extra_requests)

    # Internal helper to perform a single HTTP GET and stream to file
    def _stream_to_file(get_url: str, get_kwargs: Dict[str, Any]):
        with requests.get(get_url, **get_kwargs) as r:
            if r.status_code in (200, 206):
                total = r.headers.get("Content-Length")
                if total is not None:
                    total = int(total) + pos if "Range" in headers else int(total)
                with open(dest, mode) as f:
                    last_report = time.time()
                    downloaded = pos
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_report > 2:
                                if total:
                                    pct = downloaded / total * 100
                                    print(f"[{dest.name}] {downloaded/1e6:.1f} MB/{total/1e6:.1f} MB ({pct:.1f}%)")
                                else:
                                    print(f"[{dest.name}] {downloaded/1e6:.1f} MB")
                                last_report = now
            else:
                raise RuntimeError(f"HTTP {r.status_code} for {get_url}")

    # Try normal verified HTTPS first; on SSL verify error, retry with verify=False, then fallback to http
    try:
        _stream_to_file(url, kwargs)
    except requests.exceptions.SSLError as ssl_err:
        # If user already requested verify=False, just re-raise
        if kwargs.get("verify") is False:
            raise
        print(f"[warn] SSL verification failed for {url}: {ssl_err}. Retrying without certificate verification…")
        # Suppress noisy urllib3 warnings when we intentionally skip verification
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        insecure_kwargs = dict(kwargs)
        insecure_kwargs["verify"] = False
        try:
            _stream_to_file(url, insecure_kwargs)
        except Exception:
            # As a last resort, try plain HTTP if original was HTTPS
            parsed = urlparse(url)
            if parsed.scheme == "https":
                http_url = parsed._replace(scheme="http").geturl()
                print(f"[warn] Retrying over HTTP: {http_url}")
                _stream_to_file(http_url, insecure_kwargs)
            else:
                raise


# ---------- Extract helpers ----------

def extract_zip(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dst)


def extract_tar(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:") as tf:
        tf.extractall(dst)


def extract_targz(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:gz") as tf:
        tf.extractall(dst)


def extract_rar(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    try:
        import rarfile  # optional
        with rarfile.RarFile(src) as rf:
            rf.extractall(path=dst)
    except Exception:
        # Fallback to 'unar' (preferred over unrar)
        print(f"[{src.name}] Using system 'unar' fallback...")
        dst.mkdir(parents=True, exist_ok=True)
        # unar writes into output directory specified by -o
        result = subprocess.run(["unar", "-force-overwrite", "-o", str(dst), str(src)], capture_output=True, text=True)
        if result.returncode != 0:
            # Optional secondary fallback to 7z if available
            print(f"[{src.name}] 'unar' failed ({result.returncode}). Trying '7z' fallback...")
            result2 = subprocess.run(["7z", "x", str(src), f"-o{str(dst)}", "-y"], capture_output=True, text=True)
            if result2.returncode != 0:
                raise RuntimeError(f"Failed to extract RAR: {result.stderr.strip()} | {result2.stderr.strip()}")


def extract_any(task: dict, allow_merge: bool = False):
    ftype = task["type"]
    src, dst = task["dest_file"], task["extract_to"]
    name = task['name']

    def maybe_unwrap(destination: Path):
        if not task.get("unwrap_single_dir"):
            return False
        try:
            # Ignore common noise entries
            entries = [e for e in destination.iterdir() if e.name not in (".DS_Store", "__MACOSX")]
            dirs = [e for e in entries if e.is_dir()]
            files = [e for e in entries if e.is_file()]
            if len(dirs) == 1 and len(files) == 0:
                top = dirs[0]
                for p in top.iterdir():
                    shutil.move(str(p), str(destination / p.name))
                top.rmdir()
                print(f"[{name}] Unwrapped single top-level dir: {top.name}")
                return True
        except Exception as e:
            print(f"[{name}] Warning: failed to unwrap top-level dir: {e}")
        return False
    # Preview is only shown during dry-run; skip during normal extraction to reduce noise

    # If target directory already exists, skip extraction entirely (idempotent)
    # When allow_merge=True (multi-archive datasets), we still extract into the existing folder.
    if dst.exists() and not allow_merge:
        try:
            # Attempt unwrap on previously extracted content if there is anything to unwrap
            has_content = any(dst.iterdir())
            if has_content:
                maybe_unwrap(dst)
        except Exception:
            # If listing fails, still skip extraction as requested
            pass
        print(f"[{name}] Extract destination exists at {dst}. Skipping extraction.")
        return
    print(f"[{name}] Extracting {src.name} -> {dst}")
    if ftype == "zip":
        extract_zip(src, dst)
    elif ftype == "tar":
        extract_tar(src, dst)
    elif ftype == "targz":
        extract_targz(src, dst)
    elif ftype == "rar":
        extract_rar(src, dst)
    else:
        raise ValueError(f"Unknown type {ftype}")
    # Optionally unwrap a single top-level directory after extraction
    maybe_unwrap(dst)
    print(f"[{name}] Extracted to {dst}")


# ---------- Prepare helpers ----------

def find_dir_with_subdirs(root: Path, required: List[str]) -> Optional[Path]:
    """Find a directory under root that contains all required subdirs/files."""
    # First, check the root itself
    if all((root / p).exists() for p in required):
        return root
    # Then scan one level deep
    for child in root.glob("**/*"):
        if child.is_dir():
            if all((child / p).exists() for p in required):
                return child
    return None


def run_prepare_script(script_path: Path, args: List[str]) -> None:
    cmd = [sys.executable, str(script_path)] + args
    print(f"[prepare] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Prepare script failed: {script_path}")


def prepare_dataset(task: dict) -> None:
    prep = task.get("prepare")
    if not prep:
        print(f"[{task['name']}] No prepare step configured. Skipping.")
        return

    extracted = Path(task["extract_to"]).resolve()
    kind = prep.get("kind")

    # Dynamic arg detection helpers
    def resolve_args(arg_spec):
        if callable(arg_spec):
            return arg_spec(extracted)
        if isinstance(arg_spec, list):
            return [str(a) for a in arg_spec]
        if isinstance(arg_spec, str) and arg_spec.startswith("detect:"):
            which = arg_spec.split(":", 1)[1]
            if which == "esc50":
                # Needs audio/ and meta/ under a root
                root = find_dir_with_subdirs(extracted, ["audio", os.path.join("meta", "esc50.csv")])
                if not root:
                    raise RuntimeError("ESC-50 root with audio/ and meta/esc50.csv not found after extraction")
                return [str(root)]
            if which == "urbansound8k":
                root = find_dir_with_subdirs(extracted, ["audio", os.path.join("metadata", "UrbanSound8K.csv")])
                if not root:
                    raise RuntimeError("UrbanSound8K root with audio/ and metadata/UrbanSound8K.csv not found after extraction")
                return [str(root)]
            if which == "ucf101":
                # Requires UCF-101/ and ucfTrainTestlist/
                root = find_dir_with_subdirs(extracted, ["UCF-101"]) or extracted
                if not (root / "ucfTrainTestlist").exists():
                    # Optional: skip if split files missing
                    raise FileNotFoundError("ucfTrainTestlist/ not found; skipping prepare")
                return [str(root)]
        # default: pass extracted dir
        return [str(extracted)]

    try:
        if kind == "script":
            script_path: Path = Path(prep["path"]).resolve()
            arg_spec = prep.get("args", [str(extracted)])
            args = resolve_args(arg_spec)
            run_prepare_script(script_path, args)
        elif kind == "python":
            module_name = prep["module"]
            callable_name = prep.get("callable", None)
            sys.path.insert(0, str(DATASETS_DIR))
            mod = __import__(module_name)
            if callable_name == "auto_pipeline":
                # LLVIP: if person/background not present, generate crops then balance
                # dataset structure expected after unzip: infrared/, Annotations/
                root = extracted
                person_dir = root / "person"
                bg_dir = root / "background"
                if not person_dir.exists() or not bg_dir.exists():
                    if hasattr(mod, "generate_crops"):
                        print(f"[LLVIP] Generating crops under {root}...")
                        mod.generate_crops(str(root))
                    else:
                        print("[LLVIP] Missing generate_crops in module; skipping crop generation.")
                if hasattr(mod, "balance_and_split"):
                    print(f"[LLVIP] Balancing and splitting under {root}...")
                    mod.balance_and_split(str(root))
                else:
                    print("[LLVIP] Missing balance_and_split in module; skipping split.")
            else:
                fn = getattr(mod, callable_name) if callable_name else None
                if callable(fn):
                    fn(str(extracted))
                else:
                    raise RuntimeError(f"Invalid python prepare callable: {module_name}.{callable_name}")
        else:
            print(f"[{task['name']}] Unknown prepare kind: {kind}. Skipping.")
    except FileNotFoundError as e:
        # Respect optional flag
        if prep.get("optional"):
            print(f"[{task['name']}] Optional prepare skipped: {e}")
        else:
            raise


def run_gen_script(script_path: Path, args: List[str]) -> None:
    cmd = [sys.executable, str(script_path)] + args
    print(f"[gen] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Gen script failed: {script_path}")


def generate_jsons(task: dict) -> None:
    gen = task.get("gen")
    if not gen:
        return
    kind = gen.get("kind")
    extracted = Path(task.get("extract_to", ROOT)).resolve()

    # Reuse prepare's arg resolver
    def resolve_args(arg_spec):
        if callable(arg_spec):
            return arg_spec(extracted)
        if isinstance(arg_spec, list):
            return [str(a) for a in arg_spec]
        return [str(extracted)]

    try:
        if kind == "script":
            script_path: Path = Path(gen["path"]).resolve()
            arg_spec = gen.get("args", [str(extracted)])
            args = resolve_args(arg_spec)
            run_gen_script(script_path, args)
        else:
            print(f"[{task['name']}] Unknown gen kind: {kind}. Skipping.")
    except FileNotFoundError as e:
        if gen.get("optional"):
            print(f"[{task['name']}] Optional gen skipped: {e}")
        else:
            raise


# ---------- Worker ----------

def process_dataset(task: dict, delete_archive_after: bool = False, skip_gen: bool = False):
    name = task["name"]
    url = task.get("url")
    dest: Optional[Path] = task.get("dest_file")
    extra = task.get("extra_requests")
    requires_manual: bool = bool(task.get("requires_manual"))

    # Support multi-archive datasets via 'archives' list. Fallback to single 'url' + 'dest_file'.
    if isinstance(task.get("archives"), list) and task["archives"]:
        archives = task["archives"]
        # Download all archives first
        for arch in archives:
            a_url = arch.get("url")
            a_dest = Path(arch.get("dest_file")) if arch.get("dest_file") is not None else None
            if not a_url or a_dest is None:
                raise RuntimeError(f"[{name}] Invalid archive entry (missing url or dest_file)")
            if a_dest.exists() and a_dest.stat().st_size > 0:
                print(f"[{name}] Archive already exists at {a_dest}. Skipping download.")
            else:
                print(f"[{name}] Downloading from: {urlparse(a_url).netloc}")
                download_with_resume(a_url, a_dest, extra_requests=extra)
                print(f"[{name}] Download complete: {a_dest} ({a_dest.stat().st_size/1e6:.1f} MB)")
        # Extract each archive, allowing merge into existing destination folder
        # Special-case: For UCF101, if destination already has 'ucfTrainTestlist', skip extraction entirely
        dest_root = Path(task.get("extract_to"))
        if name.upper() == "UCF101" and dest_root.exists() and (dest_root / "ucfTrainTestlist").exists():
            print(f"[{name}] Detected existing 'ucfTrainTestlist' in {dest_root}. Skipping extraction.")
        else:
            for idx, arch in enumerate(archives, start=1):
                subtask = {
                    "name": f"{name}[{idx}]",
                    "type": arch.get("type"),
                    "dest_file": Path(arch.get("dest_file")),
                    "extract_to": task.get("extract_to"),
                    "unwrap_single_dir": task.get("unwrap_single_dir", False),
                }
                extract_any(subtask, allow_merge=True)
    else:
        # Single-archive or manual/no-download case
        if url and dest is not None:
            # Skip downloading if archive already exists (non-zero size)
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[{name}] Archive already exists at {dest}. Skipping download.")
            else:
                print(f"[{name}] Downloading from: {urlparse(url).netloc}")
                download_with_resume(url, dest, extra_requests=extra)
                print(f"[{name}] Download complete: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
            # Extract if not already extracted
            extract_any(task)

        # Manual or no external download case
        extract_to = Path(task.get("extract_to", ROOT))
        if requires_manual:
            if extract_to.exists():
                print(f"[{name}] Using existing extracted dataset at {extract_to}.")
            elif dest is not None and Path(dest).exists() and task.get("type"):
                print(f"[{name}] Found archive at {dest}; extracting manually provided file…")
                extract_any(task)
            else:
                hint = task.get("manual_hint", f"Please place the dataset under {extract_to} and re-run.")
                print(f"[{name}] Skipping — manual dataset not found. {hint}")
                return
        else:
            print(f"[{name}] No external download configured. Proceeding to prepare/gen if available.")
    # Run prepare step if configured
    try:
        prepare_dataset(task)
    except Exception as e:
        raise RuntimeError(f"prepare failed: {e}")

    # Run JSON generation step if configured
    if not skip_gen:
        try:
            generate_jsons(task)
        except Exception as e:
            raise RuntimeError(f"gen failed: {e}")
    else:
        print(f"[{name}] Skipping gen step (--skip-gen flag set)")

    if delete_archive_after and dest is not None and dest.exists():
        try:
            dest.unlink()
            print(f"[{name}] Deleted archive: {dest}")
        except Exception as e:
            print(f"[{name}] Warning: could not delete archive ({e})")


# ---------- Main ----------

def main(max_workers: Optional[int] = None, delete_archives: bool = False,
         only: Optional[set] = None, skip: Optional[set] = None, skip_gen: bool = False):
    # Normalize name filters to lowercase
    only = {n.lower() for n in only} if only else set()
    skip = {n.lower() for n in skip} if skip else set()

    # Filter datasets per user request
    selected: List[Dict[str, Any]] = []
    skipped_by_filter: List[str] = []
    for task in DATASETS:
        lname = task["name"].lower()
        if only and lname not in only:
            skipped_by_filter.append(task["name"]) 
            continue
        if skip and lname in skip:
            skipped_by_filter.append(task["name"]) 
            continue
        selected.append(task)

    print(f"Root: {ROOT}")
    if skipped_by_filter:
        print("Skipping due to filter:", ", ".join(skipped_by_filter))
    # Determine effective workers: default to number of datasets (all concurrent)
    eff_workers = max(1, len(selected)) if (max_workers is None or max_workers == 0) else max_workers
    print(f"Starting {len(selected)} datasets with {eff_workers} workers...")
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=eff_workers) as ex:
        futs = {ex.submit(process_dataset, task, delete_archives, skip_gen): task["name"] for task in selected}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
            except Exception as e:
                errors[name] = str(e)
                print(f"[{name}] ERROR: {e}", file=sys.stderr)

    print("\n=== Summary ===")
    if not errors:
        print("All downloads, extractions, and preparations completed successfully.")
    else:
        print("Some tasks failed:")
        for k, v in errors.items():
            print(f" - {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download, extract, prepare, and generate metadata for datasets.")
    ap.add_argument("--max-workers", type=int, default=0, help="Max concurrent workers (0 = all datasets concurrently)")
    ap.add_argument("--delete-archives", action="store_true", help="Delete archives after successful extraction")
    ap.add_argument("--only", action="append", default=[
        # "N-ImageNet-1K",
    ], help="Only process these dataset names (comma-separated or repeated)")
    ap.add_argument("--skip", action="append", default=[
        "N-ImageNet-1K",
        # "FSD-50K",
        "MSR-VTT",
        "ESC-50",
        "ImageNet-1K",
        "LLVIP",
        "ModelNet40",
        # "UrbanSound8K",
        "Kinetics-400",
        "Places365",
        "N-Caltech-101",
        # "UCF101",
        "COCO-Caption",
        "VQA2",
    ], help="Skip these dataset names (comma-separated or repeated)")
    ap.add_argument("--dry-run-preview", action="store_true", default=False, help="Only log the first two directory levels inside available archives; no extraction or preparation")
    ap.add_argument("--skip-gen", action="store_true", default=True, help="Skip the metadata generation step")

    args = ap.parse_args()

    # Flatten multi-append and comma-separated forms into sets
    def to_name_set(values: List[str]) -> set:
        items: set = set()
        for v in values:
            if not v:
                continue
            for tok in str(v).split(','):
                tok = tok.strip()
                if tok:
                    items.add(tok)
        return items

    only_set = to_name_set(args.only)
    skip_set = to_name_set(args.skip)

    # Set up dual logging to terminal and timestamped file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"datasets_{ts}.txt"
    setup_dual_logging(log_file)

    if args.dry_run_preview:
        # Dry run: preview archives for selected datasets without changing anything
        selected = []
        for task in DATASETS:
            lname = task["name"]
            if only_set and lname not in only_set:
                print(f"Skipping (not in --only): {task['name']}")
                continue
            if skip_set and lname in skip_set:
                print(f"Skipping (in --skip): {task['name']}")
                continue
            selected.append(task)
        print(f"Root: {ROOT}")
        print(f"Dry run: previewing archives for {len(selected)} dataset(s)...")
        for task in selected:
            preview_archive_for_task(task, force=True)
        print("Done.")
    else:
        main(max_workers=args.max_workers, delete_archives=args.delete_archives, only=only_set, skip=skip_set, skip_gen=args.skip_gen)
