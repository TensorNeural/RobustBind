import os
import json
import math
import torch
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision.transforms._transforms_video import NormalizeVideo
import torchaudio

from utils.utils import load_centre_embeddings

# ===================
# Modality Constants
# ===================
MEAN_MAP = {
    "image": [0.48145466, 0.4578275, 0.40821073],
    "event": [0.48145466, 0.4578275, 0.40821073],
    "thermal": [0.5],
    "video": [0.48145466, 0.4578275, 0.40821073],
    "audio": [-4.268],
    "point": [0.0, 0.0, 0.0],
}

STD_MAP = {
    "image": [0.26862954, 0.26130258, 0.27577711],
    "event": [0.26862954, 0.26130258, 0.27577711],
    "thermal": [0.5],
    "video": [0.26862954, 0.26130258, 0.27577711],
    "audio": [9.138],
    "point": [1.0, 1.0, 1.0],
}

# ===================
# Dataset
# ===================
class JsonDataset(Dataset):
    def __init__(self, dataset_root, data_json_path, transform, label_to_index=None, index_to_label=None, max_samples=None, debug=False):
        self.root_dir = dataset_root
        self.transform = transform
        self.label_to_index_fn = label_to_index
        self.index_to_label_fn = index_to_label

        with open(data_json_path, "r") as f:
            self.samples = [(item["data"], item["label"]) for item in json.load(f)]

        if max_samples is not None and max_samples < len(self.samples):
            indices = torch.arange(max_samples) if debug else torch.randperm(len(self.samples))[:max_samples]
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label_str = self.samples[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        tensor = self.transform([full_path], device="cpu")[0]
        label = self.label_to_index_fn[label_str] if self.label_to_index_fn else 0
        return tensor, label

# ===================
# Transforms
# ===================
def get_transform_fn(modality):
    return {
        "image": load_and_transform_vision_data,
        "event": load_and_transform_vision_data,
        "thermal": load_and_transform_thermal_data,
        "video": load_and_transform_video_data,
        "audio": load_and_transform_audio_data,
        "point": load_and_transform_point_data
    }[modality]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_MAP["image"], std=STD_MAP["image"]),
])

def load_and_transform_vision_data(image_paths, device):
    images = []
    for p in image_paths:
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        images.append(IMAGE_TRANSFORM(img).to(device))
    return torch.stack(images)

def load_and_transform_thermal_data(thermal_paths, device):
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    images = []
    for p in thermal_paths:
        with open(p, "rb") as f:
            img = Image.open(f).convert("L")
        images.append(transform(img).to(device))
    return torch.stack(images)

def load_and_transform_point_data(point_paths, device):
    return torch.stack([torch.load(p) for p in point_paths]).to(device)

def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sample_rate,
        use_energy=False, window_type="hanning", num_mel_bins=num_mel_bins,
        dither=0.0, frame_length=25, frame_shift=10
    ).transpose(0, 1)
    p = target_length - fbank.size(1)
    if abs(p) / fbank.size(1) > 0.2:
        logging.warning(f"Audio frame mismatch: {fbank.size(1)} vs {target_length}")
    fbank = torch.nn.functional.pad(fbank, (0, max(0, p)))[:, :target_length]
    return fbank.unsqueeze(0)

def load_and_transform_audio_data(audio_paths, device, num_mel_bins=128, target_length=204, sample_rate=16000, clip_duration=2, clips_per_video=3, mean=-4.268, std=9.138):
    normalize = transforms.Normalize(mean=[mean], std=[std])
    outputs = []
    sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=clips_per_video)

    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        segments = []
        for start, end in get_clip_timepoints(sampler, waveform.size(1) / sample_rate):
            clip = waveform[:, int(start * sample_rate):int(end * sample_rate)]
            spec = waveform2melspec(clip, sample_rate, num_mel_bins, target_length)
            segments.append(normalize(spec))
        outputs.append(torch.stack(segments))

    return torch.stack(outputs).to(device)

def load_and_transform_video_data(video_paths, device, clip_duration=2, clips_per_video=5, sample_rate=16000):
    transform = transforms.Compose([
        transforms.Resize(224),
        NormalizeVideo(mean=MEAN_MAP["video"], std=STD_MAP["video"]),
    ])
    outputs = []
    sampler = ConstantClipsPerVideoSampler(clip_duration, clips_per_video)

    for path in video_paths:
        video = EncodedVideo.from_path(path, decoder="decord", decode_audio=False, sample_rate=sample_rate)
        all_frames = []
        for start, end in get_clip_timepoints(sampler, video.duration):
            clip = video.get_clip(start, end)
            if clip is None or "video" not in clip:
                continue
            frames = clip["video"] / 255.0
            all_frames.append(transform(frames))
        outputs.append(torch.stack(all_frames))
    return torch.stack(outputs).to(device)

def get_clip_timepoints(clip_sampler, duration):
    clips, end = [], 0.0
    while True:
        start, end, _, _, is_last = clip_sampler(end, duration, annotation=None)
        clips.append((start, end))
        if is_last: break
    return clips

# ===================
# Loader & Utility
# ===================
def load_label_mapping(center_emb_path, device):
    raw_emb, raw_lbls = load_centre_embeddings(center_emb_path, device)
    raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
    unique_lbls = sorted(set(raw_lbls))
    lbl_to_idx = {l: i for i, l in enumerate(unique_lbls)}
    idx_to_lbl = {v: k for k, v in lbl_to_idx.items()}
    return raw_emb, raw_lbls, lbl_to_idx, idx_to_lbl

def train_data_loader(
    modality, dataset_root, train_json, label_to_index, index_to_label,
    batch_size, num_workers, max_samples=None, debug=False
):
    transform = get_transform_fn(modality)
    dataset = JsonDataset(dataset_root, train_json, transform, label_to_index, index_to_label, max_samples, debug)
    sampler = DistributedSampler(dataset, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)

def val_data_loader(
    modality, dataset_root, val_json, label_to_index, index_to_label,
    batch_size, num_workers, max_samples=None, debug=False
):
    transform = get_transform_fn(modality)
    dataset = JsonDataset(dataset_root, val_json, transform, label_to_index, index_to_label, max_samples, debug)
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)

def get_normalization_tensors(modality, device):
    mean = torch.tensor(MEAN_MAP[modality], device=device)
    std = torch.tensor(STD_MAP[modality], device=device)

    if modality == "point":
        return mean.view(1, 1, 3), std.view(1, 1, 3)
    elif modality == "thermal":
        return mean.view(1, 1, 1), std.view(1, 1, 1)
    else:
        return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
