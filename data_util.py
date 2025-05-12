import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
import torchaudio
import numpy as np

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

NUM_FRAMES = 2

# ===================
# Dataset
# ===================
class JsonDataset(Dataset):
    def __init__(self, dataset_root, data_json_path, transform, label_to_index=None, max_samples=None, debug=False):
        self.root_dir = dataset_root
        self.transform = transform
        self.label_to_index_fn = label_to_index

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
def get_transform_fn(modality, minSample=False):
    return {
        "image": load_and_transform_vision_data,
        # "event": load_and_transform_event_data,
        "event": load_and_transform_vision_data,
        "thermal": load_and_transform_thermal_data,
        "video": load_and_transform_video_data(minSample=minSample),
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

def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138
):
    normalize = transforms.Normalize(mean=[mean], std=[std])
    outputs = []
    sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # Ensure minimum total duration
        total_duration = waveform.size(1) / sample_rate
        expected_samples = int(clip_duration * sample_rate)
        if total_duration < clip_duration:
            pad_len = expected_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        segments = []
        clip_timepoints = get_clip_timepoints(sampler, waveform.size(1) / sample_rate)

        for start, end in clip_timepoints:
            clip = waveform[:, int(start * sample_rate):int(end * sample_rate)]

            # Pad clip if still too short (e.g., end too close to waveform end)
            if clip.size(1) < expected_samples:
                clip = torch.nn.functional.pad(clip, (0, expected_samples - clip.size(1)))

            spec = waveform2melspec(clip, sample_rate, num_mel_bins, target_length)
            segments.append(normalize(spec))

        outputs.append(torch.stack(segments))

    return torch.stack(outputs).to(device)

def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to perform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes

def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res

def load_and_transform_video_data(
    minSample=False,
):
    def transform(
        video_paths,
        device,
        clip_duration=2,
        clips_per_video=5,
        sample_rate=16000):
        if video_paths is None:
            return None

        video_outputs = []
        video_transform = transforms.Compose(
            [
                ShortSideScale(224),
                Normalize(
                    mean=MEAN_MAP["video"],
                    std=STD_MAP["video"],
                ),
            ]
        )

        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
        frame_sampler = UniformTemporalSubsample(num_samples=clip_duration)

        for video_path in video_paths:
            video = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
                **{"sample_rate": sample_rate},
            )

            all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

            # Drop 4 clips if minSample is True
            if minSample:
                all_clips_timepoints = all_clips_timepoints[:max(1, len(all_clips_timepoints) - 4)]

            all_video = []
            for clip_timepoints in all_clips_timepoints:
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                if clip is None:
                    raise ValueError("No clip found")
                video_clip = frame_sampler(clip["video"])
                video_clip = video_clip / 255.0  # normalize to [0,1]
                all_video.append(video_clip)

            all_video = [video_transform(clip) for clip in all_video]
            all_video = SpatialCrop(224, num_crops=3)(all_video)

            all_video = torch.stack(all_video, dim=0)
            video_outputs.append(all_video)

        return torch.stack(video_outputs, dim=0).to(device)

    return transform

def get_clip_timepoints(clip_sampler, duration):
    clips, end = [], 0.0
    while True:
        start, end, _, _, is_last = clip_sampler(end, duration, annotation=None)
        clips.append((start, end))
        if is_last: break
    return clips

def read_bin_event_file(path):
    raw = np.fromfile(path, dtype=np.uint64)
    raw = raw & ((1 << 40) - 1)

    x = (raw >> 32) & 0xFF
    y = (raw >> 24) & 0xFF
    p = (raw >> 23) & 0x1
    t = raw & 0x7FFFFF

    events = np.stack([x, y, t, p], axis=1).astype(np.float32)
    return torch.from_numpy(events).float()


def infer_resolution(events, default_H=180, default_W=240):
    if events.shape[0] == 0:
        return default_H, default_W
    H = max(int(events[:, 1].max().item()) + 1, default_H)
    W = max(int(events[:, 0].max().item()) + 1, default_W)
    return H, W


def normalize_voxel(voxel):
    min_val = voxel.min()
    max_val = voxel.max()
    if max_val <= min_val:
        return voxel.zero_()
    return (voxel - min_val) / (max_val - min_val + 1e-6)


def event_to_voxel_grid(events, num_bins, H, W):
    if events.shape[0] == 0:
        return torch.zeros((num_bins, H, W), dtype=torch.float32, device=events.device)

    t = events[:, 2]
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
    t_bins = (t_norm * (num_bins - 1)).long().clamp(0, num_bins - 1)

    x = events[:, 0].long()
    y = events[:, 1].long()

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t_bins = x[valid], y[valid], t_bins[valid]

    flat_idx = t_bins * H * W + y * W + x
    voxel = torch.zeros(num_bins * H * W, dtype=torch.float32, device=events.device)
    voxel.index_add_(0, flat_idx, torch.ones(flat_idx.shape[0], dtype=torch.float32, device=events.device))
    return normalize_voxel(voxel.view(num_bins, H, W))


def sample_event_clips(events, clips_per_file):
    if events.shape[0] == 0:
        return []
    t_min, t_max = events[:, 2].min(), events[:, 2].max()
    clip_duration = (t_max - t_min) / clips_per_file
    if clip_duration == 0:
        return [events]
    clips = []
    for i in range(clips_per_file):
        t_start = t_min + i * clip_duration
        t_end = t_start + clip_duration
        mask = (events[:, 2] >= t_start) & (events[:, 2] < t_end)
        clip = events[mask]
        if clip.shape[0] > 0:
            clips.append(clip)
    return clips


def compute_time_surface(clip, H, W):
    if clip.shape[0] == 0:
        return np.zeros((H, W), dtype=np.float32)

    t_range = (clip[:, 2].max() - clip[:, 2].min()).item()
    if t_range == 0:
        return np.zeros((H, W), dtype=np.float32)

    x = clip[:, 0].cpu().numpy()
    y = clip[:, 1].cpu().numpy()
    t = clip[:, 2].cpu().numpy()

    x = ((x - x.min()) / (x.ptp() + 1e-5) * (W - 1)).astype(int)
    y = ((y - y.min()) / (y.ptp() + 1e-5) * (H - 1)).astype(int)
    t_norm = (t - t.min()) / (t.ptp() + 1e-5)

    ts_surface = np.zeros((H, W), dtype=np.float32)
    for xi, yi, ti in zip(x, y, t_norm):
        if 0 <= yi < H and 0 <= xi < W:
            ts_surface[yi, xi] = max(ts_surface[yi, xi], ti)
    return ts_surface


def encode_frame(pos, g, out_size, normalize_fn):
    if pos.max() == 0:
        tensor = torch.zeros(3, out_size, out_size, dtype=torch.float32)
    else:
        r = (255 * np.log1p(pos) / (np.log1p(pos.max()) + 1e-5)).astype(np.uint8)
        b = np.zeros_like(r)
        g = (255 * g).astype(np.uint8)
        g[(r + b) == 0] = 0
        rgb = np.stack([r, g, b], axis=0)  # [C, H, W]
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0), size=(out_size, out_size), mode="bilinear", align_corners=False).squeeze(0)

    return normalize_fn(tensor)


def load_and_transform_event_data(
    event_paths,
    device,
    frames_per_clip: int = 2,
    clips_per_file: int = 5,
    out_size: int = 224,
):
    normalize = transforms.Normalize(mean=MEAN_MAP["event"], std=STD_MAP["event"])
    all_videos = []

    for path in event_paths:
        try:
            events = read_bin_event_file(path).to(device)
        except Exception as e:
            print(f"[ERROR] Failed to read file: {path} — {e}")
            continue

        if events.shape[0] == 0:
            print(f"[WARN] Empty event file: {path}")
            continue

        H, W = infer_resolution(events)
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        t_range = (t_max - t_min).item()

        if t_range == 0:
            print(f"[WARN] Constant timestamps in file: {path} — duplicating identical clip {clips_per_file} times.")
            clips = [events.clone() for _ in range(clips_per_file)]
        else:
            clips = sample_event_clips(events, clips_per_file)
            if len(clips) < clips_per_file:
                print(f"[WARN] Only {len(clips)}/{clips_per_file} valid clips from: {path} — padding with last clip.")
                if clips:
                    clips += [clips[-1].clone()] * (clips_per_file - len(clips))
                else:
                    print(f"[ERROR] All clips were empty for: {path}")
                    continue

        clip_tensors = []
        for i, clip in enumerate(clips):
            if clip.shape[0] == 0:
                print(f"[WARN] Empty clip {i} in file: {path} — skipping.")
                continue

            try:
                encode_g = (frames_per_clip == 1 and clips_per_file == 1)
                ts_surface = compute_time_surface(clip, H, W)
                voxel = event_to_voxel_grid(clip, frames_per_clip, H, W)

                frame_list = []
                for j in range(frames_per_clip):
                    pos = voxel[j].cpu().numpy()
                    g = ts_surface if encode_g else np.zeros_like(pos)
                    frame = encode_frame(pos, g, out_size, normalize)
                    frame_list.append(frame)

                clip_tensor = torch.stack(frame_list, dim=1)  # [C, T, H, W]
                clip_tensors.append(clip_tensor)
            except Exception as e:
                print(f"[ERROR] Failed to process clip {i} from {path} — {e}")

        if len(clip_tensors) < clips_per_file:
            print(f"[WARN] Only {len(clip_tensors)}/{clips_per_file} processed for: {path} — padding.")
            if clip_tensors:
                clip_tensors += [clip_tensors[-1].clone()] * (clips_per_file - len(clip_tensors))
            else:
                print(f"[ERROR] No valid clip tensors for: {path}")
                continue

        video_tensor = torch.stack(clip_tensors, dim=0).to(device)  # [V, C, T, H, W]
        all_videos.append(video_tensor)

    if not all_videos:
        print("[ERROR] All event files failed. Returning dummy tensor.")
        dummy = torch.zeros((1, clips_per_file, 3, frames_per_clip, out_size, out_size), dtype=torch.float32, device=device)
        return dummy

    return torch.stack(all_videos, dim=0)  # [B, V, C, T, H, W]

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
    modality, dataset_root, train_json, label_to_index,
    batch_size, num_workers, max_samples=None, debug=False
):
    transform = get_transform_fn(modality, minSample=True)
    dataset = JsonDataset(dataset_root, train_json, transform, label_to_index, max_samples, debug)
    sampler = DistributedSampler(dataset, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False, persistent_workers=True)

def val_data_loader(
    modality, dataset_root, val_json, label_to_index,
    batch_size, num_workers, max_samples=None, debug=False
):
    transform = get_transform_fn(modality, minSample=True)
    dataset = JsonDataset(dataset_root, val_json, transform, label_to_index, max_samples, debug)
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False, persistent_workers=True)

def get_normalization_tensors(modality, device):
    mean = torch.tensor(MEAN_MAP[modality], device=device)
    std = torch.tensor(STD_MAP[modality], device=device)
    if modality == "image":
        # For [B, C, H, W] images
        return mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    elif modality == "event":
        # For [B, C, H, W] event images
        return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    if modality == "point":
        # For [B, N, 3] point clouds
        return mean.view(1, 1, 3), std.view(1, 1, 3)
    elif modality == "thermal":
        # For [1, H, W] grayscale images
        return mean.view(1, 1, 1), std.view(1, 1, 1)
    elif modality == "video":
        # For [B, V, C, T, H, W] video tensors
        return mean.view(1, 1, -1, 1, 1, 1), std.view(1, 1, -1, 1, 1, 1)
    else:
        return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
