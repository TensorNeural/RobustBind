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
from imagebind.multimodal_preprocessors import SimpleTokenizer
import numpy as np
from shared_types import BindModelType, Modality
from binds.languagebind import transform_dict as lb_transform_dict

from utils.utils import load_centre_embeddings

BPE_PATH = "bpe/bpe_simple_vocab_16e6.txt.gz"

# ===================
# Modality Constants
# ===================
MEAN_MAP = {
    Modality.IMAGE: [0.48145466, 0.4578275, 0.40821073],
    Modality.EVENT: [0.48145466, 0.4578275, 0.40821073],
    Modality.THERMAL: [0.5],
    Modality.VIDEO: [0.48145466, 0.4578275, 0.40821073],
    Modality.AUDIO: [-4.268],
    Modality.POINT: [0.0, 0.0, 0.0],
}

STD_MAP = {
    Modality.IMAGE: [0.26862954, 0.26130258, 0.27577711],
    Modality.EVENT: [0.26862954, 0.26130258, 0.27577711],
    Modality.THERMAL: [0.5],
    Modality.VIDEO: [0.26862954, 0.26130258, 0.27577711],
    Modality.AUDIO: [9.138],
    Modality.POINT: [1.0, 1.0, 1.0],
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
def get_transform_fn(modality, minSample=False, model_type=None):
    return {
        Modality.TEXT: load_and_transform_text,
        Modality.IMAGE: load_and_transform_vision_data,
        Modality.EVENT: load_and_transform_event_data,
        # Modality.EVENT: load_and_transform_vision_data,
        Modality.THERMAL: load_and_transform_thermal_data(model_type=model_type),
        Modality.VIDEO: load_and_transform_video_data(minSample=minSample, model_type=model_type),
        Modality.AUDIO: load_and_transform_audio_data(model_type),
        Modality.POINT: load_and_transform_point_data
    }[modality]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_MAP[Modality.IMAGE], std=STD_MAP[Modality.IMAGE]),
])

def load_and_transform_vision_data(image_paths, device):
    images = []
    for p in image_paths:
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        images.append(IMAGE_TRANSFORM(img).to(device))
    return torch.stack(images)

def load_and_transform_thermal_data(model_type=BindModelType.IMAGEBIND):
    def transform(thermal_paths, device):
        # Determine expected channels and color mode
        if model_type == BindModelType.LANGUAGEBIND:
            expected_channels = 3
            color_mode = "RGB"
        else:  # UniBind/ImageBind expects 1 channel
            expected_channels = 1
            color_mode = "L"

        # Load mean and std, then expand if needed
        mean = MEAN_MAP[Modality.THERMAL]
        std = STD_MAP[Modality.THERMAL]
        if len(mean) == 1 and expected_channels == 3:
            mean = mean * 3
            std = std * 3

        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        images = []
        for p in thermal_paths:
            with open(p, "rb") as f:
                img = Image.open(f).convert(color_mode)
            images.append(preprocess(img).to(device))

        return torch.stack(images)

    return transform

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

def load_and_transform_audio_data(model_type):
    def transform(
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
        if model_type == BindModelType.LANGUAGEBIND:
            num_mel_bins = 112
            target_length = 1036
            clip_duration = 10.4

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
    return transform

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
    model_type=BindModelType.IMAGEBIND
):
    def transform(
        video_paths,
        device,
        sample_rate=16000,
        target_t=8  # target temporal frames for LanguageBind
    ):
        if video_paths is None:
            return None

        video_outputs = []

        if model_type == BindModelType.LANGUAGEBIND:
            # LanguageBind settings
            frame_sampler = UniformTemporalSubsample(num_samples=target_t)
            spatial_crop = None  # no multi-crop
            video_transform = transforms.Compose([
                ShortSideScale(224),
                transforms.CenterCrop(224),
                Normalize(mean=MEAN_MAP[Modality.VIDEO], std=STD_MAP[Modality.VIDEO]),
            ])
        else:
            # ImageBind settings
            frame_sampler = UniformTemporalSubsample(num_samples=2)
            spatial_crop = SpatialCrop(224, num_crops=3)
            video_transform = transforms.Compose([
                ShortSideScale(224),
                Normalize(mean=MEAN_MAP[Modality.VIDEO], std=STD_MAP[Modality.VIDEO]),
            ])

        for video_path in video_paths:
            video = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
                **{"sample_rate": sample_rate},
            )

            if model_type == BindModelType.LANGUAGEBIND:
                # One full clip
                all_clips_timepoints = [(0, video.duration)]
            else:
                # One short clip of 2 frames
                clip_duration = 2
                clips_per_video = 1
                all_clips_timepoints = get_clip_timepoints(
                    ConstantClipsPerVideoSampler(clip_duration, clips_per_video),
                    video.duration
                )

            all_video = []
            for clip_timepoints in all_clips_timepoints:
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                if clip is None:
                    continue
                video_clip = frame_sampler(clip["video"]) / 255.0  # [C, T, H, W]
                all_video.append(video_clip)

            if not all_video:
                raise ValueError(f"No valid clips in video {video_path}")

            if model_type == BindModelType.LANGUAGEBIND:
                video_tensor = video_transform(all_video[0])  # [C, T, H, W]
                T_actual = video_tensor.shape[1]
                if T_actual < 8:
                    raise ValueError(f"Video {video_path} has only {T_actual} frames, require ≥8")
                T_trim = (T_actual // 8) * 8
                video_tensor = video_tensor[:, :T_trim]  # [C, T_trim, H, W]
            else:
                all_video = [video_transform(clip) for clip in all_video]
                all_video = spatial_crop(all_video)  # list of [C, T=2, H, W]
                video_tensor = torch.stack(all_video, dim=0)  # [V=3, C, T=2, H, W]

            video_outputs.append(video_tensor)

        if model_type == BindModelType.LANGUAGEBIND:
            return torch.stack(video_outputs, dim=0).to(device)  # [B, C, T, H, W]
        else:
            return torch.stack(video_outputs, dim=0).to(device)  # [B, V, C, T=2, H, W]

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
    return torch.from_numpy(events)

def read_npz_event_file(path):
    raw = np.load(path)["event_data"]
    events_np = np.stack([raw["x"], raw["y"], raw["t"], raw["p"]], axis=1).astype(np.float32)
    return torch.from_numpy(events_np)

def load_event_file(path):
    if path.endswith(".bin"):
        return read_bin_event_file(path)
    elif path.endswith(".npz"):
        return read_npz_event_file(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

def infer_resolution(events, default_H=180, default_W=240):
    if events.shape[0] == 0:
        return default_H, default_W
    H = max(int(events[:, 1].max().item()) + 1, default_H)
    W = max(int(events[:, 0].max().item()) + 1, default_W)
    return H, W

def split_events_temporally(events, num_bins):
    events = events[events[:, 2].argsort()]
    return np.array_split(events, num_bins)

def generate_eventbind_rgb(h_pos, h_neg):
    h, w = h_pos.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = h_neg
    rgb[..., 1] = h_pos + h_neg
    rgb[..., 2] = h_pos
    return np.clip(rgb, 0, 1)

def render_event_frame(events, H, W):
    x = np.clip(events[:, 0], 0, W - 1).astype(int)
    y = np.clip(events[:, 1], 0, H - 1).astype(int)
    p = events[:, 3].astype(int)

    h_pos = np.zeros((H, W), dtype=np.float32)
    h_neg = np.zeros((H, W), dtype=np.float32)
    for xi, yi, pi in zip(x, y, p):
        if pi > 0:
            h_pos[yi, xi] += 1
        else:
            h_neg[yi, xi] += 1

    if h_pos.max() > 0: h_pos /= h_pos.max()
    if h_neg.max() > 0: h_neg /= h_neg.max()
    return generate_eventbind_rgb(h_pos, h_neg)

def load_and_transform_event_data(
    event_paths,
    device,
    frames_per_video: int = 1,
    out_size: int = 224,
    model_type=BindModelType.IMAGEBIND,
):
    normalize = transforms.Normalize(mean=MEAN_MAP[Modality.EVENT], std=STD_MAP[Modality.EVENT])
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((out_size, out_size), interpolation=transforms.InterpolationMode.BILINEAR)
    all_videos = []

    for path in event_paths:
        try:
            events = load_event_file(path).cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Failed to read {path} — {e}")
            continue

        if events.shape[0] == 0:
            print(f"[WARN] Empty event file: {path}")
            continue

        H, W = infer_resolution(torch.from_numpy(events))

        # Frame segmentation
        if model_type == BindModelType.LANGUAGEBIND:
            if frames_per_video < 8:
                raise ValueError("LanguageBind requires ≥ 8 frames.")
            segments = split_events_temporally(events, frames_per_video)
        elif model_type == BindModelType.IMAGEBIND:
            if frames_per_video not in (1, 2):
                raise ValueError("ImageBind only supports 1 (image) or 2 (video) frames.")
            segments = split_events_temporally(events, frames_per_video)
        else:
            raise ValueError("Unsupported model type.")

        # Convert segments to image tensors
        frame_tensors = []
        for segment in segments:
            rgb = render_event_frame(segment, H, W)
            img = Image.fromarray((rgb * 255).astype(np.uint8))
            img = resize(img)
            tensor = normalize(to_tensor(img))  # [3, H, W]
            frame_tensors.append(tensor)

        if model_type == BindModelType.LANGUAGEBIND:
            video_tensor = torch.stack(frame_tensors, dim=1)  # [3, T, H, W]
            T_raw = video_tensor.shape[1]
            if T_raw < 8:
                raise ValueError(f"{path} has only {T_raw} frames; LanguageBind requires ≥8.")
            T_trim = (T_raw // 8) * 8
            video_tensor = video_tensor[:, :T_trim]
            all_videos.append(video_tensor)  # [3, T_trim, H, W]

        elif model_type == BindModelType.IMAGEBIND:
            cropper = SpatialCrop(out_size, num_crops=3)

            if frames_per_video == 1:
                image_tensor = frame_tensors[0]  # [3, H, W]
                crops = cropper([image_tensor.unsqueeze(1)])  # -> list of [3, 1, H, W]
                crops = [c.squeeze(1) for c in crops]         # remove T=1 → [3, H, W]
                all_videos.append(torch.stack(crops, dim=0))  # [3, 3, H, W]

            elif frames_per_video == 2:
                video_tensor = torch.stack(frame_tensors, dim=1)  # [3, 2, H, W]
                crops = cropper([video_tensor])                   # list of [3, 2, H, W]
                all_videos.append(torch.stack(crops, dim=0))      # [3, 3, 2, H, W]

    if not all_videos:
        print("[ERROR] All event files failed. Returning dummy.")
        if model_type == BindModelType.LANGUAGEBIND:
            return torch.zeros((1, 3, 8, out_size, out_size), device=device)
        elif frames_per_video == 1:
            return torch.zeros((1, 3, 3, out_size, out_size), device=device)
        else:
            return torch.zeros((1, 3, 3, frames_per_video, out_size, out_size), device=device)

    return torch.stack(all_videos, dim=0).to(device)

def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens

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
    batch_size, num_workers, max_samples=None, debug=False,
    model_type=BindModelType.IMAGEBIND
):
    transform = get_transform_fn(modality, minSample=True, model_type=model_type)
    dataset = JsonDataset(dataset_root, train_json, transform, label_to_index, max_samples, debug)
    sampler = DistributedSampler(dataset, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False, persistent_workers=True)

def val_data_loader(
    modality, dataset_root, val_json, label_to_index,
    batch_size, num_workers, max_samples=None, debug=False,
    model_type=BindModelType.IMAGEBIND
):
    transform = get_transform_fn(modality, minSample=True, model_type=model_type)
    dataset = JsonDataset(dataset_root, val_json, transform, label_to_index, max_samples, debug)
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False, persistent_workers=True)

def get_normalization_tensors(modality, device, model_type=BindModelType.IMAGEBIND):
    mean = torch.tensor(MEAN_MAP[modality], device=device)
    std = torch.tensor(STD_MAP[modality], device=device)

    if modality == Modality.IMAGE:
        return mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    elif modality == Modality.EVENT:
        return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    elif modality == Modality.POINT:
        return mean.view(1, 1, 3), std.view(1, 1, 3)
    elif modality == Modality.THERMAL:
        return mean.view(1, 1, 1), std.view(1, 1, 1)
    elif modality == Modality.VIDEO:
        if model_type == BindModelType.LANGUAGEBIND:
            # For [B, C, T, H, W]
            return mean.view(1, 3, 1, 1, 1), std.view(1, 3, 1, 1, 1)
        else:
            # For [B, V, C, T, H, W]
            return mean.view(1, 1, 3, 1, 1, 1), std.view(1, 1, 3, 1, 1, 1)
    else:
        return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)