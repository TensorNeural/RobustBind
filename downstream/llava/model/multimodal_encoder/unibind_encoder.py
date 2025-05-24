import torch
import torch.nn as nn
import torch.nn.functional as F
from model import UniBind, MODALITY_MAP
from shared_types import Modality
from data_util import load_and_transform_patchable_image_data
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Optional
from PIL import Image

@dataclass
class UniBindVisionTowerArgs:
    # UniBind vision tower
    pretrain_weights: str
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_weights: Optional[str] = None

class ImageProcessor:
    def __init__(self, processor_fn):
        self.processor_fn = processor_fn

        self.size = {'shortest_edge': 224}
        self.crop_size = {'height': 224, 'width': 224}

    def preprocess(self, image_path, return_tensors=None):
        # Wrap single image path into a list
        tensor = self.processor_fn([image_path])  # returns [1, C, H, W]
        if return_tensors == "pt":
            return {"pixel_values": tensor}
        raise ValueError("Only return_tensors='pt' is supported")

TARGET_SIZE = 224      # Target size to resize each patch (CLIP-style)
GRID_SIZE = 24         # Number of patches per axis (14 × 14)

@dataclass
class UniBindVisionTowerArgs:
    pretrain_weights: str
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_weights: Optional[str] = None
    delay_load: bool = False

class ImageProcessor:
    def __init__(self, processor_fn):
        self.processor_fn = processor_fn

    def preprocess(self, image, return_tensors=None):
        """
        Args:
            image: PIL.Image or list of PIL.Image
        """
        if isinstance(image, Image.Image):
            images = [image]
        elif isinstance(image, list):
            images = image
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        tensor = self.processor_fn(images)  # [B, N_patches, 3, 224, 224]
        if return_tensors == "pt":
            return {"pixel_values": tensor}
        raise ValueError("Only return_tensors='pt' is supported")


class UniBindVisionTower(nn.Module):
    def __init__(self, args: UniBindVisionTowerArgs):
        super().__init__()
        self.args = args
        self.image_processor = ImageProcessor(load_and_transform_patchable_image_data)
        self.is_loaded = False

        if not args.delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print("[INFO] UniBind is already loaded. Skipping reload.")
            return

        self.unibind = UniBind(
            SimpleNamespace(
                pretrain_weights=self.args.pretrain_weights,
                modality=Modality.IMAGE
            ),
            use_flash_attention=True,
            use_lora=self.args.use_lora,
            use_fine_tune=False,
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_weights=self.args.lora_weights,
            fine_tuned_weights=None
        )
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape [B, N_patches, C, H, W]
        
        Returns:
            Tensor of shape [B, N, D], where N is output tokens per image.
        """
        if not self.is_loaded:
            raise RuntimeError("UniBind model is not loaded. Call `load_model()` first.")
        
        B, C, H, W = images.shape
        assert H % GRID_SIZE == 0 and W % GRID_SIZE == 0, \
            f"H and W must be divisible by GRID_SIZE={GRID_SIZE}"
        
        # Step 1: Extract 14x14 = 196 patches per image
        patch_H, patch_W = H // GRID_SIZE, W // GRID_SIZE
        patches = images.unfold(2, patch_H, patch_H).unfold(3, patch_W, patch_W)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, G, G, C, h, w]
        patches = patches.view(B * GRID_SIZE * GRID_SIZE, C, patch_H, patch_W)  # [B*N, 3, h, w]
        
        # Step 2: Resize each patch to 224×224
        patches = F.interpolate(patches, size=(TARGET_SIZE, TARGET_SIZE), mode="bicubic", align_corners=False)

        # Step 3: Encode using UniBind
        modality_key = MODALITY_MAP[Modality.IMAGE]
        encoded = self.unibind.encode_vision_with_mlp({modality_key: patches})  # [B*N, D]

        # Step 4: Reshape back to [B, N, D]
        encoded = encoded.view(B, GRID_SIZE * GRID_SIZE, -1)
        return encoded
    
    def patchify_and_resize(images: torch.Tensor, grid_size: int = 14, target_size: int = 224) -> torch.Tensor:
        """
        Differentiable patchify + upscale operation.

        Args:
            images: Tensor of shape [B, 3, H, W]
                    (H and W must be divisible by grid_size)
            grid_size: Number of patches per axis (14 → 14×14 = 196 total patches)
            target_size: Size to upsample each patch to (e.g., 224)

        Returns:
            Tensor of shape [B * N, 3, target_size, target_size],
            where N = grid_size^2 = number of patches per image
        """
        B, C, H, W = images.shape
        patch_H = H // grid_size
        patch_W = W // grid_size

        assert H % grid_size == 0 and W % grid_size == 0, "Input H and W must be divisible by grid_size"

        # Step 1: Extract patches [B, grid_size, grid_size, C, patch_H, patch_W]
        patches = images.unfold(2, patch_H, patch_H).unfold(3, patch_W, patch_W)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B * grid_size * grid_size, C, patch_H, patch_W)  # [B*N, 3, h, w]

        # Step 2: Resize patches to [B*N, 3, target_size, target_size]
        patches = F.interpolate(patches, size=(target_size, target_size), mode="bicubic", align_corners=False)

        return patches


    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self.unibind.device if self.is_loaded else torch.device("cpu")

    @property
    def config(self):
        class DummyConfig:
            image_size = 336
            patch_size = 14
        return DummyConfig()

