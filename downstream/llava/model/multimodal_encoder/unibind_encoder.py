from email.mime import image
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared_types import Modality
from types import SimpleNamespace
from data_util import load_and_transform_patchable_image_data
from model import MODALITY_MAP
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from model import UniBind

INPUT_IMAGE_SIZE = 336
GRID_SIZE = 14
PATCH_SIZE = INPUT_IMAGE_SIZE // GRID_SIZE
EMBEDDING_DIM = 1024

UNIBIND_IMAGE_SIZE = 224

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

        # self.size = {'shortest_edge': INPUT_IMAGE_SIZE}
        # self.crop_size = {'height': INPUT_IMAGE_SIZE, 'width': INPUT_IMAGE_SIZE}
        self.size = {'shortest_edge': UNIBIND_IMAGE_SIZE}
        self.crop_size = {'height': UNIBIND_IMAGE_SIZE, 'width': UNIBIND_IMAGE_SIZE}

    def preprocess(self, image_path, return_tensors=None):
        # Wrap single image path into a list
        tensor = self.processor_fn([image_path])  # returns [1, C, H, W]
        if return_tensors == "pt":
            return {"pixel_values": tensor}
        raise ValueError("Only return_tensors='pt' is supported")

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
        if not self.is_loaded:
            raise RuntimeError("UniBind model is not loaded. Call `load_model()` first.")
        
        # BatchSize, Channels, Height, Width
        B, C, H, W = images.shape
        assert H % GRID_SIZE == 0 and W % GRID_SIZE == 0, \
            f"H and W must be divisible by GRID_SIZE={GRID_SIZE}"
        
        # Patchify + resize
        # [BatchSize * PATCH_COUNT, Channels, UNIBIND_IMAGE_SIZE, UNIBIND_IMAGE_SIZE]
        patches = self.patchify_and_resize(images)
        modality_key = MODALITY_MAP[Modality.IMAGE]

        # [BatchSize * PATCH_COUNT, EMBEDDING_DIM]
        patch_embeddings = self.unibind.encode_vision_with_mlp({modality_key: patches})

        # PATCH_COUNT
        N = patches.size(0) // B

        # [BatchSize, PATCH_COUNT, EMBEDDING_DIM]
        patch_embeddings = patch_embeddings.view(B, N, -1)

        # TODO: support with CLS token
        # [BatchSize, EMBEDDING_DIM]
        # image_global_embeddings = self.unibind.encode_vision_with_mlp({modality_key: x})
        # [BatchSize, 1, EMBEDDING_DIM]
        # cls = image_global_embeddings.unsqueeze(1)

        # TODO: support with CLS token
        # [BatchSize, PATCH_COUNT+1, EMBEDDING_DIM]
        # return torch.cat([cls, patch_embeddings], dim=1)
        return patch_embeddings

    def patchify_and_resize(self, images: torch.Tensor, grid_size: int = GRID_SIZE, target_size: int = UNIBIND_IMAGE_SIZE):
        # BatchSize, Channels, Height, Width
        B, C, H, W = images.shape
        patch_H = H // grid_size
        patch_W = W // grid_size
        assert H % grid_size == 0 and W % grid_size == 0, "H and W must be divisible by grid_size"
        # [BatchSize, Channels, grid_size, grid_size, patch_H, patch_W]
        patches = images.unfold(2, patch_H, patch_H).unfold(3, patch_W, patch_W)

        # [BatchSize, grid_size, grid_size, Channels, patch_H, patch_W]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        # [BatchSize * PATCH_COUNT, Channels, patch_H, patch_W]
        patches = patches.view(B * grid_size * grid_size, C, patch_H, patch_W)

        # [BatchSize * PATCH_COUNT, Channels, target_size, target_size]
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
            image_size = INPUT_IMAGE_SIZE

            # TODO: fix patch size
            # patch_size = PATCH_SIZE
            patch_size = 14
        return DummyConfig()
