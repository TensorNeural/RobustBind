import torch
import torch.nn as nn
from model import UniBind, MODALITY_MAP
from shared_types import Modality
from data_util import load_and_transform_image_data
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
        self.size = {'shortest_edge': 224}
        self.crop_size = {'height': 224, 'width': 224}

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
        print(f"[DEBUG] tensor shape: {tensor.shape}")
        if return_tensors == "pt":
            return {"pixel_values": tensor}
        raise ValueError("Only return_tensors='pt' is supported")


class UniBindVisionTower(nn.Module):
    def __init__(self, args: UniBindVisionTowerArgs):
        super().__init__()
        self.args = args
        self.image_processor = ImageProcessor(load_and_transform_image_data)
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
        
        print(f"[DEBUG] images shape: {images.shape}")
        modality_key = MODALITY_MAP[Modality.IMAGE]
        outputs = []
        for i, image_patches in enumerate(images):  # image_patches: [N_patches, C, H, W]
            inp_dict = {modality_key: image_patches}  # expects batched patches
            img_output = self.unibind.encode_vision_with_mlp(inp_dict)  # [N_patches, D] or [N, D]
            outputs.append(img_output)

        output = torch.stack(outputs, dim=0)  # [B, N, D]
        print(f"[DEBUG] output shape: {output.shape}")
        return output

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self.unibind.device if self.is_loaded else torch.device("cpu")

    @property
    def config(self):
        class DummyConfig:
            image_size = 224
            patch_size = 14
        return DummyConfig()

