import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """
    Expands a 2D attention mask into a 4D attention mask for use in transformer models.

    Args:
        mask (torch.Tensor): shape (batch_size, src_len)
        dtype (torch.dtype): usually `hidden_states.dtype`
        tgt_len (Optional[int]): target length (e.g. sequence length), defaults to src_len

    Returns:
        torch.Tensor: shape (batch_size, 1, tgt_len, src_len)
                      with 0 for attend, -inf for mask
    """
    batch_size, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    # (batch_size, 1, tgt_len, src_len)
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len)
    # convert mask: 1 -> 0.0 (keep), 0 -> -inf (mask out)
    expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
    return expanded_mask

class CustomCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        # Normalize image size to (H, W)
        if isinstance(self.image_size, int):
            height = width = self.image_size
        else:
            height, width = self.image_size

        self.grid_size = (height // self.patch_size, width // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_positions = self.num_patches + 1

        # Replace position embedding with correct size
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape

        # Normalize expected image size
        if isinstance(self.image_size, int):
            expected_height = expected_width = self.image_size
        else:
            expected_height, expected_width = self.image_size

        if not interpolate_pos_encoding and (height != expected_height or width != expected_width):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({expected_height}*{expected_width})."
            )

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # (B, D, H', W')
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)                    # (B, N, D)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)             # (B, 1, D)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)              # (B, N+1, D)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings