#!/usr/bin/env python3

import torch
import os

def rename_vision_head(old_ckpt_path: str, new_ckpt_path: str):
    # Load state dict
    state_dict = torch.load(old_ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint with {len(state_dict)} parameters.")

    # Remap keys
    remapped_sd = {}
    for key, value in state_dict.items():
        if key.startswith("bind.modality_heads.vision.2."):
            new_key = key.replace("bind.modality_heads.vision.2.", "bind.modality_heads.vision.1.")
            print(f"Renamed: {key} → {new_key}")
            remapped_sd[new_key] = value
        else:
            remapped_sd[key] = value

    # Save new state dict
    torch.save(remapped_sd, new_ckpt_path)
    print(f"[✓] Saved updated checkpoint to: {os.path.abspath(new_ckpt_path)}")

if __name__ == "__main__":
    # Update these paths as needed
    old_checkpoint = "ckpts/pretrained_weights_flash_atten.pt"
    new_checkpoint = "ckpts/pretrained_weights_flash_atten_image_patchs.pt"

    rename_vision_head(old_checkpoint, new_checkpoint)
