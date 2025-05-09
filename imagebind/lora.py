import math
import torch
import torch.nn as nn
from torch.amp import autocast

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = alpha / rank

        self.lora_down = nn.Linear(self.base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        base_out = self.base_layer(x)
        with autocast('cuda', enabled=False):
            lora_x = x.to(torch.float32)
            lora_down_out = self.lora_down(lora_x)
            lora_down_out.requires_grad_(True)
            lora_up_out = self.lora_up(lora_down_out)
        lora_out = lora_up_out.to(dtype=base_out.dtype)
        return base_out + self.scaling * lora_out

def init_lora(model, state_dict):
    """
    - Initializes LoRALinear base_layer weights from a standard Linear-based state_dict.
    - Removes used weights from state_dict.
    - Freezes all parameters except LoRA adapters (lora_down, lora_up, scale if exists).
    
    Returns:
        cleaned_state_dict (safe to load with model.load_state_dict(strict=False))
    """
    new_sd = dict(state_dict)
    assigned_keys = []

    def _init_and_freeze(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(child, "base_layer") and isinstance(child.base_layer, nn.Linear):
                # Load base layer weights from original checkpoint
                w_key = f"{full_name}.weight"
                b_key = f"{full_name}.bias"

                with torch.no_grad():
                    if w_key in new_sd:
                        child.base_layer.weight.data.copy_(new_sd[w_key])
                        assigned_keys.append(w_key)
                    if b_key in new_sd:
                        child.base_layer.bias.data.copy_(new_sd[b_key])
                        assigned_keys.append(b_key)

            # Recurse into children
            _init_and_freeze(child, full_name)

        # Freeze everything by default
        for param_name, param in module.named_parameters(recurse=False):
            if not any(x in prefix for x in ["lora_up", "lora_down"]):
                param.requires_grad = False
            else:
                # Unfreeze LoRA parameters
                param.requires_grad = True

    _init_and_freeze(model)

    for k in assigned_keys:
        del new_sd[k]

    return new_sd

def lora_load_state_dict(model, state_dict, allow_missing_substrings=("lora_", "base_layer")):
    """
    Loads state_dict with strict=False, then:
    - Fails if any unexpected keys exist
    - Fails if any missing keys are not LoRA-related
    - Produces no logs
    """
    state_dict = init_lora(model, state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if unexpected_keys:
         raise ValueError(f"Unexpected keys in state_dict: {unexpected_keys}")

    for k in missing_keys:
        if not any(substr in k for substr in allow_missing_substrings):
            raise ValueError(f"Unexpected missing key: {k}")

def load_lora_weights(model, checkpoint_path):
    """
    Loads LoRA adapter weights into the model.
    Ensures all expected LoRA weights are present, and no unexpected keys exist.

    Args:
        model (nn.Module): Model with LoRA layers already initialized.
        checkpoint_path (str): Path to checkpoint containing only lora_up / lora_down weights.
    """
    lora_state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(lora_state, strict=False)

    # Expected LoRA keys in the model
    expected_lora_keys = [k for k in model.state_dict().keys() if "lora_" in k]
    missing_lora_keys = [k for k in expected_lora_keys if k in missing_keys]

    if missing_lora_keys:
        raise ValueError(f"Missing expected LoRA keys: {missing_lora_keys}")
    if unexpected_keys:
        raise ValueError(f"Unexpected keys in LoRA checkpoint: {unexpected_keys}")


def save_lora_weights(model, path, allowed_substrings=("lora_up", "lora_down")):
    """
    Save only LoRA adapter weights (lora_up, lora_down, scale) to a file.
    """
    lora_sd = {
        k: v for k, v in model.state_dict().items()
        if any(substr in k for substr in allowed_substrings)
    }
    torch.save(lora_sd, path)