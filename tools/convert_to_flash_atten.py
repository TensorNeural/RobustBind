#!/usr/bin/env python3

import os
import torch
from typing import Dict
from model import UniBind
from imagebind.transformer import MultiheadAttention, FlashAttention2
from imagebind.imagebind_model import ModalityType
import torch.nn.functional as F

###############################################################################
# Global config mappings
###############################################################################
prefix_to_config = {
    "bind.modality_trunks.vision":  (1280, 16),
    "bind.modality_trunks.text":    (1024, 16),
    "bind.modality_trunks.audio":   (768, 12),
    "bind.modality_trunks.depth":   (768, 12),
    "bind.modality_trunks.thermal": (768, 12),
    "bind.modality_trunks.imu":     (512, 8),
}

prefix_to_add_bias_kv = {
    "bind.modality_trunks.vision":  False,
    "bind.modality_trunks.text":    False,
    "bind.modality_trunks.audio":   True,
    "bind.modality_trunks.depth":   True,
    "bind.modality_trunks.thermal": True,
    "bind.modality_trunks.imu":     True,
}


###############################################################################
# 1) Helper function to split in_proj -> Q/K/V
###############################################################################
def split_in_proj(
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    embed_dim: int
):
    """
    Splits in_proj_weight ([3*D, D]) + in_proj_bias ([3*D]) => Q/K/V
    """
    q_w = in_proj_weight[:embed_dim, :]
    k_w = in_proj_weight[embed_dim : 2 * embed_dim, :]
    v_w = in_proj_weight[2 * embed_dim :, :]

    q_b = in_proj_bias[:embed_dim]
    k_b = in_proj_bias[embed_dim : 2 * embed_dim]
    v_b = in_proj_bias[2 * embed_dim :]

    return q_w, k_w, v_w, q_b, k_b, v_b


###############################################################################
# 2) Convert a state_dict from MultiheadAttention -> FlashAttention2
###############################################################################
def convert_mha_to_flash2(
    old_sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert a state_dict with MultiheadAttention in_proj_* 
    => separate q_proj/k_proj/v_proj for FlashAttention2.
    Preserves out_proj.* and all other unrelated parameters.
    """
    new_sd = {}
    visited_bias = set()

    for full_name, param in old_sd.items():
        if full_name.endswith("in_proj_weight"):
            base_name = full_name[: -len("in_proj_weight")]
            three_d, d_model = param.shape
            if three_d % 3 != 0:
                raise RuntimeError(
                    f"Expected [3*D, D], got {param.shape} in {full_name}"
                )
            embed_dim = three_d // 3

            # match in_proj_bias
            bias_name = base_name + "in_proj_bias"
            if bias_name not in old_sd:
                raise KeyError(f"Missing {bias_name} for {full_name}")
            bias_param = old_sd[bias_name]
            visited_bias.add(bias_name)

            # do the split
            q_w, k_w, v_w, q_b, k_b, v_b = split_in_proj(param, bias_param, embed_dim)

            new_sd[base_name + "q_proj.weight"] = q_w
            new_sd[base_name + "q_proj.bias"]   = q_b
            new_sd[base_name + "k_proj.weight"] = k_w
            new_sd[base_name + "k_proj.bias"]   = k_b
            new_sd[base_name + "v_proj.weight"] = v_w
            new_sd[base_name + "v_proj.bias"]   = v_b

        elif full_name.endswith("in_proj_bias"):
            # skip copying in_proj_bias directly, unless it was handled above
            if full_name not in visited_bias:
                print(f"Warning: found {full_name} with no matching in_proj_weight.")
            continue
        else:
            # copy everything else as-is
            new_sd[full_name] = param

    return new_sd


###############################################################################
# 3) Single-layer test function (optional usage)
###############################################################################
def test_single_attention_layer(
    old_attn_sd: Dict[str, torch.Tensor],
    new_attn_sd: Dict[str, torch.Tensor],
    embed_dim: int,
    num_heads: int,
    add_bias_kv: bool,
) -> float:
    """
    Compare old MHA vs. new FlashAttn on random input, returning mean difference.
    """
    # old MHA
    old_attn = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        add_bias_kv=add_bias_kv,
    )
    old_attn.load_state_dict(old_attn_sd, strict=True)
    old_attn.eval()

    # new FlashAttn
    new_attn = FlashAttention2(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=add_bias_kv,
    )
    new_attn.load_state_dict(new_attn_sd, strict=True)
    new_attn.eval()

    x = torch.randn(2, 5, embed_dim, dtype=torch.float32)
    with torch.no_grad():
        old_out = old_attn(x, attn_mask=None)[0]
        new_out = new_attn(x)

    diff = (old_out - new_out).abs().mean().item()
    return diff


def debug_single_layer_equivalence(
    prefix: str,
    old_sd: Dict[str, torch.Tensor],
    converted_sd: Dict[str, torch.Tensor],
    embed_dim: int,
    num_heads: int,
    add_bias_kv: bool
):
    """
    Compare old MHA vs. new FlashAttention2 on a single layer, 
    verifying both shape and output similarity.
    """
    # Gather the old and new per-layer state dicts
    old_attn_sd = {
        k[len(prefix):]: v
        for k, v in old_sd.items()
        if k.startswith(prefix)
    }
    new_attn_sd = {
        k[len(prefix):]: v
        for k, v in converted_sd.items()
        if k.startswith(prefix)
    }

    print(f"\n=== Debugging layer: {prefix} ===")
    print(f"old_attn_sd keys: {list(old_attn_sd.keys())}")
    print(f"new_attn_sd keys: {list(new_attn_sd.keys())}")

    # Build old MultiheadAttention
    old_attn = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=False,
        add_bias_kv=add_bias_kv
    ).eval()

    # Build new FlashAttention2
    new_attn = FlashAttention2(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=add_bias_kv
    ).eval()

    # Load state dicts
    old_attn.load_state_dict(old_attn_sd, strict=True)
    new_attn.load_state_dict(new_attn_sd, strict=True)

    # Forward pass with test input
    x = torch.randn(2, 5, embed_dim)  # shape [B, L, D]
    with torch.no_grad():
        old_out = old_attn(x, attn_mask=None)
        new_out = new_attn(x)

    # Compare shapes and values
    print(f"old_out.shape = {old_out.shape}, new_out.shape = {new_out.shape}")

    diff = (old_out - new_out).abs().mean().item()
    cos = F.cosine_similarity(old_out.flatten(), new_out.flatten(), dim=0).item()

    print(f"Mean diff: {diff:.6f}")
    print(f"Cosine similarity: {cos:.6f}")


###############################################################################
# 4) Helper for scanning all attention prefixes
###############################################################################
def find_attention_prefixes(old_sd: Dict[str, torch.Tensor]):
    """
    Return a sorted list of prefix strings that represent MHA layers
    by checking for in_proj_weight in the old checkpoint.
    """
    attn_prefixes = []
    for k in old_sd:
        if k.endswith("in_proj_weight"):
            prefix = k[: -len("in_proj_weight")]
            attn_prefixes.append(prefix)
    return sorted(set(attn_prefixes))


def test_all_attention_layers(
    old_sd: Dict[str, torch.Tensor],
    converted_sd: Dict[str, torch.Tensor]
):
    """
    Example of how you might test all attention prefixes with random input,
    using the test_single_attention_layer() function.
    """
    attn_prefixes = find_attention_prefixes(old_sd)
    print(f"Found {len(attn_prefixes)} attention prefix(es):")
    for pfx in attn_prefixes:
        print("  ", pfx)

    for prefix in attn_prefixes:
        old_attn_sd = {
            k[len(prefix):]: v
            for k, v in old_sd.items() 
            if k.startswith(prefix)
        }
        new_attn_sd = {
            k[len(prefix):]: v
            for k, v in converted_sd.items()
            if k.startswith(prefix)
        }

        # Check if it's actually an MHA
        if "in_proj_weight" not in old_attn_sd:
            continue

        w = old_attn_sd["in_proj_weight"]
        three_d, d_model = w.shape
        if three_d % 3 != 0:
            continue

        # Figure out num_heads + add_bias_kv from the global config
        matched_num_heads = None
        add_bias_kv = False

        for trunk_prefix, (cfg_embed, cfg_heads) in prefix_to_config.items():
            if prefix.startswith(trunk_prefix):
                matched_num_heads = cfg_heads
                break
        for trunk_prefix, kv_flag in prefix_to_add_bias_kv.items():
            if prefix.startswith(trunk_prefix):
                add_bias_kv = kv_flag
                break

        # If we found a match for config, do the test
        if matched_num_heads is not None:
            diff = test_single_attention_layer(
                old_attn_sd,
                new_attn_sd,
                d_model,
                matched_num_heads,
                add_bias_kv
            )
            print(f"Prefix: {prefix}, single-layer diff: {diff:.6f}")


###############################################################################
# 5) End-to-end check using UniBind
###############################################################################
class FakeArgs:
    modality = "image"  
    def __init__(self, pretrain_weights=None):
        self.pretrain_weights = pretrain_weights
        # Add other fields if your UniBind constructor expects them


def unibind_end_to_end_test(
    old_checkpoint_path: str,
    new_checkpoint_path: str
):
    """
    Loads two UniBind models (old vs new) and compares final embeddings
    on dummy input to gauge end-to-end equivalence.
    """
    model_old = UniBind(
        args=FakeArgs(old_checkpoint_path),
        use_flash_attention=False
    ).eval()

    model_new = UniBind(
        args=FakeArgs(new_checkpoint_path),
        use_flash_attention=True
    ).eval()

    # Dummy input for the chosen modality
    image = torch.randn(2, 3, 224, 224)
    image = image.cuda()
    dummy_input = {ModalityType.VISION: image}

    with torch.no_grad():
        vision_old = model_old.encode_vision(dummy_input)
        vision_new = model_new.encode_vision(dummy_input)

    vision_diff = (vision_old - vision_new).abs().mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        vision_old, vision_new, dim=-1
    ).mean().item()

    print(f"\n=== UniBind end-to-end check (modality={model_old.args.modality}) ===")
    print(f"Vision embedding diff: {vision_diff:.6f}")
    print(f"Vision cosine similarity: {cosine_sim:.6f}")


###############################################################################
# 6) Main entry point
###############################################################################
def main():
    OLD_CHECKPOINT_PATH = "./ckpts/pretrained_weights.pt"  
    NEW_CHECKPOINT_PATH = "./ckpts/pretrained_weights_flash_atten.pt"  

    if not os.path.isfile(OLD_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find {OLD_CHECKPOINT_PATH}")

    # A) Load old checkpoint
    old_ckpt = torch.load(OLD_CHECKPOINT_PATH, map_location="cpu")
    if "state_dict" in old_ckpt:
        old_sd = old_ckpt["state_dict"]
    else:
        old_sd = old_ckpt
    print(f"Loaded old checkpoint with {len(old_sd)} params.")

    # B) Convert MHA -> FlashAttn
    converted_sd = convert_mha_to_flash2(old_sd)

    # C) (Optional) Test all attention prefixes
    #    (Comment in/out if you want to run single-layer checks)
    # test_all_attention_layers(old_sd, converted_sd)

    # D) Save new checkpoint
    torch.save(converted_sd, NEW_CHECKPOINT_PATH)
    print(f"\nSaved new checkpoint to {NEW_CHECKPOINT_PATH}")

    # E) Quick debugging example on a single layer
    print("\n=== Checking Q/K/V projection shapes ===")
    target_prefix = "bind.modality_trunks.vision.blocks.0.attn."
    w = old_sd[target_prefix + "in_proj_weight"]
    b = old_sd[target_prefix + "in_proj_bias"]

    print("in_proj_weight shape:", w.shape)
    print("in_proj_bias shape:", b.shape)

    embed_dim = 1280
    num_heads = 16
    add_bias_kv = False

    q_w, k_w, v_w, q_b, k_b, v_b = split_in_proj(w, b, embed_dim)
    print("Q weight:", q_w.shape)
    print("K weight:", k_w.shape)
    print("V weight:", v_w.shape)

    # Check old vs new single-layer equivalence
    debug_single_layer_equivalence(
        prefix=target_prefix,
        old_sd=old_sd,
        converted_sd=converted_sd,
        embed_dim=embed_dim,
        num_heads=num_heads,
        add_bias_kv=add_bias_kv
    )

    # F) End-to-end check with UniBind (requires GPU for .cuda() call)
    unibind_end_to_end_test(OLD_CHECKPOINT_PATH, NEW_CHECKPOINT_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
