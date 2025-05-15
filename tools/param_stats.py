import torch
import os
import argparse
from collections import defaultdict
from datetime import datetime

def count_params(state_dict):
    return sum(v.numel() for v in state_dict.values() if v.dtype in [torch.float, torch.float32, torch.float16, torch.bfloat16])

def is_point_key(key):
    return key.startswith("point_encoder") or key.startswith("pc_projection") or "modality_head_point" in key or "modality_postprocessor_point" in key

def get_modality(key):
    for mod in ["image", "video", "audio", "thermal", "depth", "imu", "event", "text", "vision"]:
        if f".{mod}." in key:
            return mod
    return "unknown"

def is_lora_key(key):
    return "lora_" in key

def analyze_weights(ckpt_path_full, ckpt_path_lora, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        def log(msg=""):
            print(msg)
            f.write(msg + "\n")

        log(f"📦 Loading full checkpoint: {ckpt_path_full}")
        full_state = torch.load(ckpt_path_full, map_location="cpu")

        log(f"📦 Loading LoRA checkpoint: {ckpt_path_lora}")
        lora_state = torch.load(ckpt_path_lora, map_location="cpu")

        stats = {
            "lora": defaultdict(int),
            "non_lora": defaultdict(int),
            "point": 0,
            "logit_scale": 0
        }

        for key, val in full_state.items():
            if not isinstance(val, torch.Tensor):
                continue
            if key == "logit_scale":
                stats["logit_scale"] += val.numel()
            elif is_point_key(key):
                stats["point"] += val.numel()
            elif is_lora_key(key):
                continue
            else:
                mod = get_modality(key)
                stats["non_lora"][mod] += val.numel()

        for key, val in lora_state.items():
            if not isinstance(val, torch.Tensor):
                continue
            mod = get_modality(key)
            stats["lora"][mod] += val.numel()

        total_non_lora = sum(stats["non_lora"].values())
        total_lora = sum(stats["lora"].values())
        total_point = stats["point"]
        total_logit = stats["logit_scale"]
        total_all = total_non_lora + total_lora + total_logit + total_point

        log("\n=== 📊 Parameter Summary ===")
        log(f"Total Non-LoRA Params (excluding point): {total_non_lora:,}")
        log(f"Total LoRA Params: {total_lora:,}")
        log(f"Total Point Params: {total_point:,}")
        log(f"Total Logit Scale Params: {total_logit:,}")
        log(f"Total All Params: {total_all:,}\n")

        log("=== 📉 Per-Modality Breakdown ===")
        for mod in sorted(set(stats["non_lora"].keys()).union(stats["lora"].keys())):
            non_lora = stats["non_lora"].get(mod, 0)
            lora = stats["lora"].get(mod, 0)
            shrink_pct = 100 * (non_lora - lora) / non_lora if non_lora else 0
            lora_pct = 100 * lora / total_lora if total_lora else 0
            log(f"• {mod:8s} | Non-LoRA: {non_lora:,} | LoRA: {lora:,} | Shrunk: {shrink_pct:5.2f}% | LoRA% of LoRA Total: {lora_pct:5.2f}%")

        # Group: audio + thermal + vision
        group = ["audio", "thermal", "vision"]
        group_non_lora = sum(stats["non_lora"].get(mod, 0) for mod in group)
        group_lora = sum(stats["lora"].get(mod, 0) for mod in group)
        group_shrink_pct = 100 * (group_non_lora - group_lora) / group_non_lora if group_non_lora else 0
        log(f"\n• audio+thermal+vision | Non-LoRA: {group_non_lora:,} | LoRA: {group_lora:,} | Shrunk: {group_shrink_pct:5.2f}%")

        total_nl_plus_logit = total_non_lora + total_logit
        total_shrink_pct = 100 * (total_nl_plus_logit - total_lora) / total_nl_plus_logit if total_nl_plus_logit else 0
        total_lora_pct = 100 * total_lora / total_nl_plus_logit if total_nl_plus_logit else 0

        log("\n=== 📦 Total Shrink Stats (excluding point) ===")
        log(f"Overall Shrink % (Non-LoRA + Logit → LoRA): {total_shrink_pct:.2f}%")
        log(f"LoRA Params % of Non-LoRA + Logit:         {total_lora_pct:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--lora_ckpt", type=str, required=True)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"./output/param_stats_{timestamp}.log"
    analyze_weights(args.pretrained_ckpt, args.lora_ckpt, log_path)
