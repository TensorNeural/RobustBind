import os
import torch
from datetime import datetime

ckpt_dir = "./ckpts"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

modalities = ["audio", "thermal", "vision"]
eps_versions = ["eps2", "eps4"]
log_path = os.path.join(output_dir, f"merge_validate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log(msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")

def merge_lora_weights(eps):
    merged = {}
    trained_keys = {}

    for mod in modalities:
        path = os.path.join(ckpt_dir, f"{mod}_{eps}_lora_weights.pt")
        log(f"📦 Loading {mod} from {path}")
        state = load_checkpoint(path)
        trained_keys[mod] = set()

        for k, v in state.items():
            if f".{mod}." in k:
                merged[k] = v.clone()
                trained_keys[mod].add(k)
            elif k not in merged:
                merged[k] = v.clone()

        log(f"🔑 Trained keys for {mod} ({len(trained_keys[mod])}):")
        for key in sorted(trained_keys[mod]):
            log(f"  - {key}")

    out_path = os.path.join(ckpt_dir, f"{eps}_lora_weights.pt")
    torch.save(merged, out_path)
    log(f"✅ Saved merged weights to {out_path}")
    return merged, trained_keys

def validate_merged(eps, merged, trained_keys):
    log(f"\n=== ✅ Validating {eps} merge ===")
    for mod in modalities:
        path = os.path.join(ckpt_dir, f"{mod}_{eps}_lora_weights.pt")
        state = load_checkpoint(path)

        for k in trained_keys[mod]:
            if not torch.equal(merged[k], state[k]):
                diff = (merged[k] - state[k]).abs().max().item()
                log(f"[❌] Value mismatch in key {k} for modality {mod} | max abs diff = {diff:.6f}")
                raise ValueError(f"Mismatch in key {k}")
            else:
                log(f"[✅] Verified key {k} for modality {mod}")

    log(f"✅ All trained weights validated for {eps}")

def main():
    for eps in eps_versions:
        log(f"\n=== 🔄 Merging weights for {eps} ===")
        merged, trained_keys = merge_lora_weights(eps)
        validate_merged(eps, merged, trained_keys)

if __name__ == "__main__":
    main()
