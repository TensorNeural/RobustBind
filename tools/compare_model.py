#!/usr/bin/env python3

import sys
import torch

PREFIX_TO_IGNORE = "unibind.backbone."

def strip_prefix(key: str, prefix: str) -> str:
    """Remove the given prefix from key if it starts with it."""
    if key.startswith(prefix):
        return key[len(prefix):]
    return key

def normalize_state_dict(state_dict, prefix=PREFIX_TO_IGNORE):
    """
    Return a new dictionary where the specified prefix has been removed
    from each key if present.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = strip_prefix(k, prefix)
        new_sd[new_key] = v
    return new_sd

def compare_state_dicts(sd1, sd2, tol=1e-7):
    """
    Compare two state dicts:
      1) Check if they have the exact same parameter keys.
      2) Check if each tensor matches in shape and values.
    """
    # 1) Check same parameter keys
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    if keys1 != keys2:
        missing_in_sd2 = keys1 - keys2
        missing_in_sd1 = keys2 - keys1
        if missing_in_sd2:
            print(f"Keys in the first model but missing in the second: {missing_in_sd2}")
        if missing_in_sd1:
            print(f"Keys in the second model but missing in the first: {missing_in_sd1}")
        print("\nResult: The state dictionaries do NOT have the same set of keys.")
        return False
    
    # 2) Compare parameter shapes and values
    mismatch_found = False
    for k in sd1.keys():
        v1 = sd1[k]
        v2 = sd2[k]
        
        # Check shape
        if v1.shape != v2.shape:
            print(f"Parameter {k} has different shapes: {v1.shape} vs. {v2.shape}")
            mismatch_found = True
            continue
        
        # Check values elementwise
        if not torch.allclose(v1, v2, atol=tol, rtol=0):
            diff = (v1 - v2).abs()
            max_diff = diff.max().item()
            print(f"Parameter {k} differs (max diff = {max_diff}).")
            mismatch_found = True

    if mismatch_found:
        print("\nResult: The two state dictionaries differ.")
        return False
    else:
        print("All parameters match (within tolerance).")
        return True

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} model1.pt model2.pt")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    # Load the PyTorch checkpoints
    try:
        checkpoint1 = torch.load(file1, map_location='cpu')
    except Exception as e:
        print(f"Failed to load '{file1}': {e}")
        sys.exit(1)

    try:
        checkpoint2 = torch.load(file2, map_location='cpu')
    except Exception as e:
        print(f"Failed to load '{file2}': {e}")
        sys.exit(1)

    # If the loaded objects themselves are not state dicts,
    # look for 'state_dict' key often present in model checkpoints
    if isinstance(checkpoint1, dict) and "state_dict" in checkpoint1:
        checkpoint1 = checkpoint1["state_dict"]
    if isinstance(checkpoint2, dict) and "state_dict" in checkpoint2:
        checkpoint2 = checkpoint2["state_dict"]

    # Validate the loaded checkpoint is indeed a state dictionary or dict of tensors
    if not isinstance(checkpoint1, dict) or not isinstance(checkpoint2, dict):
        print("Error: Both files must be PyTorch checkpoints (state dicts).")
        sys.exit(1)

    # Normalize state dicts by removing the unibind.backbone. prefix
    sd1 = normalize_state_dict(checkpoint1, PREFIX_TO_IGNORE)
    sd2 = normalize_state_dict(checkpoint2, PREFIX_TO_IGNORE)

    print("\nStarting comparison...\n")
    compare_state_dicts(sd1, sd2, tol=1e-7)

if __name__ == "__main__":
    main()
