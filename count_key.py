import torch
from collections import defaultdict

def count_keys_by_full_prefix(weight_path):
    # Load the checkpoint
    state_dict = torch.load(weight_path, map_location='cpu')

    # Extract state_dict from common wrappers
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

    # Count using full prefix minus final parameter name (e.g., weight/bias)
    prefix_counts = defaultdict(int)
    for key in state_dict.keys():
        # Split off the parameter name (like weight/bias)
        parts = key.split('.')
        if len(parts) > 1:
            prefix = '.'.join(parts[:-1])
        else:
            prefix = parts[0]
        prefix_counts[prefix] += 1

    # Sort and print
    for prefix in sorted(prefix_counts):
        print(f"{prefix}: {prefix_counts[prefix]} keys")

# Example usage
count_keys_by_full_prefix("./ckpts/pretrained_weights.pt")
