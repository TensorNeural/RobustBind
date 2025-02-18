import pickle
import json
import torch

def map_labels_to_centers(embeddings_path, output_json, device="cpu"):
    """Maps each label to a list of center indices and writes to a JSON file."""
    try:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
    except Exception as e:
        print(f"Error loading {embeddings_path}: {e}")
        return

    label_to_centers = {}  # Mapping of label to center indices
    center_index = 0  # Unique center index counter

    for label in sorted(embeddings.keys()):  # Sort for consistency
        emb_tensor = embeddings[label]

        if isinstance(emb_tensor, torch.Tensor):  # Ensure it's a tensor
            num_centers = emb_tensor.shape[0]  # Number of centers for this label
            label_to_centers[label] = list(range(center_index, center_index + num_centers))  # Assign indices
            center_index += num_centers  # Increment center index

    # Save to JSON file
    with open(output_json, "w") as json_file:
        json.dump(label_to_centers, json_file, indent=2)

    print(f"Saved {len(label_to_centers)} label-to-centers mappings to {output_json}")

if __name__ == "__main__":
    embeddings_path = "centre_embs/image_in_center_embeddings.pkl"  # Update if needed
    output_json = "label_to_centers.json"  # Output file name

    map_labels_to_centers(embeddings_path, output_json)
