import torch
import torch.nn as nn
import torch.nn.init as init
from types import SimpleNamespace
from model import UniBind
from utils.utils import load_centre_embeddings

def init_linear_as_identity(linear_layer):
    assert linear_layer.in_features == linear_layer.out_features
    init.eye_(linear_layer.weight)
    nn.init.zeros_(linear_layer.bias)
    return linear_layer

class UniBindClassifier(nn.Module):
    def __init__(
        self,
        device,
        pretrain_weights,
        modality,
        centre_embeddings,
        centre_labels,
        label_to_index,
        index_to_label,
        load_unibind_pretrained=True
    ):
        super().__init__()
        self.unibind = UniBind(
            SimpleNamespace(pretrain_weights=pretrain_weights, modality=modality),
            load_pretrained=load_unibind_pretrained
        )
        self.modality = modality
        self.label_to_index_map = label_to_index
        self.index_to_label_map = index_to_label
        self.centre_embeddings = centre_embeddings.to(device)
        self.centre_label_indices = torch.tensor(
            [self.label_to_index_map[lbl] for lbl in centre_labels],
            dtype=torch.int64,
            device=device
        )

def check_if_trained(weight_tensor, atol=1e-6):
    identity = torch.eye(weight_tensor.size(0), device=weight_tensor.device)
    is_trained = not torch.allclose(weight_tensor, identity, atol=atol)
    return is_trained

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Update these paths as needed ===
    pretrain_weights_path = "./ckpts/pretrained_weights.pt"
    centre_embedding_path = "./centre_embs/image_in_center_embeddings.pkl"
    checkpoint_path = "./output/best_model.pt"

    # === Load center embeddings ===
    centre_embeddings, centre_labels = load_centre_embeddings(centre_embedding_path, device)
    centre_embeddings = centre_embeddings / centre_embeddings.norm(dim=-1, keepdim=True)

    unique_labels = sorted(list(set(centre_labels)))
    label_to_index = {lbl: i for i, lbl in enumerate(unique_labels)}
    index_to_label = {v: k for k, v in label_to_index.items()}

    # === Initialize model ===
    model = UniBindClassifier(
        device=device,
        pretrain_weights=pretrain_weights_path,
        modality="image",
        centre_embeddings=centre_embeddings,
        centre_labels=centre_labels,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        load_unibind_pretrained=False
    ).to(device)

    # === Load weights ===
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # === Check if trained ===
    weight_tensor = model.unibind.mlp_for_image.weight.data
    trained = check_if_trained(weight_tensor)

    if trained:
        print("✅ `mlp_for_image.weight` HAS BEEN TRAINED (updated from identity).")
    else:
        print("❌ `mlp_for_image.weight` is still the IDENTITY MATRIX (not trained).")

if __name__ == "__main__":
    main()
