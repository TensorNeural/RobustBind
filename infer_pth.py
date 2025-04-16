import torch
from imagebind.imagebind_model import ModalityType
import argparse
import logging
from utils.utils import set_env, load_centre_embeddings
from model import UniBind

logger = logging.getLogger(__name__)


def direct_evaluate_adversarial(args, model, device="cuda"):
    data_dict = torch.load(args.adv_pth_path, map_location=device)
    adv_examples_norm = data_dict["adv_complete"]
    adv_similarity = (
        data_dict["adv_similarity"] if "adv_similarity" in data_dict else None
    )
    adv_labels = data_dict["adv_labels"] if "adv_labels" in data_dict else None
    labels = data_dict["labels"]

    centre_embeddings, centre_labels = load_centre_embeddings(
        args.centre_embeddings_path, device
    )
    centre_embeddings = centre_embeddings.to(device)
    centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)

    model.eval()

    with torch.no_grad():
        inputs = {ModalityType.VISION: adv_examples_norm}
        embeddings = model.encode_vision(inputs).to(device)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

        similarity = embeddings @ centre_embeddings.t()
        logic = similarity.softmax(dim=-1)
        predicted_indices = logic.argmax(dim=-1)

    predicted_labels = [centre_labels[idx] for idx in predicted_indices.cpu().tolist()]

    correct = 0
    total = len(labels)

    for i in range(total):
        if predicted_labels[i] == labels[i]:
            correct += 1

    if adv_similarity is not None:
        diff = similarity - adv_similarity
        nonzero_rows = torch.nonzero((diff != 0).any(dim=1))
        print(f"Number of similarity diff rows: {len(nonzero_rows)}")

    adv_accuracy = (
        sum(
            adv_label == predicted_label
            for adv_label, predicted_label in zip(adv_labels, predicted_labels)
        )
        / len(adv_labels)
        * 100
    )
    print(f"Accuracy between predicted and adversarial labels: {adv_accuracy:.2f}%")

    accuracy = 100.0 * correct / total
    print(f"Accuracy on adversarial (.pth) data (centers method): {accuracy:.2f}%")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--adv_pth_path",
        type=str,
        default="/home/user/datasets/ImageNet-1K/new_val_adv/eps0/eps0_0.pth",
    )
    parser.add_argument(
        "--centre_embeddings_path",
        type=str,
        default="./centre_embs/image_in_center_embeddings.pkl",
    )
    parser.add_argument(
        "--pretrain_weights", type=str, default="./ckpts/pretrained_weights.pt"
    )
    parser.add_argument("--modality", type=str, default="image")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/val_data_zero_shot"
    )
    args = parser.parse_args()

    set_env(args, run_type=f"{args.modality}_infer")

    model = UniBind(args)
    model.to(device)

    direct_evaluate_adversarial(args, model, device=device)
