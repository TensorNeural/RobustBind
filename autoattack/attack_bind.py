import os
import abc
from types import SimpleNamespace

import torch
import torch_scatter
from torch.utils.data import DataLoader, Subset

from autoattack import AutoAttack
from imagebind.imagebind_model import ModalityType
from model import UniBind
from utils.utils import load_centre_embeddings


def unnormalize_inplace(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverse normalization in-place: (x * std) + mean, clamped to [0,1].
    """
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    x.mul_(std_t).add_(mean_t).clamp_(0, 1)
    return x


def normalize_inplace(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalizes in-place: (x - mean) / std.
    """
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    x.sub_(mean_t).div_(std_t)
    return x


class Attack(abc.ABC):
    def __init__(
        self,
        dataset,
        dataset_name="custom_dataset",
        centre_embeddings_path="./centre_embs/image_in_center_embeddings.pkl",
        save_dir="./results",
        batch_size=25,
        max_samples=50000,
        epsilons=[2 / 255, 4 / 255],
        norm="Linf",
        version="custom",
        log_root="./logs",
        modality="image",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.centre_embeddings_path = centre_embeddings_path
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.epsilons = epsilons
        self.norm = norm
        self.version = version
        self.mean = mean
        self.std = std

        if not os.path.exists(log_root):
            os.makedirs(log_root, exist_ok=True)

        self.log_path = os.path.join(log_root, f"log_{dataset_name}.txt")
        self.modality = modality

    @abc.abstractmethod
    def get_label_indices(self, centre_labels, device) -> torch.Tensor:
        pass


modality_map = {
    "image": ModalityType.VISION,
    "video": ModalityType.VISION,
    "audio": ModalityType.AUDIO,
    "thermal": ModalityType.THERMAL,
    "point": ModalityType.POINT,
    "event": ModalityType.VISION,
}


class AutoAttackRunner:
    def __init__(self, device=None):
        # Hard-coded
        self.pretrain_weights = "./ckpts/pretrained_weights.pt"

        # Decide device
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = None  # We'll load it once we know the user-specified modality

    def _init_model(self, attack_modality: str):
        print("Initializing UniBind with weights:", self.pretrain_weights)
        print(f"Attack modality: {attack_modality}")

        self.model = UniBind(
            SimpleNamespace(
                **{
                    "pretrain_weights": self.pretrain_weights,
                    "modality": attack_modality,
                }
            )
        ).to(self.device)

        self.model.eval()
        print("UniBind model ready.")

    def predict(self, x, centre_embeddings, label_indices, attack):
        modality = modality_map[attack.modality]
        x_dict = {modality: x}

        visual_embeddings = self.model.encode_vision(x_dict).to(torch.bfloat16)
        visual_embeddings = visual_embeddings / visual_embeddings.norm(
            dim=-1, keepdim=True
        )
        similarity = visual_embeddings @ centre_embeddings.t()
        similarity = similarity.to(torch.bfloat16)
        class_raw_scores, _ = torch_scatter.scatter_max(
            similarity, label_indices.expand(similarity.shape[0], -1), dim=1
        )
        return class_raw_scores

    def run(self, attack: Attack):
        if self.model is None:
            self._init_model(attack.modality)

        if attack.max_samples is not None and attack.max_samples < len(attack.dataset):
            indices = torch.randperm(len(attack.dataset))[: attack.max_samples]
            attack.dataset = Subset(attack.dataset, indices)

        loader = DataLoader(
            attack.dataset,
            batch_size=attack.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=8,
            persistent_workers=True,
        )
        os.makedirs(attack.save_dir, exist_ok=True)

        print("Loading center embeddings from:", attack.centre_embeddings_path)
        centre_embeddings, centre_labels = load_centre_embeddings(
            attack.centre_embeddings_path, self.device
        )
        centre_embeddings = centre_embeddings.to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        )
        centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)

        print("Mapping center labels to dataset IDs...")
        label_indices = attack.get_label_indices(centre_labels, self.device)

        def local_predict(x):
            return self.predict(x, centre_embeddings, label_indices, attack)

        for eps in attack.epsilons:
            print(f"Running AutoAttack for epsilon = {int(eps * 255)}/255")
            adversary = AutoAttack(
                local_predict,
                norm=attack.norm,
                eps=eps,
                log_path=attack.log_path,
                version=attack.version,
                attacks_to_run=["apgd-ce"],
            )

            for batch_idx, (x_test, y_test) in enumerate(loader):
                torch.cuda.empty_cache()
                print(f"Processing batch {batch_idx + 1} for epsilon = {int(eps * 255)}/255")

                x_test = x_test.to(self.device, dtype=torch.float32, non_blocking=True)
                y_test = y_test.to(self.device, dtype=torch.int64, non_blocking=True)

                x_test_unorm = x_test.clone().detach()
                unnormalize_inplace(x_test_unorm, attack.mean, attack.std)

                with torch.no_grad():
                    adv_examples = adversary.run_standard_evaluation(
                        x_test_unorm, y_test, bs=attack.batch_size
                    )

                normalize_inplace(adv_examples, attack.mean, attack.std)
                outpath = os.path.join(
                    attack.save_dir, f"adv_results_eps{int(eps * 255)}_{batch_idx}.pth"
                )
                torch.save(
                    {"adv_complete": adv_examples, "x_test": x_test, "y_test": y_test},
                    outpath,
                )

        print(
            "AutoAttack completed for all epsilon values. .pth files are saved in:",
            attack.save_dir,
        )
