import os
import abc
import json
from types import SimpleNamespace

import torch
import torch_scatter
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils

from concurrent.futures import ThreadPoolExecutor

from autoattack import AutoAttack
from imagebind.imagebind_model import ModalityType
from model import UniBind
from utils.utils import load_centre_embeddings


def unnormalize_inplace(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    x.mul_(std_t).add_(mean_t).clamp_(0, 1)
    return x


def normalize_inplace(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    x.sub_(mean_t).div_(std_t)
    return x


def parallel_save_images(adv_examples, eps_dir, batch_idx):
    """
    Saves each sample in adv_examples to eps_dir using multiple threads.
    Filenames will follow:
        batch{batch_idx}_idx{i}.png
    This function does NOT return anything or handle metadata.
    """
    adv_cpu = adv_examples.cpu()

    def save_image(tensor_img, path):
        vutils.save_image(tensor_img, path)

    futures = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(adv_cpu.size(0)):
            single_adv = adv_cpu[i]
            img_filename = f"batch{batch_idx}_idx{i}.png"
            img_save_path = os.path.join(eps_dir, img_filename)
            futures.append(executor.submit(save_image, single_adv, img_save_path))

    for f in futures:
        f.result()


class Attack(abc.ABC):
    def __init__(
        self,
        dataset,
        dataset_name="custom_dataset",
        centre_embeddings_path="./centre_embs/image_in_center_embeddings.pkl",
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
    def get_indices_from_labels(self, centre_labels, device) -> torch.Tensor:
        pass

    def get_labels_from_indices(self, indices) -> list:
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
    def __init__(
        self,
        dataset_adversary_root: str,
        metadata_output_root: str,
        metadata_prefix: str,
        device=None,
    ):
        self.pretrain_weights = "./ckpts/pretrained_weights.pt"
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = None
        self.dataset_adversary_root = dataset_adversary_root
        self.metadata_output_root = metadata_output_root
        self.metadata_prefix = metadata_prefix

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

    def predict(self, x, centre_embeddings, label_indices, attack: Attack):
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

        os.makedirs(self.dataset_adversary_root, exist_ok=True)
        os.makedirs(self.metadata_output_root, exist_ok=True)

        print("Loading center embeddings from:", attack.centre_embeddings_path)
        centre_embeddings, centre_labels = load_centre_embeddings(
            attack.centre_embeddings_path, self.device
        )
        centre_embeddings = centre_embeddings.to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        )
        centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)

        print("Mapping center labels to dataset IDs...")
        label_indices = attack.get_indices_from_labels(centre_labels, self.device)

        def local_predict(x):
            return self.predict(x, centre_embeddings, label_indices, attack)

        for eps in attack.epsilons:
            eps_int = int(eps * 255)
            print(f"Running AutoAttack for epsilon = {eps_int}/255")

            rel_eps_dir = f"eps{eps_int}"
            abs_eps_dir = os.path.join(self.dataset_adversary_root, rel_eps_dir)
            os.makedirs(abs_eps_dir, exist_ok=True)

            adv_metadata_eps = []

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
                print(f"Processing batch {batch_idx + 1} for epsilon = {eps_int}/255")

                x_test = x_test.to(self.device, dtype=torch.float32, non_blocking=True)
                y_test = y_test.to(self.device, dtype=torch.int64, non_blocking=True)
                labels = attack.get_labels_from_indices(y_test)

                x_test_unorm = x_test.clone().detach()
                unnormalize_inplace(x_test_unorm, attack.mean, attack.std)

                with torch.no_grad():
                    adv_examples = adversary.run_standard_evaluation(
                        x_test_unorm, y_test, bs=attack.batch_size
                    )

                unnormalize_inplace(adv_examples, attack.mean, attack.std)

                outpath = os.path.join(abs_eps_dir, f"eps{eps_int}_{batch_idx}.pth")
                torch.save(
                    {
                        "adv_complete": adv_examples,
                        "x_test": x_test,
                        "y_test": y_test,
                        "labels": labels,
                    },
                    outpath,
                )

                # parallel_save_images(adv_examples, abs_eps_dir, batch_idx)

                for idx_in_batch in range(adv_examples.size(0)):
                    img_filename = f"batch{batch_idx}_idx{idx_in_batch}.png"
                    img_save_path = os.path.join(rel_eps_dir, img_filename)
                    label_str = labels[idx_in_batch]
                    adv_metadata_eps.append(
                        {"data": os.path.abspath(img_save_path), "label": label_str}
                    )

            meta_filename = f"{self.metadata_prefix}_eps{eps_int}.json"
            meta_filepath = os.path.join(self.metadata_output_root, meta_filename)
            with open(meta_filepath, "w") as f:
                json.dump(adv_metadata_eps, f, indent=2)

            print(f"Metadata for eps={eps_int} saved to {meta_filepath}")

        print("AutoAttack completed for all epsilon values.")
