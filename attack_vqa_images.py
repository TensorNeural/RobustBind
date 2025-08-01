import os
import json
import torch
import argparse
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm
from torchvision import transforms
from types import SimpleNamespace
import logging

from model import Model, UniBind, ForwardMode, MODALITY_MAP
from attack import APGDAttack, AttackModel, two_stage_attack_l2
from transform import normalize_inplace, unnormalize_inplace
from data_util import (
    load_and_transform_vision_data,
    get_normalization_tensors
)
from shared_types import Modality

class UniBindModel(Model):
    def __init__(self, pretrain_weights, logger, lora_weights=None):
        super().__init__()
        self.unibind = UniBind(
            SimpleNamespace(pretrain_weights=pretrain_weights, modality=Modality.IMAGE),
            use_flash_attention=True,
            fine_tuned_weights=None,
            lora_weights=lora_weights,
            logger=logger,
            use_lora=(lora_weights is not None),
            lora_rank=4,
            lora_alpha=8.0,
            use_fine_tune=False
        )
        self.modality_key = MODALITY_MAP[Modality.IMAGE]

    def forward(self, x, mode=ForwardMode.EMBEDDINGS):
        return self.unibind.encode_vision_with_mlp({self.modality_key: x})

    def extract_tensor(self, x):
        return x

    def wrap_tensor(self, x):
        return x

    def data_to_device(self, x, device):
        return x.to(device)

    def load_lora_weights(self, lora_path):
        self.unibind.load_lora_weights(lora_path)


def save_adv_image(tensor, out_path):
    tensor = tensor.squeeze(0).clamp(0, 1).cpu()
    image = transforms.ToPILImage()(tensor)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def setup_logger(rank, output_path):
    logger = logging.getLogger(f"EvalLogger-Rank{rank}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[RANK {rank}] %(asctime)s - %(message)s")

    file_handler = logging.FileHandler(output_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def main(args):
    local_rank, rank, world_size, device = setup_distributed()

    args.output_dir = "output/llava/attack/vqa"
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.output_dir, f"rank{rank}_{timestamp}.log")
    logger = setup_logger(rank, log_path)

    if rank == 0:
        with open(args.val_json, "r") as f:
            all_data = json.load(f)
        if args.max_samples:
            all_data = all_data[:args.max_samples]
    else:
        all_data = None

    obj_list = [all_data]
    dist.broadcast_object_list(obj_list, src=0)
    all_data = obj_list[0]
    all_data = all_data[rank::world_size]

    mean, std = get_normalization_tensors(Modality.IMAGE, device)

    LORA_WEIGHTS = {
        "robustbind2": "./ckpts/vision_eps2_lora_weights.pt",
        "robustbind4": "./ckpts/vision_eps4_lora_weights.pt",
    }

    configs = []
    # for eps in [2, 4]:
    #     for model_tag in ["unibind", "robustbind2", "robustbind4"]:
    #         if model_tag == "unibind":
    #             lora_path = None
    #         elif model_tag == "robustbind2":
    #             lora_path = LORA_WEIGHTS["robustbind2"]
    #         elif model_tag == "robustbind4":
    #             lora_path = LORA_WEIGHTS["robustbind4"]
    #         else:
    #             continue
    #         configs.append((eps, model_tag, lora_path))
    
    for eps in [2, 4]:
        for model_tag in ["unibind"]:
            if model_tag == "unibind":
                lora_path = None
            elif model_tag == "robustbind2":
                lora_path = LORA_WEIGHTS["robustbind2"]
            elif model_tag == "robustbind4":
                lora_path = LORA_WEIGHTS["robustbind4"]
            else:
                continue
            configs.append((eps, model_tag, lora_path))

    for eps, model_tag, lora_path in configs:
        logger.info(f"Generating adversarial examples for {model_tag.upper()} at ε={eps}/255")
        model = UniBindModel(args.pretrain_weights, logger, lora_weights=lora_path).eval().to(device)
        attack_model = AttackModel(model, mean=mean, std=std)

        stage1 = APGDAttack(
            logger=logger,
            model=attack_model,
            norm="linf",
            n_restarts=1,
            n_iter=args.steps,
            eps=eps / 255.0,
            loss_type="l2",
            device=device
        )

        stage2 = APGDAttack(
            logger=logger,
            model=attack_model,
            norm="linf",
            n_restarts=1,
            n_iter=args.steps,
            eps=eps / 255.0,
            loss_type="l2",
            device=device
        )

        adv_dir = os.path.join(args.image_root, f"val_adv_eps{eps}_{model_tag}")
        adv_data_rank = []

        batches = [all_data[i:i + args.batch_size] for i in range(0, len(all_data), args.batch_size)]
        for batch in tqdm(batches, desc=f"[Rank {rank}] eps={eps} {model_tag}", disable=(rank != 0)):
            image_paths = [os.path.join(args.image_root, s["image"]) for s in batch]
            image_tensor = load_and_transform_vision_data(image_paths, device)

            with torch.no_grad():
                emb_orig = model(image_tensor, mode=ForwardMode.EMBEDDINGS)

            input = image_tensor.clone()
            unnormalize_inplace(input, mean, std)

            adv_input = two_stage_attack_l2(logger, attack_model, input, emb_orig, stage1, stage2, mean, std)

            for j, sample in enumerate(batch):
                filename = os.path.basename(sample["image"])
                out_path = os.path.join(adv_dir, filename)
                save_adv_image(adv_input[j:j + 1], out_path)
                updated_sample = sample.copy()
                updated_sample["image"] = f"{os.path.basename(adv_dir)}/{filename}"
                adv_data_rank.append(updated_sample)

        all_adv_data = [None for _ in range(world_size)]
        dist.all_gather_object(all_adv_data, adv_data_rank)

        if rank == 0:
            final_data = [x for group in all_adv_data for x in group]
            json_path = os.path.join(os.path.dirname(args.val_json), f"val_data_adv_eps{eps}_{model_tag}.json")
            with open(json_path, "w") as f:
                json.dump(final_data, f, indent=2)
            logger.info(f"[✔] Wrote {len(final_data)} entries to {json_path}")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--pretrain_weights", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    main(args)
