# ---------------------------------
# 1) Sorted Imports
# ---------------------------------
import abc
import argparse
import logging
import os
import time
from datetime import datetime  # <-- for timestamping

import torch
import torch.nn as nn
import torch_scatter
from enum import Enum
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Mixed precision
from torch.cuda.amp import autocast, GradScaler

# Optional compile if PyTorch 2.x:
# if hasattr(torch, 'compile'):
#     model = torch.compile(model)

from autoattack.autopgd_base import APGDAttack
from datasets.datasets import ImageNetDataset
from imagebind.imagebind_model import ModalityType
from model import UniBind
from utils.data_transform import IMAGE_TRANSFORM
from utils.utils import load_centre_embeddings

# ---------------------------------
# 2) Logging Setup
# ---------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# ---------------------------------
# 3) Model / Enum Definitions
# ---------------------------------
class LossType(str, Enum):
    CROSS_ENTROPY = "ce"
    LATENT_DIMENSION_REDUCTION = "ldr"

class BaseModel(nn.Module):
    @abc.abstractmethod
    def logits(self, x):
        pass

    @abc.abstractmethod
    def logits_loss_type(self) -> LossType:
        pass

    @abc.abstractmethod
    def encode(self, x):
        pass

# ---------------------------------
# 4) UniBind Model
# ---------------------------------
MODALITY_MAP = {
    "image": ModalityType.VISION,
    "video": ModalityType.VISION,
    "audio": ModalityType.AUDIO,
    "thermal": ModalityType.THERMAL,
    "point": ModalityType.POINT,
    "event": ModalityType.VISION
}

class UniBindModel(BaseModel):
    def __init__(
        self,
        device,
        pretrain_weights,
        modality,
        centre_embeddings,
        centre_labels,
        label_to_index,
        index_to_label,
        logger=None,
        load_unibind_pretrained=True
    ):
        super().__init__()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("Initializing UniBindModel...")

        from types import SimpleNamespace
        self.unibind = UniBind(
            SimpleNamespace(pretrain_weights=pretrain_weights, modality=modality),
            load_pretrained=load_unibind_pretrained,
            logger=self.logger
        )
        self.unibind.to(device)

        self.modality = modality
        self.label_to_index_map = label_to_index
        self.index_to_label_map = index_to_label

        self.logger.info("Storing centre embeddings on device...")
        self.centre_embeddings = centre_embeddings.to(device)

        self.logger.info("Building centre_label_indices...")
        self.centre_label_indices = torch.tensor(
            [self.label_to_index_map[lbl] for lbl in centre_labels],
            dtype=torch.int64,
            device=device
        )

    def logits(self, x):
        embeddings = self.encode(x)
        similarity = embeddings @ self.centre_embeddings.t()
        expanded_idx = self.centre_label_indices.expand(similarity.shape[0], -1)
        class_scores, _ = torch_scatter.scatter_max(similarity, expanded_idx, dim=1)
        return class_scores, similarity

    def logits_loss_type(self):
        return LossType.CROSS_ENTROPY

    def encode(self, x):
        modality = MODALITY_MAP[self.modality]
        inp_dict = {modality: x}
        emb = self.unibind.encode_vision_with_mlp(inp_dict)
        return emb / emb.norm(dim=-1, keepdim=True)

    def predict_label_index(self, similarity_row):
        pred_centroid = similarity_row.argmax().item()
        return self.centre_label_indices[pred_centroid].item()

# ---------------------------------
# 5) Training Helpers
# ---------------------------------
def compute_embedding_loss(emb1, emb2):
    return torch.nn.functional.mse_loss(emb1, emb2, reduction='sum')

@torch.no_grad()
def two_stage_attack(model, inputs, labels, attack_stage1: APGDAttack, attack_stage2: APGDAttack):
    """
    1) Attack with stage1
    2) Identify 'still correct' samples
    3) Attack those with stage2
    Return final adv batch.
    """
    logger.info("Running two-stage attack on batch...")

    # We'll do the forward in half precision if possible
    with autocast():
        adv_stage1 = attack_stage1.perturb(inputs, labels)

    logits_stage1, _ = model.logits(adv_stage1)
    preds_stage1 = logits_stage1.argmax(dim=1)

    correct_mask = (preds_stage1 == labels)
    adv_final = adv_stage1.clone()

    keep_idx = torch.nonzero(correct_mask).squeeze(-1)
    if len(keep_idx) > 0:
        logger.info(f"Stage1 left {len(keep_idx)}/{inputs.size(0)} samples correct. Attacking them with Stage2...")
        with autocast():
            adv_stage2 = attack_stage2.perturb(inputs[keep_idx], labels[keep_idx])
        adv_final[keep_idx] = adv_stage2

    logger.info("Two-stage attack finished.")
    return adv_final

def train_epoch(
    device,
    model_train: UniBindModel,
    model_original: UniBindModel,
    data_loader,
    optimizer,
    scheduler,
    attack: APGDAttack
):
    """
    Train for one epoch, logging the time it takes per batch and overall epoch time.
    """
    epoch_start_time = time.time()
    model_train.train()
    scaler = GradScaler()  # for half-precision training

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()

        logger.info(f"[TRAIN] Processing batch {batch_idx+1}/{len(data_loader)}")
        inp, lbl = inp.to(device), lbl.to(device)

        logger.info("Generating adversarial examples (one-stage) ...")
        model_train.eval()
        adv_inp = attack.perturb(inp, lbl)

        model_train.train()
        emb_adv = model_train.encode(adv_inp)

        with torch.no_grad():
            emb_orig = model_original.encode(inp)

        loss_val = compute_embedding_loss(emb_adv, emb_orig)
        logger.info(f"Embedding loss for this batch = {loss_val.item():.4f}")

        optimizer.zero_grad()
        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} training time: {batch_end_time - batch_start_time:.2f} seconds")

    epoch_end_time = time.time()
    logger.info(f"Epoch training time: {epoch_end_time - epoch_start_time:.2f} seconds")

@torch.no_grad()
def evaluate_robust_one_stage(model: UniBindModel, data_loader, device, one_attack: APGDAttack):
    """
    Evaluate the model under a one-stage APGDAttack, logging per-batch timing.
    """
    eval_start_time = time.time()
    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()

        logger.info(f"[EVAL ONE-STAGE] Evaluating batch {batch_idx+1}/{len(data_loader)}")
        inp, lbl = inp.to(device), lbl.to(device)

        with autocast():
            adv_inp = one_attack.perturb(inp, lbl)
        logits_adv, _ = model.logits(adv_inp)
        preds = logits_adv.argmax(dim=1)

        total_correct += (preds == lbl).sum().item()
        total_samples += inp.size(0)

        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} evaluation time (one-stage): {batch_end_time - batch_start_time:.2f} seconds")

    robust_acc = total_correct / total_samples
    eval_end_time = time.time()
    logger.info(f"Total one-stage eval time: {eval_end_time - eval_start_time:.2f} seconds")
    return robust_acc

@torch.no_grad()
def evaluate_two_stage(model: UniBindModel, data_loader, device, iteration_count=100, epsilon=2/255):
    """
    Evaluate the model under a two-stage APGDAttack, logging per-batch timing.
    """
    logger.info(f"Running two-stage robust evaluation: iteration_count={iteration_count}, epsilon={(epsilon * 255):.2f}")
    eval_start_time = time.time()

    stage1_attack = APGDAttack(
        predict=model.logits,
        norm='Linf',
        n_restarts=1,
        n_iter=iteration_count,
        eps=epsilon,
        loss=model.logits_loss_type(),
        device=device,
        logger=logger
    )
    stage2_attack = APGDAttack(
        predict=model.logits,
        norm='Linf',
        n_restarts=1,
        n_iter=iteration_count,
        eps=epsilon,
        loss=model.logits_loss_type(),
        device=device,
        logger=logger
    )

    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()

        logger.info(f"[EVAL TWO-STAGE] Evaluating batch {batch_idx+1}/{len(data_loader)}")
        inp, lbl = inp.to(device), lbl.to(device)

        adv_fin = two_stage_attack(model, inp, lbl, stage1_attack, stage2_attack)
        logits_fin, _ = model.logits(adv_fin)
        preds = logits_fin.argmax(dim=1)

        total_correct += (preds == lbl).sum().item()
        total_samples += inp.size(0)

        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} evaluation time (two-stage): {batch_end_time - batch_start_time:.2f} seconds")

    robust_acc = total_correct / total_samples
    eval_end_time = time.time()
    logger.info(f"Total two-stage eval time: {eval_end_time - eval_start_time:.2f} seconds")
    return robust_acc

@torch.no_grad()
def evaluate_clean(model: UniBindModel, data_loader, device):
    """
    Evaluate the model on clean (non-adversarial) inputs, logging per-batch timing.
    """
    logger.info("Running CLEAN evaluation (no attack).")
    eval_start_time = time.time()
    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()

        logger.info(f"[EVAL CLEAN] Evaluating batch {batch_idx+1}/{len(data_loader)}")
        inp, lbl = inp.to(device), lbl.to(device)

        logits_clean, _ = model.logits(inp)
        preds_clean = logits_clean.argmax(dim=1)

        total_correct += (preds_clean == lbl).sum().item()
        total_samples += inp.size(0)

        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} evaluation time (clean): {batch_end_time - batch_start_time:.2f} seconds")

    acc = total_correct / total_samples
    eval_end_time = time.time()
    logger.info(f"Total clean eval time: {eval_end_time - eval_start_time:.2f} seconds")
    return acc

# ---------------------------------
# 7) Main Training/Eval Flow
# ---------------------------------
def train_and_evaluate(
    model_train: UniBindModel,
    model_original: UniBindModel,
    train_loader,
    val_loader,
    device,
    out_dir,
    epsilon=2/255,
):
    logger.info(f"Starting training with epsilon={(epsilon * 255):.2f}")
    epochs = 2
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.07 * total_steps)  # 7%

    optimizer = AdamW(model_train.parameters(), lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.95))
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0.0)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    # One-stage attacks for training (10 iter) & epoch-check (50 iter)
    train_attack = APGDAttack(
        predict=model_train.logits,
        norm='Linf',
        n_restarts=1,
        n_iter=10,
        eps=epsilon,
        loss=model_train.logits_loss_type(),
        device=device,
        logger=logger
    )
    epoch_attack = APGDAttack(
        predict=model_train.logits,
        norm='Linf',
        n_restarts=1,
        n_iter=50,
        eps=epsilon,
        loss=model_train.logits_loss_type(),
        device=device,
        logger=logger
    )

    best_acc = 0.0
    for ep in range(epochs):
        epoch_start_time = time.time()

        logger.info(f"Epoch {ep+1}/{epochs} -----------------------------------------")
        train_epoch(device, model_train, model_original, train_loader, optimizer, scheduler, train_attack)

        logger.info(f"Evaluating robust accuracy with 50-iter one-stage attack, epoch {ep+1}")
        robust_acc = evaluate_robust_one_stage(model_train, val_loader, device, epoch_attack)
        logger.info(f"[Epoch {ep+1}] robust acc (one-stage 50 iter) = {robust_acc:.4f}")

        epoch_end_time = time.time()
        logger.info(f"Epoch {ep+1} total time (training + evaluation): {epoch_end_time - epoch_start_time:.2f} seconds")

        if robust_acc > best_acc:
            best_acc = robust_acc
            logger.info(f"New best checkpoint with robust acc={best_acc:.4f}. Saving ...")
            torch.save(model_train.state_dict(), os.path.join(out_dir, "best_model.pt"))

    logger.info(f"Training complete! Best robust one-stage accuracy was {best_acc:.4f}")

# ---------------------------------
# 8) Main
# ---------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset_root", type=str, default="/home/user/datasets/ImageNet-1K")
    parser.add_argument("--train_json", type=str, default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--val_json", type=str, default="./datasets/ImageNet-1K/val_data.json")
    parser.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights.pt")
    parser.add_argument("--center_emb", type=str, default="./centre_embs/image_in_center_embeddings.pkl")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=2/255)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create a timestamped filename: "training_YYYY-MM-DD_HH-MM-SS.log"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_{timestamp}.log"
    file_handler = logging.FileHandler(os.path.join(args.output_dir, log_filename), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.handlers = [console_handler, file_handler]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1) Load center embeddings
    logger.info("Loading center embeddings ...")
    raw_emb, raw_lbls = load_centre_embeddings(args.center_emb, device)
    raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
    unique_lbls = sorted(list(set(raw_lbls)))
    lbl_to_idx = {l: i for i, l in enumerate(unique_lbls)}
    idx_to_lbl = {v: k for k, v in lbl_to_idx.items()}

    # 2) Datasets
    logger.info("Loading train dataset ...")
    train_ds = ImageNetDataset(
        dataset_root=args.dataset_root,
        data_json_path=args.train_json,
        transform=IMAGE_TRANSFORM,
        max_samples=120000,  # for quick tests
        debug=True,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl
    )
    logger.info("Loading val dataset ...")
    val_ds = ImageNetDataset(
        dataset_root=args.dataset_root,
        data_json_path=args.val_json,
        transform=IMAGE_TRANSFORM,
        max_samples=3000,
        debug=True,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3) Models
    logger.info("Initializing original + training models ...")
    model_orig = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality="image",
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger
    )
    model_train = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality="image",
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger
    )

    # (optional) If PyTorch 2.0+ you can compile the entire model:
    # if hasattr(torch, 'compile'):
    #     model_orig = torch.compile(model_orig)
    #     model_train = torch.compile(model_train)

    # 4) Train & Evaluate
    train_and_evaluate(model_train, model_orig, train_loader, val_loader, device, args.output_dir, args.epsilon)

    # 5) Final checks
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt_path):
        logger.info("Loading best checkpoint for final two-stage & clean evaluations ...")
        final_model = UniBindModel(
            device=device,
            pretrain_weights=args.pretrain_weights,
            modality="image",
            centre_embeddings=raw_emb,
            centre_labels=raw_lbls,
            label_to_index=lbl_to_idx,
            index_to_label=idx_to_lbl,
            load_unibind_pretrained=False,
            logger=logger
        )
        final_model.load_state_dict(torch.load(best_ckpt_path))

        final_robust_acc = evaluate_two_stage(final_model, val_loader, device, iteration_count=100, epsilon=args.epsilon)
        logger.info(f"Final two-stage robust accuracy = {final_robust_acc:.4f}")

        final_clean_acc = evaluate_clean(final_model, val_loader, device)
        logger.info(f"Final clean accuracy = {final_clean_acc:.4f}")
    else:
        logger.warning("No best_model.pt found. Skipping final evaluations.")

if __name__ == "__main__":
    main()
