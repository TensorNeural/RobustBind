import abc
import argparse
import logging
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from enum import Enum
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import inspect
import os
import math
# from autoattack import apgd_train

# autoattack & your custom modules
from autoattack.autopgd_base import APGDAttack
from datasets.datasets import ImageNetDataset
from imagebind.imagebind_model import ModalityType
from model import UniBind
from utils.data_transform import IMAGE_TRANSFORM, IMAGE_MEAN, IMAGE_STD
from utils.utils import load_centre_embeddings

###################################
# 1) Logging Setup
###################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

class GpuMemoryTracker:
    def __init__(self, logger, label=None, device=None, message=None):
        self.logger = logger
        self.label = label
        self.message = message
        self.device = device or torch.device("cuda")

        # Capture file & line for reference
        frame = inspect.currentframe()
        outer_frame = inspect.getouterframes(frame)[1]
        self.filename = os.path.basename(outer_frame.filename)
        self.lineno = outer_frame.lineno

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_allocated = torch.cuda.memory_allocated(self.device)
        self.start_reserved = torch.cuda.memory_reserved(self.device)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_allocated = torch.cuda.memory_allocated(self.device)
        end_reserved = torch.cuda.memory_reserved(self.device)
        duration = time.time() - self.start_time
        delta_allocated = end_allocated - self.start_allocated
        delta_reserved = end_reserved - self.start_reserved

        parts = []
        parts.append("[{}:{}]".format(self.filename, self.lineno))
        if self.label:
            parts.append(self.label)
        parts.append("Duration: {:.2f}s".format(duration))
        parts.append("Allocated Δ: {}".format(self._format_bytes(delta_allocated)))
        parts.append("Reserved Δ: {}".format(self._format_bytes(delta_reserved)))

        if self.message:
            parts.append(self.message)

        self.logger.debug(" | ".join(parts))

    def _format_bytes(self, num_bytes):
        abs_bytes = abs(num_bytes)
        if abs_bytes >= 1024 ** 3:
            value = num_bytes / (1024 ** 3)
            unit = "GB"
        elif abs_bytes >= 1024 ** 2:
            value = num_bytes / (1024 ** 2)
            unit = "MB"
        elif abs_bytes >= 1024:
            value = num_bytes / 1024
            unit = "KB"
        else:
            value = num_bytes
            unit = "B"
        return "{:+.2f} {}".format(value, unit)

class ProfileModelMemory:
    def __init__(self, model, logger, label="ProfileModelMemory", device=None):
        self.model = model
        self.logger = logger
        self.label = label
        self.device = device or torch.device("cuda")
        self.hooks = []
        self.ctx_map = {}

        # Create a top-level GpuMemoryTracker (not entered yet)
        self.top_tracker = GpuMemoryTracker(
            logger=self.logger,
            label=self.label,
            device=self.device,
            message="(Top-level model forward)"
        )

    def __enter__(self):
        # Manually enter the top-level tracker
        self.top_tracker.__enter__()
        # Register hooks on all modules (including containers)
        self._register_hooks(self.model, prefix="", depth=0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

        # Exit the top-level tracker
        self.top_tracker.__exit__(exc_type, exc_val, exc_tb)

    def _register_hooks(self, module, prefix="", depth=0):
        """
        Recursively register forward_pre_hook and forward_hook on every module
        (including containers). Indentation is based on 'depth'.
        """
        # 1) Register hooks on this module
        indent = "  " * depth  # 2 spaces per depth level

        module_name = prefix if prefix else module.__class__.__name__
        # forward_pre_hook
        pre_hook_handle = module.register_forward_pre_hook(
            self._hook_pre(module_name, module, indent)
        )
        self.hooks.append(pre_hook_handle)

        # forward_hook
        post_hook_handle = module.register_forward_hook(
            self._hook_post(module_name, module, indent)
        )
        self.hooks.append(post_hook_handle)

        # 2) Recurse into children
        for child_name, child_module in module.named_children():
            full_name = f"{module_name}.{child_name}"
            self._register_hooks(child_module, full_name, depth + 1)

    def _hook_pre(self, name, module, indent):
        """Called BEFORE module.forward()."""
        def inner_pre_hook(module, inputs):
            input_shapes = [
                inp.shape for inp in inputs if hasattr(inp, 'shape')
            ]
            msg = "Input shapes: {}".format(input_shapes)
            label_str = "{}{}::Pre::{}({})".format(
                indent, self.label, name, module.__class__.__name__
            )

            tracker = GpuMemoryTracker(
                logger=self.logger,
                label=label_str,
                message=msg,
                device=self.device
            )
            tracker.__enter__()
            self.ctx_map[module] = tracker
        return inner_pre_hook

    def _hook_post(self, name, module, indent):
        """Called AFTER module.forward()."""
        def inner_post_hook(module, inputs, output):
            tracker = self.ctx_map.pop(module, None)
            if tracker is not None:
                # Update label for post-forward
                tracker.label = "{}{}::Post::{}({})".format(
                    indent, self.label, name, module.__class__.__name__
                )

                # If there's an output shape, append to the message
                if hasattr(output, 'shape'):
                    if tracker.message:
                        tracker.message = "{} | Output shape: {}".format(
                            tracker.message, list(output.shape)
                        )
                    else:
                        tracker.message = "Output shape: {}".format(list(output.shape))

                # Manually exit => logs memory usage
                tracker.__exit__(None, None, None)
        return inner_post_hook

###################################
# 2) Normalization Utilities
###################################
def unnormalize_inplace(x, mean_t, std_t):
    """
    x is currently in normalized space => (x - mean)/std
    This function brings x back into [0,1] range:
      x = x * std + mean
    """
    x.mul_(std_t).add_(mean_t).clamp_(0, 1)
    return x

def normalize_inplace(x, mean_t, std_t):
    """
    x is currently in [0,1] => we apply x = (x - mean)/std
    to produce the model's expected normalized domain.
    """
    x.sub_(mean_t).div_(std_t)
    return x

###################################
# 3) Attack Adapter (Optional if you never unnormalize)
###################################
def attack_adapter(logits_fn, mean, std):
    """
    If your loader provides [0,1] data, you'd do `predict=attack_adapter(...)`.
    But in this code, we are manually unnormalizing in train_epoch and two_stage_attack
    before calling .perturb(...). So an adapter is optional.
    """
    def predict_adapter(x):
        # x is in [0,1], so re-normalize
        x_norm = (x - mean) / std
        return logits_fn(x_norm)
    return predict_adapter

###################################
# 4) Enums & BaseModel
###################################
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

###################################
# 5) UniBind Model
###################################
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
        fine_tuned_weights=None
    ):
        super().__init__()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("Initializing UniBindModel...")

        from types import SimpleNamespace
        self.unibind = UniBind(
            SimpleNamespace(pretrain_weights=pretrain_weights, modality=modality),
            use_flash_attention=True,
            fine_tuned_weights=fine_tuned_weights,
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
        """
        x is expected to be 'normalized' if images.
        """
        embeddings = self.encode(x)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        similarity = embeddings @ self.centre_embeddings.t()
        expanded_idx = self.centre_label_indices.expand(similarity.shape[0], -1)
        class_scores, _ = torch_scatter.scatter_max(similarity, expanded_idx, dim=1)
        return class_scores, similarity

    def logits_loss_type(self):
        return LossType.CROSS_ENTROPY

    def encode(self, x):
        modality = MODALITY_MAP[self.modality]
        inp_dict = {modality: x}
        return self.unibind.encode_vision_with_mlp(inp_dict)
    
    def save_fine_tuned_weights(self, path: str):
        self.logger.info(f"[save_fine_tuned_weights] Saving fine tuned weights to '{path}'...")
        self.unibind.save_fine_tuned_weights(path)

    def load_fine_tuned_weights(self, path: str):
        self.logger.info(f"[load_fine_tuned_weights] Loading fine tuned weights from '{path}'...")
        self.unibind.load_fine_tuned_weights(path)

###################################
# 6) AverageMeter & compute_acc
###################################
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return (self.sum / self.count) if self.count else 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

def compute_acc(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().sum().item()
    return 100.0 * correct / targets.size(0)

###################################
# 7) MSE Loss for embeddings
###################################
def compute_embedding_loss(emb1, emb2):
    return nn.functional.mse_loss(emb1, emb2, reduction='none').sum(dim=1).mean()

###################################
# 8) Two‐Stage Attack (Fixed)
###################################
@torch.no_grad()
def two_stage_attack(model, inputs, labels, attack_stage1, attack_stage2, mean, std):
    """
    We do the same approach as in train_epoch:
    1) Unnormalize 'inputs' to [0,1]
    2) stage1_attack.perturb(...)
    3) Re-normalize stage1 adv
    4) Identify correct
    5) For correct ones, do stage2 on the *original* input again or the same approach
       (this code re-attacks from the clean input subset).
    6) Re-normalize stage2 adv
    7) Combine results
    """
    logger.info("Running two-stage attack on batch...")

    # 1) Unnormalize the entire batch
    inputs_unorm = inputs.clone().detach()
    unnormalize_inplace(inputs_unorm, mean, std)

    # 2) stage1 attack in [0,1]
    adv_stage1 = attack_stage1.perturb(inputs_unorm, labels)

    # 3) re-normalize stage1 adv
    normalize_inplace(adv_stage1, mean, std)

    # Evaluate which are still correct
    logits_stage1, _ = model.logits(adv_stage1)
    preds_stage1 = logits_stage1.argmax(dim=1)
    correct_mask = (preds_stage1 == labels)

    adv_final = adv_stage1.clone()
    keep_idx = torch.nonzero(correct_mask).squeeze(-1)

    if len(keep_idx) > 0:
        logger.info(f"Stage1 left {len(keep_idx)}/{inputs.size(0)} samples correct. Attacking them with Stage2...")

        # For those that remain correct, unnormalize original again
        inputs_unorm_2 = inputs[keep_idx].clone().detach()
        unnormalize_inplace(inputs_unorm_2, mean, std)

        # stage2 from [0,1]
        adv_stage2 = attack_stage2.perturb(inputs_unorm_2, labels[keep_idx])

        # re-normalize stage2 adv
        normalize_inplace(adv_stage2, mean, std)

        # place them back
        adv_final[keep_idx] = adv_stage2

    # Clean up
    del adv_stage1, inputs_unorm
    torch.cuda.empty_cache()
    logger.info("Two-stage attack finished.")
    return adv_final

###################################
# 9) Training for One Epoch
###################################
def train_epoch(
    device,
    model_train: UniBindModel,
    model_original: UniBindModel,
    mean,
    std,
    data_loader,
    optimizer,
    scheduler,
    attack: APGDAttack,
    epoch: int,
    total_epochs: int,
    loss_meter: AverageMeter,
    cos_sim_meter: AverageMeter,
    acc_meter: AverageMeter,
    racc_meter: AverageMeter,
    is_classification=False,
):
    epoch_start_time = time.time()
    step_base = epoch * len(data_loader)

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()
        step_total = step_base + batch_idx + 1
        logger.info(f"[TRAIN] Epoch {epoch+1}/{total_epochs}, batch {batch_idx+1}/{len(data_loader)}")

        with GpuMemoryTracker(logger):
            inp = inp.to(device)

        with GpuMemoryTracker(logger):
            lbl = lbl.to(device)

        model_train.eval()

        with GpuMemoryTracker(logger):
            inp_unorm = inp.clone().detach()
            unnormalize_inplace(inp_unorm, mean, std)

        with torch.no_grad():
            with GpuMemoryTracker(logger):
                emb_orig = model_original.encode(inp)

        loss_inner_wrapper = ComputeLossWrapper(
            emb_orig,
            reduction='none', loss='l2',
            logit_scale=100.
            )
        
        with GpuMemoryTracker(logger):
            # adv_inp = attack.perturb(inp_unorm, lbl)
            # adv_inp = apgd_train()
            # data_adv = apgd(
            #     model=model,
            #     loss_fn=loss_inner_wrapper,
            #     x=data,
            #     y=targets,
            #     norm=args.norm,
            #     eps=args.eps,
            #     n_iter=args.iterations_adv,
            #     verbose=True
            # )
            adv_inp = apgd_train(
                encode_fn=model_train.encode,
                logits_fn=model_train.logits,
                loss_fn=loss_inner_wrapper,
                x=inp_unorm,
                y=lbl,
                norm='Linf',
                eps=2/255,
                n_iter=10,
                initial_stepsize=1/255,
                verbose=True,
            )
            normalize_inplace(adv_inp, mean, std)

        with GpuMemoryTracker(logger):
            torch.cuda.empty_cache()

        model_train.train()

        with ProfileModelMemory(model_train, logger):
            emb_adv = model_train.encode(adv_inp)
        
        with GpuMemoryTracker(logger):
            embe_loss_val = compute_embedding_loss(emb_adv, emb_orig)

        with torch.no_grad():
            before_logits_adv, _ = model_train.logits(adv_inp)
        
        with GpuMemoryTracker(logger):
            optimizer.zero_grad()
        
        with GpuMemoryTracker(logger):
            embe_loss_val.backward()

        with GpuMemoryTracker(logger):
            optimizer.step()
        
        with GpuMemoryTracker(logger):
            scheduler.step()

        n_samples = inp.size(0)
        loss_meter.update(embe_loss_val.item(), n_samples)

        model_train.eval()
        with torch.no_grad():
            cos_sim = F.cosine_similarity(emb_adv, emb_orig, dim=1).mean()
            cos_sim_meter.update(cos_sim.item(), n_samples)

            if is_classification:
                with GpuMemoryTracker(logger):
                    logits_adv, _ = model_train.logits(adv_inp)

                with GpuMemoryTracker(logger):
                    logits_clean, _ = model_train.logits(inp)

                before_racc = compute_acc(before_logits_adv, lbl)
                racc = compute_acc(logits_adv, lbl)
                acc = compute_acc(logits_clean, lbl)
                acc_meter.update(acc, n_samples)
                racc_meter.update(racc, n_samples)
            else:
                acc = None
                racc = None

        lr_ = optimizer.param_groups[0]['lr']
        logger.info(
            f"[TRAIN] Step={step_total}, LR={lr_:.10f}, EmbeLoss={embe_loss_val.item():.6f}, CosSim={cos_sim.item():.4f}"
            + (f", Acc={acc:.2f}, BeforeAcc={before_racc:.2f}, RAcc={racc:.2f}, AvgAcc={acc_meter.avg}, AvgRAcc={racc_meter.avg}" 
               if (acc is not None and racc is not None) else "")
        )

        del inp, lbl, inp_unorm, adv_inp, emb_adv, emb_orig, embe_loss_val
        with GpuMemoryTracker(logger):
            torch.cuda.empty_cache()

        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} time: {batch_end_time - batch_start_time:.2f} seconds")

    epoch_end_time = time.time()
    logger.info(f"Epoch {epoch+1}/{total_epochs} training time: {epoch_end_time - epoch_start_time:.2f} seconds")

###################################
# 10) Evaluate Helpers (One‐Stage, Two‐Stage, Clean)
###################################
@torch.no_grad()
def evaluate_robust_one_stage(model: UniBindModel, data_loader, device, one_attack: APGDAttack, mean, std):
    eval_start_time = time.time()
    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()
        logger.info(f"[EVAL ONE-STAGE] Evaluating batch {batch_idx+1}/{len(data_loader)}")

        inp, lbl = inp.to(device), lbl.to(device)

        inp_unorm = inp.clone().detach()
        unnormalize_inplace(inp_unorm, mean, std)
        adv_inp = one_attack.perturb(inp_unorm, lbl)
        normalize_inplace(adv_inp, mean, std)

        logits_adv, _ = model.logits(adv_inp)
        preds = logits_adv.argmax(dim=1)

        total_correct += (preds == lbl).sum().item()
        total_samples += inp.size(0)

        del inp, lbl, inp_unorm, adv_inp, logits_adv, preds
        torch.cuda.empty_cache()
        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} time: {batch_end_time - batch_start_time:.2f} seconds")

    robust_acc = total_correct / total_samples
    eval_end_time = time.time()
    logger.info(f"Total one-stage eval time: {eval_end_time - eval_start_time:.2f} seconds")
    return robust_acc

@torch.no_grad()
def evaluate_two_stage(model: UniBindModel, data_loader, device, iteration_count, epsilon, mean, std):
    logger.info(f"Running two-stage robust evaluation: iteration_count={iteration_count}, eps={(epsilon * 255):.0f}/255")
    eval_start_time = time.time()

    stage1_attack = APGDAttack(
        predict=attack_adapter(model.logits, mean, std),
        norm='Linf',
        n_restarts=1,
        n_iter=iteration_count,
        eps=epsilon,
        loss=model.logits_loss_type(),
        device=device,
        logger=logger,
        verbose=True,
    )
    stage2_attack = APGDAttack(
        predict=attack_adapter(model.logits, mean, std),
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

        adv_fin = two_stage_attack(model, inp, lbl, stage1_attack, stage2_attack, mean, std)
        logits_fin, _ = model.logits(adv_fin)
        preds = logits_fin.argmax(dim=1)

        total_correct += (preds == lbl).sum().item()
        total_samples += inp.size(0)

        del inp, lbl, adv_fin, logits_fin, preds
        torch.cuda.empty_cache()
        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} time: {batch_end_time - batch_start_time:.2f} seconds")
        
    robust_acc = total_correct / total_samples
    eval_end_time = time.time()
    logger.info(f"Total two-stage eval time: {eval_end_time - eval_start_time:.2f} seconds")
    return robust_acc

@torch.no_grad()
def evaluate_clean(model: UniBindModel, data_loader, device):
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

        del inp, lbl, logits_clean, preds_clean
        torch.cuda.empty_cache()
        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} time: {batch_end_time - batch_start_time:.2f} seconds")

    acc = total_correct / total_samples
    eval_end_time = time.time()
    logger.info(f"Total clean eval time: {eval_end_time - eval_start_time:.2f} seconds")
    return acc

###################################
# 11) Main Training/Eval Flow
###################################
def train_and_evaluate(
    model_train: UniBindModel,
    model_original: UniBindModel,
    train_mean,
    train_std,
    train_loader,
    val_mean,
    val_std,
    val_loader,
    device,
    out_dir,
    epsilon,
):
    logger.info(f"Starting training for 2 epochs with epsilon={(epsilon * 255):.0f}/255")

    # 2 epochs, OneCycleLR
    epochs = 2
    steps_per_epoch = len(train_loader)

    trainable_params = [p for p in model_train.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.07 * steps_per_epoch * epochs),
        num_training_steps=steps_per_epoch * epochs,
        num_cycles=0.5,
    )

    # Attack configs (training uses 10-step APGD)
    train_attack = APGDAttack(
        predict=attack_adapter(model_train.logits, train_mean, train_std),  # we do manual unnormalize
        norm='Linf',
        n_restarts=1,
        n_iter=10,
        eps=epsilon,
        loss=model_train.logits_loss_type(),
        device=device,
        logger=logger,
        verbose=True
    )
    # Validation uses 50-step APGD
    eval_attack = APGDAttack(
        predict=attack_adapter(model_train.logits, val_mean, val_std),  # we do manual unnormalize
        norm='Linf',
        n_restarts=1,
        n_iter=50,
        eps=epsilon,
        loss=model_train.logits_loss_type(),
        device=device,
        logger=logger,
        verbose=True
    )

    loss_meter = AverageMeter()
    cos_sim_meter = AverageMeter()
    acc_meter = AverageMeter()
    racc_meter = AverageMeter()

    best_acc = 0.0
    for ep in range(epochs):
        logger.info(f"Epoch {ep+1}/{epochs} -----------------------------------------")
        epoch_start_time = time.time()

        # Train
        train_epoch(
            device=device,
            model_train=model_train,
            model_original=model_original,
            mean=train_mean,
            std=train_std,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            attack=train_attack,
            epoch=ep,
            total_epochs=epochs,
            loss_meter=loss_meter,
            cos_sim_meter=cos_sim_meter,
            acc_meter=acc_meter,
            racc_meter=racc_meter,
            is_classification=True,
        )

        # Evaluate robust accuracy (one-stage, 50 steps)
        logger.info(f"Evaluating robust accuracy with 50-iter one-stage attack, epoch {ep+1}")
        robust_acc = evaluate_robust_one_stage(
            model_train, val_loader, device, eval_attack,
            mean=val_mean, std=val_std
        )
        logger.info(f"[Epoch {ep+1}] robust acc (one-stage 50 iter) = {robust_acc:.4f}")

        epoch_end_time = time.time()
        logger.info(f"Epoch {ep+1} total time (training+eval): {epoch_end_time - epoch_start_time:.2f} seconds")

        # Best checkpoint
        if robust_acc > best_acc:
            best_acc = robust_acc
            logger.info(f"New best checkpoint: robust acc={best_acc:.4f}. Saving fine-tuned weights ...")
            model_train.save_fine_tuned_weights(os.path.join(out_dir, "best_fine_tuned_weights.pt"))

    logger.info(f"Training complete! Best robust (one-stage) accuracy was {best_acc:.4f}")


def cosine_schedule_with_warmup(
        optimizer, 
                                   num_warmup_steps: int, 
                                   num_training_steps: int, 
                                   num_cycles: float = 0.5, 
                                   last_epoch: int = -1):
    """
    Create a schedule with a learning rate that linearly warms up from 0 to the 
    initial lr set in the optimizer over `num_warmup_steps`, then decreases 
    following a cosine curve from the initial lr down to 0 over the remaining 
    `num_training_steps - num_warmup_steps` steps.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule (half-waves).
        last_epoch (int): The index of the last epoch when resuming training.

    Returns:
        LambdaLR: A PyTorch learning rate scheduler.
    """

    def lr_lambda(current_step: int):
        # 1) Warmup phase: linearly increase from 0 to 1 over num_warmup_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 2) Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def apgd_train(encode_fn, logits_fn, x, y, norm, eps, n_iter=10, use_rs=False, loss_fn=None,
               verbose=False, is_train=True, initial_stepsize=None):
    # assert not model.training
    norm = norm.replace('linf', 'Linf').replace('l2', 'L2')
    device = x.device
    ndims = len(x.shape) - 1

    if not use_rs:
        x_adv = x.clone()
    else:
        raise NotImplemented
        if norm == 'Linf':
            t = torch.rand_like(x)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)

    # set loss
    # criterion_indiv = criterion_dict[loss]


    # set params
    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.
    elif norm in ['L1']:
        k = max(int(.04 * n_iter), 1)
        init_topk = .05 if is_train else .2
        topk = init_topk * torch.ones([x.shape[0]], device=device)
        sp_old = n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.
        alpha = 1.

    if initial_stepsize:
        alpha = initial_stepsize / eps

    step_size = alpha * eps * torch.ones(
        [x.shape[0], *[1] * ndims],
        device=device
        )
    counter3 = 0

    x_adv.requires_grad_()
    # grad = torch.zeros_like(x)
    # for _ in range(self.eot_iter)
    # with torch.enable_grad()
    output = encode_fn(x_adv)
    loss_indiv = loss_fn(output, y)
    loss = loss_indiv.sum()
    # grad += torch.autograd.grad(loss, [x_adv])[0].detach()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    # grad /= float(self.eot_iter)
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()

    logits, _ = logits_fn(x_adv)
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0

    u = torch.arange(x.shape[0], device=device)
    x_adv_old = x_adv.clone().detach()

    for i in range(n_iter):
        ### gradient step
        if True:  # with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            loss_curr = loss.detach().mean()

            a = 0.75 if i > 0 else 1.0

            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv_1,
                            x - eps
                            ), x + eps
                        ), 0.0, 1.0
                    )
                x_adv_1 = torch.clamp(
                    torch.min(
                        torch.max(
                            x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                            x - eps
                        ), x + eps
                    ), 0.0, 1.0
                )

            # elif norm == 'L2':
            #     x_adv_1 = x_adv + step_size * grad / (L2_norm(
            #         grad,
            #         keepdim=True
            #         ) + 1e-12)
            #     x_adv_1 = torch.clamp(
            #         x + (x_adv_1 - x) / (L2_norm(
            #             x_adv_1 - x,
            #             keepdim=True
            #             ) + 1e-12) * torch.min(
            #             eps * torch.ones_like(x),
            #             L2_norm(x_adv_1 - x, keepdim=True)
            #             ), 0.0, 1.0
            #         )
            #     x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
            #     x_adv_1 = torch.clamp(
            #         x + (x_adv_1 - x) / (L2_norm(
            #             x_adv_1 - x,
            #             keepdim=True
            #             ) + 1e-12) * torch.min(
            #             eps * torch.ones_like(x),
            #             L2_norm(x_adv_1 - x, keepdim=True)
            #             ), 0.0, 1.0
            #         )

            # elif norm == 'L1':
            #     grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
            #     topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
            #     grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
            #     sparsegrad = grad * (grad.abs() >= grad_topk).float()
            #     x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
            #             sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
            #                 -1, 1, 1, 1
            #             ) + 1e-10)

            #     delta_u = x_adv_1 - x
            #     delta_p = L1_projection(x, delta_u, eps)
            #     x_adv_1 = x + delta_u + delta_p

            # elif norm == 'L0':
            #     L1normgrad = grad / (grad.abs().view(grad.shape[0], -1).sum(
            #         dim=-1, keepdim=True
            #     ) + 1e-12).view(
            #         grad.shape[0], *[1] * (
            #                 len(grad.shape) - 1)
            #         )
            #     x_adv_1 = x_adv + step_size * L1normgrad * n_fts
            #     x_adv_1 = L0_projection(x_adv_1, x, eps)
            #     # TODO: add momentum

            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        # grad = torch.zeros_like(x)
        # for _ in range(self.eot_iter)
        # with torch.enable_grad()
        output = encode_fn(x_adv)
        loss_indiv = loss_fn(output, y)
        loss = loss_indiv.sum()

        # grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        if i < n_iter - 1:
            # save one backward pass
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        # grad /= float(self.eot_iter)
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()

        logits, _ = logits_fn(x_adv)
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0).nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        if verbose:
            str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                step_size.mean(), topk.mean() * n_fts
            ) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                step_size.mean()
            )
            print(
                'iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats
                )
            )
            # print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))

        ### check step size
        if True:  # with torch.no_grad()
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1

            if counter3 == k:
                if norm in ['Linf', 'L2']:
                    fl_oscillation = check_oscillation(
                        loss_steps, i, k,
                        loss_best, k3=thr_decr
                        )
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(
                        fl_oscillation,
                        fl_reduce_no_impr
                        )
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    counter3 = 0
                    k = max(k - size_decr, n_iter_min)

                # elif norm == 'L1':
                #     # adjust sparsity
                #     sp_curr = L0_norm(x_best - x)
                #     fl_redtopk = (sp_curr / sp_old) < .95
                #     topk = sp_curr / n_fts / 1.5
                #     step_size[fl_redtopk] = alpha * eps
                #     step_size[~fl_redtopk] /= adasp_redstep
                #     step_size.clamp_(alpha * eps / adasp_minstep, alpha * eps)
                #     sp_old = sp_curr.clone()

                #     x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                #     grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                #     counter3 = 0

    #return x_best, acc, loss_best, x_best_adv
    return x_best_adv

def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()

class ComputeLossWrapper:
    def __init__(self, embedding_orig, reduction='mean', loss=None,
                 logit_scale=100.):
        self.embedding_orig = embedding_orig
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale

    def __call__(self, embedding, targets):
        return compute_loss(
            loss_str=self.loss_str, embedding=embedding, targets=targets,
            embedding_orig=self.embedding_orig, logit_scale=self.logit_scale,
            reduction=self.reduction
            )

def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
                 embedding_text_labels_norm=None, reduction='mean'):
    if loss_str == 'l2':
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        loss = ce(
            out=embedding @ (logit_scale * embedding_text_labels_norm),
            targets=targets,
            reduction=reduction
        )
    else:
        raise ValueError(f'loss {loss_str} not supported')
    return loss

def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch

def ce(out, targets, reduction='mean'):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1

    return F.cross_entropy(out, targets, reduction=reduction)

###################################
# 12) Main
###################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset_root", type=str, default="/home/user/datasets/ImageNet-1K")
    parser.add_argument("--train_json", type=str, default="./datasets/ImageNet-1K/train_data.json")
    parser.add_argument("--val_json", type=str, default="./datasets/ImageNet-1K/val_data.json")
    parser.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten.pt")
    parser.add_argument("--center_emb", type=str, default="./centre_embs/image_in_center_embeddings.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=2/255)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging to file + console
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
        # max_samples=2000,  # example
        debug=False,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl
    )
    logger.info("Loading val dataset ...")
    val_ds = ImageNetDataset(
        dataset_root=args.dataset_root,
        data_json_path=args.val_json,
        transform=IMAGE_TRANSFORM,
        max_samples=5000,
        debug=False,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl
    )

    # NOTE: Typically shuffle=True for training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
        logger=logger,
        fine_tuned_weights=None
    )
    model_train = UniBindModel(
        device=device,
        pretrain_weights=args.pretrain_weights,
        modality="image",
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger,
        fine_tuned_weights=None
    )

    first_param = next(model_train.parameters(), None)
    if first_param is not None:
        logger.info(f"model_train first param dtype: {first_param.dtype}")
    else:
        logger.info("No parameters in model_train!")

    # 4) Train & Evaluate
    mean_t = torch.tensor(IMAGE_MEAN, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(IMAGE_STD, device=device).view(1, -1, 1, 1)

    train_and_evaluate(
        model_train=model_train,
        model_original=model_orig,
        train_mean=mean_t,
        train_std=std_t,
        train_loader=train_loader,
        val_mean=mean_t,
        val_std=std_t,
        val_loader=val_loader,
        device=device,
        out_dir=args.output_dir,
        epsilon=args.epsilon
    )

    # 5) Final two-stage + clean checks (optional)
    best_fine_tuned_ckpt_path = os.path.join(args.output_dir, "best_fine_tuned_weights.pt")
    if os.path.exists(best_fine_tuned_ckpt_path):
        logger.info("Loading best fine tuned weights for final two-stage & clean evaluations ...")
        final_model = UniBindModel(
            device=device,
            pretrain_weights=args.pretrain_weights,
            modality="image",
            centre_embeddings=raw_emb,
            centre_labels=raw_lbls,
            label_to_index=lbl_to_idx,
            index_to_label=idx_to_lbl,
            logger=logger,
            fine_tuned_weights=best_fine_tuned_ckpt_path
        )

        final_robust_acc = evaluate_two_stage(
            final_model, val_loader, device,
            iteration_count=100, 
            epsilon=args.epsilon,
            mean=mean_t, 
            std=std_t
        )
        logger.info(f"Final two-stage robust accuracy = {final_robust_acc:.4f}")

        final_clean_acc = evaluate_clean(final_model, val_loader, device)
        logger.info(f"Final clean accuracy = {final_clean_acc:.4f}")
    else:
        logger.warning("No best_fine_tuned_weights.pt found. Skipping final evaluations.")

if __name__ == "__main__":
    main()
