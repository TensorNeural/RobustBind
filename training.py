import time
from shared_types import Modality
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from attack import Attack
from perf.profiling import GpuMemoryTracker, ProfileModelMemory
from perf.debug import log_grad
from model import MODALITY_MAP, Model, ForwardMode
from meter import AverageMeter
from transform import unnormalize_inplace, normalize_inplace
from loss import l2_loss, ce_loss
import torch.distributed as dist

from utils.utils import gen_label, loss_fun, all_gather_batch_with_grad, all_gather_batch
from imagebind.imagebind_model import ModalityType

def train_alignment_epoch(
    logger,
    device,
    model,
    data_loader,
    optimizer,
    scheduler,
    epoch: int,
    total_epochs: int,
    writer: SummaryWriter,
    modality: Modality,
    align_temperature: float = 1.0,
    align_symmetric: bool = False,
    align_all_gather: bool = False,
    align_label_smoothing: float = 0.0,
    align_mask_same_label: bool = False,
):
    epoch_start_time = time.time()
    step_base = epoch * len(data_loader)
    total_samples = 0

    model.train()
    for batch_idx, batch in enumerate(data_loader):
        batch_start_time = time.time()
        step_total = step_base + batch_idx + 1

        inputs_tensor = batch['inputs']
        descriptions = batch['descriptions']
        _ = batch['labels']

        # to device
        inputs_tensor = inputs_tensor.to(device, non_blocking=True)
        descriptions = descriptions.to(device, non_blocking=True)
        labels_tensor = batch.get('labels', None)
        if labels_tensor is not None:
            labels_tensor = labels_tensor.to(device, non_blocking=True)

        merged_inputs = {
            MODALITY_MAP[modality]: inputs_tensor,
            MODALITY_MAP[Modality.TEXT]: descriptions
        }

        with GpuMemoryTracker(logger):
            text_embeddings, modality_embeddings = model(merged_inputs)

        # Optionally gather negatives across GPUs for stronger contrast
        if align_all_gather and dist.is_initialized() and dist.get_world_size() > 1:
            text_embeddings, modality_embeddings = all_gather_batch_with_grad([
                text_embeddings, modality_embeddings
            ])
            # Keep labels in sync with embeddings if masking is enabled
            if align_mask_same_label and labels_tensor is not None:
                (labels_tensor,) = all_gather_batch([labels_tensor])

        with GpuMemoryTracker(logger):
            logits = modality_embeddings @ text_embeddings.t()
            # Temperature scaling (divide by tau)
            if align_temperature is not None and align_temperature > 0:
                logits = logits / align_temperature
            labels = gen_label(logits, device)
            # Optionally mask negatives that share the same class label
            if align_mask_same_label and labels_tensor is not None:
                B = logits.size(0)
                diag = torch.eye(B, dtype=torch.bool, device=device)
                same = labels_tensor.unsqueeze(1) == labels_tensor.unsqueeze(0)
                neg_same = same & (~diag)
                logits = logits.masked_fill(neg_same, -1e9)

            if align_symmetric:
                # Text->modality direction
                logits_t2m = text_embeddings @ modality_embeddings.t()
                if align_temperature is not None and align_temperature > 0:
                    logits_t2m = logits_t2m / align_temperature
                # Use label smoothing if provided; else default loss_fun
                if align_label_smoothing and align_label_smoothing > 0.0:
                    loss_m2t = F.cross_entropy(logits, labels, reduction='mean', label_smoothing=align_label_smoothing)
                    loss_t2m = F.cross_entropy(logits_t2m, labels, reduction='mean', label_smoothing=align_label_smoothing)
                else:
                    loss_m2t = loss_fun(logits, labels)
                    loss_t2m = loss_fun(logits_t2m, labels)
                loss_val = 0.5 * (loss_m2t + loss_t2m)
            else:
                if align_label_smoothing and align_label_smoothing > 0.0:
                    loss_val = F.cross_entropy(logits, labels, reduction='mean', label_smoothing=align_label_smoothing)
                else:
                    loss_val = loss_fun(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        n_samples = inputs_tensor.size(0)
        total_samples += n_samples
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"[ALIGN] Epoch={epoch+1}/{total_epochs}, Step={batch_idx+1}/{len(data_loader)}, "
            f"LR={lr:.6f}, Loss={loss_val.item():.6f}, tau={align_temperature:.4f}, sym={align_symmetric}, gather={align_all_gather}, ls={align_label_smoothing}, mask_same={align_mask_same_label}"
        )
        if dist.get_rank() == 0 and writer is not None:
            writer.add_scalar("alignment/loss", loss_val.item(), step_total)
            writer.add_scalar("alignment/lr", lr, step_total)

        del inputs_tensor, descriptions, text_embeddings, modality_embeddings, logits, labels, loss_val
        torch.cuda.empty_cache()

        batch_end_time = time.time()
        logger.info(f"[ALIGN] Batch {batch_idx+1} time: {batch_end_time - batch_start_time:.2f} seconds")

    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    epoch_end_time = time.time()
    logger.info(f"[ALIGN] Epoch {epoch+1}/{total_epochs} time: {epoch_end_time - epoch_start_time:.2f} seconds")
    logger.info(f"[ALIGN] Total samples processed (all ranks): {sample_tensor.item()}")

def train_robust_epoch(
    logger,
    device,
    model_train: Model,     # UniBindClassifier
    model_original: Model,  # UniBindClassifier
    mean,
    std,
    data_loader,
    optimizer,
    scheduler,
    attack: Attack,
    train_loss_type,
    epoch: int,
    total_epochs: int,
    loss_meter: AverageMeter,
    cos_sim_meter: AverageMeter,
    rcos_sim_meter: AverageMeter,
    acc_meter: AverageMeter,
    racc_meter: AverageMeter,
    writer: SummaryWriter,
):
    epoch_start_time = time.time()
    step_base = epoch * len(data_loader)
    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        inputs_tensor = batch['inputs']
        lbl = batch['labels']

        batch_size = inputs_tensor.size(0)
        total_samples += batch_size
        batch_start_time = time.time()
        step_total = step_base + batch_idx + 1
        logger.info(f"[ROBUST] Epoch {epoch+1}/{total_epochs}, batch {batch_idx+1}/{len(data_loader)}, batch size={batch_size}")

        inputs_tensor = inputs_tensor.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)

        model_train.eval()

        # Unnormalized copy for attack
        inp_unorm = inputs_tensor.clone().detach()
        unnormalize_inplace(inp_unorm, mean, std)

        emb_orig = None
        if train_loss_type == 'l2':
            with torch.no_grad():
                emb_orig = model_original(inputs_tensor, mode=ForwardMode.EMBEDDINGS)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            adv_inp = attack.perturb(inp_unorm, lbl, emb_orig)
        normalize_inplace(adv_inp, mean, std)

        # ===== Initial metrics (before update) =====
        if train_loss_type == 'l2':
            with torch.no_grad():
                clean_emb = model_train(inputs_tensor, mode=ForwardMode.EMBEDDINGS)
                robust_emb = model_train(adv_inp, mode=ForwardMode.EMBEDDINGS)
                clean_cos = F.cosine_similarity(clean_emb, emb_orig, dim=1).mean()
                robust_cos = F.cosine_similarity(robust_emb, emb_orig, dim=1).mean()
            logger.info(f"[ROBUST][Init] Step={step_total}, CleanCos={clean_cos.item():.4f}, RobustCos={robust_cos.item():.4f}")

        elif train_loss_type == 'ce':
            with torch.no_grad():
                logits_clean, _ = model_train(inputs_tensor, mode=ForwardMode.LOGITS)
                logits_robust, _ = model_train(adv_inp, mode=ForwardMode.LOGITS)
                clean_acc = compute_acc(logits_clean, lbl)
                robust_acc = compute_acc(logits_robust, lbl)
            logger.info(f"[ROBUST][Init] Step={step_total}, CleanAcc={clean_acc:.2f}, RobustAcc={robust_acc:.2f}")

        # ===== Training step =====
        model_train.train()
        optimizer.zero_grad()
        adv_inp.requires_grad = True

        if train_loss_type == 'l2':
            emb_adv = model_train(adv_inp, mode=ForwardMode.EMBEDDINGS)
            loss_val = l2_loss(emb_adv, emb_orig)
        elif train_loss_type == 'ce':
            logits_adv, _ = model_train(adv_inp, mode=ForwardMode.LOGITS)
            loss_val = ce_loss(logits_adv, lbl)
        else:
            raise ValueError(f"Unknown loss type: {train_loss_type}")

        loss_val.backward()
        optimizer.step()
        scheduler.step()

        n_samples = batch_size
        loss_meter.update(loss_val.item(), n_samples)

        # ===== Final metrics (after update) =====
        model_train.eval()
        lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            if train_loss_type == 'l2':
                final_clean_emb = model_train(inputs_tensor, mode=ForwardMode.EMBEDDINGS)
                final_robust_emb = model_train(adv_inp, mode=ForwardMode.EMBEDDINGS)
                clean_cos = F.cosine_similarity(final_clean_emb, emb_orig, dim=1).mean()
                robust_cos = F.cosine_similarity(final_robust_emb, emb_orig, dim=1).mean()
                cos_sim_meter.update(clean_cos.item(), n_samples)
                rcos_sim_meter.update(robust_cos.item(), n_samples)
                logger.info(
                    f"[ROBUST][Final] Step={step_total}, LR={lr:.6f}, Loss={loss_val.item():.6f}, "
                    f"CleanCos={clean_cos.item():.4f}, RobustCos={robust_cos.item():.4f}, "
                    f"AvgCleanCos={cos_sim_meter.avg:.4f}, AvgRobustCos={rcos_sim_meter.avg:.4f}"
                )
                if dist.get_rank() == 0 and writer is not None:
                    writer.add_scalar("train/loss", loss_val.item(), step_total)
                    writer.add_scalar("train/clean_cos", clean_cos.item(), step_total)
                    writer.add_scalar("train/robust_cos", robust_cos.item(), step_total)
                    writer.add_scalar("train/lr", lr, step_total)

            elif train_loss_type == 'ce':
                final_logits_clean, _ = model_train(inputs_tensor, mode=ForwardMode.LOGITS)
                final_logits_robust, _ = model_train(adv_inp, mode=ForwardMode.LOGITS)
                clean_acc = compute_acc(final_logits_clean, lbl)
                robust_acc = compute_acc(final_logits_robust, lbl)
                acc_meter.update(clean_acc, n_samples)
                racc_meter.update(robust_acc, n_samples)
                logger.info(
                    f"[ROBUST][Final] Step={step_total}, LR={lr:.6f}, Loss={loss_val.item():.6f}, "
                    f"CleanAcc={clean_acc:.2f}, RobustAcc={robust_acc:.2f}, "
                    f"AvgCleanAcc={acc_meter.avg:.2f}, AvgRobustAcc={racc_meter.avg:.2f}"
                )
                if dist.get_rank() == 0 and writer is not None:
                    writer.add_scalar("train/loss", loss_val.item(), step_total)
                    writer.add_scalar("train/clean_acc", clean_acc, step_total)
                    writer.add_scalar("train/robust_acc", robust_acc, step_total)
                    writer.add_scalar("train/lr", lr, step_total)

        del inputs_tensor, lbl, inp_unorm, adv_inp, loss_val
        torch.cuda.empty_cache()

        batch_end_time = time.time()
        logger.info(f"[ROBUST] Batch {batch_idx+1} time: {batch_end_time - batch_start_time:.2f} seconds")

    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    epoch_end_time = time.time()
    logger.info(f"[ROBUST] Epoch {epoch+1}/{total_epochs} time: {epoch_end_time - epoch_start_time:.2f} seconds")
    logger.info(f"[ROBUST] Total samples processed: {sample_tensor.item()}")

def predict(logits):
    return logits.argmax(dim=1)

def compute_acc(logits, targets):
    correct = (predict(logits) == targets).float().sum().item()
    return 100.0 * correct / targets.size(0)