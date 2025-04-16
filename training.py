import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from attack import Attack
from perf.profiling import GpuMemoryTracker, ProfileModelMemory
from model import Model
from meter import AverageMeter
from transform import unnormalize_inplace, normalize_inplace
from loss import l2_loss, ce_loss

def train_epoch(
    logger,
    device,
    model_train: Model,
    model_original: Model,
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
    acc_meter: AverageMeter,
    racc_meter: AverageMeter,
    writer: SummaryWriter,
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

        with GpuMemoryTracker(logger):
            adv_inp = attack.perturb(inp_unorm, lbl)
            normalize_inplace(adv_inp, mean, std)

        with GpuMemoryTracker(logger):
            torch.cuda.empty_cache()

        model_train.train()
        with GpuMemoryTracker(logger):
            optimizer.zero_grad()
        
        if train_loss_type == 'l2':
            with GpuMemoryTracker(logger):
                emb_adv = model_train.encode(adv_inp)

            with torch.no_grad():
                with GpuMemoryTracker(logger):
                    emb_orig = model_original.encode(inp)

            with GpuMemoryTracker(logger):
                loss_val = l2_loss(emb_adv, emb_orig)
            
            with torch.no_grad():
                with GpuMemoryTracker(logger):
                    cos_sim = F.cosine_similarity(emb_adv, emb_orig, dim=1).mean()

                logger.info(f"[TRAIN] (Initial) Step={step_total}, CosSim={cos_sim.item():.4f}")
        elif train_loss_type == 'ce':
            with ProfileModelMemory(model_train, logger):
                logits_adv, _ = model_train.logits(adv_inp)
                
            with GpuMemoryTracker(logger):
                loss_val = ce_loss(logits_adv, lbl)

            with torch.no_grad():
                with GpuMemoryTracker(logger):
                    logits_clean, _ = model_train.logits(inp)
    
                acc = compute_acc(logits_clean, lbl)
                racc = compute_acc(logits_adv, lbl)
                logger.info(f"[TRAIN] (Initial) Step={step_total}, Clean Acc={acc:.2f}, RobustAcc={racc:.2f}")
        else:
            raise ValueError(f"Unknown loss type: {train_loss_type}")
        
        with GpuMemoryTracker(logger):
            loss_val.backward()

        with GpuMemoryTracker(logger):
            optimizer.step()
        
        for name, param in model_train.named_parameters():
            if param.requires_grad and param.grad is not None:
                delta = param.grad.detach().norm().item()
                logger.info(f"[GradNorm] {name}: {delta:.6f}")
        
        with GpuMemoryTracker(logger):
            scheduler.step()

        n_samples = inp.size(0)
        loss_meter.update(loss_val.item(), n_samples)

        model_train.eval()
        lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            if train_loss_type == 'l2':
                with GpuMemoryTracker(logger):
                    final_emb_adv = model_train.encode(adv_inp)

                cos_sim = F.cosine_similarity(final_emb_adv, emb_orig, dim=1).mean()
                cos_sim_meter.update(cos_sim.item(), n_samples)
                logger.info(f"[TRAIN] Step={step_total}, LR={lr:.6f}, Loss={loss_val.item():.6f}, CosSim={cos_sim.item():.4f}")

                writer.add_scalar("train/loss", loss_val.item(), step_total)
                writer.add_scalar("train/cos_sim", cos_sim.item(), step_total)
                writer.add_scalar("train/lr", lr, step_total)

                del emb_adv, emb_orig, cos_sim
            elif train_loss_type == 'ce':
                with GpuMemoryTracker(logger):
                    final_logits_adv, _ = model_train.logits(adv_inp)
                    final_logits_clean, _ = model_train.logits(inp)
                
                final_racc = compute_acc(final_logits_adv, lbl)
                final_acc = compute_acc(final_logits_clean, lbl)
                acc_meter.update(final_acc, n_samples)
                racc_meter.update(final_racc, n_samples)
                logger.info(
                    f"[TRAIN] (Final) Step={step_total}, LR={lr:.6f}, Loss={loss_val.item():.6f}, "
                    f"Acc={final_acc:.2f}, RobustAcc={final_racc:.2f}, AvgAcc={acc_meter.avg:.2f}, AvgRobustAcc={racc_meter.avg:.2f}"
                )

                writer.add_scalar("train/loss", loss_val.item(), step_total)
                writer.add_scalar("train/clean_acc", final_acc, step_total)
                writer.add_scalar("train/robust_acc", final_racc, step_total)
                writer.add_scalar("train/lr", lr, step_total)
                
                del logits_clean, logits_adv

        del inp, lbl, inp_unorm, adv_inp, loss_val
        with GpuMemoryTracker(logger):
            torch.cuda.empty_cache()

        batch_end_time = time.time()
        logger.info(f"Batch {batch_idx+1} time: {batch_end_time - batch_start_time:.2f} seconds")

    epoch_end_time = time.time()
    logger.info(f"Epoch {epoch+1}/{total_epochs} training time: {epoch_end_time - epoch_start_time:.2f} seconds")

def predict(logits):
    return logits.argmax(dim=1)

def compute_acc(logits, targets):
    correct = (predict(logits) == targets).float().sum().item()
    return 100.0 * correct / targets.size(0)