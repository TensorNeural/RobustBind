import time
import torch
import torch.distributed as dist
from attack import AttackModel, APGDAttack, two_stage_attack
from model import UniBindClassifier, ForwardMode
from transform import unnormalize_inplace, normalize_inplace


def evaluate_robust_one_stage(logger, device, model: UniBindClassifier, data_loader, one_attack: APGDAttack, mean, std):
    eval_start_time = time.time()
    model.eval()
    total_correct = 0
    total_samples = 0

    eps = getattr(one_attack, 'eps', None)
    logger.info(f"[EVAL ONE-STAGE] Start | batches={len(data_loader)} | eps={'' if eps is None else f'{eps*255:.0f}/255'}")

    for batch_idx, batch in enumerate(data_loader):
        batch_start_time = time.time()
        inputs = batch['inputs'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # Unnormalize inputs, run attack, then re-normalize for model
        inputs_unorm = inputs.detach().clone()
        unnormalize_inplace(inputs_unorm, mean, std)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            adv_inputs_unorm = one_attack.perturb(inputs_unorm, labels)
        normalize_inplace(adv_inputs_unorm, mean, std)

        # Inference on adversarial normalized inputs
        wrapped = model.wrap_tensor(adv_inputs_unorm)
        logits_adv, _ = model(wrapped, mode=ForwardMode.LOGITS)
        preds = logits_adv.argmax(dim=1)

        correct_batch = (preds == labels).sum().item()
        total_correct += correct_batch
        total_samples += labels.size(0)

        batch_acc = correct_batch / labels.size(0)
        running_acc = (total_correct / total_samples) if total_samples > 0 else 0.0

        logger.info(f"[EVAL ONE-STAGE][{batch_idx+1}/{len(data_loader)}] size={labels.size(0)} | time={time.time() - batch_start_time:.2f}s | batch_acc={batch_acc:.4f} | running_acc={running_acc:.4f} | total_samples={total_samples}")

        # Cleanup
        del inputs, labels, inputs_unorm, adv_inputs_unorm, wrapped, logits_adv, preds
        torch.cuda.empty_cache()


    correct_tensor = torch.tensor(total_correct, dtype=torch.float64, device=device)
    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)

    if sample_tensor.item() == 0:
        logger.warning("[EVAL ONE-STAGE] No samples processed.")
        return 0.0

    acc = (correct_tensor / sample_tensor).item()
    logger.info(f"[EVAL ONE-STAGE] Total samples: {int(sample_tensor.item())}")
    logger.info(f"[EVAL ONE-STAGE] Total correct: {int(correct_tensor.item())}")
    logger.info(f"[EVAL ONE-STAGE] Total time: {time.time() - eval_start_time:.2f}s | acc={acc:.4f}")
    return acc


@torch.no_grad()
def evaluate_alignment_acc(logger, device, model_val, val_loader):
    """Evaluate alignment accuracy with DDP support and detailed logging.

    Returns accuracy in percent (same convention as previous implementation).
    """
    logger.info(f"[EVAL ALIGN] Start | batches={len(val_loader)}")
    eval_start_time = time.time()
    model_val.eval()
    total_correct_local = 0
    total_samples_local = 0

    for batch_idx, batch in enumerate(val_loader):
        batch_start = time.time()
        inputs = batch['inputs'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        wrapped = inputs  # UniBindClassifier expects tensors directly
        logits, _ = model_val(wrapped, mode=ForwardMode.LOGITS)
        preds = logits.argmax(dim=1)
        correct_batch = (preds == labels).sum().item()
        total_correct_local += correct_batch
        total_samples_local += labels.size(0)

        batch_acc = correct_batch / labels.size(0)
        running_acc = (total_correct_local / total_samples_local) if total_samples_local > 0 else 0.0

        logger.info(f"[EVAL ALIGN][{batch_idx+1}/{len(val_loader)}] size={int(batch_acc * 0 + 1) * 0 + int(batch_acc * 0 + 0) + labels.size(0)} | time={time.time() - batch_start:.2f}s | batch_acc={batch_acc:.4f} | running_acc={running_acc:.4f} | total_samples={total_samples_local}")

        # Cleanup per batch
        del inputs, labels, logits, preds
        torch.cuda.empty_cache()

    # Distributed aggregation
    correct_tensor = torch.tensor(total_correct_local, dtype=torch.float64, device=device)
    samples_tensor = torch.tensor(total_samples_local, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

    total_correct = correct_tensor.item()
    total_samples = samples_tensor.item()

    if total_samples == 0:
        logger.warning("[EVAL ALIGN] No samples processed.")
        return 0.0

    acc_fraction = total_correct / total_samples
    acc_percent = 100.0 * acc_fraction
    logger.info(f"[EVAL ALIGN] Total samples: {int(total_samples)}")
    logger.info(f"[EVAL ALIGN] Total correct: {int(total_correct)}")
    logger.info(f"[EVAL ALIGN] Total time: {time.time() - eval_start_time:.2f}s | acc={acc_fraction:.4f}")
    return acc_percent


def evaluate_two_stage(logger, device, model: UniBindClassifier, data_loader, attack_loss_type, iteration_count, epsilon, mean, std):
    logger.info(f"[EVAL TWO-STAGE] Start | batches={len(data_loader)} | iters={iteration_count} | eps={epsilon*255:.0f}/255")

    eval_start_time = time.time()

    attack_model = AttackModel(model, mean, std)
    stage1_attack = APGDAttack(
        model=attack_model,
        norm='linf',
        n_restarts=1,
        n_iter=iteration_count,
        eps=epsilon,
        loss_type=attack_loss_type,
        device=device,
        logger=logger
    )
    stage2_attack = APGDAttack(
        model=attack_model,
        norm='linf',
        n_restarts=1,
        n_iter=iteration_count,
        eps=epsilon,
        loss_type=attack_loss_type,
        device=device,
        logger=logger
    )

    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        batch_start_time = time.time()
        inputs = batch['inputs'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        adv_fin = two_stage_attack(logger, model, inputs, labels, stage1_attack, stage2_attack, mean, std)

        wrapped_adv_fin = model.wrap_tensor(adv_fin)
        logits_fin, _ = model(wrapped_adv_fin, mode=ForwardMode.LOGITS)
        preds = logits_fin.argmax(dim=1)
        correct_batch = (preds == labels).sum().item()
        total_correct += correct_batch
        total_samples += labels.size(0)

        batch_acc = correct_batch / labels.size(0)
        running_acc = (total_correct / total_samples) if total_samples > 0 else 0.0

        logger.info(f"[EVAL TWO-STAGE][{batch_idx+1}/{len(data_loader)}] size={labels.size(0)} | time={time.time() - batch_start_time:.2f}s | batch_acc={batch_acc:.4f} | running_acc={running_acc:.4f} | total_samples={total_samples}")

        del inputs, labels, adv_fin, wrapped_adv_fin, logits_fin, preds
        torch.cuda.empty_cache()
        
    
    correct_tensor = torch.tensor(total_correct, dtype=torch.float64, device=device)
    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)

    if sample_tensor.item() == 0:
        logger.warning("[EVAL TWO-STAGE] No samples processed.")
        return 0.0
    
    acc = correct_tensor.item() / sample_tensor.item()
    logger.info(f"[EVAL TWO-STAGE] Total samples: {int(sample_tensor.item())}")
    logger.info(f"[EVAL TWO-STAGE] Total correct: {int(correct_tensor.item())}")
    logger.info(f"[EVAL TWO-STAGE] Total time: {time.time() - eval_start_time:.2f}s | acc={acc:.4f}")
    return acc


@torch.no_grad()
def evaluate_clean(logger, device, model: UniBindClassifier, data_loader):
    logger.info(f"[EVAL CLEAN] Start | batches={len(data_loader)}")

    eval_start_time = time.time()
    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(data_loader):
        batch_start_time = time.time()
        inputs = batch['inputs'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        wrapped_inp = model.wrap_tensor(inputs)
        logits_clean, _ = model(wrapped_inp, mode=ForwardMode.LOGITS)
        preds_clean = logits_clean.argmax(dim=1)
        
        correct_batch = (preds_clean == labels).sum().item()
        total_correct += correct_batch
        total_samples += labels.size(0)

        batch_acc = correct_batch / labels.size(0)
        running_acc = (total_correct / total_samples) if total_samples > 0 else 0.0

        logger.info(f"[EVAL CLEAN][{batch_idx+1}/{len(data_loader)}] size={labels.size(0)} | time={time.time() - batch_start_time:.2f}s | batch_acc={batch_acc:.4f} | running_acc={running_acc:.4f} | total_samples={total_samples}")

        del inputs, wrapped_inp, labels, logits_clean, preds_clean
        torch.cuda.empty_cache()

    
    correct_tensor = torch.tensor(total_correct, dtype=torch.float64, device=device)
    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    total_correct = correct_tensor.item()
    total_samples = sample_tensor.item()

    if total_samples == 0:
        logger.warning("[EVAL CLEAN] No samples processed.")
        return 0.0
    
    acc = total_correct / total_samples
    logger.info(f"[EVAL CLEAN] Total samples: {int(sample_tensor.item())}")
    logger.info(f"[EVAL CLEAN] Total correct: {int(total_correct)}")
    logger.info(f"[EVAL CLEAN] Total time: {time.time() - eval_start_time:.2f}s | acc={acc:.4f}")
    return acc
