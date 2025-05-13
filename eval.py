import time
import torch
import torch.distributed as dist
from attack import AttackModel, APGDAttack, two_stage_attack
from model import UniBindModel, ForwardMode
from transform import unnormalize_inplace, normalize_inplace

def evaluate_robust_one_stage(logger, device, model: UniBindModel, data_loader, one_attack: APGDAttack, mean, std):
    eval_start_time = time.time()
    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()
        logger.info(f"[EVAL ONE-STAGE] Evaluating batch {batch_idx+1}/{len(data_loader)}, batch size={inp.size(0)}")

        inp, lbl = inp.to(device), lbl.to(device)
        inp_unorm = inp.clone().detach()
        unnormalize_inplace(inp_unorm, mean, std)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            adv_inp = one_attack.perturb(inp_unorm, lbl)
        normalize_inplace(adv_inp, mean, std)

        logits_adv, _ = model(adv_inp, mode=ForwardMode.LOGITS)
        preds = logits_adv.argmax(dim=1)

        total_correct += (preds == lbl).sum().item()
        total_samples += inp.size(0)

        del inp, lbl, inp_unorm, adv_inp, logits_adv, preds
        torch.cuda.empty_cache()
        
        logger.info(f"Batch {batch_idx+1} time: {time.time() - batch_start_time:.2f} seconds")
        logger.info(f"Samples processed: {total_samples}")

    correct_tensor = torch.tensor(total_correct, dtype=torch.float64, device=device)
    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)

    if sample_tensor.item() == 0:
        logger.warning("No samples processed during evaluation.")
        return 0.0

    robust_acc = (correct_tensor / sample_tensor).item()
    logger.info(f"Total robust samples: {sample_tensor.item()}")
    logger.info(f"Total robust correct: {correct_tensor.item()}")
    logger.info(f"Total one-stage eval time: {time.time() - eval_start_time:.2f} seconds")
    return robust_acc

def evaluate_two_stage(logger, device, model: UniBindModel, data_loader, attack_loss_type, iteration_count, epsilon, mean, std):
    logger.info(f"Running two-stage robust evaluation: iteration_count={iteration_count}, eps={(epsilon * 255):.0f}/255")

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

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()
        
        logger.info(f"[EVAL TWO-STAGE] Evaluating batch {batch_idx+1}/{len(data_loader)}, batch size={inp.size(0)}")

        inp, lbl = inp.to(device), lbl.to(device)
        adv_fin = two_stage_attack(logger, model, inp, lbl, stage1_attack, stage2_attack, mean, std)
        wrapped_adv_fin = model.wrap_tensor(adv_fin)
        logits_fin, _ = model(wrapped_adv_fin, mode=ForwardMode.LOGITS)
        preds = logits_fin.argmax(dim=1)

        total_correct += (preds == lbl).sum().item()
        total_samples += inp.size(0)

        del inp, lbl, adv_fin, wrapped_adv_fin, logits_fin, preds
        torch.cuda.empty_cache()
        
        logger.info(f"Batch {batch_idx+1} time: {time.time() - batch_start_time:.2f} seconds")
        logger.info(f"Samples processed: {total_samples}")
    
    correct_tensor = torch.tensor(total_correct, dtype=torch.float64, device=device)
    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)

    if sample_tensor.item() == 0:
        logger.warning("No samples processed during evaluation.")
        return 0.0
    
    robust_acc = correct_tensor.item() / sample_tensor.item()
    logger.info(f"Total robust samples: {sample_tensor.item()}")
    logger.info(f"Total robust correct: {correct_tensor.item()}")
    logger.info(f"Total two-stage eval time: {time.time() - eval_start_time:.2f} seconds")
    return robust_acc

@torch.no_grad()
def evaluate_clean(logger, device, model: UniBindModel, data_loader):
    logger.info("Running CLEAN evaluation (no attack).")

    eval_start_time = time.time()
    model.eval()
    total_correct = 0
    total_samples = 0

    for batch_idx, (inp, lbl) in enumerate(data_loader):
        batch_start_time = time.time()
        logger.info(f"[EVAL CLEAN] Evaluating batch {batch_idx+1}/{len(data_loader)}, batch size={inp.size(0)}")

        inp, lbl = inp.to(device), lbl.to(device)
        wrapped_inp = model.wrap_tensor(inp)
        logits_clean, _ = model(wrapped_inp, mode=ForwardMode.LOGITS)
        preds_clean = logits_clean.argmax(dim=1)

        total_correct += (preds_clean == lbl).sum().item()
        total_samples += inp.size(0)

        del inp, wrapped_inp, lbl, logits_clean, preds_clean
        torch.cuda.empty_cache()

        logger.info(f"Batch {batch_idx+1} time: {time.time() - batch_start_time:.2f} seconds")
        logger.info(f"Samples processed: {total_samples}")
    
    correct_tensor = torch.tensor(total_correct, dtype=torch.float64, device=device)
    sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    total_correct = correct_tensor.item()
    total_samples = sample_tensor.item()

    if total_samples == 0:
        logger.warning("No samples processed during evaluation.")
        return 0.0
    
    acc = total_correct / total_samples
    logger.info(f"Total clean samples: {sample_tensor}")
    logger.info(f"Total clean correct: {total_correct}")
    logger.info(f"Total clean eval time: {time.time() - eval_start_time:.2f} seconds")
    return acc
