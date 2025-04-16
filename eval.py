import time
import torch
from attack import AttackModel
from autoattack.autopgd_base import APGDAttack
from attack import two_stage_attack
from model import UniBindModel
from transform import unnormalize_inplace, normalize_inplace

@torch.no_grad()
def evaluate_robust_one_stage(logger, device, model: UniBindModel, data_loader, one_attack: APGDAttack, mean, std):
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
def evaluate_two_stage(logger, device, model: UniBindModel, data_loader, attack_loss_type, iteration_count, epsilon, mean, std):
    logger.info(f"Running two-stage robust evaluation: iteration_count={iteration_count}, eps={(epsilon * 255):.0f}/255")
    eval_start_time = time.time()

    attack_model = AttackModel(model, mean, std)
    stage1_attack = APGDAttack(
        predict=attack_model.logits,
        norm='Linf',
        n_restarts=1,
        n_iter=iteration_count,
        eps=epsilon,
        loss=attack_loss_type,
        device=device,
        logger=logger,
        verbose=True,
    )
    stage2_attack = APGDAttack(
        predict=attack_model.logits,
        norm='Linf',
        n_restarts=1,
        n_iter=iteration_count,
        eps=epsilon,
        loss=attack_loss_type,
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

        adv_fin = two_stage_attack(logger, model, inp, lbl, stage1_attack, stage2_attack, mean, std)
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
def evaluate_clean(logger, device, model: UniBindModel, data_loader):
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