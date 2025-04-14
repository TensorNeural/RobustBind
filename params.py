from model import UniBindModel
from autoattack.autopgd_base import APGDAttack
from attack import attack_adapter
from loss import l2_loss, ce_loss
import torch
from torch.optim import AdamW
from transform import unnormalize_inplace, normalize_inplace
import matplotlib.pyplot as plt

def find_lr(
    logger,
    device,
    raw_emb,
    raw_lbls,
    lbl_to_idx,
    idx_to_lbl,
    pretrain_weights,
    use_flash_attention,
    train_mean,
    train_std,
    train_loader,
    attack_loss_type,
    train_loss_type,
    epsilon,
    steps=100,
):
    logger.info("Finding learning rate ...")
    logger.info("Initializing original + training models ...")
    model_original = UniBindModel(
        device=device,
        pretrain_weights=pretrain_weights,
        modality="image",
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger,
        use_flash_attention=use_flash_attention,
        fine_tuned_weights=None
    )
    model_train = UniBindModel(
        device=device,
        pretrain_weights=pretrain_weights,
        modality="image",
        centre_embeddings=raw_emb,
        centre_labels=raw_lbls,
        label_to_index=lbl_to_idx,
        index_to_label=idx_to_lbl,
        logger=logger,
        use_flash_attention=use_flash_attention,
        fine_tuned_weights=None
    )

    trainable_params = [p for p in model_train.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.95))
    attack = APGDAttack(
        predict=attack_adapter(model_train.logits, train_mean, train_std),
        norm='Linf',
        n_restarts=1,
        n_iter=10,
        eps=epsilon,
        loss=attack_loss_type,
        device=device,
        logger=logger,
        verbose=True
    )
    model_train.train()
    model_original.eval()
    logger.info("Running LR finder ...")
    
    num_batches = min(len(train_loader), steps)
    init_value = 1e-7
    final_value = 1.0
    lr_multiplier = (final_value / init_value) ** (1.0 / max(1, num_batches - 1))
    lr = init_value
    logger.info(f"Initial learning rate: {lr:.6f}, final learning rate: {final_value:.6f}, multiplier: {lr_multiplier:.6f}")

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    beta = 0.98
    avg_loss = 0.0
    best_loss = float('inf')
    batch_num = 0

    losses = []
    lrs = []

    for batch_idx, (inp, lbl) in enumerate(train_loader):
        if batch_idx >= steps:
            break

        logger.info(f"Batch {batch_idx + 1}/{steps} ...")
        logger.info(f"Learning rate: {lr:.6f}")
        batch_num += 1
        inp, lbl = inp.to(device), lbl.to(device)

        model_train.eval()
        inp_unorm = inp.clone().detach()
        unnormalize_inplace(inp_unorm, train_mean, train_std)

        adv_inp = attack.perturb(inp_unorm, lbl)
        normalize_inplace(adv_inp, train_mean, train_std)

        model_train.train()
        optimizer.zero_grad()

        if train_loss_type == 'l2':
            with torch.no_grad():
                emb_orig = model_original.encode(inp)

            emb_adv = model_train.encode(adv_inp)
            loss = l2_loss(emb_orig, emb_adv)
            del emb_orig, emb_adv
        elif train_loss_type == 'ce':
            logits_adv, _ = model_train.logits(adv_inp)
            loss = ce_loss(logits_adv, lbl)
            del logits_adv
        else: 
            raise ValueError(f"Unknown loss type: {train_loss_type}")

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        logger.info(f"Loss: {loss.item()}, Smoothed loss: {smoothed_loss:.6f}, best loss: {best_loss:.6f}")

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        lrs.append(lr)

        loss.backward()
        optimizer.step()

        lr *= lr_multiplier
        logger.info(f"Updated learning rate: {lr:.6f}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if smoothed_loss > 4 * best_loss:
            logger.info("Stopping early due to loss explosion.")
            break
        
        del inp, lbl, inp_unorm, adv_inp, loss
        torch.cuda.empty_cache()
    
    logger.info("Finished running LR finder.")
    return lrs, losses