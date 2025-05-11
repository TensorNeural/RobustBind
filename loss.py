from torch import nn
import torch.nn.functional as F

def l2_loss(emb1, emb2):
    return nn.functional.mse_loss(emb1, emb2, reduction='none').sum(dim=1).mean()

def ce_loss(logits, labels):
    return nn.functional.cross_entropy(logits, labels, reduction='mean')

def ce_loss_targeted(logits, targets):
    return -F.cross_entropy(logits, targets)

def dlr_loss(logits, labels):
    B = logits.size(0)
    sorted_logits, sorted_idx = logits.sort(dim=1, descending=True)
    label_logits = logits[torch.arange(B), labels]
    second_best = torch.where(
        sorted_idx[:, 0] == labels,
        sorted_logits[:, 1],
        sorted_logits[:, 0]
    )
    denom = sorted_logits[:, 0] - sorted_logits[:, 2] + 1e-12
    return -((label_logits - second_best) / denom).mean()


def dlr_loss_targeted(logits, labels, targets):
    B = logits.size(0)
    sorted_logits, _ = logits.sort(dim=1)
    label_logits = logits[torch.arange(B), labels]
    target_logits = logits[torch.arange(B), targets]
    denom = sorted_logits[:, -1] - 0.5 * (sorted_logits[:, -3] + sorted_logits[:, -4]) + 1e-12
    return -((label_logits - target_logits) / denom).mean()
