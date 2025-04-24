from torch import nn

def l2_loss(emb1, emb2):
    return nn.functional.mse_loss(emb1, emb2, reduction='none').sum(dim=1).mean()

def ce_loss(logits, labels):
    return nn.functional.cross_entropy(logits, labels, reduction='mean')