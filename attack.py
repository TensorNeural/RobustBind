import torch
from transform import unnormalize_inplace, normalize_inplace

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

@torch.no_grad()
def two_stage_attack(logger, model, inputs, labels, attack_stage1, attack_stage2, mean, std):
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

# import torch.nn as nn
# import torch.nn.functional as F
# from abc import ABC, abstractmethod

# class AttackModel(ABC):
#     @abstractmethod
#     def logits(self, x: torch.Tensor):
#         pass

#     @abstractmethod
#     def encode(self, x: torch.Tensor):
#         pass


# class Attack(ABC):
#     @abstractmethod
#     def perturb(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
#         pass

# class APGDAttack(Attack):
#     def __init__(
#         self,
#         model: AttackModel,
#         n_iter=10,
#         norm='Linf',
#         eps=2/255,
#         loss='ce',
#         step_size=None,
#         device=None,
#         logger=None,
#         verbose=False
#     ):
#         self.model = model
#         self.n_iter = n_iter
#         self.norm = norm
#         self.eps = eps
#         self.loss = loss
#         self.step_size = step_size
#         self.device = device or torch.device("cuda")
#         self.logger = logger
#         self.verbose = verbose

#     def perturb(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
#         x = x.detach().to(self.device)
#         if y is not None:
#             y = y.detach().to(self.device)

#         if self.step_size is None:
#             if self.norm == 'Linf':
#                 self.step_size = 1.0 / 255.0
#             else:
#                 self.step_size = 0.1

#         x_adv = x.clone()
#         x_adv.requires_grad = True

#         for _ in range(self.n_iter):
#             if self.loss == 'ce':
#                 logits, _ = self.model.logits(x_adv)
#                 loss_val = nn.CrossEntropyLoss()(logits, y)
#             elif self.loss == 'l2':
#                 emb_adv = self.model.encode(x_adv)
#                 loss_val = F.mse_loss(emb_adv, y, reduction='mean')
#             else:
#                 raise ValueError(f"Unknown loss mode: {self.loss}")

#             grad = torch.autograd.grad(loss_val, x_adv)[0]

#             with torch.no_grad():
#                 if self.norm == 'Linf':
#                     x_adv = x_adv + self.step_size * grad.sign()
#                     x_adv = torch.max(torch.min(x_adv, x + self.eps), x - self.eps)
#                 elif self.norm == 'L2':
#                     grad_norm = grad.view(grad.size(0), -1).norm(dim=1, keepdim=True) + 1e-9
#                     scaled_grad = grad / grad_norm.view(-1,1,1,1)
#                     x_adv = x_adv + self.step_size * scaled_grad
#                     delta = x_adv - x
#                     delta_norm = delta.view(delta.size(0), -1).norm(dim=1, keepdim=True) + 1e-9
#                     mask = (delta_norm > self.eps).float()
#                     delta = delta * (self.eps / delta_norm) * mask.view(-1,1,1,1)
#                     x_adv = x + delta
#                 else:
#                     raise ValueError("Unsupported norm: 'Linf' or 'L2' only.")

#                 x_adv = torch.clamp(x_adv, 0, 1)
#                 x_adv.requires_grad = True

#         return x_adv.detach()