import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from model import Model, ForwardMode
from transform import unnormalize_inplace, normalize_inplace
from loss import ce_loss, l2_loss, dlr_loss, dlr_loss_targeted, ce_loss_targeted
from perf.profiling import ProfileModelMemory, ProfileModelGradient
from contextlib import contextmanager

# =========================== Base ===========================

class Attack(ABC):
    @abstractmethod
    def perturb(self, x: torch.Tensor, y: torch.Tensor = None, emb_orig=None) -> torch.Tensor:
        pass


class AttackModel(Model):
    def __init__(self, model: Model, mean=0, std=1):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def forward(self, x, mode=ForwardMode.EMBEDDINGS):
        x = (x - self.mean) / self.std
        wrapped_x = self.model.wrap_tensor(x)
        return self.model(wrapped_x, mode)


# =========================== PGD ===========================

class PGDAttack(Attack):
    def __init__(self, logger, model: AttackModel,
                 epsilon=8 / 255, alpha=2 / 255, steps=10,
                 norm="linf", random_start=True,
                 clamp_min=0.0, clamp_max=1.0,
                 loss_type="ce"):
        assert norm in ("linf", "l2", "l1")
        assert loss_type in ("ce", "l2", "dlr", "dlr-targeted", "ce-targeted")

        self.logger = logger
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.random_start = random_start
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.loss_type = loss_type
        self.y_target = None

    def perturb(self, x, y=None, emb_orig=None):
        with no_model_grads(self.model):
            x_adv = x.detach().clone().requires_grad_(True)

            if self.loss_type in {"ce", "dlr", "ce-targeted", "dlr-targeted"}:
                if y is None:
                    raise ValueError("This loss type requires labels.")
                with torch.no_grad():
                    acc = self._acc_with_x(x, y)
                self.logger.info(f"[PGD] Initial accuracy: {acc.item() * 100:.2f}%")
            elif self.loss_type == "l2":
                if emb_orig is None:
                    raise ValueError("L2 loss requires original embeddings.")
                with torch.no_grad():
                    cos_sim = self._cos_sim_with_x(x, emb_orig)
                self.logger.info(f"[PGD] Initial cosine similarity: {cos_sim.item():.4f}")

            if self.random_start:
                self.logger.debug(f"[PGD] Applying random start for norm={self.norm}")
                x_adv = {
                    "linf": random_start_linf,
                    "l2": random_start_l2,
                    "l1": random_start_l1
                }[self.norm](x, self.epsilon, self.clamp_min, self.clamp_max)

            for step in range(self.steps):
                x_adv = x_adv.detach().clone().requires_grad_(True)
                model_input = x_adv.clone()

                if self.loss_type == "ce":
                    with ProfileModelMemory(self.model, self.logger):
                        logits, _ = self.model(model_input, mode=ForwardMode.LOGITS)
                    loss = ce_loss(logits, y)
                elif self.loss_type == "ce-targeted":
                    logits, _ = self.model(model_input, mode=ForwardMode.LOGITS)
                    loss = ce_loss_targeted(logits, self.y_target)
                elif self.loss_type == "dlr":
                    logits, _ = self.model(model_input, mode=ForwardMode.LOGITS)
                    loss = dlr_loss(logits, y)
                elif self.loss_type == "dlr-targeted":
                    logits, _ = self.model(model_input, mode=ForwardMode.LOGITS)
                    loss = dlr_loss_targeted(logits, y, self.y_target)
                elif self.loss_type == "l2":
                    x_adv_emb = self.model(model_input, mode=ForwardMode.EMBEDDINGS)
                    loss = l2_loss(x_adv_emb, emb_orig)

                grad = torch.autograd.grad(loss, x_adv)[0]
                grad_norm = grad.norm().item()
                del loss
                x_adv = x_adv.detach()

                if self.norm == "linf":
                    x_adv = project_linf(x_adv + self.alpha * grad.sign(), x, self.epsilon)
                elif self.norm == "l2":
                    step_dir = self.alpha * grad / (L2_norm(grad, keepdim=True) + 1e-12)
                    x_adv = project_l2(x_adv + step_dir, x, self.epsilon)
                elif self.norm == "l1":
                    step_dir = self.alpha * grad / (L1_norm(grad, keepdim=True) + 1e-12)
                    x_adv = x + project_l1(x_adv + step_dir - x, self.epsilon)

                x_adv = x_adv.clamp(self.clamp_min, self.clamp_max)

                # === Log step-wise metrics ===
                self.logger.debug(f"[PGD][Step {step+1}/{self.steps}] Grad norm: {grad_norm:.4f}")
                if self.loss_type == "l2":
                    with torch.no_grad():
                        cos_sim_step = self._cos_sim_with_x(x_adv, emb_orig)
                    self.logger.debug(f"[PGD][Step {step+1}] Cosine similarity: {cos_sim_step.item():.4f}")
                elif self.loss_type in {"ce", "dlr", "ce-targeted", "dlr-targeted"}:
                    with torch.no_grad():
                        acc_step = self._acc_with_x(x_adv, y)
                    self.logger.debug(f"[PGD][Step {step+1}] Accuracy: {acc_step.item() * 100:.2f}%")
                    self.logger.debug(f"[PGD][Step {step+1}] Perturbed sample range: "
                                      f"[{x_adv.min().item():.4f}, {x_adv.max().item():.4f}]")

                del grad

            # === Final metric ===
            if self.loss_type == "l2":
                with torch.no_grad():
                    final_cos_sim = self._cos_sim_with_x(x_adv, emb_orig)
                self.logger.info(f"[PGD] Final cosine similarity: {final_cos_sim.item():.4f}")
            elif self.loss_type in {"ce", "dlr", "ce-targeted", "dlr-targeted"}:
                with torch.no_grad():
                    final_acc = self._acc_with_x(x_adv, y)
                self.logger.info(f"[PGD] Final accuracy: {final_acc.item() * 100:.2f}%")

            self.logger.info("[PGD] Attack completed.")
            return x_adv

    def _acc_with_x(self, x, y):
        logits, _ = self.model(x, mode=ForwardMode.LOGITS)
        return self._acc_with_logits(logits, y)

    def _acc_with_logits(self, logits, y):
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean()
    
    def _cos_sim_with_x(self, x, original_emb):
        x_emb = self.model(x, mode=ForwardMode.EMBEDDINGS)
        return self._cos_sim_with_emb(x_emb, original_emb)

    def _cos_sim_with_emb(self, emb, original_emb):
        return F.cosine_similarity(emb, original_emb, dim=1).mean()


# =========================== APGD ===========================

class APGDAttack(Attack):
    def __init__(self, logger, model: AttackModel,
                 n_iter=100, norm="linf", n_restarts=1,
                 eps=8 / 255, loss_type="ce",
                 eot_iter=1, best_loss=True,
                 device=None):
        assert norm in ("linf", "l2", "l1")
        assert loss_type in ("ce", "l2", "dlr", "dlr-targeted", "ce-targeted")

        self.logger = logger
        self.model = model
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.eps = eps
        self.norm = norm
        self.loss_type = loss_type
        self.eot_iter = eot_iter
        self.best_loss = best_loss
        self.device = device
        self.y_target = None

    def perturb(self, x, y=None, emb_orig=None):
        with no_model_grads(self.model):
            if y is None and self.loss_type in {"ce", "dlr", "dlr-targeted", "ce-targeted"}:
                raise ValueError("This attack requires labels.")

            x = x.detach().clone().to(self.device)
            y = y.to(self.device) if y is not None else None
            emb_orig = emb_orig.to(self.device) if emb_orig is not None else None

            best_adv = x.clone()

            for restart in range(self.n_restarts):
                self.logger.info(f"[APGD] Restart {restart + 1}/{self.n_restarts}")
                delta = (2 * torch.rand_like(x) - 1) if self.norm == "linf" else torch.randn_like(x)
                delta = self._normalize(delta) * self.eps
                x_adv = (x + delta).clamp(0, 1)

                for i in range(self.n_iter):
                    x_adv = x_adv.detach().clone().requires_grad_(True)
                    loss = 0.0

                    with ProfileModelGradient(self.model, self.logger):
                        for _ in range(self.eot_iter):
                            if self.loss_type == "ce":
                                logits, _ = self.model(x_adv, ForwardMode.LOGITS)
                                loss += ce_loss(logits, y)
                            elif self.loss_type == "ce-targeted":
                                logits, _ = self.model(x_adv, ForwardMode.LOGITS)
                                loss += ce_loss_targeted(logits, self.y_target)
                            elif self.loss_type == "dlr":
                                logits, _ = self.model(x_adv, ForwardMode.LOGITS)
                                loss += dlr_loss(logits, y)
                            elif self.loss_type == "dlr-targeted":
                                logits, _ = self.model(x_adv, ForwardMode.LOGITS)
                                loss += dlr_loss_targeted(logits, y, self.y_target)
                            elif self.loss_type == "l2":
                                x_emb = self.model(x_adv, ForwardMode.EMBEDDINGS)
                                loss += l2_loss(x_emb, emb_orig)

                    loss /= self.eot_iter
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    grad_norm = grad.norm().item()
                    x_adv = x_adv.detach()

                    if self.norm == "linf":
                        x_adv = project_linf(x_adv + self.eps * grad.sign(), x, self.eps)
                    elif self.norm == "l2":
                        step = grad / (L2_norm(grad, keepdim=True) + 1e-12)
                        x_adv = project_l2(x_adv + self.eps * step, x, self.eps)
                    elif self.norm == "l1":
                        step = grad / (L1_norm(grad, keepdim=True) + 1e-12)
                        x_adv = x + project_l1(x_adv + self.eps * step - x, self.eps)

                    x_adv = x_adv.clamp(0, 1)

                    # === Log step-wise metrics ===
                    self.logger.debug(f"[APGD][Restart {restart+1}][Iter {i+1}/{self.n_iter}] Grad norm: {grad_norm:.4f}")
                    if self.loss_type == "l2":
                        with torch.no_grad():
                            cos_sim_step = self._cos_sim_with_x(x_adv, emb_orig)
                        self.logger.debug(f"[APGD][Restart {restart+1}][Iter {i+1}] Cosine similarity: {cos_sim_step.item():.4f}")
                    elif self.loss_type in {"ce", "dlr", "ce-targeted", "dlr-targeted"}:
                        with torch.no_grad():
                            acc_step = self._acc_with_x(x_adv, y)
                        self.logger.debug(f"[APGD][Restart {restart+1}][Iter {i+1}] Accuracy: {acc_step.item() * 100:.2f}%")
                        self.logger.debug(f"[APGD][Restart {restart+1}][Iter {i+1}] Perturbed sample range: "
                                          f"[{x_adv.min().item():.4f}, {x_adv.max().item():.4f}]")

                    del grad

                best_adv = x_adv

            # === Final metric ===
            if self.loss_type == "l2":
                with torch.no_grad():
                    final_cos_sim = self._cos_sim_with_x(best_adv, emb_orig)
                self.logger.info(f"[APGD] Final cosine similarity: {final_cos_sim.item():.4f}")
            elif self.loss_type in {"ce", "dlr", "ce-targeted", "dlr-targeted"}:
                with torch.no_grad():
                    final_acc = self._acc_with_x(best_adv, y)
                self.logger.info(f"[APGD] Final accuracy: {final_acc.item() * 100:.2f}%")

            self.logger.info("[APGD] Attack completed.")
            return best_adv

    def _normalize(self, x):
        if self.norm == "linf":
            return x.sign()
        elif self.norm == "l2":
            return x / (x.view(x.size(0), -1).norm(p=2, dim=1, keepdim=True) + 1e-12)
        elif self.norm == "l1":
            return x / (x.view(x.size(0), -1).abs().sum(dim=1, keepdim=True) + 1e-12)

    def _acc_with_x(self, x, y):
        logits, _ = self.model(x, mode=ForwardMode.LOGITS)
        return (logits.argmax(dim=1) == y).float().mean()

    def _cos_sim_with_x(self, x, emb_orig):
        x_emb = self.model(x, mode=ForwardMode.EMBEDDINGS)
        return F.cosine_similarity(x_emb, emb_orig, dim=1).mean()


# =========================== Two-Stage Attack ===========================
def two_stage_attack(logger, model, inputs, labels, attack_stage1, attack_stage2, mean, std):
    logger.info("Running two-stage attack...")
    inputs_unorm = inputs.detach().clone()

    unnormalize_inplace(inputs_unorm, mean, std)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        adv_stage1 = attack_stage1.perturb(inputs_unorm, labels)

    normalize_inplace(adv_stage1, mean, std)

    wrapped_adv_stage1 = model.wrap_tensor(adv_stage1)
    with torch.no_grad():
        logits_stage1, _ = model(wrapped_adv_stage1, ForwardMode.LOGITS)
        preds_stage1 = logits_stage1.argmax(dim=1)
        correct_mask = preds_stage1 == labels
        keep_idx = correct_mask.nonzero(as_tuple=True)[0]
        adv_final = adv_stage1.detach()
    if len(keep_idx) > 0:
        logger.info(f"Stage1 left {len(keep_idx)}/{inputs.size(0)} samples correct. Applying Stage2...")
        
        with torch.no_grad():
            inputs_unorm2 = inputs[keep_idx]
            inputs_unorm2 = inputs_unorm2.detach().clone()
            unnormalize_inplace(inputs_unorm2, mean, std)

        adv_stage2 = attack_stage2.perturb(inputs_unorm2, labels[keep_idx])

        with torch.no_grad():
            normalize_inplace(adv_stage2, mean, std)
            adv_final[keep_idx] = adv_stage2

    return adv_final


# =========================== Norm Helpers ===========================

def L2_norm(x, keepdim=False):
    norm = x.view(x.shape[0], -1).norm(p=2, dim=1)
    return norm.view(-1, *[1] * (x.dim() - 1)) if keepdim else norm

def L1_norm(x, keepdim=False):
    norm = x.view(x.shape[0], -1).abs().sum(dim=1)
    return norm.view(-1, *[1] * (x.dim() - 1)) if keepdim else norm

def project_linf(x_adv, x_orig, eps):
    return torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

def project_l2(x_adv, x_orig, eps):
    delta = x_adv - x_orig
    norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
    factor = torch.clamp(eps / (norm + 1e-12), max=1.0)
    delta = delta * factor.view(-1, *[1] * (delta.ndim - 1))
    return x_orig + delta

def project_l1(delta, eps):
    flat = delta.view(delta.size(0), -1)
    norm = flat.abs().sum(dim=1, keepdim=True)
    factor = (eps / (norm + 1e-12)).clamp(max=1.0)
    flat = flat * factor
    return flat.view_as(delta)

def random_start_linf(x, eps, clamp_min, clamp_max):
    delta = (2.0 * torch.rand_like(x) - 1.0) * eps
    return (x + delta).clamp(clamp_min, clamp_max)

def random_start_l2(x, eps, clamp_min, clamp_max):
    delta = torch.randn_like(x)
    norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
    delta = delta * (eps / (norm + 1e-12)).view(-1, *[1] * (x.dim() - 1))
    return (x + delta).clamp(clamp_min, clamp_max)

def random_start_l1(x, eps, clamp_min, clamp_max):
    delta = torch.randn_like(x)
    delta = project_l1(delta, eps)
    return (x + delta).clamp(clamp_min, clamp_max)


@contextmanager
def no_model_grads(model):
    backup = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = False
    try:
        yield
    finally:
        for name, p in model.named_parameters():
            p.requires_grad = backup[name]
