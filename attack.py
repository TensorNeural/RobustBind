import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from model import Model, ForwardMode
from transform import unnormalize_inplace, normalize_inplace
from loss import ce_loss, l2_loss
from perf.profiling import ProfileModelMemory


# =========================== Base ===========================

class Attack(ABC):
    @abstractmethod
    def perturb(self, x: torch.Tensor, y: torch.Tensor = None, emb_orig = None) -> torch.Tensor:
        pass


class AttackModel(Model):
    def __init__(self, model: Model, mean=0, std=1):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def forward(self, x, mode=ForwardMode.EMBEDDINGS):
        x = (x - self.mean) / self.std
        return self.model(x, mode=mode)


# =========================== PGD ===========================

class PGDAttack(Attack):
    def __init__(
        self,
        logger,
        model: AttackModel,
        epsilon=8 / 255,
        alpha=2 / 255,
        steps=10,
        norm="linf",
        random_start=True,
        clamp_min=0.0,
        clamp_max=1.0,
        loss_type="ce",
    ):
        self.logger = logger
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.norm = norm.lower()
        self.random_start = random_start
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.loss_type = loss_type.lower()

        assert self.norm in ("linf", "l2", "l1")
        assert self.loss_type in ("ce", "l2")

    def perturb(self, x, y=None, emb_orig=None):
        x_adv = x.clone().detach()

        if self.loss_type == "ce":
            if y is None:
                raise ValueError("Cross-entropy loss requires labels.")
            
            with torch.no_grad():
                acc = self._acc(x, y)
            self.logger.info(f"[PGDAttack] Initial accuracy: {acc.item() * 100:.4f}%")
        elif self.loss_type == "l2":
            if emb_orig is None:
                raise ValueError("L2 loss requires original embeddings.")
            
            with torch.no_grad():
                cos_sim = self._cos_sim(x, emb_orig)
            self.logger.info(f"[PGDAttack] Initial cosine similarity: {cos_sim.item():.4f}")
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if self.random_start:
            if self.norm == "linf":
                x_adv = random_start_linf(x, self.epsilon, self.clamp_min, self.clamp_max)
            elif self.norm == "l2":
                x_adv = random_start_l2(x, self.epsilon, self.clamp_min, self.clamp_max)
            elif self.norm == "l1":
                x_adv = random_start_l1(x, self.epsilon, self.clamp_min, self.clamp_max)

        for step in range(self.steps):
            x_adv.requires_grad_(True)

            if self.loss_type == "ce":
                logits, _ = self.model(x_adv, mode=ForwardMode.LOGITS)
                loss = ce_loss(logits, y)

                with torch.no_grad():
                    acc = self._acc(x_adv, y)
                self.logger.debug(f"[PGDAttack] Step{step} accuracy: {acc.item() * 100:.4f}%")
            elif self.loss_type == "l2":
                with ProfileModelMemory(self.model, self.logger):
                    x_adv_emb = self.model(x_adv, mode=ForwardMode.EMBEDDINGS)
                loss = l2_loss(x_adv_emb, emb_orig)

                with torch.no_grad():
                    cos_sim = self._cos_sim(x_adv, emb_orig)
                self.logger.debug(f"[PGDAttack] Step{step} cosine similarity: {cos_sim.item():.4f}")
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")

            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach()

            if self.norm == "linf":
                x_adv = project_linf(x_adv + self.alpha * grad.sign(), x, self.epsilon)
            elif self.norm == "l2":
                step = self.alpha * grad / (L2_norm(grad, keepdim=True) + 1e-12)
                x_adv = project_l2(x_adv + step, x, self.epsilon)
            elif self.norm == "l1":
                step = self.alpha * grad / (L1_norm(grad, keepdim=True) + 1e-12)
                x_adv = x + project_l1(x_adv + step - x, self.epsilon)

            x_adv = x_adv.clamp(self.clamp_min, self.clamp_max)

        if self.loss_type == "ce":
            with torch.no_grad():
                acc = self._acc(x_adv, y)
            self.logger.info(f"[PGDAttack] Final accuracy: {acc.item() * 100:.4f}%")
        elif self.loss_type == "l2":
            with torch.no_grad():
                cos_sim = self._cos_sim(x_adv, emb_orig)
            self.logger.info(f"[PGDAttack] Final cosine similarity: {cos_sim.item():.4f}")

        return x_adv

    def _acc(self, x, y):
        logits, _ = self.model(x, mode=ForwardMode.LOGITS)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean()

    def _cos_sim(self, x, original_emb):
        x_emb = self.model(x, mode=ForwardMode.EMBEDDINGS)
        return F.cosine_similarity(x_emb, original_emb, dim=1).mean()


# =========================== APGD ===========================

class APGDAttack(Attack):
    def __init__(
        self,
        model,
        n_iter=100,
        norm="Linf",
        n_restarts=1,
        eps=None,
        seed=0,
        loss="ce",
        eot_iter=1,
        rho=0.75,
        verbose=False,
        device=None,
        logger=None,
    ):
        super().__init__()
        assert norm in ["Linf", "L2", "L1"]
        assert loss in ["ce", "l2"]
        assert eps is not None

        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.rho = rho
        self.verbose = verbose
        self.device = device
        self.logger = logger

    def normalize(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(dim=1)[0]
        elif self.norm == "L2":
            t = x.view(x.shape[0], -1).norm(p=2, dim=1)
        elif self.norm == "L1":
            t = x.view(x.shape[0], -1).abs().sum(dim=1)
        return x / (t.view(-1, *([1] * (x.dim() - 1))) + 1e-12)

    def perturb(self, x, y=None, emb_orig=None):
        if self.loss == "ce" and y is None:
            raise ValueError("Must provide labels y for CE loss")
        if self.loss == "l2" and emb_orig is None:
            raise ValueError("Must provide original embeddings for L2 loss")

        x = x.detach().clone().float().to(self.device)
        if y is not None:
            y = y.detach().clone().long().to(self.device)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        if self.loss == "l2":
            with torch.no_grad():
                x_emb = self.model(x, mode=ForwardMode.EMBEDDINGS)
        else:
            x_emb = None

        adv_best = x.clone()

        if self.loss == "ce":
            with torch.no_grad():
                acc = self._acc(x, y)
            self.logger.info(f"[APGDAttack] Initial accuracy: {acc.item() * 100:.2f}%")
        elif self.loss == "l2":
            with torch.no_grad():
                cos_sim = F.cosine_similarity(x_emb, emb_orig, dim=1).mean()
            self.logger.info(f"[APGDAttack] Initial cosine similarity: {cos_sim.item():.2f}")
        else:
            raise ValueError(f"Invalid loss: {self.loss}")

        for _ in range(self.n_restarts):
            x_adv = x.clone().detach()
            delta = (2 * torch.rand_like(x) - 1) if self.norm == "Linf" else torch.randn_like(x)
            x_adv = (x + self.eps * self.normalize(delta)).clamp(0, 1)

            step_size = self.eps
            cos_sim_prev = 1.0
            oscillation_counter = 0

            for iteration in range(self.n_iter):
                x_adv.requires_grad_(True)
                grad = torch.zeros_like(x_adv)

                with torch.enable_grad():
                    for _ in range(self.eot_iter):
                        if self.loss == "ce":
                            logits, _ = self.model(x_adv, mode=ForwardMode.LOGITS)
                            loss = F.cross_entropy(logits, y, reduction="none").sum()
                            with torch.no_grad():
                                acc = self._acc(x_adv, y)
                            self.logger.debug(f"[APGDAttack] Iteration {iteration}, accuracy: {acc.item() * 100:.2f}%")
                        elif self.loss == "l2":
                            x_adv_emb = self.model(x_adv, mode=ForwardMode.EMBEDDINGS)
                            diff = (x_adv_emb - emb_orig).view(x.shape[0], -1)
                            loss = (diff ** 2).sum()
                            with torch.no_grad():
                                cos_sim = F.cosine_similarity(x_adv_emb, emb_orig, dim=1).mean()
                            self.logger.debug(f"[APGDAttack] Iteration {iteration}, cosine similarity: {cos_sim.item():.2f}")
                        else:
                            raise ValueError(f"Invalid loss: {self.loss}")
                        grad += torch.autograd.grad(loss, [x_adv])[0].detach()

                grad /= self.eot_iter
                x_adv = x_adv.detach()

                # Step
                if self.norm == "Linf":
                    x_adv = project_linf(x_adv + step_size * grad.sign(), x, self.eps)
                elif self.norm == "L2":
                    step = step_size * grad / (L2_norm(grad, keepdim=True) + 1e-12)
                    x_adv = project_l2(x_adv + step, x, self.eps)
                elif self.norm == "L1":
                    step = step_size * grad / (L1_norm(grad, keepdim=True) + 1e-12)
                    x_adv = x + project_l1(x_adv + step - x, self.eps)
                    x_adv = x_adv.clamp(0.0, 1.0)

                # Cosine oscillation handling
                if self.loss == "l2":
                    with torch.no_grad():
                        x_adv_emb = self.model(x_adv, mode=ForwardMode.EMBEDDINGS)
                        cos_sim_curr = F.cosine_similarity(x_adv_emb, emb_orig, dim=1).mean().item()

                    if cos_sim_curr > cos_sim_prev - 1e-4:  # not improving
                        oscillation_counter += 1
                    else:
                        oscillation_counter = 0

                    cos_sim_prev = cos_sim_curr

                    if oscillation_counter >= 3:
                        step_size *= 0.5
                        oscillation_counter = 0
                        self.logger.debug(f"[APGDAttack] Reduced step size to {step_size:.6f} at iteration {iteration}")

                adv_best = x_adv

        if self.loss == "ce":
            with torch.no_grad():
                acc = self._acc(adv_best, y)
            self.logger.info(f"[APGDAttack] Final accuracy: {acc.item() * 100:.2f}%")
        elif self.loss == "l2":
            with torch.no_grad():
                adv_best_emb = self.model(adv_best, mode=ForwardMode.EMBEDDINGS)
                cos_sim = F.cosine_similarity(adv_best_emb, emb_orig, dim=1).mean()
            self.logger.info(f"[APGDAttack] Final cosine similarity: {cos_sim.item():.2f}")
        else:
            raise ValueError(f"Invalid loss: {self.loss}")

        return adv_best

    def _acc(self, x, y):
        logits, _ = self.model(x, mode=ForwardMode.LOGITS)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean()

# =========================== Two-Stage Attack ===========================

@torch.no_grad()
def two_stage_attack(logger, model, inputs, labels, attack_stage1, attack_stage2, mean, std):
    logger.info("Running two-stage attack...")

    inputs_unorm = inputs.clone()
    unnormalize_inplace(inputs_unorm, mean, std)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        adv_stage1 = attack_stage1.perturb(inputs_unorm, labels)
    normalize_inplace(adv_stage1, mean, std)

    logits_stage1, _ = model(adv_stage1, mode=ForwardMode.LOGITS)
    preds_stage1 = logits_stage1.argmax(dim=1)
    correct_mask = preds_stage1 == labels
    keep_idx = correct_mask.nonzero(as_tuple=True)[0]

    adv_final = adv_stage1.clone()

    if len(keep_idx) > 0:
        logger.info(f"Stage1 left {len(keep_idx)}/{inputs.size(0)} samples correct. Applying Stage2...")
        inputs_unorm2 = inputs[keep_idx].clone()
        unnormalize_inplace(inputs_unorm2, mean, std)
        adv_stage2 = attack_stage2.perturb(inputs_unorm2, labels[keep_idx])
        normalize_inplace(adv_stage2, mean, std)
        adv_final[keep_idx] = adv_stage2

    return adv_final


# =========================== Helpers ===========================

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
