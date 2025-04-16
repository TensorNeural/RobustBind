import torch
import time
import math
from model import Model
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transform import unnormalize_inplace, normalize_inplace
from loss import l2_loss, ce_loss

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

class AttackModel(Model):
    def __init__(self, model: Model, mean=0, std=1):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def logits(self, x):
        x_norm = (x - self.mean) / self.std
        return self.model.logits(x_norm)

    def encode(self, x):
        x_norm = (x - self.mean) / self.std
        return self.model.encode(x_norm)

class Attack(ABC):
    @abstractmethod
    def perturb(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pass

class PGDAttack(Attack):
    """
    PGD with selectable norm (L∞, L2, L1) and selectable loss:
      - Cross-entropy ('ce') requires labels y, uses .logits(...)
      - L2 ('l2') uses MSE in model encoding space, uses .encode(...)
    
    Args:
        model (AttackModel): Model with `.logits(...)` and `.encode(...)`.
        epsilon (float): Norm bound on the perturbation.
        alpha (float): Step size in each PGD iteration.
        steps (int): Number of update steps.
        norm (str): 'linf', 'l2', or 'l1'.
        random_start (bool): If True, sample initial perturbation inside the norm-ball.
        clamp_min (float): Minimum clamp value (e.g., 0.0 for images).
        clamp_max (float): Maximum clamp value (e.g., 1.0 for normalized images).
        loss_type (str): 'ce' or 'l2'.
    """
    def __init__(
        self,
        logger,
        model: AttackModel,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        steps: int = 10,
        norm: str = "linf",
        random_start: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        loss_type: str = "ce",
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

        assert self.norm in ("linf", "l2", "l1"), f"Unsupported norm {self.norm}"
        assert self.loss_type in ("ce", "l2"), f"Unsupported loss {self.loss_type}"

    def perturb(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Generate adversarial examples from input x (with optional labels y).
          - For "ce" (cross-entropy), y is required.
          - For "l2" (MSE in latent space), y is ignored.
        """
        # Clone to avoid modifying the original
        x_adv = x.clone().detach()

        with torch.no_grad():
            if self.loss_type == "ce":
                acc = self._acc(x, y)
                self.logger.info(f"[PGDAttack] Initial accuracy: {acc.item() * 100:.2f}%")
            elif self.loss_type == "l2":
                original_emb = self.model.encode(x)
                cos_sim = self._cos_sim(x, original_emb)
                self.logger.info(f"[PGDAttack] Initial cosine similarity: {cos_sim.item():.2f}")

        # Random start
        if self.random_start:
            if self.norm == "linf":
                x_adv = random_start_linf(x, self.epsilon, self.clamp_min, self.clamp_max)
            elif self.norm == "l2":
                x_adv = random_start_l2(x, self.epsilon, self.clamp_min, self.clamp_max)
            else:  # 'l1'
                x_adv = random_start_l1(x, self.epsilon, self.clamp_min, self.clamp_max)

        # PGD iterations
        for step in range(self.steps):
            self.logger.debug(f"[PGDAttack] PGD step: {step+1}/{self.steps}")
            x_adv.requires_grad = True

            if self.loss_type == "ce":
                if y is None:
                    raise ValueError("Cross-entropy loss requires label tensor y.")
                
                logits, _ = self.model.logits(x_adv)
                loss = ce_loss(logits, y)

                with torch.no_grad():
                    y_pred = logits.argmax(dim=1)
                    acc = (y_pred == y).float().sum() / y_pred.shape[0]

                self.logger.debug(f"[PGDAttack] Accuracy: {acc.item() * 100:.2f}%, Loss: {loss.mean().item():.2f}")

            elif self.loss_type == 'l2':  # "l2" loss in encoding space
                x_adv_emb = self.model.encode(x_adv)
                loss = l2_loss(x_adv_emb, original_emb)

                with torch.no_grad():
                    cos_sim = F.cosine_similarity(x_adv_emb, original_emb, dim=1).mean()

                self.logger.debug(f"[PGDAttack] Cosine similarity: {cos_sim.item():.2f}, Loss: {loss.item():.2f}")
            else:
                raise ValueError(f"Unsupported loss type {self.loss_type}")
            
            self.logger.debug(f"[PGDAttack] Loss: {loss.item():.2f}")

            # Backprop to get gradient
            loss.backward()
            grad = x_adv.grad.detach()

            # 1) Gradient step
            if self.norm == "linf":
                # direction is sign(grad)
                step = self.alpha * grad.sign()
                x_adv = x_adv + step
                # 2) Project onto L∞ ball
                x_adv = project_linf(x_adv, x, self.epsilon)

            elif self.norm == "l2":
                # direction is grad / ||grad||_2
                gnorm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True)
                step = self.alpha * grad / (gnorm + 1e-12)
                x_adv = x_adv + step
                # 2) Project onto L2 ball
                x_adv = project_l2(x_adv, x, self.epsilon)

            else:  # "l1"
                # direction is grad / ||grad||_1
                gnorm = grad.view(grad.size(0), -1).abs().sum(dim=1, keepdim=True)
                step = self.alpha * grad / (gnorm + 1e-12)
                x_adv = x_adv + step
                # 2) Project onto L1 ball
                delta = x_adv - x
                delta = project_l1(delta, self.epsilon)  # rescale if needed
                x_adv = (x + delta).clamp(self.clamp_min, self.clamp_max)

            # Finally clamp to [clamp_min, clamp_max]
            x_adv = x_adv.clamp(self.clamp_min, self.clamp_max)
            x_adv = x_adv.detach()
        
        with torch.no_grad():
            if self.loss_type == "ce":
                acc = self._acc(x_adv, y)
                self.logger.info(f"[PGDAttack] Final accuracy: {acc.item() * 100:.2f}%")
            elif self.loss_type == "l2":
                cos_sim = self._cos_sim(x_adv, original_emb)
                self.logger.info(f"[PGDAttack] Final cosine similarity: {cos_sim.item():.2f}")

        return x_adv
    
    def _acc(self, x, y):
        """
        Compute accuracy given logits and labels.
        """
        logits, _ = self.model.logits(x)
        _, preds = logits.max(1)
        correct = (preds == y).float().sum()
        acc = correct / y.shape[0]
        return acc
    
    def _cos_sim(self, x, original_emb):
        x_emb = self.model.encode(x)
        cos_sim = F.cosine_similarity(x_emb, original_emb, dim=1)
        return cos_sim.mean()

class APGDAttack(Attack):
    """
    AutoPGD Attack (multi-restart, L∞/L2/L1, EOT, adaptive step size)
    with support for:
      - loss = 'ce'  (cross-entropy)
      - loss = 'l2'  (MSE in the model's encoding space)

    Reference: https://arxiv.org/abs/2003.01690

    :param predict:      A model (or wrapper) that, when called, returns
                         (logits, encoding) or at least logits. We store
                         it in self.model.
    :param norm:         Which norm to bound perturbations: 'Linf', 'L2', or 'L1'
    :param n_restarts:   Number of random restarts
    :param n_iter:       Number of iterations
    :param eps:          Bound on the perturbation norm
    :param seed:         Random seed for reproducible restarts
    :param loss:         'ce' or 'l2'
    :param eot_iter:     Expectation over Transformation (EOT) iterations
    :param rho:          Parameter used to reduce step size (thr_decr)
    :param topk:         Used in L1 attacks for top-k selection (optional)
    :param verbose:      If True, prints/logs progress
    :param device:       Torch device (CPU or GPU)
    :param use_largereps If True, uses the large-eps scheduling approach
    :param is_tf_model:  If True, indicates a TF-like model interface
                         (not used much here, but included for parity)
    :param logger:       Optional logger for printing
    """
    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',            # 'ce' or 'l2'
            eot_iter=1,
            rho=0.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None
        ):

        super().__init__()
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho  # used in oscillation checks
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.use_largereps = use_largereps
        self.n_iter_orig = n_iter
        self.eps_orig = eps
        self.is_tf_model = is_tf_model
        self.logger = logger

        # For bounding norm
        assert self.norm in ['Linf', 'L2', 'L1'], f"Unknown norm {self.norm}"
        # For objective
        assert self.loss in ['ce', 'l2'], f"Unknown loss {self.loss}"
        assert self.eps is not None, "eps must be specified"

        # For the "oscillation" checks
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

        # If you want to store target labels for a targeted setup, you can.
        self.y_target = None

    # ----------------------------------------------------------------
    # Helper: initialize hyperparams (device, etc.)
    # ----------------------------------------------------------------
    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    # ----------------------------------------------------------------
    # Helper: check oscillation to decide step-size shrink
    # ----------------------------------------------------------------
    def check_oscillation(self, loss_steps, j, k, loss_best, k3=0.75):
        """
        The 'oscillation' check from the original code: 
        looks at the last k steps in 'loss_steps' (for the sample j)
        and sees if there's not enough improvement. If the fraction 
        of “improved steps” is below k3, we indicate we should shrink step size.
        """
        t = torch.zeros(loss_steps.shape[1], device=self.device)
        for counter5 in range(k):
            t += (loss_steps[j - counter5] > loss_steps[j - counter5 - 1]).float()
        return (t <= k * k3 * torch.ones_like(t)).float()

    # ----------------------------------------------------------------
    # Helper: normalize a tensor to unit norm in L∞, L2, or L1 sense
    # ----------------------------------------------------------------
    def normalize(self, x):
        """
        Normalizes input x according to self.norm (L∞, L2, L1),
        used mostly for random initialization or gradient steps.
        """
        if self.norm == 'Linf':
            # L∞ normalization is basically sign, but here we do an absolute max
            t = x.abs().view(x.shape[0], -1).max(dim=1)[0]
        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        elif self.norm == 'L1':
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        return x / (t.view(-1, *([1]*self.ndims)) + 1e-12)

    # ----------------------------------------------------------------
    # The single-run attack (no multiple restarts). 
    # You can call this for each random restart.
    # ----------------------------------------------------------------
    def attack_single_run(self, x, y, x_init=None, x_enc=None):
        """
        Runs a single instance of APGD given:
          - x:      clean input
          - y:      labels (used for 'ce'), ignored for 'l2'
          - x_init: optional custom initialization
          - x_enc:  the encoding of x (only needed if self.loss == 'l2') 
        Returns:
          (x_best, acc, loss_best, x_best_adv)
        """
        # If x or y are smaller than the batch shape, unsqueeze them
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        # 1) Random / custom initialization
        if x_init is not None:
            x_adv = x_init.clone()
        else:
            if self.norm == 'Linf':
                t = 2 * torch.rand_like(x) - 1
                x_adv = x + self.eps * self.normalize(t)
            elif self.norm == 'L2':
                t = torch.randn_like(x)
                x_adv = x + self.eps * self.normalize(t)
            elif self.norm == 'L1':
                t = torch.randn_like(x)
                delta = L1_projection(x, t, self.eps)
                x_adv = x + t + delta
        
        # Clamp to [0,1] if needed (assuming image inputs)
        x_adv = x_adv.clamp(0., 1.)

        # We'll track "best so far" in two ways:
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_best = -1e10 * torch.ones(x.shape[0], device=self.device)
        acc = torch.ones_like(loss_best).bool()  # track which samples remain robust

        # We'll store the loss each iteration for each sample:
        loss_steps = torch.zeros([self.n_iter, x.shape[0]], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]], device=self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        # Set up the appropriate criterion (CE vs. L2).
        # For 'l2', we do MSE in encoding space. For 'ce', standard cross-entropy.
        def criterion_indiv_ce(logits, labels):
            return F.cross_entropy(logits, labels, reduction='none')
        
        def criterion_indiv_l2(x_enc_adv, x_enc_clean):
            # MSE with reduction='none' => shape [B, ...] => sum spatial dims
            # so we end up with a per-sample scalar
            # We sum across feature dims so we get a single loss value per sample
            diff = x_enc_adv - x_enc_clean
            diff_flat = diff.view(diff.shape[0], -1)
            return (diff_flat**2).sum(dim=1)  # shape [B]

        # Precompute the "clean" encodings if needed
        # (so that each step we only encode x_adv).
        if self.loss == 'l2':
            if x_enc is None:
                # if not already provided, compute once
                with torch.no_grad():
                    # model(...) should return (logits, encoding) or use encode(...)
                    _, x_enc_clean = self.model(x)
                x_enc = x_enc_clean
        else:
            x_enc = None

        # Evaluate the starting gradient + loss
        x_adv.requires_grad_(True)
        grad = torch.zeros_like(x)
        with torch.enable_grad():
            # EOT: we do multiple forward/back passes
            for _ in range(self.eot_iter):
                logits, x_adv_enc = self.model(x_adv)  # shape: (B,#classes), (B,encodingdim,...)

                if self.loss == 'ce':
                    loss_indiv = criterion_indiv_ce(logits, y)
                elif self.loss == 'l2':
                    loss_indiv = criterion_indiv_l2(x_adv_enc, x_enc)
                
                loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        x_adv = x_adv.detach()

        # Evaluate who is currently misclassified
        if self.loss == 'ce':
            pred = logits.detach().max(1)[1] == y
            acc = pred.clone()
        else:
            # For 'l2', there's no direct "misclassification" to track.
            # We can interpret "acc" as 'did not degrade the correct class?'
            # Or we can just keep it to see if the final model prediction changes:
            with torch.no_grad():
                pred_class = logits.detach().max(1)[1]
                acc = (pred_class == y)
        acc_steps[0] = acc.float()

        # Record the best known loss
        loss_best = loss_indiv.detach().clone()
        loss_best_steps[0] = loss_best

        # Some step-size heuristics from the reference code
        if self.norm in ['Linf', 'L2']:
            alpha = 2.0  # base factor
        else:  # L1
            alpha = 1.0
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1]*self.ndims)],
                                                 device=self.device)
        
        x_adv_old = x_adv.clone()
        grad_best = grad.clone()

        k = self.n_iter_2
        n_fts = math.prod(self.orig_dim)

        if self.norm == 'L1':
            k = max(int(0.04 * self.n_iter), 1)
        
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        # =========================
        # Main iteration loop
        # =========================
        for i in range(self.n_iter):
            # 1) Momentum-like update from the reference (x_adv_1, then combine with x_adv + grad2)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                # Weighted combination factor 'a'
                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    # sign(grad)
                    x_adv_1 = x_adv + step_size * grad.sign()
                    # project to L∞ ball
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps),
                                         x + self.eps), 0.0, 1.0)
                    # combine
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    # project again
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps),
                                         x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    # step in direction of normalized grad
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    # project to L2 ball
                    rad = torch.min(self.eps * torch.ones_like(x), 
                                    L2_norm(x_adv_1 - x, keepdim=True))
                    # clamp to [0,1]
                    x_adv_1 = x + self.normalize(x_adv_1 - x) * rad
                    x_adv_1 = x_adv_1.clamp(0.0, 1.0)
                    # combine
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    rad = torch.min(self.eps * torch.ones_like(x),
                                    L2_norm(x_adv_1 - x, keepdim=True))
                    x_adv_1 = x + self.normalize(x_adv_1 - x) * rad
                    x_adv_1 = x_adv_1.clamp(0.0, 1.0)

                elif self.norm == 'L1':
                    # pick topk for gradient
                    grad_topk_vals = grad.abs().view(x.shape[0], -1).sort(dim=-1)[0]
                    if self.topk is not None:
                        # if user provided a topk fraction
                        topk_curr = torch.clamp(
                            (1. - self.topk) * n_fts, min=0, max=n_fts - 1
                        ).long()
                    else:
                        # or adapt topk fraction from iteration to iteration
                        # here we do a naive approach: pick the median?
                        topk_curr = (0.8 * n_fts) * torch.ones([x.shape[0]], device=self.device)
                        topk_curr = topk_curr.long()
                    # threshold
                    threshold = grad_topk_vals[torch.arange(x.shape[0], device=self.device), topk_curr]
                    threshold = threshold.view(-1, *([1]*(x.dim()-1)))

                    sparsegrad = grad * (grad.abs() >= threshold).float()

                    # step
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (L1_norm(sparsegrad.sign(), keepdim=True)+1e-10)
                    # project to L1 ball
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = (x + delta_u + delta_p).clamp(0.0, 1.0)

                    # combine
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1-a)
                    # project again
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = (x + delta_u + delta_p).clamp(0.0, 1.0)

                x_adv = x_adv_1.clone()

            # 2) Compute new gradient
            x_adv.requires_grad_(True)
            grad.zero_()
            with torch.enable_grad():
                for _ in range(self.eot_iter):
                    logits, x_adv_enc = self.model(x_adv)

                    if self.loss == 'ce':
                        loss_indiv = criterion_indiv_ce(logits, y)
                    else:  # 'l2'
                        loss_indiv = criterion_indiv_l2(x_adv_enc, x_enc)

                    loss = loss_indiv.sum()
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad /= float(self.eot_iter)
            x_adv = x_adv.detach()

            # 3) Update book-keeping for best losses
            with torch.no_grad():
                # Evaluate final prediction if using CE
                if self.loss == 'ce':
                    pred = logits.detach().max(1)[1] == y
                    acc = torch.min(acc, pred)  # once we are misclassified, we remain misclassified
                else:
                    # for 'l2', you might check if it changes the predicted class
                    pred_class = logits.detach().max(1)[1]
                    still_correct = (pred_class == y)
                    acc = torch.min(acc, still_correct)

                acc_steps[i+1] = acc.float()
                ind_pred = (acc == 0).nonzero().squeeze()  # samples that are now misclassified
                x_best_adv[ind_pred] = x_adv[ind_pred].clone()

                # Check if the sample's loss improved
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1
                improved_mask = (y1 > loss_best)
                x_best[improved_mask] = x_adv[improved_mask].clone()
                grad_best[improved_mask] = grad[improved_mask].clone()
                loss_best[improved_mask] = y1[improved_mask]
                loss_best_steps[i+1] = loss_best

                # Possibly do the “oscillation check”
                counter3 += 1
                if counter3 == k:
                    if self.norm in ['Linf', 'L2']:
                        fl_osc = self.check_oscillation(loss_steps, i, k, loss_best, k3=self.thr_decr)
                        # also check "no improvement" since last check
                        fl_reduce_no_impr = (1. - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                        fl_osc = torch.max(fl_osc, fl_reduce_no_impr)
                        reduced_last_check = fl_osc.clone()
                        loss_best_last_check = loss_best.clone()

                        # For those that haven't improved, reduce step size and revert to best
                        if fl_osc.sum() > 0:
                            ind_fl_osc = (fl_osc > 0).nonzero().squeeze()
                            step_size[ind_fl_osc] /= 2.0
                            x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                            grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                        k = max(k - self.size_decr, self.n_iter_min)

                    elif self.norm == 'L1':
                        # L1 does a slightly different scheme
                        sp_curr = (x_best - x).abs().view(x.shape[0], -1).sum(dim=1)
                        fl_redtopk = (sp_curr < 0.95 * sp_curr.max())  # naive check
                        step_size[fl_redtopk] = step_size[fl_redtopk].clone()  # could reset or reduce
                        # or do the more advanced logic from original code
                        x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                        grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

        return x_best, acc, loss_best, x_best_adv

    # ----------------------------------------------------------------
    # The main entry point: multi-restart APGD
    # ----------------------------------------------------------------
    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:        Clean images (B,C,H,W)
        :param y:        Clean labels. Required if loss='ce'. 
                         If loss='l2', we ignore y, but you can still pass it.
        :param best_loss If True, returns the examples that attain the highest loss
                         for each sample, rather than the first successful adv.
        :param x_init:   Optional custom init for each sample. Usually None.
        :return:         adversarial examples
        """
        if self.loss == 'ce' and y is None:
            raise ValueError("Must provide labels y when loss='ce'.")
        
        # Make sure x,y are on correct device
        x = x.detach().clone().float()
        if self.device is None:
            self.device = x.device
        x = x.to(self.device)
        if y is not None:
            y = y.detach().clone().long().to(self.device)

        # Possibly get predictions on the clean input
        with torch.no_grad():
            logits, x_enc = self.model(x)
            y_pred = logits.max(dim=1)[1]
        if y is None:
            # for 'l2' with no provided labels, we won't need y
            y = y_pred.clone()

        # We'll store final adversaries in 'adv'
        adv = x.clone()
        # For CE, "acc" is whether each sample is still correct
        acc = (y_pred == y) if self.loss == 'ce' else torch.ones_like(y).bool()
        loss = -1e10 * torch.ones_like(acc).float()

        # If 'l2' loss, precompute the clean encodings if you'd like
        x_enc_clean = None
        if self.loss == 'l2':
            x_enc_clean = x_enc.detach().clone()

        if self.verbose and self.logger is not None:
            self.logger.info(f'Running AutoPGD with norm={self.norm}, eps={self.eps}, loss={self.loss}...')
            self.logger.info(f'Initial accuracy: {acc.float().mean():.2%}')

        # Possibly do large-eps scheduling
        if self.use_largereps and self.norm == 'L1':
            # Example approach from your snippet (for L1 only). 
            # If you need the same for L2/L∞, adapt similarly:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [math.ceil(0.3*self.n_iter_orig), math.ceil(0.3*self.n_iter_orig),
                     self.n_iter_orig - 2*math.ceil(0.3*self.n_iter_orig)]
        else:
            epss = [self.eps_orig]
            iters = [self.n_iter_orig]

        # ---------------------------
        # Multi-Restart logic
        # ---------------------------
        start_time = time.time()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        if not best_loss:
            # Standard approach: we return the "first successful adversarial" example
            # for each sample across restarts
            for rst in range(self.n_restarts):
                # attack only those still robust
                ind_to_fool = (acc == True).nonzero().squeeze()
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    x_init_to_fool = x_init[ind_to_fool] if (x_init is not None) else None
                    if self.loss == 'l2':
                        x_enc_to_fool = x_enc_clean[ind_to_fool] if (x_enc_clean is not None) else None
                    else:
                        x_enc_to_fool = None

                    # Possibly large-eps scheduling
                    if len(epss) > 1 and self.norm == 'L1':
                        # Decreasing eps in multiple stages
                        # You can adapt for L2 or Linf if you want
                        x_best, acc_curr, loss_curr, adv_curr = self.decr_eps_pgd(
                            x_to_fool, y_to_fool, epss, iters, x_init=x_init_to_fool, x_enc=x_enc_to_fool
                        )
                    else:
                        # normal single-run with self.n_iter
                        x_best, acc_curr, loss_curr, adv_curr = self.attack_single_run(
                            x_to_fool, y_to_fool, x_init=x_init_to_fool, x_enc=x_enc_to_fool
                        )

                    # Merge results back
                    ind_fail = (acc_curr == 0).nonzero().squeeze()  # those newly fooled
                    acc[ind_to_fool[ind_fail]] = False
                    adv[ind_to_fool[ind_fail]] = adv_curr[ind_fail].clone()

                    if self.verbose and self.logger is not None:
                        self.logger.info(f'restart {rst} - robust accuracy: {acc.float().mean():.2%} '
                                         f'- time: {time.time() - start_time:.1f}s')

            return adv

        else:
            # "best_loss" mode: keep the points that yield the highest final loss
            adv_best = x.clone()
            loss_best = -1e10 * torch.ones(x.shape[0], device=self.device)
            for rst in range(self.n_restarts):
                if len(epss) > 1 and self.norm == 'L1':
                    x_best, acc_curr, loss_curr, x_best_adv = self.decr_eps_pgd(
                        x, y, epss, iters, x_init=x_init, x_enc=x_enc_clean
                    )
                else:
                    x_best, acc_curr, loss_curr, x_best_adv = self.attack_single_run(
                        x, y, x_init=x_init, x_enc=x_enc_clean
                    )

                # pick which improved
                improved = (loss_curr > loss_best)
                adv_best[improved] = x_best[improved].clone()
                loss_best[improved] = loss_curr[improved]

                if self.verbose and self.logger is not None:
                    self.logger.info(f'Restart {rst} - sum of best loss: {loss_best.sum():.5f}')

            return adv_best

    # ----------------------------------------------------------------
    # Large-eps scheduling for L1 (optional). 
    # Called if self.use_largereps is True and norm=='L1'.
    # ----------------------------------------------------------------
    def decr_eps_pgd(self, x, y, epss, iters, x_init=None, x_enc=None):
        """
        Decreases epsilon in steps: 
          epss: list of eps values
          iters: list of #iters for each step
        Then merges the final results.
        """
        assert len(epss) == len(iters)
        assert self.norm == 'L1'
        self.use_rs = False  # if false, skip random init
        if x_init is not None:
            x_adv_curr = x_init.clone()
        else:
            # start with random init on the largest eps
            t = torch.randn_like(x)
            delta = L1_projection(x, t, epss[0])
            x_adv_curr = (x + t + delta).clamp(0., 1.)

        for (eps_i, niter_i) in zip(epss, iters):
            # override the attack's iteration count & eps
            old_iter = self.n_iter
            old_eps = self.eps

            self.n_iter = niter_i
            self.eps = eps_i

            # run single-run
            x_best, acc, loss_best, x_best_adv = self.attack_single_run(
                x, y, x_init=x_adv_curr, x_enc=x_enc
            )

            # for the next step, use final from this step
            x_adv_curr = x_best.clone()

            # restore iteration count & eps
            self.n_iter = old_iter
            self.eps = old_eps

        return x_best, acc, loss_best, x_best_adv

def L2_norm(x, keepdim=False):
    """
    Returns the L2 norm of x (over all dimensions except batch).
    """
    norm = x.view(x.shape[0], -1).norm(p=2, dim=1)
    if keepdim:
        return norm.view(-1, *[1]*(x.dim()-1))
    else:
        return norm

def L1_norm(x, keepdim=False):
    """
    Returns the L1 norm of x (over all dimensions except batch).
    """
    norm = x.view(x.shape[0], -1).abs().sum(dim=1)
    if keepdim:
        return norm.view(-1, *[1]*(x.dim()-1))
    else:
        return norm

def L1_projection(x_orig, delta, eps):
    """
    Projects perturbation 'delta' onto the L1-ball of radius eps around x_orig.
    The usual approach is to solve the linearized L1 projection. 
    """
    # One simple approach: if the L1 norm of delta exceeds eps, we rescale.
    # But your code might do something more advanced. For brevity:
    mask = (L1_norm(delta) > eps)
    if mask.any():
        # Rescale only those that exceed
        factor = eps / (L1_norm(delta[mask]) + 1e-12)
        factor = factor.view(-1, *[1]*(delta.dim()-1))
        delta[mask] = delta[mask] * factor
    return delta

def random_start_linf(x, eps, clamp_min, clamp_max):
    """Random start in L∞ ball."""
    delta = 2.0 * torch.rand_like(x) - 1.0
    delta = delta * eps
    x_adv = (x + delta).clamp(clamp_min, clamp_max)
    return x_adv

def random_start_l2(x, eps, clamp_min, clamp_max):
    """Random start in L2 ball."""
    # Sample normal and project to radius eps
    delta = torch.randn_like(x)
    # Compute L2 norm per sample
    delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
    # Scale to have norm = eps
    delta = delta * (eps / (delta_norm + 1e-12)).view(-1, *[1]*(x.ndim - 1))
    x_adv = (x + delta).clamp(clamp_min, clamp_max)
    return x_adv

def random_start_l1(x, eps, clamp_min, clamp_max):
    """Random start in L1 ball."""
    # Sample normal
    delta = torch.randn_like(x)
    # Project onto L1 ball
    delta = project_l1(delta, eps)
    x_adv = (x + delta).clamp(clamp_min, clamp_max)
    return x_adv

def project_linf(x_adv, x_orig, eps):
    """Project x_adv into the L∞ ball of radius eps around x_orig."""
    return torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

def project_l2(x_adv, x_orig, eps):
    """Project x_adv into the L2 ball of radius eps around x_orig."""
    delta = x_adv - x_orig
    mask = delta.view(delta.size(0), -1)
    norm = mask.norm(p=2, dim=1, keepdim=True)
    # if norm > eps, rescale
    factor = torch.clamp(eps / (norm + 1e-12), max=1.0)
    delta = delta * factor.view(-1, *[1]*(delta.ndim - 1))
    return x_orig + delta

def project_l1(delta, eps):
    """
    Project 'delta' onto the L1 ball of radius eps.
    Simplest approach: if ||delta||_1 > eps, rescale.
    """
    flat = delta.view(delta.size(0), -1)
    norm = flat.abs().sum(dim=1, keepdim=True)
    factor = (eps / (norm + 1e-12)).clamp(max=1.0)
    flat = flat * factor
    return flat.view_as(delta)