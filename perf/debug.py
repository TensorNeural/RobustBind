def register_forward_hooks(model, logger):
    def make_hook(name):
        def hook(module, input, output):
            input_requires_grad = [
                i.requires_grad if isinstance(i, torch.Tensor) else 'NonTensor'
                for i in input
            ]
            logger.info(f"[FORWARD] {name}.{module} | input_requires_grad: {input_requires_grad}")
        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(make_hook(name))

def register_backward_hooks(model, logger):
    def make_hook(name):
        def hook(module, grad_input, grad_output):
            logger.info(f"[BACKWARD] {name}.{module} received grad_input: {[g.norm().item() if g is not None else None for g in grad_input]}")
        return hook

    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            module.register_full_backward_hook(make_hook(name))

def log_grad(model, logger):
    for name, param in model.named_parameters():
        if 'lora_down' in name or 'lora_up' in name:
            if param.requires_grad:
                weight_norm = param.data.norm().item()
                if param.grad is None:
                    logger.debug(f"[LoRA Grad] {name} ❌ grad is None | weight norm: {weight_norm:.6e}")
                else:
                    grad_norm = param.grad.detach().norm().item()
                    logger.debug(f"[LoRA GradNorm] {name} ✅ grad norm: {grad_norm:.6e} | weight norm: {weight_norm:.6e}")