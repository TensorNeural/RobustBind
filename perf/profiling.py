import os
import inspect
import time
import torch

class GpuMemoryTracker:
    def __init__(self, logger, label=None, device=None, message=None):
        self.logger = logger
        self.label = label
        self.message = message
        self.device = device or torch.device("cuda")

        # Capture file & line for reference
        frame = inspect.currentframe()
        outer_frame = inspect.getouterframes(frame)[1]
        self.filename = os.path.basename(outer_frame.filename)
        self.lineno = outer_frame.lineno

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_allocated = torch.cuda.memory_allocated(self.device)
        self.start_reserved = torch.cuda.memory_reserved(self.device)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_allocated = torch.cuda.memory_allocated(self.device)
        end_reserved = torch.cuda.memory_reserved(self.device)
        duration = time.time() - self.start_time
        delta_allocated = end_allocated - self.start_allocated
        delta_reserved = end_reserved - self.start_reserved

        parts = []
        parts.append("[{}:{}]".format(self.filename, self.lineno))
        if self.label:
            parts.append(self.label)
        parts.append("Duration: {:.2f}s".format(duration))
        parts.append("Allocated Δ: {}".format(self._format_bytes(delta_allocated)))
        parts.append("Reserved Δ: {}".format(self._format_bytes(delta_reserved)))

        if self.message:
            parts.append(self.message)

        self.logger.debug(" | ".join(parts))

    def _format_bytes(self, num_bytes):
        abs_bytes = abs(num_bytes)
        if abs_bytes >= 1024 ** 3:
            value = num_bytes / (1024 ** 3)
            unit = "GB"
        elif abs_bytes >= 1024 ** 2:
            value = num_bytes / (1024 ** 2)
            unit = "MB"
        elif abs_bytes >= 1024:
            value = num_bytes / 1024
            unit = "KB"
        else:
            value = num_bytes
            unit = "B"
        return "{:+.2f} {}".format(value, unit)

class ProfileModelMemory:
    def __init__(self, model, logger, label="ProfileModelMemory", device=None):
        self.model = model
        self.logger = logger
        self.label = label
        self.device = device or torch.device("cuda")
        self.hooks = []
        self.ctx_map = {}

        # Create a top-level GpuMemoryTracker (not entered yet)
        self.top_tracker = GpuMemoryTracker(
            logger=self.logger,
            label=self.label,
            device=self.device,
            message="(Top-level model forward)"
        )

    def __enter__(self):
        # Manually enter the top-level tracker
        self.top_tracker.__enter__()
        # Register hooks on all modules (including containers)
        # self._register_hooks(self.model, prefix="", depth=0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

        # Exit the top-level tracker
        self.top_tracker.__exit__(exc_type, exc_val, exc_tb)

    def _register_hooks(self, module, prefix="", depth=0):
        """
        Recursively register forward_pre_hook and forward_hook on every module
        (including containers). Indentation is based on 'depth'.
        """
        # 1) Register hooks on this module
        indent = "  " * depth  # 2 spaces per depth level

        module_name = prefix if prefix else module.__class__.__name__
        # forward_pre_hook
        pre_hook_handle = module.register_forward_pre_hook(
            self._hook_pre(module_name, module, indent)
        )
        self.hooks.append(pre_hook_handle)

        # forward_hook
        post_hook_handle = module.register_forward_hook(
            self._hook_post(module_name, module, indent)
        )
        self.hooks.append(post_hook_handle)

        # 2) Recurse into children
        for child_name, child_module in module.named_children():
            full_name = f"{module_name}.{child_name}"
            self._register_hooks(child_module, full_name, depth + 1)

    def _hook_pre(self, name, module, indent):
        """Called BEFORE module.forward()."""
        def inner_pre_hook(module, inputs):
            input_shapes = [
                inp.shape for inp in inputs if hasattr(inp, 'shape')
            ]
            msg = "Input shapes: {}".format(input_shapes)
            label_str = "{}{}::Pre::{}({})".format(
                indent, self.label, name, module.__class__.__name__
            )

            tracker = GpuMemoryTracker(
                logger=self.logger,
                label=label_str,
                message=msg,
                device=self.device
            )
            tracker.__enter__()
            self.ctx_map[module] = tracker
        return inner_pre_hook

    def _hook_post(self, name, module, indent):
        """Called AFTER module.forward()."""
        def inner_post_hook(module, inputs, output):
            tracker = self.ctx_map.pop(module, None)
            if tracker is not None:
                # === Set post-forward label
                tracker.label = "{}{}::Post::{}({})".format(
                    indent, self.label, name, module.__class__.__name__
                )

                # === Compute memory size in bytes ===
                def tensor_nbytes(t):
                    return t.numel() * t.element_size()

                def compute_memory_stats(out):
                    grad_bytes = 0
                    non_grad_bytes = 0
                    if isinstance(out, torch.Tensor):
                        if out.requires_grad:
                            grad_bytes += tensor_nbytes(out)
                        else:
                            non_grad_bytes += tensor_nbytes(out)
                    elif isinstance(out, (list, tuple)):
                        for o in out:
                            if isinstance(o, torch.Tensor):
                                if o.requires_grad:
                                    grad_bytes += tensor_nbytes(o)
                                else:
                                    non_grad_bytes += tensor_nbytes(o)
                    return grad_bytes, non_grad_bytes

                def any_input_requires_grad(inputs):
                    if isinstance(inputs, torch.Tensor):
                        return inputs.requires_grad
                    elif isinstance(inputs, (list, tuple)):
                        return any(
                            isinstance(i, torch.Tensor) and i.requires_grad for i in inputs
                        )
                    return False

                input_grad = any_input_requires_grad(inputs)
                grad_bytes, non_grad_bytes = compute_memory_stats(output)

                # === Format and append to message ===
                grad_str = tracker._format_bytes(grad_bytes)
                nongrad_str = tracker._format_bytes(non_grad_bytes)

                tracker.message = (tracker.message or "")
                tracker.message += (
                    f" | input_requires_grad: {input_grad}"
                    f" | Grad Output: {grad_str}"
                    f" | Non-Grad Output: {nongrad_str}"
                )

                if hasattr(output, 'shape'):
                    tracker.message += f" | Output shape: {list(output.shape)}"

                tracker.__exit__(None, None, None)
        return inner_post_hook

class ProfileModelGradient:
    def __init__(self, model, logger, label="ProfileModelGradient", device=None):
        self.model = model
        self.logger = logger
        self.label = label
        self.device = device or torch.device("cuda")
        self.hooks = []

    def __enter__(self):
        # self._register_hooks(self.model, prefix="", depth=0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _register_hooks(self, module, prefix="", depth=0):
        module_name = prefix if prefix else module.__class__.__name__

        def _backward_hook(mod, grad_input, grad_output):
            # No need to check gradients here; will only be called if gradients flow
            pass

        def _forward_hook(mod, input, output):
            input_requires_grad = any(
                isinstance(i, torch.Tensor) and i.requires_grad for i in input
            )
            if isinstance(output, torch.Tensor):
                output_requires_grad = output.requires_grad
            elif isinstance(output, (list, tuple)):
                output_requires_grad = any(
                    isinstance(o, torch.Tensor) and o.requires_grad for o in output
                )
            else:
                output_requires_grad = False

            msg = (
                f"input_requires_grad: {input_requires_grad} | "
                f"output_requires_grad: {output_requires_grad}"
            )
            self.logger.info(f"{self.label}::{module_name} | {msg}")

        # Only use forward hook to detect requirement status
        fwd_handle = module.register_forward_hook(_forward_hook)
        self.hooks.append(fwd_handle)

        for child_name, child_module in module.named_children():
            full_name = f"{module_name}.{child_name}"
            self._register_hooks(child_module, prefix=full_name, depth=depth + 1)
