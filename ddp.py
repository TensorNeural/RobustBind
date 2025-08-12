from torch.nn.parallel import DistributedDataParallel as DDP

class ProxyDDP(DDP):
    """DistributedDataParallel wrapper that forwards unknown attribute access to the wrapped module.
    Allows calling custom model methods directly on the DDP wrapper (e.g., ddp.enable_lora())."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)