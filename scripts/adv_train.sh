torchrun \
  --nproc_per_node=$(nvidia-smi -L | wc -l)\
  train_robust.py