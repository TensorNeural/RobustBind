torchrun \
  --nproc_per_node=$(nvidia-smi -L | wc -l)\
  attack_imagenet.py