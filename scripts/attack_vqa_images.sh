torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) attack_vqa_images.py \
  --val_json ./datasets/VQA2/val_data.json \
  --image_root /home/user/datasets/VQA2 \
  --pretrain_weights ./ckpts/pretrained_weights_flash_atten.pt