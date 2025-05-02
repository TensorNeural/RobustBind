torchrun \
  --nproc_per_node=$(nvidia-smi -L | wc -l) \
  train_robust.py \
  --output_dir output \
  --dataset_root /home/user/datasets/ImageNet-1K \
  --train_json ./datasets/ImageNet-1K/train_data.json \
  --val_json ./datasets/ImageNet-1K/val_data.json \
  --pretrain_weights ./ckpts/pretrained_weights_flash_atten.pt \
  --use_flash_attention \
  --center_emb ./centre_embs/image_in_center_embeddings.pkl \
  --train_batch_size 70 \
  --val_batch_size 70 \
  --num_workers 2 \
  --train_max_samples None \
  --val_max_samples 3000 \
  --train_attack_loss l2 \
  --val_attack_loss ce \
  --train_loss l2 \
  --epsilon 4/255 \
  --tensorboard_data_dir tensorboard
