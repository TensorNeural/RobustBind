# ImageNet-1K
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#   --test_dataset_dir /home/user/datasets/ImageNet-1K \
#   --test_data_path ./datasets/ImageNet-1K/val_data.json \
#   --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
#   --pretrain_weights ./ckpts/pretrained_weights.pt \
#   --output_dir ./outputs/val_data_zero_shot \
#   --modality image \
#   --val_batch_size 2000 \
#   --num_workers 4 \
#   --seed 1234

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --test_dataset_dir /home/user/datasets/ImageNet-1K/val_adv \
  --test_data_path ./datasets/ImageNet-1K/val_adv_eps0.json \
  --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
  --pretrain_weights ./ckpts/pretrained_weights.pt \
  --output_dir ./outputs/val_data_zero_shot \
  --modality image \
  --val_batch_size 2000 \
  --num_workers 4 \
  --seed 1234

# CUDA_VISIBLE_DEVICES=0 python infer.py \
#   --test_dataset_dir /home/user/datasets/ImageNet-1K/val_adv \
#   --test_data_path ./datasets/ImageNet-1K/val_adv_eps2.json \
#   --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
#   --pretrain_weights ./ckpts/pretrained_weights.pt \
#   --output_dir ./outputs/val_data_zero_shot \
#   --modality image \
#   --val_batch_size 2000 \
#   --num_workers 4 \
#   --seed 1234

# CUDA_VISIBLE_DEVICES=0 python infer.py \
#   --test_dataset_dir /home/user/datasets/ImageNet-1K/val_adv \
#   --test_data_path ./datasets/ImageNet-1K/val_adv_eps4.json \
#   --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
#   --pretrain_weights ./ckpts/pretrained_weights.pt \
#   --output_dir ./outputs/val_data_zero_shot \
#   --modality image \
#   --val_batch_size 2000 \
#   --num_workers 4 \
#   --seed 1234