# cd ../
# CUDA_VISIBLE_DEVICES=0 python infer.py --test_dataset_dir ./datasets/xx/test_dataset \
#  --test_data_path ./datasets/xx/test_data.json \
#  --centre_embeddings_path ./centre_embs/xx_center_embeddings.pkl \
#  --pretrain_weights ./ckpts/pretrained_weights.pt \
#  --output_dir ./outputs/image_xx \
#  --modality image \
#  --val_batch_size 16 \
#  --num_workers 0 \
#  --seed 1234 \
# cd ../
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#   --test_dataset_dir /home/user/datasets/ImageNet-1K \
#   --test_data_path ./datasets/ImageNet-1K/crane_test_data.json \
#   --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
#   --pretrain_weights ./ckpts/pretrained_weights.pt \
#   --output_dir ./outputs/crane_zero_shot \
#   --modality image \
#   --val_batch_size 16 \
#   --num_workers 4 \
#   --seed 1234

cd ../
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --test_dataset_dir /home/user/datasets/ImageNet-1K \
  --test_data_path ./datasets/ImageNet-1K/val_data.json \
  --centre_embeddings_path ./centre_embs/image_in_center_embeddings.pkl \
  --pretrain_weights ./ckpts/pretrained_weights.pt \
  --output_dir ./outputs/val_data_zero_shot \
  --modality image \
  --val_batch_size 200 \
  --num_workers 4 \
  --seed 1234