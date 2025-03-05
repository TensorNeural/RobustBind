cd ../

# Places365
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --test_dataset_dir /home/user/datasets/places365 \
  --test_data_path ./datasets/Places365/val_data.json \
  --centre_embeddings_path ./centre_embs/image_p365_center_embeddings.pkl \
  --pretrain_weights ./ckpts/pretrained_weights.pt \
  --output_dir ./outputs/places365_val_data_zero_shot \
  --modality image \
  --val_batch_size 2000 \
  --num_workers 4 \
  --seed 1234