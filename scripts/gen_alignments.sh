python -m tools.tsne_ranked_all_triplets \
  --dataset_root /data/datasets \
  --pretrain_weights ./ckpts/pretrained_weights_flash_atten_image_patchs.pt \
  --use_flash_attention
