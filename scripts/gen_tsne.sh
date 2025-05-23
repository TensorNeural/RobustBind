python ./tool/tsne_visualization.py \
  --dataset_root /home/user/datasets \
  --val_json_image ./datasets/Places365/val_data.json \
  --val_json_event ./datasets/N-ImageNet-1K/val_data.json \
  --val_json_audio ./datasets/ESC-50/val_data.json \
  --center_emb_image ./centre_embs/image_p365_center_embeddings.pkl \
  --center_emb_event ./centre_embs/event_nin_center_embeddings.pkl \
  --center_emb_audio ./centre_embs/audio_esc_center_embeddings.pkl \
  --pretrain_weights ./ckpts/pretrained_weights_flash_atten.pt \
  --use_flash_attention
