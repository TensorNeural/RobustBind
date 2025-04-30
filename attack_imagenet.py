# import argparse
# import logging
# import os
# from datetime import datetime

# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist

# from attack_bind import AttackEval, AttackEvalRunner
# from utils.data_transform import IMAGE_TRANSFORM, IMAGE_MEAN, IMAGE_STD
# from datasets.datasets import ImageNetDataset
# from attack import APGDAttack, AttackModel
# from utils.utils import load_centre_embeddings

# # --------------------
# # Logger Formatter
# # --------------------
# class RelativePathFormatter(logging.Formatter):
#     def __init__(self, rank, fmt=None, datefmt=None, style='%', validate=True):
#         super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
#         self.rank = rank

#     def format(self, record):
#         run_dir = os.getcwd()
#         abs_path = os.path.abspath(record.pathname)
#         try:
#             record.relativepath = os.path.relpath(abs_path, run_dir)
#             record.rank = self.rank
#         except ValueError:
#             record.relativepath = record.pathname
#         return super().format(record)

# # --------------------
# # ImageNetEval subclass
# # --------------------
# class ImageNetAttackEval(AttackEval):
#     def __init__(
#         self,
#         device,
#         dataset_root,
#         data_json_path,
#         centre_embeddings_path="./centre_embs/image_in_center_embeddings.pkl",
#         batch_size=25,
#         max_samples=50000,
#         epsilons=[2 / 255],
#         modality="image",
#     ):
#         dataset_name = "ImageNet_1K"
#         centre_embeddings_path = "./centre_embs/image_in_center_embeddings.pkl"

#         raw_emb, raw_lbls = load_centre_embeddings(centre_embeddings_path, device)
#         raw_emb = raw_emb / raw_emb.norm(dim=-1, keepdim=True)
#         unique_lbls = sorted(list(set(raw_lbls)))
#         lbl_to_idx = {l: i for i, l in enumerate(unique_lbls)}
#         idx_to_lbl = {v: k for k, v in lbl_to_idx.items()}

#         ds = ImageNetDataset(
#             dataset_root=dataset_root,
#             data_json_path=data_json_path,
#             transform=IMAGE_TRANSFORM,
#             max_samples=max_samples,
#             label_to_index=lbl_to_idx,
#             index_to_label=idx_to_lbl,
#         )
#         super().__init__(
#             dataset=ds,
#             centre_labels=unique_lbls,
#             centre_embeddings=raw_emb,
#             dataset_name=dataset_name,
#             batch_size=batch_size,
#             max_samples=max_samples,
#             epsilons=epsilons,
#             modality=modality,
#             mean=IMAGE_MEAN,
#             std=IMAGE_STD,
#         )
    
#     def get_attack(self, model: AttackModel, eps):
#         return APGDAttack(
#             model=model,
#             norm=self.norm.lower(),
#             eps=eps,
#             n_iter=100,
#             n_restarts=1,
#             loss="ce",
#             device=self.device,
#             logger=self.logger,
#             verbose=False,
#         )

# # --------------------
# # Main
# # --------------------
# def main():
#     parser = argparse.ArgumentParser()

#     # Dataset group
#     parser.add_argument("--dataset_root", type=str, default="/home/user/datasets/ImageNet-1K")
#     parser.add_argument("--data_json_path", type=str, default="./datasets/ImageNet-1K/val_data_5000.json")
#     parser.add_argument("--center_label_to_wordnet_path", type=str, default="./datasets/ImageNet-1K/center_to_wordnet.json")

#     # Output group
#     parser.add_argument("--dataset_adversary_root", type=str, default="/home/user/datasets/ImageNet-1K/val_5000_adv")
#     parser.add_argument("--metadata_output_root", type=str, default="./datasets/ImageNet-1K/")
#     parser.add_argument("--metadata_prefix", type=str, default="val_5000_adv")

#     # Attack group
#     parser.add_argument("--batch_size", type=int, default=160)
#     parser.add_argument("--max_samples", type=int, default=5000)
#     parser.add_argument("--epsilons", nargs="+", type=float, default=[2/255, 4/255])
#     parser.add_argument("--norm", type=str, default="Linf")
#     parser.add_argument("--version", type=str, default="custom")
#     parser.add_argument("--modality", type=str, default="image")
#     parser.add_argument("--log_root", type=str, default="./logs")
#     parser.add_argument("--pretrain_weights", type=str, default="./ckpts/pretrained_weights_flash_atten.pt")

#     args = parser.parse_args()

#     try:
#         # Multi-GPU setup
#         local_rank = int(os.environ.get("LOCAL_RANK", "0"))
#         torch.cuda.set_device(local_rank)
#         device = torch.device("cuda", local_rank)

#         dist.init_process_group(backend="nccl")
#         rank = dist.get_rank()

#         # Logging
#         os.makedirs(args.log_root, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         log_filename = f"rank{rank}_attack_{timestamp}.log"

#         formatter = RelativePathFormatter(rank=rank, fmt='[RANK %(rank)d] %(asctime)s - %(relativepath)s:%(lineno)d - [%(levelname)s] - %(message)s')

#         file_handler = logging.FileHandler(os.path.join(args.log_root, log_filename), mode='w')
#         file_handler.setLevel(logging.INFO)
#         file_handler.setFormatter(formatter)

#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
#         console_handler.setFormatter(formatter)

#         logger = logging.getLogger(__name__)
#         logger.setLevel(logging.INFO)
#         logger.handlers = [console_handler, file_handler]

#         logger.info(f"Rank: {rank}, Local Rank: {local_rank}")
#         logger.info(f"Using device: {device}")

#         cudnn.benchmark = True

#         # Create Eval
#         eval = ImageNetAttackEval(
#             device=device,
#             dataset_root=args.dataset_root,
#             data_json_path=args.data_json_path,
#             batch_size=args.batch_size,
#             max_samples=args.max_samples,
#             epsilons=args.epsilons,
#             modality=args.modality,
#         )

#         runner = AttackEvalRunner(
#             logger=logger,
#             dataset_adversary_root=args.dataset_adversary_root,
#             metadata_output_root=args.metadata_output_root,
#             metadata_prefix=args.metadata_prefix,
#             pretrain_weights_path=args.pretrain_weights,
#             device=device,
#             debug=False,
#         )

#         runner.run(
#             eval=eval,
#         )

#     finally:
#         dist.destroy_process_group()

# if __name__ == "__main__":
#     main()
