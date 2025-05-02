# import os
# import abc
# import json
# import time
# from concurrent.futures import ThreadPoolExecutor

# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader, Subset
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torchvision.utils as vutils

# from utils.utils import load_centre_embeddings
# from model import UniBindModel
# from transform import normalize_inplace, unnormalize_inplace
# from attack import AttackModel, APGDAttack, PGDAttack, run_standard_evaluation

# # --------------------
# # Eval Abstract Class
# # --------------------

# class AttackEval(abc.ABC):
#     def __init__(
#         self,
#         dataset,
#         centre_labels,
#         centre_embeddings,
#         dataset_name="custom_dataset",
#         batch_size=128,
#         max_samples=50000,
#         epsilons=[2/255, 4/255],
#         modality="image",
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ):
#         self.dataset = dataset
#         self.centre_labels = centre_labels
#         self.centre_embeddings = centre_embeddings
#         self.dataset_name = dataset_name
#         self.batch_size = batch_size
#         self.max_samples = max_samples
#         self.epsilons = epsilons
#         self.mean = mean
#         self.std = std
#         self.modality = modality

#     @abc.abstractmethod
#     def get_attack(self, model: AttackModel, eps):
#         """Abstract: get the attack object."""
#         pass

# # --------------------
# # AutoAttackRunner
# # --------------------

# class AttackEvalRunner:
#     def __init__(
#         self,
#         logger,
#         dataset_adversary_root: str,
#         metadata_output_root: str,
#         metadata_prefix: str,
#         pretrain_weights_path: str,
#         device=None,
#         debug=False,
#     ):
#         self.logger = logger
#         self.device = device
#         self.dataset_adversary_root = dataset_adversary_root
#         self.metadata_output_root = metadata_output_root
#         self.metadata_prefix = metadata_prefix
#         self.pretrain_weights_path = pretrain_weights_path
#         self.debug = debug
#         self.model = None

#     @torch.no_grad()
#     def run(self, eval, num_workers=4):
#         dataset = eval.dataset
#         batch_size = eval.batch_size
#         epsilons = eval.epsilons
#         max_samples = eval.max_samples
#         mean = eval.mean
#         std = eval.std
#         modality = eval.modality

#         rank = dist.get_rank() if dist.is_initialized() else 0
#         is_main = not dist.is_initialized() or rank == 0

#         if max_samples is not None and max_samples < len(dataset):
#             indices = torch.arange(max_samples) if self.debug else torch.randperm(len(dataset))[:max_samples]
#             dataset = Subset(dataset, indices)

#         sampler = DistributedSampler(dataset, shuffle=False)
#         loader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             sampler=sampler,
#             num_workers=num_workers,
#             pin_memory=True,
#             persistent_workers=True,
#             prefetch_factor=4,
#         )

#         if is_main:
#             os.makedirs(self.dataset_adversary_root, exist_ok=True)
#             os.makedirs(self.metadata_output_root, exist_ok=True)

#         if self.model is None:
#             self._init_model(modality, eval.centre_embeddings, centre_labels_indices)

#         mean_t = torch.tensor(mean, device=self.device).view(1, -1, 1, 1)
#         std_t = torch.tensor(std, device=self.device).view(1, -1, 1, 1)

#         attack_model = AttackModel(self.model, mean_t, std_t)

#         for eps in epsilons:
#             eps_int = int(eps * 255)
#             self.logger.info(f"Running attack for eps={eps_int}/255")
#             attack = eval.get_attack(attack_model, eps)

#             rel_eps_dir = f"eps{eps_int}"
#             abs_eps_dir = os.path.join(self.dataset_adversary_root, rel_eps_dir)
#             if is_main:
#                 os.makedirs(abs_eps_dir, exist_ok=True)

#             adv_metadata_eps = []

#             for batch_idx, (inp, lbl) in enumerate(loader):
#                 batch_start_time = time.time()

#                 inp = inp.to(self.device, non_blocking=True)
#                 lbl = lbl.to(self.device, non_blocking=True)

#                 self.logger.info(f"Attacking batch {batch_idx+1}/{len(loader)}, batch size={inp.size(0)}")

#                 inp_unorm = inp.clone()
#                 unnormalize_inplace(inp_unorm, mean, std)

#                 adv_inp = run_standard_evaluation(
#                     logger=self.logger,
#                     device=self.device,
#                     model=attack_model,
#                     attack=attack,
#                     x_orig=inp_unorm,
#                     y_orig=lbl,
#                     batch_size=batch_size,
#                     return_labels=False,
#                 )

#                 normalize_inplace(adv_inp, mean, std)

#                 outpath = os.path.join(abs_eps_dir, f"rank{rank}_eps{eps_int}_{batch_idx}.pth")
#                 torch.save(
#                     {
#                         "adv_complete": adv_inp,
#                         "x_test": inp,
#                     },
#                     outpath,
#                 )

#                 if is_main:
#                     self._parallel_save_images(adv_inp, abs_eps_dir, batch_idx)

#                 batch_end_time = time.time()
#                 self.logger.info(f"Batch {batch_idx+1} done in {batch_end_time - batch_start_time:.2f} seconds")
#                 torch.cuda.empty_cache()

#             if is_main:
#                 meta_filename = f"{self.metadata_prefix}_eps{eps_int}.json"
#                 meta_filepath = os.path.join(self.metadata_output_root, meta_filename)
#                 with open(meta_filepath, "w") as f:
#                     json.dump(adv_metadata_eps, f, indent=2)

#                 self.logger.info(f"Metadata for eps={eps_int} saved to {meta_filepath}")

#         self.logger.info("Attack completed for all epsilon values.")
    
#     def _init_model(self, modality, centre_embeddings, centre_labels_indices, label_to_index, index_to_label):
#         self.logger.info(f"Initializing UniBindModel for modality: {modality}")

#         model = UniBindModel(
#             device=self.device,
#             pretrain_weights=self.pretrain_weights_path,
#             modality=modality,
#             centre_embeddings=centre_embeddings,
#             centre_labels=centre_labels_indices,
#             label_to_index=label_to_index,
#             index_to_label=index_to_label,
#             use_flash_attention=True,
#             logger=self.logger,
#             fine_tuned_weights=None,
#         ).to(self.device)

#         self.model = DDP(model, device_ids=[self.device.index], output_device=self.device.index)
#         self.model.eval()
#         self.logger.info("Model initialized successfully.")

#     def _parallel_save_images(self, adv_examples, eps_dir, batch_idx):
#         adv_cpu = adv_examples.cpu()

#         def save_image(tensor_img, path):
#             vutils.save_image(tensor_img, path)

#         futures = []
#         with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#             for i in range(adv_cpu.size(0)):
#                 img_filename = f"batch{batch_idx}_idx{i}.png"
#                 img_save_path = os.path.join(eps_dir, img_filename)
#                 futures.append(executor.submit(save_image, adv_cpu[i], img_save_path))

#         for f in futures:
#             f.result()
