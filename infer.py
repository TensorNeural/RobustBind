import torch
import models.PointBind_models as models
from imagebind.imagebind_model import ModalityType
import argparse
import json
import pickle
from tqdm import tqdm
from data.process_data import EvalVisionDataset
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import logging
import random
import numpy as np
import os
import pickle
from utils.utils import set_env, gen_label, loss_fun, load_centre_embeddings
from model import UniBind

logger = logging.getLogger(__name__)

def evaluate(args, model, val_data_loader, device):
    """Evaluates accuracy batch-wise with detailed logging for each stage."""
    model.eval()
    logger.info("Loading centre embeddings...")

    # Load and normalize centre embeddings
    centre_embeddings, centre_labels = load_centre_embeddings(args.centre_embeddings_path, device)
    centre_embeddings = centre_embeddings.to(device)
    centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)

    # Save centre embeddings
    torch.save(centre_embeddings.cpu(), os.path.join(args.output_dir, "centre_embeddings.pt"))
    torch.save(centre_labels, os.path.join(args.output_dir, "centre_labels.pt"))

    logger.info("Starting batch processing...")

    acc, total_samples = 0, 0
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each batch separately
    for batch_idx, batch in enumerate(tqdm(val_data_loader)):
        with torch.no_grad():
            logger.info(f"[Batch {batch_idx}] Generating visual embeddings...")
            embeddings = model.encode_vision_with_mlp(batch["inputs"]).to(device)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            # Save embeddings and labels for this batch
            torch.save(embeddings.cpu(), os.path.join(args.output_dir, f"visual_embeddings_{batch_idx}.pt"))
            torch.save(batch["labels"], os.path.join(args.output_dir, f"visual_labels_{batch_idx}.pt"))

            logger.info(f"[Batch {batch_idx}] Computing similarity with centre embeddings...")
            logic = (embeddings @ centre_embeddings.t()).softmax(dim=-1)

            logger.info(f"[Batch {batch_idx}] Classifying and calculating accuracy...")
            batch_correct = 0
            batch_size = logic.shape[0]

            for i in range(batch_size):
                predicted_index = logic[i].argmax().item()  # Get index of highest probability
                predicted_label = centre_labels[predicted_index]
                if batch["labels"][i] == predicted_label:
                    batch_correct += 1

            batch_accuracy = batch_correct / batch_size
            acc += batch_correct
            total_samples += batch_size

            logger.info(f"[Batch {batch_idx}] Accuracy: {batch_correct}/{batch_size} ({batch_accuracy:.4f})")

            # Free memory after each batch
            del embeddings, logic
            torch.cuda.empty_cache()

    final_acc = acc / total_samples
    logger.info(f"Final Accuracy: {final_acc:.4f}")
    return final_acc

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")

    parser.add_argument("--test_dataset_dir", type=str, default='', required=True)
    parser.add_argument("--test_data_path", type=str, default='', required=True)
    parser.add_argument("--centre_embeddings_path", type=str, default='', required=True)
    parser.add_argument("--output_dir", type=str, default='', required=True)
    parser.add_argument("--pretrain_weights", type=str, default='', required=True)
    parser.add_argument("--modality", type=str, default='vision', required=True)
    parser.add_argument("--val_batch_size", type=int, default=8, required=True)
    parser.add_argument("--num_workers", type=int, default=0, required=True)
    parser.add_argument("--seed", type=int, default=1234, required=True)

    args = parser.parse_args()
    log_name = args.modality + '_infer'
    set_env(args, log_name)

    val_data = EvalVisionDataset(args, device, infer_type="test")
    val_sampler = SequentialSampler(val_data)
    val_data_reader = DataLoader(dataset=val_data, sampler=val_sampler, num_workers=args.num_workers,
                                batch_size=args.val_batch_size, collate_fn=val_data.Collector, drop_last=False)
    
    model = UniBind(args, use_flash_attention=True)
    model.to(device)
    acc = evaluate(args, model, val_data_reader, device)
    logger.info(f"top 1 Acc: {acc}")