import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from autoattack import AutoAttack
from imagebind.imagebind_model import ModalityType
from utils.utils import load_centre_embeddings
import numpy as np
from torch.utils.data import Subset
import torch_scatter

# Import UniBind Model
from model import UniBind
import models.PointBind_models as models  # Ensure dependencies are available

class ImageNetDataset(datasets.ImageFolder):
    """Custom dataset to return image paths along with images and labels."""
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image_path = self.samples[index][0]
        return image, label, image_path

def load_images_parallel(dataset_root, batch_size, max_samples=None):
    """Loads ImageNet dataset with image paths included and applies `max_samples` limit."""
    transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageNetDataset(root=dataset_root, transform=transform)
    class_to_index = dataset.class_to_idx

    # Apply max_samples limit if specified
    if max_samples is not None and max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:max_samples]  # Random subset
        # indices = torch.arange(max_samples)  # Sequential subset
        dataset = Subset(dataset, indices)  # Create a Subset of the dataset

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True,
        prefetch_factor=4, persistent_workers=True
    )

    return dataloader, class_to_index

def unnormalize_inplace(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    x.mul_(std_t).add_(mean_t).clamp_(0,1)
    return x

def normalize_inplace(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    x.sub_(mean_t).div_(std_t)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help="Root path to ImageNet dataset")
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilons', nargs='+', type=float, default=[2/255, 4/255],
                        help="List of epsilon values to test (space-separated)")
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--max_samples', type=int, default=50000, help="Limit the number of images processed")
    parser.add_argument('--log_path', type=str, default='./autoattack/examples/results/log_file.txt')
    parser.add_argument('--version', type=str, default='custom')
    parser.add_argument('--state-path', type=str, default=None)
    parser.add_argument("--pretrain_weights", type=str, default='./ckpts/pretrained_weights.pt')
    parser.add_argument("--modality", type=str, default='image')
    parser.add_argument("--centre_embeddings_path", type=str, default='./centre_embs/image_in_center_embeddings.pkl',
                        help="Path to center embeddings")
    parser.add_argument("--center_label_to_wordnet_path", type=str, 
                        default='./datasets/ImageNet-1K/center_to_wordnet.json', 
                        help="Path to center-label-to-WordNet mapping")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Step 1: Load dataset
    print("Loading dataset using torchvision ImageFolder...")
    dataloader, class_to_index = load_images_parallel(args.dataset_root, args.batch_size, args.max_samples)
    print("Dataset loaded successfully.")

    # Step 2: Load center label to WordNet mapping
    print("Loading center label to WordNet mapping...")
    with open(args.center_label_to_wordnet_path, "r") as f:
        center_label_to_wordnet = json.load(f)

    # Step 3: Load and process center embeddings
    print("Loading and processing center embeddings...")
    centre_embeddings, centre_labels = load_centre_embeddings(args.centre_embeddings_path, device)
    centre_embeddings = centre_embeddings.to(device, dtype=torch.bfloat16, non_blocking=True)
    centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)

    centre_labels_indices_np = np.array(
        [class_to_index.get(center_label_to_wordnet.get(label, ""), 0) for label in centre_labels], dtype=np.int64
    )
    centre_labels_indices = torch.tensor(centre_labels_indices_np, dtype=torch.int64, device=device)

    print("Center embeddings and label mappings processed.")

    # Step 4: Initialize model
    print("Initializing model...")
    model = UniBind(args).to(device)
    model.eval()
    print("Model initialized and set to evaluation mode.")

    def predict(x):
        x = {ModalityType.VISION: x}
        visual_embeddings = model.encode_vision(x).to(torch.bfloat16)
        visual_embeddings_norm = visual_embeddings / visual_embeddings.norm(dim=-1, keepdim=True)
        similarity = (visual_embeddings_norm @ centre_embeddings.t()).to(torch.bfloat16)

        batch_size = similarity.shape[0]
        class_raw_scores = torch.zeros(batch_size, 1000, device=device, dtype=torch.bfloat16)
        class_raw_scores, _ = torch_scatter.scatter_max(
            similarity, centre_labels_indices.expand(batch_size, -1), dim=1
        )
        return class_raw_scores
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Step 5: Run AutoAttack for multiple epsilon values
    for eps in args.epsilons:
        print(f"Running AutoAttack for epsilon = {eps:.6f}")
        adversary = AutoAttack(predict, norm=args.norm, eps=eps, log_path=args.log_path, version=args.version, attacks_to_run=['apgd-ce'])

        for batch_idx, (x_test, y_test, _) in enumerate(dataloader):
            torch.cuda.empty_cache()
            print(f"Processing batch {batch_idx + 1} for epsilon = {eps:.6f}")

            x_test = x_test.to(device, dtype=torch.float32, non_blocking=True)
            y_test = y_test.to(device, dtype=torch.int64, non_blocking=True)

            x_test_unorm = x_test.clone().detach().to(torch.float32)
            unnormalize_inplace(x_test_unorm)
            with torch.no_grad():
                adv_examples = adversary.run_standard_evaluation(x_test_unorm, y_test, bs=args.batch_size)

            normalize_inplace(adv_examples)
            noise = x_test - adv_examples

            torch.save(
                {'adv_complete': adv_examples, 'x_test': x_test, 'y_test': y_test},
                os.path.join(args.save_dir, f"adv_results_eps{eps:.5f}_{batch_idx}.pth")
            )

    print("AutoAttack completed for all epsilon values. Only .pth files are saved.")
