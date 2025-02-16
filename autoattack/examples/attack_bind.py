import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from autoattack import AutoAttack
from imagebind.imagebind_model import ModalityType  # Ensure ModalityType is imported
from utils.utils import load_centre_embeddings
import numpy as np

# Import UniBind Model
from model import UniBind
import models.PointBind_models as models  # Ensure dependencies are available


class ImageDataset(Dataset):
    def __init__(self, dataset, base_dir, transform):
        self.image_paths = [os.path.join(base_dir, entry["data"]) for entry in dataset]
        self.labels = [entry["label"] for entry in dataset]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def load_images_parallel(dataset, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_dataset = ImageDataset(dataset, args.test_base_dir, transform)
    dataloader = DataLoader(
        image_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False
    )

    return iter(dataloader)  # ✅ Returns an iterator instead of full dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=2. / 255.)
    parser.add_argument('--num_examples', type=int, default=50000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./autoattack/examples/results')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--log_path', type=str, default='./autoattack/examples/results/log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    parser.add_argument("--pretrain_weights", type=str, default='./ckpts/pretrained_weights.pt')
    parser.add_argument("--modality", type=str, default='image')
    parser.add_argument("--centre_embeddings_path", type=str, default='./centre_embs/image_in_center_embeddings.pkl')
    parser.add_argument("--label_to_index_path", type=str, default='./datasets/ImageNet-1k/label_to_index.json')
    parser.add_argument("--test_base_dir", type=str, default='/root/autodl-tmp/imagenet/val')
    parser.add_argument("--test_data_path", type=str, default='./datasets/ImageNet-1k/auto_attack_data.json')
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--is_tf_model", action='store_true', help="Set this flag if the model is a TensorFlow model")
    parser.add_argument("--cache_path", type=str, default='./autoattack/examples/results/cache_test.pth',
                        help="Path to cache x_test and y_test")

    args = parser.parse_args()

    if torch.cuda.is_available():
        # torch.cuda.set_per_process_memory_fraction(0.90, 0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.test_data_path, "r") as f:
        dataset = json.load(f)
        dataset = dataset[:args.num_examples]  # Limit dataset to num_examples
    batch_iterator = load_images_parallel(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    centre_embeddings, centre_labels = load_centre_embeddings(args.centre_embeddings_path, device)
    centre_embeddings = centre_embeddings.to(device, dtype=torch.float16, non_blocking=True)
    centre_embeddings /= centre_embeddings.norm(dim=-1, keepdim=True)

    # Load label-to-index mapping
    with open(args.label_to_index_path, "r") as f:
        label_to_index = json.load(f)

    centre_labels_indices_np = np.array(
        [label_to_index.get(label, 0) for label in centre_labels], dtype=np.int64
    )
    centre_labels_indices_np[centre_labels_indices_np > 1000] = 0  # Set out-of-range indices to 0
    centre_labels_indices = torch.tensor(centre_labels_indices_np, dtype=torch.int64, device=device)

    model = UniBind(args).to(device)
    model.eval()


    def predict(x):
        x = {ModalityType.VISION: x}

        visual_embeddings = model.encode_vision(x).to(torch.float16)
        visual_embeddings_norm = visual_embeddings / visual_embeddings.norm(dim=-1, keepdim=True)
        similarity = (visual_embeddings_norm @ centre_embeddings.t()).to(
            torch.float16)  # Shape: [batch_size, len(centre_labels)]

        batch_size = similarity.shape[0]
        class_raw_scores = torch.zeros(batch_size, 1001, device=device, dtype=torch.float16)
        class_raw_scores.scatter_add_(1, centre_labels_indices.expand(batch_size, -1), similarity)

        del visual_embeddings, visual_embeddings_norm, similarity
        torch.cuda.empty_cache()

        return class_raw_scores


    for batch_idx, (x_test, y_test) in enumerate(batch_iterator):
        torch.cuda.empty_cache()
        print(f"Processing batch {batch_idx + 1}")  # ✅ Print batch index

        # Move tensors to device
        x_test = x_test.to(device, dtype=torch.float32, non_blocking=True)
        y_test = y_test.to(device, dtype=torch.int64, non_blocking=True)

        adversary = AutoAttack(predict, norm=args.norm, eps=args.epsilon,
                               log_path=args.log_path, version=args.version, is_tf_model=args.is_tf_model)

        # Run adversarial attack
        with torch.no_grad():
            if args.version == 'custom':
                adversary.attacks_to_run = ['apgd-ce', 'fab']
                adversary.apgd.n_restarts = 2
                adversary.fab.n_restarts = 2

            if not args.individual:
                adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size,
                                                                 state_path=args.state_path)

                # Create save directory if not exists
                os.makedirs(args.save_dir, exist_ok=True)

                # Save adversarial images
                save_path = os.path.join(args.save_dir, f"adv_unibind_imagenet_{batch_idx}.pth")
                torch.save({'adv_complete': adv_complete, 'x_test': x_test, 'y_test': y_test}, save_path)
                print(f"Adversarial images saved to {save_path}")
            else:
                # Individual attack: each attack is run on all test points
                adv_complete = adversary.run_standard_evaluation_individual(x_test, y_test, bs=args.batch_size)

                save_path = os.path.join(args.save_dir, f"adv_unibind_imagenet_individual_{batch_idx}.pth")
                torch.save({'adv_complete': adv_complete, 'x_test': x_test, 'y_test': y_test}, save_path)
                print(f"Adversarial images (individual) saved to {save_path}")

            del x_test, y_test
