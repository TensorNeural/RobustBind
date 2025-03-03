import os
import argparse
from pathlib import Path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from autoattack import AutoAttack

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to ImageNet dataset")
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=0.0)  # Epsilon should be in range [0,1]
    parser.add_argument('--n_ex', type=int, default=1000, help="Number of examples to attack")
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='custom')
    parser.add_argument('--state-path', type=Path, default=None)
    args = parser.parse_args()

    # **1. Load ImageNet Pretrained Model**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # **2. Load ImageNet Validation Set**
    transform = transforms.Compose([
        transforms.Resize(256),  # Standard ImageNet preprocessing
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=transform)
    test_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # **3. Create Save Directory**
    os.makedirs(args.save_dir, exist_ok=True)

    # **4. Load AutoAttack**
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path, version=args.version)

    # **5. Prepare Data for Attack**
    x_test_list, y_test_list = [], []
    for x, y in test_loader:
        x_test_list.append(x)
        y_test_list.append(y)
        if len(x_test_list) * args.batch_size >= args.n_ex:
            break  # Stop early if we've collected enough samples

    x_test = torch.cat(x_test_list, 0)[:args.n_ex].to(device)
    y_test = torch.cat(y_test_list, 0)[:args.n_ex].to(device)

    print(f"Loaded {x_test.shape[0]} ImageNet examples for attack.")

    # **6. Run AutoAttack**
    with torch.no_grad():
        adv_examples = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, state_path=args.state_path)

    # **7. Save Results**
    torch.save({'adv_complete': adv_examples, 'x_test': x_test, 'y_test': y_test},
               os.path.join(args.save_dir, f"adv_results_eps{args.epsilon:.5f}.pth"))

    print(f"Saved adversarial examples to {args.save_dir}")
