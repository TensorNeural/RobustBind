import torch
import matplotlib.pyplot as plt
import argparse
import os

def load_pth_images(pth_file):
    """Loads images from a .pth file and ensures tensors are on CPU."""
    data = torch.load(pth_file, map_location=torch.device('cpu'))  # Force CPU loading
    adv_images = data.get('adv_complete', None)  # Adversarial images
    orig_images = data.get('x_test', None)  # Original images
    labels = data.get('y_test', None)  # Labels
    return orig_images, adv_images, labels

def visualize_images(orig_images, adv_images, labels=None, num_images=10, save_dir=None):
    """Displays original, adversarial, and difference images using matplotlib and optionally saves them."""
    num_images = min(num_images, orig_images.shape[0])
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    
    for i in range(num_images):
        orig_img = orig_images[i].cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
        adv_img = adv_images[i].cpu().numpy().transpose(1, 2, 0)
        diff_img = (adv_img - orig_img) * 5  # Amplify difference for visibility
        
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())  # Normalize for display
        adv_img = (adv_img - adv_img.min()) / (adv_img.max() - adv_img.min())
        diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min())
        
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        axes[0, i].set_title("Original" if labels is None else f"Orig: {labels[i].item()}")
        
        axes[1, i].imshow(adv_img)
        axes[1, i].axis('off')
        axes[1, i].set_title("Adversarial")
        
        axes[2, i].imshow(diff_img)
        axes[2, i].axis('off')
        axes[2, i].set_title("Difference")
        
        if save_dir:
            plt.imsave(os.path.join(save_dir, f'orig_{i}.png'), orig_img)
            plt.imsave(os.path.join(save_dir, f'adv_{i}.png'), adv_img)
            plt.imsave(os.path.join(save_dir, f'diff_{i}.png'), diff_img)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_file', type=str, required=True, help='Path to the .pth file')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to visualize')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save images')
    args = parser.parse_args()
    
    orig_images, adv_images, labels = load_pth_images(args.pth_file)
    if orig_images is not None and adv_images is not None:
        visualize_images(orig_images, adv_images, labels=labels, num_images=args.num_images, save_dir=args.save_dir)
    else:
        print("Error: Original or adversarial images not found in the .pth file.")
