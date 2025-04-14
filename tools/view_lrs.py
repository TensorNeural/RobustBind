import argparse
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_lr_finder_discrete_sweet_spot(lrs, losses, window_size=5, min_lr_idx=5):
    """
    Plot learning rate finder results and detect sweet spot:
    1. Smooth the loss curve using a rolling average.
    2. Calculate discrete slopes.
    3. Detect the LR right before the slope increases (i.e. loss flattens or diverges).
    """

    lrs = np.array(lrs)
    losses = np.array(losses)
    log_lrs = np.log10(lrs)

    # Smooth the losses
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    smoothed_log_lrs = log_lrs[(window_size - 1)//2 : -(window_size // 2)]

    # Compute slope (gradient) in log-space
    slopes = np.diff(smoothed_losses) / np.diff(smoothed_log_lrs)

    # Find the sweet spot: first local minimum in slope followed by increasing trend
    valid_range = range(min_lr_idx, len(slopes) - 1)
    for i in valid_range:
        if slopes[i] > slopes[i - 1] and slopes[i - 1] < -0.02:
            sweet_spot_idx = i
            break
    else:
        sweet_spot_idx = np.argmin(slopes)  # fallback to steepest drop

    best_lr = 10 ** smoothed_log_lrs[sweet_spot_idx]
    best_loss = smoothed_losses[sweet_spot_idx]

    # Plot raw + smoothed loss
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses, linewidth=1, label="Raw Loss")
    plt.plot(10**smoothed_log_lrs, smoothed_losses, linewidth=2, label="Smoothed Loss")
    plt.scatter([best_lr], [best_loss], color='red', s=50, zorder=5,
                label=f"Sweet Spot LR = {best_lr:.2e}")

    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder: Sweet Spot Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Suggested sweet spot LR: {best_lr:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot LR Finder and Detect Sweet Spot")
    parser.add_argument("--lr_finder_file", type=str, default="./lr_finder.json",
                        help="Path to JSON file containing 'lrs' and 'losses'")
    args = parser.parse_args()

    with open(args.lr_finder_file, "r") as f:
        data = json.load(f)
        lrs = data["lrs"]
        losses = data["losses"]

    plot_lr_finder_discrete_sweet_spot(lrs, losses)
