import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_lr_finder_with_slope(
    lrs,
    losses,
    window_size=5,
    min_lr_idx=20,
    negative_slope_threshold=-0.05,
    rebound_slope_threshold=-0.01
):
    """
    1) Smooth the losses using a rolling average.
    2) Detect 'sweet spot' LR by looking for slope transition:
       from strongly negative (< negative_slope_threshold)
       to flatter/positive (> rebound_slope_threshold).
    3) Plot (a) raw vs. smoothed loss and (b) slope of smoothed loss.
    """

    # --- Convert data to numpy arrays ---
    lrs_arr = np.array(lrs, dtype=float)
    losses_arr = np.array(losses, dtype=float)
    log_lrs_arr = np.log10(lrs_arr)

    # --- Smooth losses via rolling average ---
    kernel = np.ones(window_size) / window_size
    smoothed_losses = np.convolve(losses_arr, kernel, mode="valid")

    # Align the log-lr array with the smoothed losses
    offset = (window_size - 1) // 2
    smoothed_log_lrs = log_lrs_arr[offset : offset + len(smoothed_losses)]

    # --- Compute slope of smoothed losses in log(LR) space ---
    slopes = np.diff(smoothed_losses) / np.diff(smoothed_log_lrs)

    # --- Detect sweet spot index ---
    # Skip the first min_lr_idx slope points to avoid extremely low LRs
    sweet_spot_idx = None
    for i in range(min_lr_idx, len(slopes)):
        if slopes[i - 1] < negative_slope_threshold and slopes[i] > rebound_slope_threshold:
            sweet_spot_idx = i
            break

    # If no transition found, fallback to global min after 'min_lr_idx'
    if sweet_spot_idx is None:
        fallback_slice = slice(min_lr_idx, None)
        sweet_spot_idx = int(
            np.argmin(smoothed_losses[fallback_slice]) + min_lr_idx
        )

    # Convert sweet_spot_idx to actual LR
    best_lr = 10 ** smoothed_log_lrs[sweet_spot_idx]
    best_loss = smoothed_losses[sweet_spot_idx]

    # ---------------- Plot 1: Raw + Smoothed Loss ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(lrs_arr, losses_arr, linewidth=1, label="Raw Loss")
    plt.plot(10**smoothed_log_lrs, smoothed_losses, linewidth=2, label="Smoothed Loss")
    plt.scatter([best_lr], [best_loss], color="red", s=60, zorder=5,
                label=f"Sweet Spot LR = {best_lr:.2e}")

    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder: Loss vs. LR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------- Plot 2: Slope of Smoothed Loss ----------------
    # For plotting, we take midpoints of each consecutive pair of log_lrs
    mid_log_lrs = 0.5 * (smoothed_log_lrs[:-1] + smoothed_log_lrs[1:])
    mid_lrs = 10**mid_log_lrs

    plt.figure(figsize=(8, 4))
    plt.plot(mid_lrs, slopes, linewidth=2, label="Slope of Smoothed Loss")
    plt.axvline(best_lr, color="red", linestyle="--", 
                label=f"Sweet Spot LR = {best_lr:.2e}")

    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Slope")
    plt.title("Slope of Smoothed Loss vs. Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Suggested sweet spot LR: {best_lr:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Plot LR Finder and Slope")
    parser.add_argument(
        "--lr_finder_file", type=str, default="./lr_finder.json",
        help="JSON file containing 'lrs' and 'losses'"
    )
    parser.add_argument(
        "--window_size", type=int, default=5,
        help="Rolling average window size"
    )
    parser.add_argument(
        "--min_lr_idx", type=int, default=20,
        help="Number of slope points to skip from start (avoid extremely small LRs)"
    )
    parser.add_argument(
        "--negative_slope_threshold", type=float, default=-0.05,
        help="Slope below this is strongly negative"
    )
    parser.add_argument(
        "--rebound_slope_threshold", type=float, default=-0.01,
        help="Slope above this indicates flattening or rebound"
    )
    args = parser.parse_args()

    with open(args.lr_finder_file, "r") as f:
        data = json.load(f)
        lrs = data["lrs"]
        losses = data["losses"]

    plot_lr_finder_with_slope(
        lrs,
        losses,
        window_size=args.window_size,
        min_lr_idx=args.min_lr_idx,
        negative_slope_threshold=args.negative_slope_threshold,
        rebound_slope_threshold=args.rebound_slope_threshold
    )

if __name__ == "__main__":
    main()
