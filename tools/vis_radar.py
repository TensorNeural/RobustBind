import argparse
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
# helper imports

# ============================================================
# Styling configuration (adjust here)
# ============================================================

# Per-model color palette
MODEL_COLORS = {
    "LanguageBind": "#FFB300",      # Yellow
    "ImageBind": "#4CAF50",        # Green
    "UniBind": "#29B6F6",          # Light Blue
    "RobustBind$^2$": "#E91E63",   # Pink
    "RobustBind$^4$": "#AB47BC",   # Purple
}

# Polygon & line styling
POLYGON_LINE_WIDTH = 2.0
FILL_ALPHA = 0.1               # Overlap fill opacity
ROBUST_EXTRA_ZORDER = 4        # Robust polygons on top
BASE_POLYGON_ZORDER = 2

# Grid & axis styling
GRID_COLOR = "#E0E0E0"         # Dotted grid lines color
GRID_DASH = None                # Solid grid lines (no dash pattern)
GRID_LINE_WIDTH = 0.8
GRID_ALPHA = 0.85
AXIS_SPINE_COLOR = "#E0E0E0"   # Polar spine & radial axis lines color
AXIS_SPINE_WIDTH = 0.9
RADIAL_AXIS_LINE_COLOR = "#E0E0E0"
RADIAL_AXIS_LINE_WIDTH = 0.6

# Label / text styling
AXIS_LABEL_COLOR = "#424242"
LEGEND_LABEL_COLOR = "#424242"
TITLE_COLOR = "#212121"
LABEL_FONT_WEIGHT = "bold"
COLUMN_TITLE_OFFSET = 0.035     # Relative offset above top row axes
ROW_TITLE_X = 0.01              # X position for left row titles
ROW_TITLE_FONT_SIZE = 12
COLUMN_TITLE_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 12  # originally 8 -> scaled 1.5x
Y_TICK_LABEL_SIZE = 11     # originally 7 -> scaled 1.5x

# Spacing / padding for labels (in points)
XTICK_LABEL_PAD = 8        # distance between angular tick labels and chart
YTICK_LABEL_PAD = 6        # distance between radial tick labels and chart
TITLE_PAD = 15             # title padding (was 10)


# ============================================================
# 1. Axes and modalities
# ============================================================

axes_all = [
    "IN-1K", "Places365",   # Image
    "ESC-50", "Urban-S",    # Audio
    "LLVIP",                # Thermal
    "MSR-VTT", "UCF-101",   # Video
    "N-Cal", "N-IN-1K",     # Event
]

axes_modalities_all = [
    "Image", "Image",
    "Audio", "Audio",
    "Thermal",
    "Video", "Video",
    "Event", "Event",
]

axes_no_event = axes_all[:-2]
axes_modalities_no_event = axes_modalities_all[:-2]

# ============================================================
# 2. Raw data (values in %, NaN for missing event for LB/IB)
#    Order matches axes_all.
# ============================================================

data_clean = {
    "LanguageBind": [77.02, 43.89, 94.03, 84.18, 51.97, 44.27, 71.16, np.nan, np.nan],
    "ImageBind":    [75.36, 44.61, 63.68, 51.99, 26.77, 40.48, 63.18, np.nan, np.nan],
    "UniBind":      [83.28, 54.74, 69.40, 58.82, 79.55, 43.99, 72.00, 6.06, 12.48],
    "RobustBind$^2$": [71.88, 45.73, 69.15, 59.30, 81.74, 16.70, 33.15, 5.24, 0.26],
    "RobustBind$^4$": [71.88, 45.73, 67.16, 56.52, 79.04, 16.70, 33.15, 5.24, 0.26],
}

data_2_255 = {
    "LanguageBind": [2.28, 1.08, 17.91, 25.66, 8.69, 0.50, 1.99, np.nan, np.nan],
    "ImageBind":    [4.68, 2.64, 1.74, 13.10, 0.08, 0.25, 0.25, np.nan, np.nan],
    "UniBind":      [4.76, 3.21, 2.49, 5.25, 6.16, 2.61, 0.53, 0.00, 0.00],
    "RobustBind$^2$": [55.00, 29.72, 32.09, 30.25, 75.24, 6.81, 13.89, 4.47, 0.06],
    "RobustBind$^4$": [55.03, 29.71, 30.85, 31.16, 76.71, 6.95, 13.97, 4.47, 0.06],
}

data_4_255 = {
    "LanguageBind": [1.26, 0.50, 10.20, 17.69, 3.86, 0.12, 0.62, np.nan, np.nan],
    "ImageBind":    [1.38, 0.92, 1.00, 11.17, 0.16, 0.37, 0.00, np.nan, np.nan],
    "UniBind":      [1.42, 1.33, 1.99, 4.95, 3.31, 1.64, 0.03, 0.00, 0.00],
    "RobustBind$^2$": [33.73, 15.42, 23.88, 22.89, 69.69, 2.94, 4.31, 2.45, 0.02],
    "RobustBind$^4$": [33.77, 15.37, 24.13, 25.30, 75.17, 2.87, 4.25, 2.41, 0.02],
}

eval_order = ["clean", "2_255", "4_255"]
EVAL_MAP = {
    "clean": data_clean,
    "2_255": data_2_255,
    "4_255": data_4_255,
}
EVAL_TITLES = {
    "clean": "Clean",
    "2_255": r"$\ell_{\infty} = \frac{2}{255}$",
    "4_255": r"$\ell_{\infty} = \frac{4}{255}$",
}

# ============================================================
# 3. Helpers: normalization + radar utils
# ============================================================

def radar_angles(n):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.concatenate([ang, [ang[0]]])

def compute_axis_max_for_chart(models, axis_indices):
    """Compute per-axis max over all evals and selected models."""
    vals = []
    for eval_name in eval_order:
        data = EVAL_MAP[eval_name]
        for m in models:
            v = np.array(data[m], dtype=float)[axis_indices]
            vals.append(v)
    vals = np.stack(vals, axis=0)  # (n_models * n_evals, n_axes)
    axis_max = np.nanmax(vals, axis=0)
    axis_max[axis_max == 0] = 1.0
    return axis_max

def normalize_for_eval(eval_name, models, axis_indices, axis_max):
    """Return dict model -> normalized vector for one eval, given precomputed axis_max."""
    raw_data = EVAL_MAP[eval_name]
    norm_data = {}
    for m in models:
        v = np.array(raw_data[m], dtype=float)[axis_indices]
        v = v / axis_max
        v = np.nan_to_num(v, nan=0.0)
        norm_data[m] = v.tolist()
    return norm_data

def draw_modality_background(ax, angles, modalities, max_radius=1.0):
    modality_color = {
        "Image":   "#fde0dd",
        "Audio":   "#e0ecf4",
        "Thermal": "#fff7bc",
        "Video":   "#e5f5e0",
        "Event":   "#f2e5ff",
    }
    current = modalities[0]
    start_idx = 0
    for i in range(1, len(modalities) + 1):
        if i == len(modalities) or modalities[i] != current:
            end_idx = i
            theta_start = angles[start_idx]
            theta_end = angles[end_idx] if end_idx < len(angles) else angles[0]

            if theta_end < theta_start:
                theta_range = np.linspace(theta_start, theta_end + 2 * np.pi, 100)
            else:
                theta_range = np.linspace(theta_start, theta_end, 100)

            r = np.full_like(theta_range, max_radius)
            ax.fill_between(theta_range, 0, r,
                            color=modality_color.get(current, "#ffffff"),
                            alpha=0.25, linewidth=0)

            if i < len(modalities):
                current = modalities[i]
                start_idx = i

    # NOTE: This background helper is defined for possible future use but is not
    # currently invoked by the default plotting pipeline. It intentionally leaves
    # the axes state unchanged (fills under the chart) so callers may enable it
    # selectively.

def plot_radar_subplot(ax, norm_data, axis_labels, modalities, title, colors):
    n_axes = len(axis_labels)
    angles = radar_angles(n_axes)
    # Removed modality background fill for transparent grid appearance.

    # plot each model
    # Draw non-robust first, robust last to keep robust on top (keys contain 'RobustBind').
    ordered_items = sorted(norm_data.items(), key=lambda kv: ("RobustBind" not in kv[0], kv[0]))
    for model, vals in ordered_items:
        v = np.array(vals, dtype=float)
        v = np.concatenate([v, [v[0]]])
        z_line = ROBUST_EXTRA_ZORDER if "RobustBind" in model else BASE_POLYGON_ZORDER
        z_fill = ROBUST_EXTRA_ZORDER if "RobustBind" in model else BASE_POLYGON_ZORDER - 1
        ax.plot(angles, v, color=colors[model], lw=POLYGON_LINE_WIDTH, label=model, zorder=z_line)
        ax.fill(angles, v, color=colors[model], alpha=FILL_ALPHA, zorder=z_fill)

    # Draw grid and spines (keep beneath chart elements)
    if GRID_DASH:
        ax.grid(color=GRID_COLOR, linestyle=GRID_DASH, linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
    else:
        ax.grid(color=GRID_COLOR, linestyle='-', linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
    if hasattr(ax, 'spines') and 'polar' in ax.spines:
        ax.spines['polar'].set_color(AXIS_SPINE_COLOR)
        ax.spines['polar'].set_linewidth(AXIS_SPINE_WIDTH)
    ax.tick_params(colors=AXIS_LABEL_COLOR)
    # Ensure the axes patch/background does not cover labels
    try:
        ax.patch.set_alpha(0.0)
        ax.patch.set_facecolor('none')
    except Exception:
        pass
    # Draw radial axis lines beneath all chart elements
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1.05], color=RADIAL_AXIS_LINE_COLOR, lw=RADIAL_AXIS_LINE_WIDTH, zorder=0)
    # NOTE: Do not set tick labels or titles here. Labels are drawn after all charts are created
    # so label text is guaranteed to be on top. See `draw_labels_on_ax` below.


def draw_labels_on_ax(ax, axis_labels, modalities, title=None):
    """Draw axis tick labels, y-tick labels and title on top of chart elements.

    This function should be called after the chart polygons and fills have been plotted
    so that labels render on top.
    """
    n_axes = len(axis_labels)
    angles = radar_angles(n_axes)
    ax.set_xticks(angles[:-1])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=Y_TICK_LABEL_SIZE)
    # Multiline labels: modality first line, dataset second line
    axis_labels_with_mod = [f"{mod}\n{lbl}" for lbl, mod in zip(axis_labels, modalities)]
    ax.set_xticklabels(axis_labels_with_mod, fontsize=AXIS_LABEL_FONT_SIZE, fontweight=LABEL_FONT_WEIGHT, color=AXIS_LABEL_COLOR)
    # Increase padding between tick labels and the chart
    try:
        ax.tick_params(axis='x', pad=XTICK_LABEL_PAD)
        ax.tick_params(axis='y', pad=YTICK_LABEL_PAD)
    except Exception:
        pass
    # Ensure labels are above polygons and not clipped
    ax.set_axisbelow(False)
    for tl in ax.get_xticklabels():
        tl.set_zorder(300)
        tl.set_clip_on(False)
    for tl in ax.get_yticklabels():
        tl.set_zorder(300)
        tl.set_clip_on(False)
    if title:
        ax.set_title(title, fontsize=COLUMN_TITLE_FONT_SIZE, pad=TITLE_PAD)
        ax.title.set_zorder(300)
        ax.title.set_clip_on(False)

# ============================================================
# 4. Matplotlib global styling
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def generate_figures(save_dir: str) -> None:
    """Generate six separate radar chart PDFs (one per eval × row).

    Each output file is named: radar_{rowkey}_{eval_name}.pdf
    """
    os.makedirs(save_dir, exist_ok=True)

    # Unified color mapping for legend consistency
    model_colors = MODEL_COLORS

    # Define rows and labels
    row_defs = [
        ("all_modalities", "All Modalities", ["UniBind", "RobustBind$^2$", "RobustBind$^4$"], axes_all, axes_modalities_all, list(range(len(axes_all)))),
        ("all_models", "All Models", ["LanguageBind", "ImageBind", "UniBind", "RobustBind$^2$", "RobustBind$^4$"], axes_no_event, axes_modalities_no_event, list(range(len(axes_no_event)))),
    ]

    # Precompute axis max per row
    axis_max_cache = {}
    for key, _, models, _, _, axis_indices in row_defs:
        axis_max_cache[key] = compute_axis_max_for_chart(models, axis_indices)

    # For each row and each evaluation, create a single-chart PDF
    for key, row_label, models, axis_labels, modalities, axis_indices in row_defs:
        axis_max = axis_max_cache[key]
        for eval_name in eval_order:
            # New figure for this single chart
            fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True), figsize=(6.0, 6.0))

            norm_data_eval = normalize_for_eval(eval_name, models, axis_indices, axis_max)
            colors_map = {m: model_colors[m] for m in models}

            # Draw the chart elements (polygons, grid, radial lines)
            plot_radar_subplot(ax, norm_data_eval, axis_labels, modalities, title=None, colors=colors_map)

            # Draw labels on top (NO title/annotation for individual chart PDFs)
            draw_labels_on_ax(ax, axis_labels, modalities, title=None)

            # No legend for individual chart PDFs (overview can include legend)

            fig.subplots_adjust(top=0.92, bottom=0.14, left=0.08, right=0.99)
            plt.tight_layout(rect=[0.08, 0.08, 0.99, 0.92])

            out_name = f"radar_{key}_{eval_name}.pdf"
            out_path = os.path.join(save_dir, out_name)
            # Transparent figure background
            _set_transparent_figure(fig)
            _save_transparent_pdf(fig, out_path)
            plt.close(fig)
            print(f"Saved: {out_path}")


def _set_transparent_figure(fig):
    """Make a Matplotlib figure and its children use a transparent background."""
    try:
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor('none')
    except Exception:
        pass
    for child in fig.get_children():
        try:
            child.set_facecolor('none')
        except Exception:
            # Not all children support set_facecolor; ignore.
            pass


def _save_transparent_pdf(fig, out_path: str) -> None:
    """Save figure to PDF with transparent background, ensuring tight bbox."""
    try:
        fig.savefig(out_path, bbox_inches="tight", transparent=True)
    except Exception:
        # Fall back to a standard save in case of driver/backend issues
        fig.savefig(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate radar charts for cross-modality performance")
    p.add_argument("--output-dir", type=str, default="/data/output", help="Base output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Determine session directory: always output_dir/radar/<timestamp>
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, "radar", ts)
    os.makedirs(save_dir, exist_ok=True)
    generate_figures(save_dir)


if __name__ == "__main__":
    main()
