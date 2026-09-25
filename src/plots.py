"""Shared matplotlib style and figure-export helpers for the blog's figures.

Palette follows the project's validated colorblind-safe default (see the
data-viz skill's references/palette.md): fixed categorical hue order, single-hue
sequential ramp, blue<->red diverging pair with a neutral midpoint.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

CATEGORICAL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CAT_ORDER = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]
CAT_LIST = [CATEGORICAL[k] for k in CAT_ORDER]

CORRECT_COLOR = CATEGORICAL["blue"]
INCORRECT_COLOR = CATEGORICAL["red"]
ID_COLOR = CATEGORICAL["blue"]
OOD_COLOR = CATEGORICAL["orange"]

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIVERGING = {"low": CATEGORICAL["blue"], "mid": "#f0efec", "high": CATEGORICAL["red"]}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "figure.dpi": 100,
        }
    )


def save_fig(fig, name: str, width_px: int = 1600, dpi_scale: int = 2):
    """Save `fig` as PNG (~width_px wide, 2x dpi) and SVG under figures/{name}.{png,svg}."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_w_in = fig.get_size_inches()[0]
    base_dpi = width_px / fig_w_in
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    svg_path = os.path.join(FIGURES_DIR, f"{name}.svg")
    fig.savefig(png_path, dpi=base_dpi * dpi_scale / dpi_scale, bbox_inches="tight")
    # re-save at requested pixel width precisely
    fig.set_dpi(base_dpi)
    fig.savefig(png_path, dpi=base_dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"saved {png_path} and {svg_path}")


def reliability_diagram(ax, ax_hist, accs, confs, counts, bin_edges, title=""):
    """Draw a reliability diagram (accuracy vs confidence bars) with a count histogram below."""
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = np.diff(bin_edges) * 0.9

    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1.5, label="Perfect calibration")
    ax.bar(centers, accs, width=width, color=CATEGORICAL["blue"], alpha=0.85, label="Accuracy")
    gap = confs - accs
    for c, a, g, w in zip(centers, accs, gap, width):
        if abs(g) > 1e-6:
            ax.bar(
                c, g, bottom=a, width=w, color=CATEGORICAL["red"], alpha=0.35,
                hatch="//", edgecolor=CATEGORICAL["red"], linewidth=0,
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")

    ax_hist.bar(centers, counts, width=width, color=INK_MUTED)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(False)


def id_ood_hist(ax, id_scores, ood_scores, xlabel, id_label="ID", ood_label="OOD", bins=40):
    lo = min(id_scores.min(), ood_scores.min())
    hi = max(id_scores.max(), ood_scores.max())
    bin_edges = np.linspace(lo, hi, bins)
    ax.hist(id_scores, bins=bin_edges, color=ID_COLOR, alpha=0.6, label=id_label, density=True)
    ax.hist(ood_scores, bins=bin_edges, color=OOD_COLOR, alpha=0.6, label=ood_label, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
