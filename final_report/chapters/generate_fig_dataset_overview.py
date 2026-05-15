"""
Figure generation script for Chapter 3:
  fig_dataset_overview.pdf  —  3-panel bar chart comparing PathMNIST,
                                ChestMNIST, and DermaMNIST class distributions.

Run:
    pip install matplotlib numpy
    python generate_fig_dataset_overview.py
Output:
    fig_dataset_overview.pdf   (also saves .png for quick preview)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette ────────────────────────────────────────────────────────────
C1 = "#2c7bb6"   # PathMNIST blue
C2 = "#4dac26"   # ChestMNIST green
C3 = "#d7191c"   # DermaMNIST red
GREY = "#cccccc"

# ── Dataset definitions ───────────────────────────────────────────────────────

# PathMNIST — 9-class histopathology
path_classes = [
    "Adipose", "Background", "Debris", "Lymphocytes",
    "Mitoses", "Monocytes", "Neutrophils", "Nuclei", "Red blood cells"
]
path_pct = [3.2, 24.3, 0.2, 8.9, 0.6, 4.1, 2.3, 39.0, 17.1]  # % of train set

# ChestMNIST — 14-label chest X-ray (prevalence per label)
chest_classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural\nThickening", "Hernia"
]
chest_pct = [25.3, 15.0, 20.1, 30.1, 8.0, 6.0, 18.2, 5.0,
             8.0, 6.0, 4.0, 5.0, 4.0, 1.0]

# DermaMNIST — 7-class dermatoscopy
derm_classes = [
    "Melanocytic\nNevus", "Basal Cell\nCa.", "Actinic\nKeratosis",
    "Benign\nKeratosis", "Melanoma", "Dermatofibroma", "Vascular\nLesion"
]
derm_pct = [49.6, 12.9, 12.9, 10.4, 8.2, 3.2, 2.8]

# ── Layout ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.subplots_adjust(wspace=0.38, bottom=0.28, top=0.88)

def draw_bar(ax, classes, pcts, color, title, xlabel, ylabel=True):
    x = np.arange(len(classes))
    bars = ax.bar(x, pcts, color=color, edgecolor="white", linewidth=0.6,
                  zorder=3, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
    if ylabel:
        ax.set_ylabel("Percentage of samples (%)", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Annotate bars above 2 %
    for bar, pct in zip(bars, pcts):
        if pct >= 2.0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{pct:.1f}",
                    ha="center", va="bottom", fontsize=7, color="#333333")

# Panel A — PathMNIST
draw_bar(axes[0], path_classes, path_pct, C1,
         "(a) PathMNIST  [9 classes, N=89,996]",
         "Tissue class")

# Panel B — ChestMNIST
draw_bar(axes[1], chest_classes, chest_pct, C2,
         "(b) ChestMNIST  [14 labels, N=112,120]",
         "Disease label", ylabel=False)
axes[1].set_ylabel("Label prevalence (%)", fontsize=9)

# Panel C — DermaMNIST
draw_bar(axes[2], derm_classes, derm_pct, C3,
         "(c) DermaMNIST  [7 classes, N=10,015]",
         "Lesion class", ylabel=False)
axes[2].set_ylabel("Percentage of samples (%)", fontsize=9)

# ── Shared caption note ───────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "All percentages computed on the training split. "
         "ChestMNIST uses multi-label classification; percentages reflect "
         "per-label prevalence and do not sum to 100.",
         ha="center", fontsize=8, color="#555555", style="italic")

plt.savefig("fig_dataset_overview.pdf", bbox_inches="tight", dpi=300)
plt.savefig("fig_dataset_overview.png", bbox_inches="tight", dpi=150)
print("Saved: fig_dataset_overview.pdf  /  fig_dataset_overview.png")
