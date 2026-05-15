"""
Figure generation script for Chapter 4:
  fig_system_architecture.pdf  —  end-to-end pipeline diagram showing
  data → model → UQ → triage → output.

Run:
    pip install matplotlib --break-system-packages
    python generate_fig_system_architecture.py
Output:
    fig_system_architecture.pdf  /  fig_system_architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 5.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 5.5)
ax.axis("off")

# ── colour palette ─────────────────────────────────────────────────
COL = {
    "data":   "#2166ac",
    "model":  "#1a9641",
    "uq":     "#d7191c",
    "triage": "#f46d43",
    "output": "#762a83",
    "arrow":  "#555555",
    "light":  "#f7f7f7",
}

def box(ax, x, y, w, h, color, label, sublabel="", radius=0.25):
    fancy = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           linewidth=1.5,
                           edgecolor=color,
                           facecolor=color + "22",   # ~13% opacity hex
                           zorder=3)
    ax.add_patch(fancy)
    # header strip
    header = FancyBboxPatch((x, y + h - 0.55), w, 0.55,
                            boxstyle=f"round,pad=0.0,rounding_size=0",
                            linewidth=0,
                            edgecolor="none",
                            facecolor=color,
                            zorder=4, clip_on=True)
    ax.add_patch(header)
    ax.text(x + w/2, y + h - 0.275, label,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=5)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.15, sublabel,
                ha="center", va="center", fontsize=8,
                color="#333333", zorder=5,
                multialignment="center")

def arrow(ax, x1, x2, y=2.75, color="#555555"):
    ax.annotate("",
                xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=16),
                zorder=6)

# ── Boxes ──────────────────────────────────────────────────────────
# 1. Input Data
box(ax, 0.2, 1.25, 2.2, 3.0, COL["data"], "INPUT DATA",
    "PathMNIST\nChestMNIST\nDermaMNIST\n\n28×28 pixels\nRGB / Greyscale")

# 2. Preprocessing
box(ax, 2.8, 1.25, 2.2, 3.0, COL["data"], "PREPROCESSING",
    "Normalise [0,1]\nImageNet stats\nAugmentation\n(flip, rotate,\ncolour jitter)")

# 3. Model
box(ax, 5.4, 1.25, 2.2, 3.0, COL["model"], "DEEP MODEL",
    "ResNet18\nDenseNet121\nEfficientNet-B3\nViT-Small\n(ImageNet pretrained)")

# 4. UQ
box(ax, 8.0, 1.25, 2.2, 3.0, COL["uq"], "UNCERTAINTY QU.",
    "MC Dropout\n(T=10 passes)\nDeep Ensemble\n(M=5 models)\nTemp. Scaling")

# 5. Triage
box(ax, 10.6, 1.25, 2.2, 3.0, COL["triage"], "TRIAGE POLICY",
    "u(x) > τ ?\n─────────\nDefer → Human\nAutomate → AI\nThreshold opt.")

# 6. Output
box(ax, 13.2, 1.25, 1.6, 3.0, COL["output"], "OUTPUT",
    "System\naccuracy\nCost\nanalysis\nGrad-CAM")

# ── Arrows ─────────────────────────────────────────────────────────
for x1, x2 in [(2.4, 2.8), (5.0, 5.4), (7.6, 8.0), (10.2, 10.6), (12.8, 13.2)]:
    arrow(ax, x1, x2)

# ── Human-in-the-loop feedback arc ─────────────────────────────────
# Dashed arc from Triage back up and annotated
ax.annotate("",
            xy=(11.7, 4.3), xytext=(11.7, 4.3),
            zorder=6)
# Draw a curved path
import matplotlib.patches as mp
style = "arc3,rad=-0.4"
ax.annotate("Human\nExpert",
            xy=(11.7, 4.4),
            xytext=(11.7, 4.4),
            fontsize=8.5, ha="center", color=COL["triage"],
            fontweight="bold", zorder=7)
# curved arrow from triage top down to output
arc_patch = mp.FancyArrowPatch(
    (11.7, 4.25), (13.35, 4.25),
    connectionstyle="arc3,rad=-0.35",
    arrowstyle="-|>",
    color=COL["triage"],
    lw=1.6, mutation_scale=14,
    linestyle="dashed",
    zorder=6
)
ax.add_patch(arc_patch)

# ── Bottom legend ───────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=COL["data"]   + "44", edgecolor=COL["data"],   label="Data pipeline"),
    mpatches.Patch(facecolor=COL["model"]  + "44", edgecolor=COL["model"],  label="Model backbone"),
    mpatches.Patch(facecolor=COL["uq"]     + "44", edgecolor=COL["uq"],     label="Uncertainty quantification"),
    mpatches.Patch(facecolor=COL["triage"] + "44", edgecolor=COL["triage"], label="Triage & deferral"),
    mpatches.Patch(facecolor=COL["output"] + "44", edgecolor=COL["output"], label="Evaluation & output"),
]
ax.legend(handles=legend_items, loc="lower center",
          bbox_to_anchor=(0.5, -0.04), ncol=5,
          fontsize=8, frameon=True, edgecolor="#cccccc")

ax.set_title("End-to-End System Architecture: Uncertainty-Aware Triage Learning Framework",
             fontsize=11, fontweight="bold", pad=8)

plt.tight_layout()
plt.savefig("fig_system_architecture.pdf", bbox_inches="tight", dpi=300)
plt.savefig("fig_system_architecture.png", bbox_inches="tight", dpi=150)
print("Saved: fig_system_architecture.pdf  /  fig_system_architecture.png")
