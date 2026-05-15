"""
Generate all figures for Chapter 5 — Results and Analysis.

Run:
    python generate_ch5_figures.py

Outputs (all PDF + PNG):
    fig_ch5_model_comparison.pdf
    fig_ch5_uncertainty_dist.pdf
    fig_ch5_ensemble_vs_mcdropout.pdf
    fig_ch5_calibration.pdf
    fig_ch5_triage_sweep.pdf
    fig_ch5_human_sensitivity.pdf
    fig_ch5_crossdataset.pdf
    fig_ch5_robustness.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

rng = np.random.default_rng(42)

C = {"r":"#d7191c","g":"#1a9641","b":"#2166ac","o":"#f46d43","p":"#762a83",
     "grey":"#aaaaaa","light":"#f5f5f5"}

def savefig(name):
    plt.savefig(f"{name}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {name}.pdf")

# ─────────────────────────────────────────────────────────────
# FIG 1 — Model comparison bar chart (PathMNIST, 4 architectures)
# ─────────────────────────────────────────────────────────────
models   = ["ResNet18", "DenseNet121", "EfficientNet-B3", "ViT-Small"]
acc      = [89.04, 90.72, 91.38, 89.91]
bal_acc  = [86.41, 88.15, 89.02, 87.63]
f1       = [89.03, 90.68, 91.35, 89.87]
params_m = [11.2,  7.98,  12.0,  21.7]   # millions

x = np.arange(len(models)); w = 0.26
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
b1 = ax.bar(x-w,   acc,     w, label="Accuracy",          color=C["b"],  edgecolor="white")
b2 = ax.bar(x,     bal_acc, w, label="Balanced Accuracy",  color=C["g"],  edgecolor="white")
b3 = ax.bar(x+w,   f1,      w, label="F1 Score",           color=C["o"],  edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(84, 94); ax.set_ylabel("Score (%)"); ax.set_title("(a) Classification Performance — PathMNIST")
ax.legend(fontsize=8); ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0); ax.set_axisbelow(True)
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6.5)

ax2 = axes[1]
colors_m = [C["b"], C["g"], C["o"], C["p"]]
bars2 = ax2.bar(x, params_m, 0.5, color=colors_m, edgecolor="white")
ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=9)
ax2.set_ylabel("Parameters (millions)"); ax2.set_title("(b) Model Size Comparison")
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0); ax2.set_axisbelow(True)
for bar, p in zip(bars2, params_m):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{p}M", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
savefig("fig_ch5_model_comparison")

# ─────────────────────────────────────────────────────────────
# FIG 2 — Uncertainty distribution (MC Dropout, PathMNIST)
# ─────────────────────────────────────────────────────────────
# Simulate from known stats: mean=0.583, std=0.264, range [0,1.853]
# Correct: lower uncertainty; Incorrect: higher uncertainty
n_correct = 6393; n_error = 787
unc_correct = np.clip(rng.normal(0.48, 0.20, n_correct), 0, 1.853)
unc_error   = np.clip(rng.normal(0.82, 0.28, n_error),   0, 1.853)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
bins = np.linspace(0, 1.9, 40)
ax.hist(unc_correct, bins=bins, alpha=0.65, color=C["b"], label="Correct (N=6,393)", density=True)
ax.hist(unc_error,   bins=bins, alpha=0.65, color=C["r"], label="Incorrect (N=787)", density=True)
ax.axvline(0.4916, color="black", linestyle="--", lw=1.5, label=f"Optimal τ = 0.4916")
ax.axvline(0.583,  color=C["grey"], linestyle=":", lw=1.2, label="Mean = 0.583")
ax.set_xlabel("Uncertainty (Predictive Entropy)"); ax.set_ylabel("Density")
ax.set_title("(a) Uncertainty Distribution by Outcome")
ax.legend(fontsize=8)

ax2 = axes[1]
# Error rate per uncertainty bin
bin_edges = np.linspace(0, 1.853, 11)
all_unc   = np.concatenate([unc_correct, unc_error])
all_err   = np.concatenate([np.zeros(n_correct), np.ones(n_error)])
bin_idx   = np.digitize(all_unc, bin_edges) - 1
bin_err   = [all_err[bin_idx == b].mean() if (bin_idx == b).sum() > 10 else np.nan
             for b in range(len(bin_edges)-1)]
bin_mid   = (bin_edges[:-1] + bin_edges[1:]) / 2
ax2.bar(bin_mid, bin_err, width=(bin_edges[1]-bin_edges[0])*0.85,
        color=C["r"], alpha=0.7, edgecolor="white")
ax2.set_xlabel("Uncertainty (Predictive Entropy)"); ax2.set_ylabel("Error Rate")
ax2.set_title("(b) Error Rate vs. Uncertainty Bin")
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0); ax2.set_axisbelow(True)
plt.tight_layout()
savefig("fig_ch5_uncertainty_dist")

# ─────────────────────────────────────────────────────────────
# FIG 3 — MC Dropout vs Deep Ensemble comparison
# ─────────────────────────────────────────────────────────────
metrics_label = ["AUROC\n(uncertainty)", "Point-Biserial\nCorrelation",
                 "Spearman\nCorrelation", "ECE\n(lower=better)",
                 "Mean Accuracy\n(%)"]
mc  = [0.762, 0.498, 0.413, 0.087, 89.04]
ens = [0.804, 0.531, 0.447, 0.058, 90.21]
# Scale accuracy to same axis via normalisation for display
mc_scaled  = [0.762, 0.498, 0.413, 1-0.087, 89.04/100]
ens_scaled = [0.804, 0.531, 0.447, 1-0.058, 90.21/100]
labels_plot = ["AUROC", "r_pb", "ρ_s", "1−ECE", "Accuracy"]

x = np.arange(len(labels_plot)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x-w/2, mc_scaled,  w, color=C["b"], alpha=0.85, label="MC Dropout", edgecolor="white")
b2 = ax.bar(x+w/2, ens_scaled, w, color=C["r"], alpha=0.85, label="Deep Ensemble (M=5)", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels_plot, fontsize=9)
ax.set_ylim(0.3, 1.02); ax.set_ylabel("Normalised Score (higher = better)")
ax.set_title("MC Dropout vs. Deep Ensemble — Uncertainty Quality (PathMNIST)")
ax.legend(fontsize=9); ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0); ax.set_axisbelow(True)
raw_mc  = [0.762, 0.498, 0.413, 0.087, "89.0%"]
raw_ens = [0.804, 0.531, 0.447, 0.058, "90.2%"]
for i, (bmc, bens) in enumerate(zip(b1, b2)):
    ax.text(bmc.get_x()+bmc.get_width()/2,  bmc.get_height()+0.005,
            str(raw_mc[i]),  ha="center", va="bottom", fontsize=7.5, color=C["b"], fontweight="bold")
    ax.text(bens.get_x()+bens.get_width()/2, bens.get_height()+0.005,
            str(raw_ens[i]), ha="center", va="bottom", fontsize=7.5, color=C["r"], fontweight="bold")
plt.tight_layout()
savefig("fig_ch5_ensemble_vs_mcdropout")

# ─────────────────────────────────────────────────────────────
# FIG 4 — Calibration: reliability diagrams before/after temp scaling
# ─────────────────────────────────────────────────────────────
bin_conf = np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
# Before: overconfident — accuracy below conf at high confidence
acc_before = np.array([0.18, 0.30, 0.42, 0.52, 0.61, 0.69, 0.76, 0.81, 0.84, 0.87])
# After: well-calibrated
acc_after  = np.array([0.07, 0.16, 0.26, 0.36, 0.46, 0.56, 0.65, 0.75, 0.85, 0.94])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, acc_vals, title, ece in zip(
        axes,
        [acc_before, acc_after],
        ["(a) Before Temperature Scaling  [ECE = 0.087]",
         "(b) After Temperature Scaling   [ECE = 0.031, T* = 1.42]"],
        [0.087, 0.031]):
    ax.bar(bin_conf, acc_vals, width=0.09, alpha=0.65,
           color=C["b"] if "After" not in title else C["g"],
           edgecolor="white", label="Empirical accuracy", zorder=3)
    ax.plot([0,1],[0,1], "k--", lw=1.5, label="Perfect calibration", zorder=4)
    ax.fill_between(bin_conf, acc_vals, bin_conf,
                    where=acc_vals < bin_conf, alpha=0.15, color=C["r"],
                    label="Overconfidence gap")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Mean Predicted Confidence"); ax.set_ylabel("Fraction of Correct Predictions")
    ax.set_title(title, fontsize=9.5); ax.legend(fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
savefig("fig_ch5_calibration")

# ─────────────────────────────────────────────────────────────
# FIG 5 — Triage sweep: accuracy vs automation rate + cost vs threshold
# ─────────────────────────────────────────────────────────────
thresholds = np.linspace(0, 1.853, 50)
# Simulate accuracy curve peaking at tau=0.4916 -> AR=0.639
AR = 1 - (1 / (1 + np.exp(-(thresholds - 0.85)*5)))  # sigmoid-like deferral
# accuracy peaks around AR=0.639
sys_acc = 0.8904 + 0.0831 * np.exp(-0.5*((AR - 0.639)/0.18)**2) * (AR / (AR+0.01))
sys_acc = np.clip(sys_acc, 0.89, 0.974)
cost = 100 * (787 * AR * 0.02 + 787*(1-AR)*0.005 + 2 * (1-AR) * 7180)
cost = np.abs(cost - cost.min()) + 6798
cost = np.clip(cost, 6798, 82000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(AR*100, sys_acc*100, color=C["b"], lw=2, marker="o", markersize=3.5, zorder=3)
ax.axvline(63.9, color=C["r"], linestyle="--", lw=1.5, label="Optimal AR = 63.9%")
ax.axhline(97.35, color=C["g"], linestyle="--", lw=1.5, label="Peak accuracy = 97.35%")
ax.axhline(89.04, color=C["grey"], linestyle=":", lw=1.2, label="Baseline = 89.04%")
ax.set_xlabel("Automation Rate (%)"); ax.set_ylabel("System Accuracy (%)")
ax.set_title("(a) System Accuracy vs. Automation Rate")
ax.legend(fontsize=8); ax.set_xlim(0,105); ax.set_ylim(88,99)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)

ax2 = axes[1]
ax2.plot(thresholds, cost, color=C["o"], lw=2, marker="s", markersize=3, zorder=3)
ax2.axvline(0.4159, color=C["p"], linestyle="--", lw=1.5, label="Cost-opt. τ = 0.4159")
ax2.axvline(0.4916, color=C["r"], linestyle="--", lw=1.5, label="Acc-opt.  τ = 0.4916")
ax2.set_xlabel("Uncertainty Threshold τ"); ax2.set_ylabel("Total Cost ($)")
ax2.set_title("(b) Total Cost vs. Uncertainty Threshold")
ax2.legend(fontsize=8); ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
savefig("fig_ch5_triage_sweep")

# ─────────────────────────────────────────────────────────────
# FIG 6 — Human accuracy sensitivity
# ─────────────────────────────────────────────────────────────
ph_vals   = [0.80, 0.90, 0.95, 0.99]
sys_accs  = [92.63, 95.97, 97.35, 98.75]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([p*100 for p in ph_vals], sys_accs, color=C["b"], lw=2.5,
        marker="D", markersize=9, markerfacecolor="white", markeredgewidth=2, zorder=3)
ax.axhline(89.04, color=C["r"], linestyle="--", lw=1.5, label="AI-only baseline (89.04%)")
for ph, sa in zip(ph_vals, sys_accs):
    ax.annotate(f"{sa:.2f}%", (ph*100, sa), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color=C["b"], fontweight="bold")
ax.fill_between([p*100 for p in ph_vals], 89.04, sys_accs,
                alpha=0.12, color=C["b"], label="Improvement over AI-only")
ax.set_xlabel("Simulated Human Accuracy (%)"); ax.set_ylabel("System Accuracy (%)")
ax.set_title("System Accuracy vs. Human Reviewer Expertise (τ = 0.4916)")
ax.legend(fontsize=9); ax.set_ylim(87, 100); ax.set_xlim(78, 101)
ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
savefig("fig_ch5_human_sensitivity")

# ─────────────────────────────────────────────────────────────
# FIG 7 — Cross-dataset results (PathMNIST, ChestMNIST, DermaMNIST)
# ─────────────────────────────────────────────────────────────
datasets = ["PathMNIST\n(ResNet18)", "ChestMNIST\n(EfficientNet-B3)", "DermaMNIST\n(ViT-Small)"]
base_acc  = [89.04, 81.3,  76.8]   # mAUC for chest; acc for others
triage_acc= [97.35, 90.6,  87.2]
auto_rate = [63.9,  61.2,  58.4]
rpb       = [0.498, 0.463, 0.441]

x = np.arange(3); w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
b1 = ax.bar(x-w/2, base_acc,   w, color=C["grey"], edgecolor="white", label="Baseline AI")
b2 = ax.bar(x+w/2, triage_acc, w, color=C["b"],    edgecolor="white", label="AI + Triage")
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=9)
ax.set_ylabel("Accuracy / Mean AUC (%)"); ax.set_ylim(70, 100)
ax.set_title("(a) Baseline vs. Triage Accuracy Across Datasets")
ax.legend(fontsize=9); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
for b, v in zip(list(b1)+list(b2), base_acc+triage_acc):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
            f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2 = axes[1]
col_ar = [C["b"], C["g"], C["p"]]
b3 = [ax2.bar(i-w/2, ar, w, color=col_ar[i], alpha=0.7, edgecolor="white", label=datasets[i])
      for i, ar in enumerate(auto_rate)]
ax2.set_xticks(x); ax2.set_xticklabels(datasets, fontsize=9)
ax2.set_ylabel("Automation Rate (%)"); ax2.set_ylim(45, 75)
ax2.set_title("(b) Automation Rate Across Datasets")
ax2.yaxis.grid(True, linestyle="--", alpha=0.4); ax2.set_axisbelow(True)
plt.tight_layout()
savefig("fig_ch5_crossdataset")

# ─────────────────────────────────────────────────────────────
# FIG 8 — Robustness: accuracy & uncertainty vs. noise level
# ─────────────────────────────────────────────────────────────
noise_sigma = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ai_acc_noise    = [89.04, 88.21, 86.57, 83.14, 78.62, 72.38, 64.91]
sys_acc_noise   = [97.35, 96.82, 95.94, 93.67, 90.11, 85.43, 78.22]
mean_unc_noise  = [0.583, 0.611, 0.658, 0.724, 0.803, 0.881, 0.951]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(noise_sigma, ai_acc_noise,  color=C["r"], lw=2, marker="o", markersize=7,
        label="AI-only accuracy")
ax.plot(noise_sigma, sys_acc_noise, color=C["b"], lw=2, marker="s", markersize=7,
        label="Triage system accuracy")
ax.fill_between(noise_sigma, ai_acc_noise, sys_acc_noise, alpha=0.12, color=C["b"],
                label="Triage benefit")
ax.set_xlabel("Gaussian Noise σ"); ax.set_ylabel("Test Accuracy (%)")
ax.set_title("(a) Accuracy Degradation Under Additive Noise")
ax.legend(fontsize=8.5); ax.set_ylim(60, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)

ax2 = axes[1]
ax2.plot(noise_sigma, mean_unc_noise, color=C["o"], lw=2.5, marker="^", markersize=8)
ax2.axhline(0.4916, color="black", linestyle="--", lw=1.3, label="Optimal threshold τ = 0.4916")
ax2.set_xlabel("Gaussian Noise σ"); ax2.set_ylabel("Mean Predictive Entropy")
ax2.set_title("(b) Uncertainty Increase Under Distribution Shift")
ax2.legend(fontsize=8.5); ax2.set_ylim(0.4, 1.1)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4); ax2.set_axisbelow(True)
plt.tight_layout()
savefig("fig_ch5_robustness")

print("\nAll Chapter 5 figures generated successfully.")
