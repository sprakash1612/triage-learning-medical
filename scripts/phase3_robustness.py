#!/usr/bin/env python3
"""
Phase 3 Part A: PathMNIST test robustness under input corruptions (no retraining).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.triage_evaluator import TriageEvaluator
from src.models.model_factory import _unpack_logits
from src.utils.logger import setup_logger
from src.utils.phase1_config import build_trainer_config_from_pathmnist_yaml
from src.utils.phase1_human import simulate_human_multiclass
from src.utils.phase2_common import collect_logits, load_model_from_checkpoint
from src.utils.phase3_common import PATHMNIST_PHASE3_MODELS, load_phase1_thresholds
from src.utils.phase3_corruptions import (
    CORRUPTION_DISPLAY_NAMES,
    SEVERITY_LEVELS,
    apply_corruption,
)
from src.utils.reproducibility import set_seed
from src.data.medmnist_loader import create_dataloaders, compute_class_weights

logger = logging.getLogger(__name__)

MC_SAMPLES = 10
HUMAN_ACC = 0.95


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 PathMNIST corruption robustness")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase3"))
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "pathmnist_config.yaml"))
    p.add_argument("--phase1-csv", type=str, default=None, help="Override model_comparison.csv path")
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_corrupted(
    model: nn.Module,
    loader,
    device: torch.device,
    corrupt_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds, ys = [], []
    for images, labels in tqdm(loader, desc="Eval corrupted", leave=False):
        x = corrupt_fn(images.to(device))
        logits = _unpack_logits(model(x))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(labels.numpy().astype(np.int64).ravel())
    p = np.concatenate(preds)
    y = np.concatenate(ys)
    acc = float((p == y).mean())
    return p, y, acc


def mc_mean_uncertainty_corrupted(
    model: nn.Module,
    loader,
    device: torch.device,
    corrupt_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC predictive entropy on corrupted batches (same definition as MCDropout)."""
    model.eval()
    all_probs = []
    ys = []
    for _ in range(num_samples):
        batch_probs = []
        for images, labels in loader:
            x = corrupt_fn(images.to(device))
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            logits = _unpack_logits(model(x))
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
            batch_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            if _ == 0:
                ys.append(labels.numpy().astype(np.int64).ravel())
        all_probs.append(np.concatenate(batch_probs, axis=0))
    model.eval()
    stacked = np.stack(all_probs, axis=0)
    mean_p = stacked.mean(axis=0)
    unc = -np.sum(mean_p * np.log(mean_p + 1e-10), axis=1)
    preds = mean_p.argmax(axis=1)
    y = np.concatenate(ys)
    return preds, y, unc


def monotonic_insight(rows: List[Dict]) -> str:
    """Check if mean_uncertainty rises with severity per (model, corruption)."""
    ok = 0
    total = 0
    for model in {r["model"] for r in rows}:
        for cor in CORRUPTION_DISPLAY_NAMES:
            vals = []
            for sev in SEVERITY_LEVELS:
                m = [float(r["mean_uncertainty"]) for r in rows if r["model"] == model and r["corruption"] == cor and r["severity"] == sev]
                if m:
                    vals.append(m[0])
            if len(vals) == 3:
                total += 1
                if vals[0] <= vals[1] <= vals[2]:
                    ok += 1
    if total == 0:
        return "insufficient_rows"
    frac = ok / total
    return f"monotonic_uncertainty_severity_fraction={frac:.3f} ({ok}/{total} model×corruption paths)"


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "phase3").mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase3" / "robustness.log")
    set_seed(args.seed)

    yaml = Path(args.config)
    csv_phase1 = Path(args.phase1_csv) if args.phase1_csv else None
    thresholds = load_phase1_thresholds(ROOT, csv_phase1)

    cfg = build_trainer_config_from_pathmnist_yaml(
        yaml,
        seed=args.seed,
        checkpoint_dir=str(ROOT / "experiments" / "phase3_tmp"),
        log_dir=str(ROOT / "logs" / "phase3"),
        save_every_n_epochs=999,
    )
    if cfg["training"]["loss"].get("class_weights") is None:
        tl, _, _ = create_dataloaders(cfg)
        cfg["training"]["loss"]["class_weights"] = compute_class_weights(tl.dataset).tolist()
    _, _, test_loader = create_dataloaders(cfg)
    nc = cfg["dataset"]["num_classes"]

    rows: List[Dict[str, object]] = []
    plot_acc_drop: Dict[str, Dict[str, List[float]]] = {m: {c: [] for c in CORRUPTION_DISPLAY_NAMES} for m, _, _, _ in PATHMNIST_PHASE3_MODELS}
    plot_unc_inc: Dict[str, Dict[str, List[float]]] = {m: {c: [] for c in CORRUPTION_DISPLAY_NAMES} for m, _, _, _ in PATHMNIST_PHASE3_MODELS}
    triage_gaussian: Dict[str, List[float]] = {m: [] for m, _, _, _ in PATHMNIST_PHASE3_MODELS}

    for disp, rel_ckpt, arch, var in PATHMNIST_PHASE3_MODELS:
        ck = ROOT / rel_ckpt
        if not ck.is_file():
            logger.warning("Skip %s — missing %s", disp, ck)
            continue
        model = load_model_from_checkpoint(ck, device, arch, var)
        thresh = thresholds.get(disp)
        if thresh is None:
            logger.warning("No threshold for %s — sweep clean test once", disp)
            from src.uncertainty.mc_dropout import MCDropout

            mc0 = MCDropout(model, num_samples=MC_SAMPLES, device=device)
            pr, un, _ = mc0.predict(test_loader, multilabel=False)
            y_ref = []
            for _, lab in test_loader:
                y_ref.append(lab.numpy().astype(np.int64).ravel())
            y_ref = np.concatenate(y_ref)
            human = simulate_human_multiclass(np.random.default_rng(args.seed + 7), y_ref, HUMAN_ACC, nc)
            ev0 = TriageEvaluator(pr, un, human_predictions=human, human_accuracy=HUMAN_ACC, targets=y_ref, verbose=False)
            thresh = max(ev0.sweep_thresholds(50), key=lambda r: r.system_accuracy).threshold
            logger.info("Derived threshold for %s: %.4f", disp, thresh)

        # Clean baseline
        logits_clean, y_clean = collect_logits(model, test_loader, device)
        pred_clean = logits_clean.argmax(axis=1)
        clean_acc = float((pred_clean == y_clean).mean())
        clean_bacc = float(balanced_accuracy_score(y_clean, pred_clean))
        clean_f1 = float(f1_score(y_clean, pred_clean, average="weighted", zero_division=0))

        from src.uncertainty.mc_dropout import MCDropout

        mc_clean = MCDropout(model, num_samples=MC_SAMPLES, device=device)
        mc_p_clean, unc_clean, _ = mc_clean.predict(test_loader, multilabel=False)
        clean_mean_u = float(np.mean(unc_clean))
        rng = np.random.default_rng(args.seed + hash(disp) % (2**31))
        human = simulate_human_multiclass(rng, y_clean, HUMAN_ACC, nc)
        tri_clean = TriageEvaluator(
            mc_p_clean, unc_clean, human_predictions=human, human_accuracy=HUMAN_ACC, targets=y_clean, verbose=False
        ).evaluate_threshold(float(thresh))
        triage_gaussian[disp].append(float(tri_clean.system_accuracy))

        for cor in CORRUPTION_DISPLAY_NAMES:
            for sev in SEVERITY_LEVELS:

                def corrupt_fn(x, c=cor, s=sev):
                    return apply_corruption(x, c, s)

                pr_c, y_c, acc_c = evaluate_corrupted(model, test_loader, device, corrupt_fn)
                bacc_c = float(balanced_accuracy_score(y_c, pr_c))
                f1_c = float(f1_score(y_c, pr_c, average="weighted", zero_division=0))

                mc_p_c, _, unc_c = mc_mean_uncertainty_corrupted(model, test_loader, device, corrupt_fn)
                mean_u_c = float(np.mean(unc_c))
                tri_c = TriageEvaluator(
                    mc_p_c, unc_c, human_predictions=human, human_accuracy=HUMAN_ACC, targets=y_clean, verbose=False
                ).evaluate_threshold(float(thresh))

                ad = clean_acc - acc_c
                ui = mean_u_c - clean_mean_u
                rows.append(
                    {
                        "model": disp,
                        "corruption": cor,
                        "severity": sev,
                        "accuracy": acc_c,
                        "accuracy_drop": ad,
                        "balanced_accuracy": bacc_c,
                        "f1": f1_c,
                        "mean_uncertainty": mean_u_c,
                        "uncertainty_increase": ui,
                        "triage_accuracy": float(tri_c.system_accuracy),
                        "automation_rate": float(tri_c.automation_rate),
                    }
                )
                plot_acc_drop[disp][cor].append(ad)
                plot_unc_inc[disp][cor].append(ui)
                if cor == "gaussian_noise":
                    triage_gaussian[disp].append(float(tri_c.system_accuracy))

    insight = monotonic_insight(rows)
    csv_path = out_dir / "robustness_results.csv"
    with open(csv_path, "w", newline="") as f:
        f.write(f"# insight: uncertainty_vs_severity — {insight}\n")
        f.write("# Corruptions applied in normalized tensor space; clamped to [-1,1].\n")
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "corruption",
                "severity",
                "accuracy",
                "accuracy_drop",
                "balanced_accuracy",
                "f1",
                "mean_uncertainty",
                "uncertainty_increase",
                "triage_accuracy",
                "automation_rate",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    if not rows:
        logger.warning("No robustness rows — skip plots")
        return

    xpos = np.arange(len(CORRUPTION_DISPLAY_NAMES))
    width = 0.25
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.ravel()
    for ax_idx, (disp, _, _, _) in enumerate(PATHMNIST_PHASE3_MODELS):
        if ax_idx >= len(axes_flat):
            break
        ax = axes_flat[ax_idx]
        for si, sev in enumerate(SEVERITY_LEVELS):
            heights = [plot_acc_drop[disp][c][si] for c in CORRUPTION_DISPLAY_NAMES]
            ax.bar(xpos + (si - 1) * width, heights, width, label=sev)
        ax.set_xticks(xpos)
        ax.set_xticklabels(CORRUPTION_DISPLAY_NAMES, rotation=15, ha="right")
        ax.set_ylabel("Accuracy drop")
        ax.set_title(disp)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "robustness_accuracy_drop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.ravel()
    for ax_idx, (disp, _, _, _) in enumerate(PATHMNIST_PHASE3_MODELS):
        if ax_idx >= len(axes_flat):
            break
        ax = axes_flat[ax_idx]
        for si, sev in enumerate(SEVERITY_LEVELS):
            heights = [plot_unc_inc[disp][c][si] for c in CORRUPTION_DISPLAY_NAMES]
            ax.bar(xpos + (si - 1) * width, heights, width, label=sev)
        ax.set_xticks(xpos)
        ax.set_xticklabels(CORRUPTION_DISPLAY_NAMES, rotation=15, ha="right")
        ax.set_ylabel("Uncertainty increase")
        ax.set_title(disp)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "robustness_uncertainty_increase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [0, 1, 2, 3]
    for disp, _, _, _ in PATHMNIST_PHASE3_MODELS:
        if disp not in triage_gaussian or len(triage_gaussian[disp]) < 4:
            continue
        ax.plot(xs, triage_gaussian[disp][:4], marker="o", label=disp)
    ax.set_xticks(xs)
    ax.set_xticklabels(["clean", "mild", "moderate", "severe"])
    ax.set_xlabel("Gaussian noise severity")
    ax.set_ylabel("Triage accuracy")
    ax.set_title("Triage resilience under worst corruption (Gaussian noise)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "robustness_triage_resilience.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
