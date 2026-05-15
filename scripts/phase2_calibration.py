#!/usr/bin/env python3
"""
Phase 2 Part A: Temperature scaling on PathMNIST val; calibration metrics + triage before/after.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.triage_evaluator import TriageEvaluator
from src.uncertainty.calibration import CalibrationMetrics
from src.uncertainty.mc_dropout import MCDropoutUncertainty
from src.uncertainty.temperature_scaling import TemperatureScaling
from src.utils.logger import setup_logger
from src.utils.phase2_common import (
    DEFAULT_PHASE2_MODELS,
    collect_logits,
    default_pathmnist_yaml,
    get_pathmnist_loaders,
    load_model_from_checkpoint,
    make_human_predictions,
    reliability_bin_data,
    softmax_np,
)

logger = logging.getLogger(__name__)

HUMAN_ACC = 0.95
HUMAN_SEED = 4242
MC_SAMPLES = 10
TRIAGE_THRESHOLDS = 50
NUM_BINS = 15


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 temperature scaling + calibration")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase2"))
    p.add_argument("--config", type=str, default=None, help="PathMNIST yaml")
    p.add_argument("--baseline-path", type=str, default=None)
    p.add_argument("--phase1-dir", type=str, default=None, help="Override dir for phase1 checkpoints (prefix)")
    p.add_argument("--models", type=str, default=None, help="Comma list: resnet18,densenet121,...")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_for_name(name: str, args, project_root: Path) -> Path:
    if name == "resnet18" and args.baseline_path:
        return Path(args.baseline_path)
    if args.phase1_dir and name != "resnet18":
        return Path(args.phase1_dir) / name / "best_model.pt"
    for disp, rel, arch, var in DEFAULT_PHASE2_MODELS:
        if disp == name:
            return project_root / rel
    raise KeyError(name)


def plot_reliability_two_panel(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    ece_b: float,
    ece_a: float,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    for col, (probs, title_suffix, ece) in enumerate(
        [(probs_before, "before", ece_b), (probs_after, "after", ece_a)]
    ):
        bc, acc, conf, cnt = reliability_bin_data(probs, labels, NUM_BINS)
        ax = axes[0, col]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.plot(conf, acc, "o-", color="steelblue", markersize=5)
        ax.set_xlabel("Mean predicted confidence (bin)")
        ax.set_ylabel("Fraction of positives (accuracy)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{model_name} {title_suffix}  ECE={ece:.4f}")
        ax.legend(loc="lower right", fontsize=8)
        axb = axes[1, col]
        axb.bar(bc, cnt, width=1.0 / NUM_BINS * 0.9, color="gray", alpha=0.7)
        axb.set_xlabel("Bin center (confidence)")
        axb.set_ylabel("Count")
    fig.suptitle(f"{model_name} reliability (ECE before={ece_b:.4f}, after={ece_a:.4f})", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def triage_best_mc(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    targets: np.ndarray,
    human_predictions: np.ndarray,
    num_classes: int,
) -> float:
    mc = MCDropoutUncertainty(model, num_samples=MC_SAMPLES, device=device)
    preds, unc, _mean_probs = mc.predict(test_loader, multilabel=False)
    ev = TriageEvaluator(
        preds,
        unc,
        human_predictions=human_predictions,
        human_accuracy=HUMAN_ACC,
        targets=targets,
        verbose=False,
    )
    sweep = ev.sweep_thresholds(num_thresholds=TRIAGE_THRESHOLDS)
    return float(max(sweep, key=lambda r: r.system_accuracy).system_accuracy)


def triage_best_confidence(
    preds: np.ndarray,
    probs: np.ndarray,
    targets: np.ndarray,
    human_predictions: np.ndarray,
) -> float:
    unc = 1.0 - np.max(probs, axis=1)
    ev = TriageEvaluator(
        preds,
        unc,
        human_predictions=human_predictions,
        human_accuracy=HUMAN_ACC,
        targets=targets,
        verbose=False,
    )
    sweep = ev.sweep_thresholds(num_thresholds=TRIAGE_THRESHOLDS)
    return float(max(sweep, key=lambda r: r.system_accuracy).system_accuracy)


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase2" / "calibration.log")

    yaml_path = Path(args.config) if args.config else default_pathmnist_yaml(ROOT)
    (train_loader, val_loader, test_loader), cfg = get_pathmnist_loaders(ROOT, yaml_path, args.seed)

    model_filter = None
    if args.models:
        model_filter = {x.strip() for x in args.models.split(",") if x.strip()}

    rows = []
    ece_before_list = []
    ece_after_list = []
    names_order = []

    for disp_name, _rel, arch, var in DEFAULT_PHASE2_MODELS:
        if model_filter and disp_name not in model_filter:
            continue
        ckpt = checkpoint_for_name(disp_name, args, ROOT)
        if not ckpt.is_file():
            logger.warning("Missing checkpoint %s — skip %s", ckpt, disp_name)
            continue
        logger.info("Processing %s from %s", disp_name, ckpt)
        model = load_model_from_checkpoint(ckpt, device, arch, var)
        ts = TemperatureScaling(model, device=device)
        ts.fit_temperature_simple(val_loader, num_epochs=100, learning_rate=0.05)
        T = float(ts.temperature)
        logger.info("%s temperature T=%.4f", disp_name, T)

        test_logits, y = collect_logits(model, test_loader, device)
        probs_before = softmax_np(test_logits)
        ece_b = CalibrationMetrics.compute_ece(probs_before, y, num_bins=NUM_BINS)
        mce_b = CalibrationMetrics.compute_mce(probs_before, y, num_bins=NUM_BINS)
        brier_b = CalibrationMetrics.compute_brier_score(probs_before, y)

        lt = torch.from_numpy(test_logits).float().to(device)
        scaled = ts.apply_temperature_scaling(lt).cpu().numpy()
        probs_after = softmax_np(scaled)
        ece_a = CalibrationMetrics.compute_ece(probs_after, y, num_bins=NUM_BINS)
        mce_a = CalibrationMetrics.compute_mce(probs_after, y, num_bins=NUM_BINS)
        brier_a = CalibrationMetrics.compute_brier_score(probs_after, y)

        ece_red = (
            float((ece_b - ece_a) / ece_b * 100.0) if ece_b > 1e-12 else float("nan")
        )

        human = make_human_predictions(HUMAN_SEED, y, HUMAN_ACC, cfg["dataset"]["num_classes"])
        tri_before = triage_best_mc(model, test_loader, device, y, human, cfg["dataset"]["num_classes"])
        preds_after = probs_after.argmax(axis=1)
        tri_after = triage_best_confidence(preds_after, probs_after, y, human)

        rows.append(
            {
                "model": disp_name,
                "temperature_T": T,
                "ece_before": ece_b,
                "ece_after": ece_a,
                "mce_before": mce_b,
                "mce_after": mce_a,
                "brier_before": brier_b,
                "brier_after": brier_a,
                "ece_reduction_pct": ece_red,
                "triage_acc_before": tri_before,
                "triage_acc_after": tri_after,
                "triage_improvement": tri_after - tri_before,
            }
        )
        names_order.append(disp_name)
        ece_before_list.append(ece_b)
        ece_after_list.append(ece_a)

        plot_reliability_two_panel(
            probs_before,
            probs_after,
            y,
            disp_name,
            ece_b,
            ece_a,
            out_dir / f"reliability_{disp_name}.png",
        )

    if names_order:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names_order))
        w = 0.35
        ax.bar(x - w / 2, ece_before_list, width=w, label="ECE before", color="coral")
        ax.bar(x + w / 2, ece_after_list, width=w, label="ECE after", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(names_order, rotation=20, ha="right")
        ax.set_ylabel("ECE")
        ax.legend()
        ax.set_title("ECE before vs after temperature scaling")
        fig.savefig(out_dir / "calibration_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    csv_path = out_dir / "calibration_results.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
