#!/usr/bin/env python3
"""
Phase 2 Part D: Calibration vs triage interaction (ResNet18 baseline).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

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
    collect_logits,
    default_pathmnist_yaml,
    get_pathmnist_loaders,
    load_model_from_checkpoint,
    make_human_predictions,
    softmax_np,
)
from src.models.model_factory import _unpack_logits

logger = logging.getLogger(__name__)

HUMAN_ACC = 0.95
HUMAN_SEED = 4242
MC_K = 10
N_SWEEP = 50
FIXED_THRESHOLD = 0.4916


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 calibration-triage interaction")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase2"))
    p.add_argument("--model-path", type=str, default=str(ROOT / "experiments/resnet18_pathmnist/best_model.pt"))
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def system_accuracy(
    pred: np.ndarray,
    unc: np.ndarray,
    y: np.ndarray,
    human: np.ndarray,
    thresh: float,
) -> Tuple[float, float]:
    defer = unc > thresh
    automation = float((~defer).mean())
    final = pred.copy()
    final[defer] = human[defer]
    return float((final == y).mean()), automation


def error_reduction_metric(ai_acc: float, sys_acc: float) -> float:
    e_ai = 1.0 - ai_acc
    e_sys = 1.0 - sys_acc
    return float((e_ai - e_sys) / e_ai) if e_ai > 1e-12 else float("nan")


def ece_accepted_subset(probs: np.ndarray, y: np.ndarray, defer: np.ndarray) -> float:
    mask = ~defer
    if mask.sum() < 15:
        return float("nan")
    return float(CalibrationMetrics.compute_ece(probs[mask], y[mask], num_bins=15))


def sweep_curve(
    pred: np.ndarray,
    unc: np.ndarray,
    y: np.ndarray,
    human: np.ndarray,
    probs: np.ndarray,
    ai_acc_full: float,
    n: int = N_SWEEP,
) -> Dict[str, np.ndarray]:
    lo, hi = float(unc.min()), float(unc.max())
    if hi - lo < 1e-8:
        hi = lo + 1e-6
    thresholds = np.linspace(lo, hi, n)
    defer_rates = []
    sys_accs = []
    err_reds = []
    eces = []
    for t in thresholds:
        defer = unc > t
        dr = float(defer.mean())
        sa, _ = system_accuracy(pred, unc, y, human, t)
        defer_rates.append(dr)
        sys_accs.append(sa)
        err_reds.append(error_reduction_metric(ai_acc_full, sa))
        eces.append(ece_accepted_subset(probs, y, defer))
    return {
        "deferral_rate": np.array(defer_rates),
        "system_accuracy": np.array(sys_accs),
        "error_reduction": np.array(err_reds),
        "ece_accepted": np.array(eces),
    }


def collect_mc_calibrated_mean_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    ts: TemperatureScaling,
    mc_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K dropout passes; each pass applies temperature to logits then softmax; return mean probs and entropy."""
    all_probs = []
    for _ in range(mc_k):
        batch_probs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                for m in model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.train()
                logits = _unpack_logits(model(images))
                for m in model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.eval()
                scaled = ts.apply_temperature_scaling(logits)
                batch_probs.append(torch.softmax(scaled, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(batch_probs, axis=0))
    model.eval()
    stacked = np.stack(all_probs, axis=0)
    mean_p = stacked.mean(axis=0)
    unc = -np.sum(mean_p * np.log(mean_p + 1e-10), axis=1)
    pred = mean_p.argmax(axis=1)
    return mean_p, unc, pred


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs/phase2/calibration_triage.log")

    yaml = Path(args.config) if args.config else default_pathmnist_yaml(ROOT)
    (_, val_loader, test_loader), cfg = get_pathmnist_loaders(ROOT, yaml, args.seed)
    nc = cfg["dataset"]["num_classes"]

    ck = Path(args.model_path)
    model = load_model_from_checkpoint(ck, device, "resnet", "18")

    test_logits, y_test = collect_logits(model, test_loader, device)
    human = make_human_predictions(HUMAN_SEED, y_test, HUMAN_ACC, nc)
    probs_uncal = softmax_np(test_logits)
    pred_det = probs_uncal.argmax(axis=1)
    ai_acc_full = float((pred_det == y_test).mean())

    # --- Strategy A: MC entropy, fixed threshold ---
    mc = MCDropoutUncertainty(model, num_samples=MC_K, device=device)
    pred_a, unc_a, mean_probs_a = mc.predict(test_loader, multilabel=False)
    ev_a = TriageEvaluator(
        pred_a,
        unc_a,
        human_predictions=human,
        human_accuracy=HUMAN_ACC,
        targets=y_test,
        verbose=False,
    )
    res_a = ev_a.evaluate_threshold(FIXED_THRESHOLD, strategy_name="A_fixed")
    sys_a = res_a.system_accuracy
    auto_a = res_a.automation_rate
    defer_a = unc_a > FIXED_THRESHOLD
    ece_a_acc = ece_accepted_subset(probs_uncal, y_test, defer_a)

    # --- Fit temperature on val ---
    ts = TemperatureScaling(model, device=device)
    ts.fit_temperature_simple(val_loader, num_epochs=100, learning_rate=0.05)
    T = float(ts.temperature)
    probs_cal = softmax_np(ts.apply_temperature_scaling(torch.from_numpy(test_logits).float().to(device)).cpu().numpy())
    pred_cal = probs_cal.argmax(axis=1)
    unc_b = 1.0 - np.max(probs_cal, axis=1)

    # Strategy B: optimize threshold on val for confidence uncertainty
    val_logits, y_val = collect_logits(model, val_loader, device)
    probs_val = softmax_np(ts.apply_temperature_scaling(torch.from_numpy(val_logits).float().to(device)).cpu().numpy())
    unc_val = 1.0 - np.max(probs_val, axis=1)
    pred_val = probs_val.argmax(axis=1)
    human_val = make_human_predictions(HUMAN_SEED + 1, y_val, HUMAN_ACC, nc)
    ev_val = TriageEvaluator(
        pred_val,
        unc_val,
        human_predictions=human_val,
        human_accuracy=HUMAN_ACC,
        targets=y_val,
        verbose=False,
    )
    sweep_val = ev_val.sweep_thresholds(num_thresholds=N_SWEEP)
    best_t = max(sweep_val, key=lambda r: r.system_accuracy).threshold
    sys_b, _ = system_accuracy(pred_cal, unc_b, y_test, human, best_t)
    defer_b = unc_b > best_t
    ece_b_acc = ece_accepted_subset(probs_cal, y_test, defer_b)

    # Strategy C: threshold on val; test metrics with same threshold
    mean_p_cv, unc_cv, pred_cv = collect_mc_calibrated_mean_probs(model, val_loader, device, ts, MC_K)
    ev_cv_c = TriageEvaluator(
        pred_cv,
        unc_cv,
        human_predictions=human_val,
        human_accuracy=HUMAN_ACC,
        targets=y_val,
        verbose=False,
    )
    sweep_cv_c = ev_cv_c.sweep_thresholds(num_thresholds=N_SWEEP)
    best_tc = max(sweep_cv_c, key=lambda r: r.system_accuracy).threshold

    mean_p_c, unc_c, pred_c = collect_mc_calibrated_mean_probs(model, test_loader, device, ts, MC_K)
    sys_c, _ = system_accuracy(pred_c, unc_c, y_test, human, best_tc)
    defer_c = unc_c > best_tc
    ece_c_acc = ece_accepted_subset(mean_p_c, y_test, defer_c)

    rows = [
        {
            "strategy": "A_mc_entropy_fixed_T",
            "threshold": FIXED_THRESHOLD,
            "triage_accuracy": sys_a,
            "automation_rate": float(auto_a),
            "error_reduction": error_reduction_metric(ai_acc_full, sys_a),
            "ece_ai_accepted": ece_a_acc,
            "temperature_T": T,
            "variant_note": "ECE on accepted uses single-pass uncalibrated softmax(test logits), not MC mean probs.",
        },
        {
            "strategy": "B_calibrated_confidence_optT",
            "threshold": float(best_t),
            "triage_accuracy": sys_b,
            "automation_rate": float((unc_b <= best_t).mean()),
            "error_reduction": error_reduction_metric(ai_acc_full, sys_b),
            "ece_ai_accepted": ece_b_acc,
            "temperature_T": T,
            "variant_note": "Threshold chosen on val (max system accuracy sweep); 1 - max calibrated prob uncertainty.",
        },
        {
            "strategy": "C_calibrated_mc_entropy_optT",
            "threshold": float(best_tc),
            "triage_accuracy": sys_c,
            "automation_rate": float((unc_c <= best_tc).mean()),
            "error_reduction": error_reduction_metric(ai_acc_full, sys_c),
            "ece_ai_accepted": ece_c_acc,
            "temperature_T": T,
            "variant_note": "Per MC forward: dropout on, logits/T, softmax; mean probs; predictive entropy of mean; threshold from val sweep.",
        },
    ]
    csv_path = out_dir / "calibration_triage_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Curves on test (same human; probs_* used for ECE on AI-accepted only)
    curve_a = sweep_curve(pred_a, unc_a, y_test, human, mean_probs_a, ai_acc_full)
    curve_b = sweep_curve(pred_cal, unc_b, y_test, human, probs_cal, ai_acc_full)
    curve_c = sweep_curve(pred_c, unc_c, y_test, human, mean_p_c, ai_acc_full)

    # Three panels: system accuracy, error reduction, ECE (accepted) vs deferral rate
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for cu, label in [(curve_a, "A"), (curve_b, "B"), (curve_c, "C")]:
        axes[0].plot(cu["deferral_rate"], cu["system_accuracy"], label=label)
    axes[0].set_xlabel("Deferral rate")
    axes[0].set_ylabel("System accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for cu, label in [(curve_a, "A"), (curve_b, "B"), (curve_c, "C")]:
        axes[1].plot(cu["deferral_rate"], cu["error_reduction"], label=label)
    axes[1].set_xlabel("Deferral rate")
    axes[1].set_ylabel("Error reduction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    for cu, label in [(curve_a, "A"), (curve_b, "B"), (curve_c, "C")]:
        axes[2].plot(cu["deferral_rate"], cu["ece_accepted"], label=label)
    axes[2].set_xlabel("Deferral rate")
    axes[2].set_ylabel("ECE accepted")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "calibration_triage_interaction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote %s and calibration_triage_interaction.png", csv_path)


if __name__ == "__main__":
    main()
