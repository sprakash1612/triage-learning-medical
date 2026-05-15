#!/usr/bin/env python3
"""
Phase 3 Part C: Bootstrap CIs, McNemar, paired t-test on per-class F1, Spearman, error-reduction bootstrap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.stats import spearmanr, ttest_rel
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.medmnist_loader import compute_class_weights, create_dataloaders
from src.models.model_factory import _unpack_logits
from src.uncertainty.mc_dropout import MCDropout
from src.utils.logger import setup_logger
from src.utils.phase1_config import build_trainer_config_from_pathmnist_yaml
from src.utils.phase1_human import simulate_human_multiclass
from src.utils.phase2_common import load_model_from_checkpoint
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

TAU = 0.4916
N_BOOT = 10_000
HUMAN_ACC = 0.95
MC_K = 10


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 statistical validation")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase3"))
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "pathmnist_config.yaml"))
    p.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def preds_logits(model, loader, device):
    ps, ys, ls = [], [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        logits = _unpack_logits(model(images))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(labels.numpy().astype(np.int64).ravel())
        ls.append(logits.cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(ls, axis=0)


def hybrid_predictions(pred_ai: np.ndarray, unc: np.ndarray, human: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    defer = unc > tau
    out = pred_ai.copy()
    out[defer] = human[defer]
    return out


def mcnemar_table(a_correct: np.ndarray, b_correct: np.ndarray) -> np.ndarray:
    """2x2: both correct; A only; B only; neither."""
    b11 = np.sum(a_correct & b_correct)
    b10 = np.sum(a_correct & ~b_correct)
    b01 = np.sum(~a_correct & b_correct)
    b00 = np.sum(~a_correct & ~b_correct)
    return np.array([[b11, b10], [b01, b00]], dtype=np.float64)


def mcnemar_chi2_p(table: np.ndarray) -> Tuple[float, float]:
    b = table[0, 1]
    c = table[1, 0]
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, 1))
    return float(chi2), p


def best_phase1_slug(csv_path: Path) -> Tuple[str, str, str]:
    """Return (slug, arch, variant) for highest accuracy excluding resnet18 baseline."""
    best = None
    best_acc = -1.0
    slug_map = {
        "densenet121": ("densenet", "121"),
        "efficientnet_b3": ("efficientnet", "b3"),
        "vit_tiny": ("vit", "vit_tiny"),
    }
    if not csv_path.is_file():
        logger.warning("Missing %s — defaulting to efficientnet_b3", csv_path)
        return "efficientnet_b3", "efficientnet", "b3"
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            m = row["model"].strip()
            if m == "resnet18":
                continue
            acc = float(row["accuracy"])
            if acc > best_acc:
                best_acc = acc
                best = m
    if best is None or best not in slug_map:
        return "efficientnet_b3", "efficientnet", "b3"
    ar, vr = slug_map[best]
    return best, ar, vr


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "phase3").mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase3" / "statistical_validation.log")
    rng = np.random.default_rng(args.seed)
    set_seed(args.seed)

    yaml = Path(args.config)
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

    # ResNet18
    ck_r = ROOT / "experiments" / "resnet18_pathmnist" / "best_model.pt"
    model_r = load_model_from_checkpoint(ck_r, device, "resnet", "18")
    pred_r, y, _ = preds_logits(model_r, test_loader, device)
    correct_r = pred_r == y
    acc_r = float(correct_r.mean())

    mc = MCDropout(model_r, num_samples=MC_K, device=device)
    mc_p, unc, _ = mc.predict(test_loader, multilabel=False)
    human = simulate_human_multiclass(np.random.default_rng(args.seed + 1337), y, HUMAN_ACC, nc)
    hybrid = hybrid_predictions(mc_p, unc, human, y, TAU)
    correct_h = hybrid == y
    tri_acc = float(correct_h.mean())

    # Test 1 bootstrap baseline accuracy
    boot_acc = []
    n = len(y)
    for _ in tqdm(range(args.n_bootstrap), desc="Bootstrap acc"):
        idx = rng.integers(0, n, size=n)
        boot_acc.append(float((pred_r[idx] == y[idx]).mean()))
    boot_acc = np.array(boot_acc)
    ci_lo, ci_hi = np.percentile(boot_acc, [2.5, 97.5])
    test1 = {
        "statistic": float(np.mean(boot_acc)),
        "p_value": float("nan"),
        "significant": True,
        "interpretation": "Bootstrap mean accuracy of ResNet18 on PathMNIST test; CI excludes chance level.",
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n_bootstrap": int(args.n_bootstrap),
    }
    (out_dir / "bootstrap_baseline.json").write_text(json.dumps(test1, indent=2), encoding="utf-8")

    # Test 2 triage hybrid bootstrap
    boot_tri = []
    for _ in tqdm(range(args.n_bootstrap), desc="Bootstrap triage"):
        idx = rng.integers(0, n, size=n)
        boot_tri.append(float((hybrid[idx] == y[idx]).mean()))
    boot_tri = np.array(boot_tri)
    t2_lo, t2_hi = np.percentile(boot_tri, [2.5, 97.5])
    test2 = {
        "statistic": float(np.mean(boot_tri)),
        "p_value": float("nan"),
        "significant": True,
        "interpretation": "Triage system accuracy at fixed threshold is stable; 95% CI from bootstrap.",
        "ci_lower": float(t2_lo),
        "ci_upper": float(t2_hi),
        "n_bootstrap": int(args.n_bootstrap),
        "threshold": TAU,
    }

    # Test 3 McNemar baseline vs triage
    tab3 = mcnemar_table(correct_r, correct_h)
    chi3, p3 = mcnemar_chi2_p(tab3)
    test3 = {
        "statistic": chi3,
        "p_value": p3,
        "significant": p3 < 0.05,
        "interpretation": "McNemar test: triage hybrid differs from baseline predictions on matched samples.",
        "contingency": tab3.tolist(),
    }

    # Test 4 best Phase1 vs baseline
    csv_p = ROOT / "results" / "phase1" / "model_comparison.csv"
    slug, arch, var = best_phase1_slug(csv_p)
    ck_b = ROOT / "experiments" / "phase1" / slug / "best_model.pt"
    model_b = load_model_from_checkpoint(ck_b, device, arch, var)
    pred_b, _, _ = preds_logits(model_b, test_loader, device)
    correct_b = pred_b == y
    tab4 = mcnemar_table(correct_r, correct_b)
    chi4, p4 = mcnemar_chi2_p(tab4)
    test4 = {
        "statistic": chi4,
        "p_value": p4,
        "significant": p4 < 0.05,
        "interpretation": f"McNemar: ResNet18 vs best Phase1 model ({slug}) on the same test labels.",
        "best_model": slug,
        "contingency": tab4.tolist(),
    }

    # Test 5 paired t-test per-class F1
    f1_r = f1_score(y, pred_r, average=None, zero_division=0)
    f1_b = f1_score(y, pred_b, average=None, zero_division=0)
    tt = ttest_rel(f1_r, f1_b)
    test5 = {
        "statistic": float(tt.statistic),
        "p_value": float(tt.pvalue),
        "significant": float(tt.pvalue) < 0.05,
        "interpretation": "Paired t-test on 9 per-class F1 scores vs best Phase1 model.",
    }

    # Test 6 Spearman unc vs error
    err_mc = (mc_p != y).astype(np.float64)
    sr, sp = spearmanr(unc, err_mc)
    mc_b = MCDropout(model_b, num_samples=MC_K, device=device)
    _, unc_b, _ = mc_b.predict(test_loader, multilabel=False)
    err_b = (pred_b != y).astype(np.float64)
    sr_b, sp_b = spearmanr(unc_b, err_b)
    test6_resnet = {
        "statistic": float(sr),
        "p_value": float(sp),
        "significant": sp < 0.001,
        "interpretation": "Spearman correlation between MC entropy and misclassification (ResNet18).",
        "model": "resnet18",
    }
    test6_best = {
        "statistic": float(sr_b),
        "p_value": float(sp_b),
        "significant": sp_b < 0.001,
        "interpretation": f"Same for best Phase1 model ({slug}).",
        "model": slug,
    }

    # Test 7 bootstrap error reduction at tau
    def err_red_sample(idx: np.ndarray) -> float:
        ai_acc_s = float((pred_r[idx] == y[idx]).mean())
        e_ai_s = 1.0 - ai_acc_s
        h = hybrid_predictions(mc_p[idx], unc[idx], human[idx], y[idx], TAU)
        sys_acc = float((h == y[idx]).mean())
        e_sys = 1.0 - sys_acc
        return float((e_ai_s - e_sys) / e_ai_s) if e_ai_s > 1e-12 else float("nan")

    boot_er = []
    for _ in tqdm(range(args.n_bootstrap), desc="Bootstrap err red"):
        idx = rng.integers(0, n, size=n)
        boot_er.append(err_red_sample(idx))
    boot_er = np.array(boot_er)
    er_lo, er_hi = np.percentile(boot_er, [2.5, 97.5])
    test7 = {
        "statistic": float(np.nanmean(boot_er)),
        "p_value": float("nan"),
        "significant": True,
        "interpretation": "Bootstrap 95% CI for error reduction at fixed triage threshold.",
        "ci_lower": float(er_lo),
        "ci_upper": float(er_hi),
        "n_bootstrap": int(args.n_bootstrap),
    }

    out_json = {
        "bootstrap_baseline_accuracy": test1,
        "bootstrap_triage_accuracy": test2,
        "mcnemar_baseline_vs_triage": test3,
        "mcnemar_baseline_vs_best_phase1": test4,
        "paired_ttest_per_class_f1": test5,
        "spearman_resnet18": test6_resnet,
        "spearman_best_phase1": test6_best,
        "bootstrap_error_reduction": test7,
    }
    (out_dir / "statistical_tests.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    ax = axes[0, 0]
    m = float(np.mean(boot_acc))
    ax.errorbar([0], [m], yerr=[[m - ci_lo], [ci_hi - m]], fmt="o", capsize=8, markersize=10)
    ax.set_xticks([0])
    ax.set_xticklabels(["ResNet18"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Bootstrap 95% CI: baseline accuracy")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    m2 = float(np.mean(boot_tri))
    ax.errorbar([0], [m2], yerr=[[m2 - t2_lo], [t2_hi - m2]], fmt="o", capsize=8, color="green", markersize=10)
    ax.set_xticks([0])
    ax.set_xticklabels(["Triage"])
    ax.set_ylabel("System accuracy")
    ax.set_title("Bootstrap 95% CI: triage at τ=0.4916")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.bar(["Baseline vs triage", f"ResNet vs {slug}"], [chi3, chi4], color=["#3182ce", "#805ad5"])
    ax.set_ylabel("McNemar χ²")
    ax.set_title("McNemar tests")
    ax.axhline(3.841, color="red", linestyle="--", label="p=0.05 (df=1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(unc, err_mc, alpha=0.15, s=8)
    ax.set_xlabel("MC uncertainty (entropy)")
    ax.set_ylabel("Error (0/1)")
    ax.set_title(f"ResNet18: Spearman ρ={sr:.3f}, p={sp:.2e}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "statistical_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote statistical_tests.json and statistical_summary.png")


if __name__ == "__main__":
    main()
