#!/usr/bin/env python3
"""
Phase 3 Part D: MC uncertainty as OOD detector (clean PathMNIST vs severe corruption).
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.medmnist_loader import compute_class_weights, create_dataloaders
from src.models.model_factory import _unpack_logits
from src.utils.logger import setup_logger
from src.utils.phase1_config import build_trainer_config_from_pathmnist_yaml
from src.utils.phase2_common import load_model_from_checkpoint
from src.utils.phase3_corruptions import CORRUPTION_DISPLAY_NAMES, apply_corruption
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

MC_K = 10
OOD_MODELS = [
    ("resnet18", "experiments/resnet18_pathmnist/best_model.pt", "resnet", "18"),
    ("efficientnet_b3", "experiments/phase1/efficientnet_b3/best_model.pt", "efficientnet", "b3"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 OOD uncertainty detector")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase3"))
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "pathmnist_config.yaml"))
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mc_entropy_batch(model, images: torch.Tensor, device: torch.device) -> np.ndarray:
    """Predictive entropy of mean softmax over MC_K passes (images on device)."""
    import torch.nn as nn

    model.eval()
    all_p = []
    for _ in range(MC_K):
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        logits = _unpack_logits(model(images))
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        all_p.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    model.eval()
    stacked = np.stack(all_p, axis=0)
    mean_p = stacked.mean(axis=0)
    ent = -np.sum(mean_p * np.log(mean_p + 1e-10), axis=1)
    return ent


def collect_uncertainties(
    model,
    loader,
    device: torch.device,
    corrupt_fn,
) -> np.ndarray:
    out = []
    for images, _ in tqdm(loader, desc="MC unc", leave=False):
        x = corrupt_fn(images.to(device))
        out.append(mc_entropy_batch(model, x, device))
    return np.concatenate(out, axis=0)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns (opt_t, precision, recall, f1) maximizing F1 for OOD-positive."""
    thr_grid = np.unique(np.percentile(scores, np.linspace(0, 100, 200)))
    best_f1 = -1.0
    best_t = float(np.median(scores))
    best_p = best_r = 0.0
    for t in thr_grid:
        pred = (scores >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_p = float(precision_score(y_true, pred, zero_division=0))
            best_r = float(recall_score(y_true, pred, zero_division=0))
    return best_t, best_p, best_r, best_f1


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "phase3").mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase3" / "ood_detector.log")
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

    rows = []
    roc_store_eff: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    for disp, rel, arch, var in OOD_MODELS:
        ck = ROOT / rel
        if not ck.is_file():
            logger.warning("Skip %s", ck)
            continue
        model = load_model_from_checkpoint(ck, device, arch, var)

        for cor in CORRUPTION_DISPLAY_NAMES:
            sev = "severe"

            def corrupt_id(x):
                return x

            def corrupt_ood(x, c=cor):
                return apply_corruption(x, c, sev)

            u_id = collect_uncertainties(model, test_loader, device, corrupt_id)
            u_ood = collect_uncertainties(model, test_loader, device, corrupt_ood)
            y = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_ood))])
            s = np.concatenate([u_id, u_ood])
            try:
                roc_auc = float(roc_auc_score(y, s))
            except ValueError:
                roc_auc = float("nan")
            fpr, tpr, _ = roc_curve(y, s)
            opt_t, prec, rec, f1m = best_f1_threshold(y, s)
            rows.append(
                {
                    "model": disp,
                    "corruption": cor,
                    "auroc": roc_auc,
                    "optimal_threshold": opt_t,
                    "precision": prec,
                    "recall": rec,
                    "f1_ood_detection": f1m,
                }
            )
            if disp == "efficientnet_b3":
                roc_store_eff[cor] = (fpr, tpr, roc_auc)

    csv_path = out_dir / "ood_detection_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "corruption", "auroc", "optimal_threshold", "precision", "recall", "f1_ood_detection"],
        )
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, cor in enumerate(CORRUPTION_DISPLAY_NAMES):
        if cor not in roc_store_eff:
            continue
        fpr, tpr, au = roc_store_eff[cor]
        ax.plot(fpr, tpr, styles[i % len(styles)], label=f"{cor} (AUROC={au:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("OOD detection ROC (EfficientNet-B3, severe corruption)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "ood_detection_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote ood_detection_results.csv and ood_detection_roc.png")


if __name__ == "__main__":
    main()
