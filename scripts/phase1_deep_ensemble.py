#!/usr/bin/env python3
"""
Phase 1 Part C: Train 5 ResNet18 seeds on PathMNIST; ensemble vs MC Dropout comparison CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.medmnist_loader import create_dataloaders, compute_class_weights
from src.evaluation.metrics import compute_calibration_metrics
from src.evaluation.triage_evaluator import TriageEvaluator
from src.evaluation.uncertainty_metrics import UncertaintyMetrics
from src.models.model_factory import create_model
from src.models.resnet import ResNetClassifier
from src.training.trainer import Trainer
from src.uncertainty.deep_ensemble import DeepEnsembleUncertainty
from src.uncertainty.mc_dropout import MCDropout
from src.utils.logger import setup_logger
from src.utils.phase1_config import build_trainer_config_from_pathmnist_yaml
from src.utils.phase1_human import simulate_human_multiclass
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1337]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 deep ensemble PathMNIST")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--seed", type=int, default=42, help="Logging / default reproducibility anchor")
    p.add_argument("--config", type=str, default=str(ROOT / "configs/pathmnist_config.yaml"))
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--max-epochs", type=int, default=None)
    return p.parse_args()


def _device(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_resnet_config(seed: int, max_epochs: int | None) -> Dict:
    ckpt_root = ROOT / "experiments" / "phase1" / "ensemble"
    train_dir = ckpt_root / f"train_seed_{seed}"
    cfg = build_trainer_config_from_pathmnist_yaml(
        ROOT / "configs/pathmnist_config.yaml",
        seed=seed,
        checkpoint_dir=str(train_dir),
        log_dir=str(ROOT / "logs" / "phase1"),
        save_every_n_epochs=10,
        overrides={
            "experiment": {"name": f"phase1_ensemble_seed_{seed}"},
            "model": {
                "architecture": "resnet",
                "variant": "18",
                "pretrained": True,
                "dropout_rate": 0.3,
                "input_channels": 3,
            },
            "training": {
                "loss": {"type": "cross_entropy", "label_smoothing": 0.1, "class_weights": None},
            },
        },
    )
    if max_epochs is not None:
        cfg["training"]["num_epochs"] = int(max_epochs)
        if cfg["training"]["scheduler"]["type"] == "cosine":
            cfg["training"]["scheduler"]["T_max"] = int(max_epochs)
    return cfg


def train_all_seeds(device, args) -> None:
    yaml = Path(args.config)
    for seed in SEEDS:
        set_seed(seed)
        cfg = build_resnet_config(seed, args.max_epochs)
        if cfg["training"]["loss"].get("class_weights") is None:
            tl = create_dataloaders(cfg)
            cfg["training"]["loss"]["class_weights"] = compute_class_weights(tl[0].dataset).tolist()
        train_loader, val_loader, test_loader = create_dataloaders(cfg)
        model = create_model(cfg)
        out_name = f"model_seed_{seed}.pt"
        out_path = ROOT / "experiments" / "phase1" / "ensemble" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_training and out_path.is_file():
            logger.info("Skip train seed %s", seed)
            continue
        model.to(device)
        Trainer(model, cfg, device).train(train_loader, val_loader)
        best = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"
        if best.is_file():
            shutil.copy2(best, out_path)
            logger.info("Saved %s", out_path)


def load_models(device) -> List[torch.nn.Module]:
    models = []
    for seed in SEEDS:
        path = ROOT / "experiments" / "phase1" / "ensemble" / f"model_seed_{seed}.pt"
        ck = torch.load(path, map_location=device)
        m = ResNetClassifier(
            variant="resnet18",
            num_classes=9,
            pretrained=False,
            dropout_rate=0.3,
        )
        m.load_state_dict(ck["model_state_dict"])
        m.to(device)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def ensemble_predict(
    models: List[torch.nn.Module],
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean softmax probs across models; predictive entropy; class preds."""
    all_probs_batches = []
    ys = []
    for images, labels in tqdm(loader, desc="Ensemble infer"):
        images = images.to(device)
        B = images.size(0)
        stack = []
        for m in models:
            logits = m(images)
            stack.append(F.softmax(logits, dim=1).cpu().numpy())
        mean_p = np.mean(np.stack(stack, axis=0), axis=0)
        all_probs_batches.append(mean_p)
        ys.append(labels.numpy().ravel())
    mean_probs = np.concatenate(all_probs_batches, axis=0)
    y = np.concatenate(ys)
    preds = mean_probs.argmax(axis=1)
    ent = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    return preds, y, mean_probs, ent


def ece_on_mask(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 5:
        return float("nan")
    return float(compute_calibration_metrics(probs[mask], y[mask], num_bins=15)["ece"])


def triage_best(
    preds: np.ndarray,
    unc: np.ndarray,
    y: np.ndarray,
    human: np.ndarray,
    n: int,
    ai_acc: float,
) -> Dict:
    err = (preds != y).astype(np.int32)
    uq = UncertaintyMetrics.compute_uncertainty_quality(unc, err)
    ev = TriageEvaluator(
        model_predictions=preds,
        model_uncertainties=unc,
        human_predictions=human,
        human_accuracy=0.95,
        targets=y,
        verbose=False,
    )
    sweep = ev.sweep_thresholds(num_thresholds=n)
    best = max(sweep, key=lambda r: r.system_accuracy)
    e_ai = 1.0 - ai_acc
    e_sys = 1.0 - float(best.system_accuracy)
    err_red = float((e_ai - e_sys) / e_ai) if e_ai > 1e-12 else float("nan")
    return {
        "mean_uncertainty": float(np.mean(unc)),
        "spearman_corr": float(uq["spearman_corr"]),
        "uncertainty_auroc": float(uq["auroc"]),
        "triage_accuracy": float(best.system_accuracy),
        "automation_rate": float(best.automation_rate),
        "error_reduction": err_red,
        "threshold": float(best.threshold),
    }


def main():
    args = parse_args()
    device = _device(args.device)
    setup_logger(ROOT / "logs" / "phase1" / "ensemble.log")
    train_all_seeds(device, args)

    cfg_ref = build_resnet_config(SEEDS[0], args.max_epochs)
    _, _, test_loader = create_dataloaders(cfg_ref)
    models = load_models(device)

    preds_e, y, mean_probs, unc_e = ensemble_predict(models, test_loader, device)
    ai_acc_e = float((preds_e == y).mean())
    rng = np.random.default_rng(args.seed + 7)
    human = simulate_human_multiclass(rng, y, 0.95, 9)
    tri_e = triage_best(preds_e, unc_e, y, human, 50, ai_acc_e)
    defer = unc_e > tri_e["threshold"]
    ece_e_before = float(compute_calibration_metrics(mean_probs, y)["ece"])
    ece_e_after = ece_on_mask(mean_probs, y, ~defer)

    seed42_path = ROOT / "experiments" / "phase1" / "ensemble" / "model_seed_42.pt"
    ck42 = torch.load(seed42_path, map_location=device)
    m42 = ResNetClassifier(variant="resnet18", num_classes=9, pretrained=False, dropout_rate=0.3)
    m42.load_state_dict(ck42["model_state_dict"])
    m42.to(device)
    mc = MCDropout(m42, num_samples=10, device=device)
    mc_preds, unc_m, mean_p_m = mc.predict(test_loader, multilabel=False)
    ai_acc_m = float((mc_preds == y).mean())
    rng2 = np.random.default_rng(args.seed + 7)
    human_m = simulate_human_multiclass(rng2, y, 0.95, 9)
    tri_m = triage_best(mc_preds, unc_m, y, human_m, 50, ai_acc_m)
    defer_m = unc_m > tri_m["threshold"]
    ece_m_before = float(compute_calibration_metrics(mean_p_m, y)["ece"])
    ece_m_after = ece_on_mask(mean_p_m, y, ~defer_m)

    # Optional: call DeepEnsembleUncertainty API on one batch (reuse check)
    de = DeepEnsembleUncertainty(ResNetClassifier, num_models=1, device=device)
    de.models = [m42]
    de.trained_models = [m42]
    xb, _ = next(iter(test_loader))
    _ = de.compute_ensemble_uncertainty(xb[:4].to(device), uncertainty_type="entropy")

    logger.info(
        "ECE ensemble before=%.4f after_ai_subset=%.4f | MC before=%.4f after=%.4f",
        ece_e_before,
        ece_e_after,
        ece_m_before,
        ece_m_after,
    )
        {
            "method": "ensemble_entropy_mean_probs",
            "mean_uncertainty": tri_e["mean_uncertainty"],
            "spearman_corr": tri_e["spearman_corr"],
            "uncertainty_auroc": tri_e["uncertainty_auroc"],
            "triage_accuracy": tri_e["triage_accuracy"],
            "automation_rate": tri_e["automation_rate"],
            "error_reduction": tri_e["error_reduction"],
        },
        {
            "method": "mc_dropout_resnet18_seed42",
            "mean_uncertainty": tri_m["mean_uncertainty"],
            "spearman_corr": tri_m["spearman_corr"],
            "uncertainty_auroc": tri_m["uncertainty_auroc"],
            "triage_accuracy": tri_m["triage_accuracy"],
            "automation_rate": tri_m["automation_rate"],
            "error_reduction": tri_m["error_reduction"],
        },
    ]
    out_csv = ROOT / "results" / "phase1" / "ensemble_vs_mcdropout.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
