#!/usr/bin/env python3
"""
Phase 1 Part B: EfficientNet-B3 on ChestMNIST (multi-label), MC dropout, triage; JSON vs PathMNIST.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.medmnist_loader import create_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer, _unpack_logits
from src.uncertainty.mc_dropout import MCDropout
from src.utils.logger import setup_logger
from src.utils.phase1_config import build_trainer_config_from_chestmnist_yaml
from src.utils.phase1_human import simulate_human_multilabel
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 ChestMNIST experiment")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default=str(ROOT / "configs/chestmnist_config.yaml"))
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--max-epochs", type=int, default=None)
    return p.parse_args()


def _device(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_config(seed: int, max_epochs: int | None) -> Dict[str, Any]:
    ckpt = ROOT / "experiments" / "phase1" / "chestmnist_efficientnet_b3"
    log_dir = ROOT / "logs" / "phase1"
    cfg = build_trainer_config_from_chestmnist_yaml(
        ROOT / "configs/chestmnist_config.yaml",
        seed=seed,
        checkpoint_dir=str(ckpt),
        log_dir=str(log_dir),
        save_every_n_epochs=10,
        force_rgb=True,
        overrides={
            "experiment": {"name": "phase1_chestmnist_efficientnet_b3"},
            "model": {
                "architecture": "efficientnet",
                "variant": "b3",
                "pretrained": True,
                "dropout_rate": 0.3,
                "input_channels": 3,
            },
        },
    )
    if max_epochs is not None:
        cfg["training"]["num_epochs"] = int(max_epochs)
        if cfg["training"]["scheduler"]["type"] == "cosine":
            cfg["training"]["scheduler"]["T_max"] = int(max_epochs)
    return cfg


@torch.no_grad()
def per_class_auc(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    model.eval()
    ys, scores = [], []
    for x, y in tqdm(loader, desc="Test AUC"):
        x = x.to(device)
        logits = _unpack_logits(model(x))
        ys.append(y.numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.vstack(ys)
    y_score = np.vstack(scores)
    out: Dict[str, Any] = {}
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if len(np.unique(col)) < 2:
            continue
        try:
            a = roc_auc_score(col, y_score[:, c])
            aucs.append(a)
            out[f"auc_class_{c}"] = float(a)
        except ValueError:
            continue
    out["auc_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def multilabel_triage_best(
    ai_pred: np.ndarray,
    unc: np.ndarray,
    targets: np.ndarray,
    human_pred: np.ndarray,
    num_thresholds: int,
    model_subset_acc: float,
) -> Dict[str, float]:
    thresholds = np.linspace(float(unc.min()), float(unc.max()), num_thresholds)
    best = None
    for t in thresholds:
        defer = unc > t
        final = ai_pred.copy()
        final[defer] = human_pred[defer]
        sys_acc = float((final == targets).all(axis=1).mean())
        auto = float((~defer).mean())
        rec = {"threshold": float(t), "triage_accuracy": sys_acc, "automation_rate": auto}
        if best is None or rec["triage_accuracy"] > best["triage_accuracy"]:
            best = rec
    e_ai = 1.0 - model_subset_acc
    e_sys = 1.0 - float(best["triage_accuracy"])
    err_red = float((e_ai - e_sys) / e_ai) if e_ai > 1e-12 else float("nan")
    best["error_reduction"] = err_red
    return best


def load_pathmnist_reference() -> Dict[str, Any]:
    csv_path = ROOT / "results" / "phase1" / "model_comparison.csv"
    ref = {
        "source": "embedded",
        "resnet18_baseline": {
            "accuracy": 0.8904,
            "triage_accuracy": 0.9735,
            "automation_rate": 0.639,
            "error_reduction": 0.758,
        },
    }
    if not csv_path.is_file():
        return ref
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ref
    r0 = rows[0]
    if r0.get("model") == "resnet18":
        ref["source"] = "model_comparison.csv"
        ref["resnet18_baseline"] = {
            "accuracy": float(r0["accuracy"]),
            "triage_accuracy": float(r0["triage_accuracy"]),
            "automation_rate": float(r0["automation_rate"]),
            "error_reduction": float(r0["error_reduction"]),
        }
    return ref


def main():
    args = parse_args()
    device = _device(args.device)
    results_dir = ROOT / "results" / "phase1"
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase1" / "chestmnist.log")
    set_seed(args.seed)

    cfg = build_config(args.seed, args.max_epochs)
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    model = create_model(cfg)
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"

    if args.skip_training and ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device)
    else:
        model.to(device)
        Trainer(model, cfg, device).train(train_loader, val_loader)

    aucs = per_class_auc(model, test_loader, device)

    mc = MCDropout(model, num_samples=10, device=device)
    mc_pred, uncertainties, mean_probs = mc.predict(test_loader, multilabel=True)
    targets_list = []
    for _, y in test_loader:
        targets_list.append(y.numpy())
    targets = np.vstack(targets_list)
    mean_u = float(np.mean(uncertainties))
    subset_acc = float((mc_pred == targets).all(axis=1).mean())

    human_acc = float(cfg.get("triage", {}).get("human_expert", {}).get("accuracy", 0.92))
    rng = np.random.default_rng(args.seed + 999)
    human_pred = simulate_human_multilabel(rng, targets, human_acc)
    tri = multilabel_triage_best(mc_pred, uncertainties, targets, human_pred, 50, subset_acc)

    path_ref = load_pathmnist_reference()
    out = {
        "dataset": "chestmnist",
        "task": "multi-label",
        "triage_definition": "exact_multilabel_match",
        "uncertainty": "mean_per_label_entropy_mc10",
        "per_class_auc": aucs,
        "test_subset_accuracy_exact_match": subset_acc,
        "mean_uncertainty": mean_u,
        "triage": tri,
        "pathmnist_reference": path_ref,
    }
    out_path = results_dir / "chestmnist_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
