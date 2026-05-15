#!/usr/bin/env python3
"""
Phase 3 Part B: DermaMNIST transfer (EfficientNet-B3 head-only vs full fine-tune) + ResNet18 scratch baseline.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.medmnist_loader import compute_class_weights, create_dataloaders
from src.evaluation.triage_evaluator import TriageEvaluator
from src.models.model_factory import ModelFactory, _unpack_logits, create_model
from src.training.trainer import Trainer
from src.uncertainty.mc_dropout import MCDropout
from src.utils.logger import setup_logger
from src.utils.phase1_config import build_trainer_config_from_dermamnist_yaml
from src.utils.phase1_human import simulate_human_multiclass
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

HUMAN_ACC = 0.95
MC_SAMPLES = 10
DERMA_YAML = ROOT / "configs" / "dermamnist_config.yaml"
EFF_CKPT = ROOT / "experiments" / "phase1" / "efficientnet_b3" / "best_model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 DermaMNIST experiments")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "phase3"))
    return p.parse_args()


def device_arg(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def replace_efficientnet_head(model: nn.Module, num_classes: int) -> nn.Module:
    clf = model.classifier
    if isinstance(clf, nn.Sequential):
        layers = list(clf.children())
        last = layers[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("Expected final classifier layer to be Linear")
        in_f = last.in_features
        layers[-1] = nn.Linear(in_f, num_classes)
        model.classifier = nn.Sequential(*layers)
    else:
        raise TypeError("Expected Sequential classifier on EfficientNetClassifier")
    return model


def load_efficientnet_b3_from_pathmnist(device: torch.device) -> nn.Module:
    model = ModelFactory.create(
        "efficientnet",
        "b3",
        num_classes=9,
        pretrained=True,
        dropout_rate=0.3,
        num_channels=3,
        image_size=28,
    )
    ck = torch.load(EFF_CKPT, map_location=device, weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.to(device)
    return model


def build_base_derma_config(
    seed: int,
    checkpoint_dir: Path,
    num_epochs: int,
    lr: float,
    early: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = build_trainer_config_from_dermamnist_yaml(
        DERMA_YAML,
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(ROOT / "logs" / "phase3"),
        save_every_n_epochs=999,
        overrides={
            "model": {
                "architecture": "efficientnet",
                "variant": "efficientnet_b3",
                "pretrained": True,
                "num_classes": 7,
                "dropout_rate": 0.3,
                "input_channels": 3,
            },
            "training": {
                "num_epochs": num_epochs,
                "batch_size": 64,
                "optimizer": {
                    "type": "adamw",
                    "lr": lr,
                    "weight_decay": 1e-4,
                    "momentum": 0.9,
                    "betas": (0.9, 0.999),
                },
                "scheduler": {"type": "cosine", "T_max": num_epochs, "step_size": 30, "gamma": 0.1, "patience": 10},
                "loss": {"type": "cross_entropy", "label_smoothing": 0.0, "class_weights": None},
                "early_stopping": early or {},
            },
        },
    )
    return cfg


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ps, ys, ls = [], [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        logits = _unpack_logits(model(images))
        ps.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(labels.numpy().astype(np.int64).ravel())
        ls.append(logits.cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(ls, axis=0)


def run_experiment_report(
    name: str,
    model: nn.Module,
    test_loader,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    preds, y, logits = evaluate_model(model, test_loader, device)
    acc = float((preds == y).mean())
    bacc = float(balanced_accuracy_score(y, preds))
    f1w = float(f1_score(y, preds, average="weighted", zero_division=0))

    mc = MCDropout(model, num_samples=MC_SAMPLES, device=device)
    mc_preds, uncertainties, _ = mc.predict(test_loader, multilabel=False)
    mean_u = float(np.mean(uncertainties))
    err = (mc_preds != y).astype(np.float64)
    sp_r, sp_p = spearmanr(uncertainties, err)

    rng = np.random.default_rng(seed + 11)
    human = simulate_human_multiclass(rng, y, HUMAN_ACC, 7)
    ev = TriageEvaluator(mc_preds, uncertainties, human_predictions=human, human_accuracy=HUMAN_ACC, targets=y, verbose=False)
    sweep = ev.sweep_thresholds(num_thresholds=50)
    best = max(sweep, key=lambda r: r.system_accuracy)
    e_ai = 1.0 - acc
    e_sys = 1.0 - float(best.system_accuracy)
    err_red = float((e_ai - e_sys) / e_ai) if e_ai > 1e-12 else float("nan")

    return {
        "experiment": name,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1": f1w,
        "mean_uncertainty": mean_u,
        "spearman_corr": float(sp_r) if sp_r == sp_r else float("nan"),
        "spearman_p": float(sp_p) if sp_p == sp_p else float("nan"),
        "triage_accuracy": float(best.system_accuracy),
        "automation_rate": float(best.automation_rate),
        "error_reduction": err_red,
    }


def train_with_trainer(model: nn.Module, cfg: Dict[str, Any], device: torch.device, train_loader, val_loader) -> None:
    trainer = Trainer(model, cfg, device)
    trainer.train(train_loader, val_loader)


def main():
    args = parse_args()
    device = device_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "phase3").mkdir(parents=True, exist_ok=True)
    setup_logger(ROOT / "logs" / "phase3" / "dermamnist.log")
    set_seed(args.seed)

    # --- DermaMNIST loaders (one base config for splits / aug) ---
    base_cfg = build_base_derma_config(args.seed, ROOT / "experiments" / "phase3" / "_tmp_loaders", 1, 1e-3)
    if base_cfg["training"]["loss"].get("class_weights") is None:
        tl, _, _ = create_dataloaders(base_cfg)
        base_cfg["training"]["loss"]["class_weights"] = compute_class_weights(tl.dataset).tolist()
    train_loader, val_loader, test_loader = create_dataloaders(base_cfg)

    results: List[Dict[str, Any]] = []

    # ----- B1: head only -----
    model_b1 = load_efficientnet_b3_from_pathmnist(device)
    model_b1 = replace_efficientnet_head(model_b1, 7)
    for p in model_b1.backbone.parameters():
        p.requires_grad = False
    for p in model_b1.classifier.parameters():
        p.requires_grad = True

    cfg_b1 = build_base_derma_config(
        args.seed,
        ROOT / "experiments" / "phase3" / "dermamnist_head",
        num_epochs=10,
        lr=0.01,
    )
    cfg_b1["training"]["loss"]["class_weights"] = base_cfg["training"]["loss"]["class_weights"]
    train_with_trainer(model_b1, cfg_b1, device, train_loader, val_loader)
    ck_b1 = torch.load(Path(cfg_b1["paths"]["checkpoint_dir"]) / "best_model.pt", map_location=device, weights_only=False)
    model_b1.load_state_dict(ck_b1["model_state_dict"])
    rep_b1 = run_experiment_report("B1_head_only_10ep", model_b1, test_loader, device, args.seed)
    results.append(rep_b1)

    # ----- B2: full fine-tune + early stopping -----
    model_b2 = load_efficientnet_b3_from_pathmnist(device)
    model_b2 = replace_efficientnet_head(model_b2, 7)
    for p in model_b2.parameters():
        p.requires_grad = True
    model_b2.to(device)

    cfg_b2 = build_base_derma_config(
        args.seed,
        ROOT / "experiments" / "phase3" / "dermamnist_finetuned",
        num_epochs=30,
        lr=1e-3,
        early={
            "enable": True,
            "patience": 5,
            "monitor_metric": "accuracy",
            "mode": "max",
            "min_delta": 1e-6,
        },
    )
    cfg_b2["training"]["loss"]["class_weights"] = base_cfg["training"]["loss"]["class_weights"]
    cfg_b2["training"]["optimizer"] = {
        "type": "adamw",
        "betas": (0.9, 0.999),
        "param_groups": [
            {"params": list(model_b2.backbone.parameters()), "lr": 1e-3, "weight_decay": 1e-4},
            {"params": list(model_b2.classifier.parameters()), "lr": 1e-2, "weight_decay": 1e-4},
        ],
    }
    train_with_trainer(model_b2, cfg_b2, device, train_loader, val_loader)
    ck_b2 = torch.load(Path(cfg_b2["paths"]["checkpoint_dir"]) / "best_model.pt", map_location=device, weights_only=False)
    model_b2.load_state_dict(ck_b2["model_state_dict"])
    rep_b2 = run_experiment_report("B2_full_finetune_earlystop", model_b2, test_loader, device, args.seed)
    results.append(rep_b2)

    # ----- Scratch ResNet18 -----
    cfg_rn = build_base_derma_config(
        args.seed,
        ROOT / "experiments" / "phase3" / "dermamnist_scratch",
        num_epochs=50,
        lr=1e-3,
    )
    cfg_rn["model"] = {
        "architecture": "resnet",
        "variant": "18",
        "pretrained": False,
        "num_classes": 7,
        "dropout_rate": 0.3,
        "input_channels": 3,
    }
    cfg_rn["training"]["loss"]["class_weights"] = base_cfg["training"]["loss"]["class_weights"]
    cfg_rn["training"]["scheduler"]["T_max"] = 50
    model_rn = create_model(cfg_rn).to(device)
    train_with_trainer(model_rn, cfg_rn, device, train_loader, val_loader)
    ck_rn = torch.load(Path(cfg_rn["paths"]["checkpoint_dir"]) / "best_model.pt", map_location=device, weights_only=False)
    model_rn.load_state_dict(ck_rn["model_state_dict"])
    rep_rn = run_experiment_report("scratch_resnet18_50ep", model_rn, test_loader, device, args.seed)
    results.append(rep_rn)

    csv_path = out_dir / "dermamnist_comparison.csv"
    keys = [
        "experiment",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "mean_uncertainty",
        "spearman_corr",
        "triage_accuracy",
        "automation_rate",
        "error_reduction",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    names = [r["experiment"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, [r["accuracy"] for r in results], color=["#3182ce", "#38a169", "#dd6b20"])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test accuracy")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(names, [r["triage_accuracy"] for r in results], color=["#3182ce", "#38a169", "#dd6b20"])
    axes[1].set_ylabel("Triage accuracy")
    axes[1].set_title("Triage (best threshold sweep)")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(out_dir / "dermamnist_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
