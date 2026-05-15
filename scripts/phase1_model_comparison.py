#!/usr/bin/env python3
"""
Phase 1 Part A: Train DenseNet121, EfficientNet-B3, ViT-tiny on PathMNIST;
evaluate, MC dropout, triage sweep; CSV + HTML comparison (includes ResNet18 baseline row).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.medmnist_loader import create_dataloaders, compute_class_weights
from src.evaluation.metrics import compute_calibration_metrics
from src.evaluation.triage_evaluator import TriageEvaluator
from src.evaluation.uncertainty_metrics import UncertaintyMetrics
from src.models.model_factory import ModelFactory, create_model
from src.training.trainer import Trainer, _unpack_logits
from src.uncertainty.mc_dropout import MCDropout
from src.utils.logger import setup_logger
from src.utils.phase1_config import build_trainer_config_from_pathmnist_yaml
from src.utils.phase1_human import simulate_human_multiclass
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

MODEL_JOBS: List[Tuple[str, str, str]] = [
    ("densenet121", "densenet", "121"),
    ("efficientnet_b3", "efficientnet", "b3"),
    ("vit_tiny", "vit", "vit_tiny"),
]

BASELINE_ROW = {
    "model": "resnet18",
    "accuracy": 0.8904,
    "balanced_accuracy": 0.8641,
    "f1": 0.8903,
    "ece": float("nan"),
    "mean_uncertainty": 0.583,
    "spearman_corr": 0.498,
    "triage_accuracy": 0.9735,
    "automation_rate": 0.639,
    "error_reduction": 0.758,
    "threshold": 0.4916,
}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 model comparison on PathMNIST")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default=str(ROOT / "configs/pathmnist_config.yaml"))
    p.add_argument("--skip-training", action="store_true", help="Load best_model.pt if present and skip train")
    p.add_argument("--max-epochs", type=int, default=None, help="Override num epochs for smoke tests")
    return p.parse_args()


def _device(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def logits_from_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _unpack_logits(model(x))


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list, y_list = [], []
    for images, labels in tqdm(loader, desc="Standard eval"):
        images = images.to(device)
        logits = logits_from_model(model, images)
        logits_list.append(logits.cpu().numpy())
        y_list.append(labels.numpy().astype(np.int64).ravel())
    logits = np.concatenate(logits_list, axis=0)
    y = np.concatenate(y_list)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = logits.argmax(axis=1)
    return preds, y, probs


def classification_metrics(preds: np.ndarray, y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    acc = float((preds == y).mean())
    bacc = float(balanced_accuracy_score(y, preds))
    f1 = float(f1_score(y, preds, average="weighted", zero_division=0))
    cal = compute_calibration_metrics(probs, y, num_bins=15)
    return {"accuracy": acc, "balanced_accuracy": bacc, "f1": f1, "ece": cal["ece"]}


def run_triage_best(
    preds: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    human_predictions: np.ndarray,
    model_accuracy: float,
    num_thresholds: int = 50,
) -> Dict[str, float]:
    ev = TriageEvaluator(
        model_predictions=preds,
        model_uncertainties=uncertainties,
        human_predictions=human_predictions,
        human_accuracy=0.95,
        targets=targets,
        verbose=False,
    )
    sweep = ev.sweep_thresholds(num_thresholds=num_thresholds)
    best = max(sweep, key=lambda r: r.system_accuracy)
    e_ai = 1.0 - model_accuracy
    e_sys = 1.0 - float(best.system_accuracy)
    if e_ai <= 1e-12:
        err_red = float("nan")
    else:
        err_red = float((e_ai - e_sys) / e_ai)
    return {
        "triage_accuracy": float(best.system_accuracy),
        "automation_rate": float(best.automation_rate),
        "error_reduction": err_red,
        "threshold": float(best.threshold),
    }


def train_one(
    config: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    train_loader,
    val_loader,
) -> None:
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, val_loader)


def build_config(
    yaml_path: Path,
    slug: str,
    arch: str,
    variant: str,
    seed: int,
    max_epochs: int | None,
) -> Dict[str, Any]:
    ckpt_dir = ROOT / "experiments" / "phase1" / slug
    log_dir = ROOT / "logs" / "phase1"
    cfg = build_trainer_config_from_pathmnist_yaml(
        yaml_path,
        seed=seed,
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        save_every_n_epochs=10,
        overrides={
            "experiment": {"name": f"phase1_{slug}"},
            "model": {
                "architecture": arch,
                "variant": variant,
                "pretrained": True,
                "dropout_rate": 0.3,
                "input_channels": 3,
            },
            "training": {
                # YAML default 128 assumes native 28x28; DenseNet/ViT/EfficientNet use ~ImageNet
                # spatial sizes and OOM on typical Kaggle GPUs if batch stays at 128.
                "batch_size": 48,
                "loss": {
                    "type": "cross_entropy",
                    "label_smoothing": 0.1,
                    "class_weights": None,
                }
            },
        },
    )
    if max_epochs is not None:
        cfg["training"]["num_epochs"] = int(max_epochs)
        if cfg["training"]["scheduler"]["type"] == "cosine":
            cfg["training"]["scheduler"]["T_max"] = int(max_epochs)
    return cfg


def fig_to_base64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def write_html(rows: List[Dict[str, Any]], uncertainties_per_model: List[Tuple[str, np.ndarray]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    models = [r["model"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    triage_a = [r["triage_accuracy"] for r in rows]
    auto = [r["automation_rate"] for r in rows]

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(models, accs, color=["#2c5282", "#2f855a", "#b7791f", "#805ad5"][: len(models)])
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Test accuracy by model")
    ax1.tick_params(axis="x", rotation=25)
    img1 = fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(auto, triage_a, s=80, c=range(len(models)), cmap="viridis")
    for i, m in enumerate(models):
        ax2.annotate(m, (auto[i], triage_a[i]), textcoords="offset points", xytext=(4, 4))
    ax2.set_xlabel("Automation rate")
    ax2.set_ylabel("Triage system accuracy")
    ax2.set_title("Triage: accuracy vs automation")
    ax2.grid(True, alpha=0.3)
    img2 = fig_to_base64(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    for name, unc in uncertainties_per_model:
        ax3.hist(unc, bins=40, alpha=0.35, label=name, density=True)
    ax3.set_xlabel("Predictive entropy (MC mean)")
    ax3.set_title("Uncertainty distribution overlay")
    ax3.legend()
    img3 = fig_to_base64(fig3)

    table_html = "<table border='1' cellpadding='6'><tr>"
    cols = list(BASELINE_ROW.keys())
    table_html += "".join(f"<th>{c}</th>" for c in cols)
    table_html += "</tr>"
    for r in rows:
        table_html += "<tr>"
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float) and (v != v):
                v = ""
            table_html += f"<td>{v}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Phase 1 model comparison</title></head>
<body>
<h1>Phase 1 — PathMNIST model comparison</h1>
<h2>Accuracy</h2><img src="data:image/png;base64,{img1}" />
<h2>Triage scatter</h2><img src="data:image/png;base64,{img2}" />
<h2>Uncertainty overlay</h2><img src="data:image/png;base64,{img3}" />
<h2>Summary table</h2>
{table_html}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    device = _device(args.device)
    yaml_path = Path(args.config)
    results_dir = ROOT / "results" / "phase1"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "phase1" / "model_comparison.log"
    setup_logger(log_path)
    set_seed(args.seed)

    human_accuracy = 0.95
    rows: List[Dict[str, Any]] = []
    unc_for_plot: List[Tuple[str, np.ndarray]] = []

    for slug, arch, variant in tqdm(MODEL_JOBS, desc="Models"):
        cfg = build_config(yaml_path, slug, arch, variant, args.seed, args.max_epochs)
        if cfg["training"]["loss"].get("class_weights") is None:
            tmp_loaders = create_dataloaders(cfg)
            cw = compute_class_weights(tmp_loaders[0].dataset)
            cfg["training"]["loss"]["class_weights"] = cw.tolist()

        train_loader, val_loader, test_loader = create_dataloaders(cfg)
        num_classes = cfg["dataset"]["num_classes"]
        model = create_model(cfg)
        ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"

        if args.skip_training and ckpt_path.is_file():
            logger.info("Skipping training; loading %s", ckpt_path)
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            model.to(device)
        else:
            model.to(device)
            train_one(cfg, model, device, train_loader, val_loader)

        preds, y, probs = collect_predictions(model, test_loader, device)
        m_std = classification_metrics(preds, y, probs)

        rng_human = np.random.default_rng(args.seed + 1337)
        human_preds = simulate_human_multiclass(rng_human, y, human_accuracy, num_classes)

        mc = MCDropout(model, num_samples=10, device=device)
        mc_preds, uncertainties, mean_probs = mc.predict(test_loader, multilabel=False)
        mean_u = float(np.mean(uncertainties))
        errors = (mc_preds != y).astype(np.int32)
        uq = UncertaintyMetrics.compute_uncertainty_quality(uncertainties, errors)
        spear = float(uq["spearman_corr"])
        ece_mc = compute_calibration_metrics(mean_probs, y, num_bins=15)["ece"]

        tri = run_triage_best(mc_preds, uncertainties, y, human_preds, m_std["accuracy"], 50)
        unc_for_plot.append((slug, uncertainties.copy()))

        rows.append(
            {
                "model": slug,
                "accuracy": m_std["accuracy"],
                "balanced_accuracy": m_std["balanced_accuracy"],
                "f1": m_std["f1"],
                "ece": ece_mc,
                "mean_uncertainty": mean_u,
                "spearman_corr": spear,
                "triage_accuracy": tri["triage_accuracy"],
                "automation_rate": tri["automation_rate"],
                "error_reduction": tri["error_reduction"],
                "threshold": tri["threshold"],
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_rows = [BASELINE_ROW] + rows

    csv_path = results_dir / "model_comparison.csv"
    fieldnames = list(BASELINE_ROW.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: ("" if (isinstance(r[k], float) and np.isnan(r[k])) else r[k]) for k in fieldnames})

    write_html(all_rows, unc_for_plot, results_dir / "model_comparison_report.html")
    logger.info("Wrote %s and HTML report", csv_path)


if __name__ == "__main__":
    main()
