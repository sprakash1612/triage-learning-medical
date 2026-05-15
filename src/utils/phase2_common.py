"""
Shared helpers for Phase 2 scripts: PathMNIST loaders, checkpoints, logits, triage human sim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.medmnist_loader import create_dataloaders, compute_class_weights
from src.models.model_factory import ModelFactory, _unpack_logits
from src.utils.phase1_config import build_trainer_config_from_pathmnist_yaml
from src.utils.phase1_human import simulate_human_multiclass

# (display_name, checkpoint default, architecture, variant)
DEFAULT_PHASE2_MODELS: List[Tuple[str, str, str, str]] = [
    ("resnet18", "experiments/resnet18_pathmnist/best_model.pt", "resnet", "18"),
    ("densenet121", "experiments/phase1/densenet121/best_model.pt", "densenet", "121"),
    ("efficientnet_b3", "experiments/phase1/efficientnet_b3/best_model.pt", "efficientnet", "b3"),
    ("vit_tiny", "experiments/phase1/vit_tiny/best_model.pt", "vit", "vit_tiny"),
]


def default_pathmnist_yaml(project_root: Path) -> Path:
    return project_root / "configs" / "pathmnist_config.yaml"


def build_eval_config(
    project_root: Path,
    yaml_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    ypath = yaml_path or default_pathmnist_yaml(project_root)
    cfg = build_trainer_config_from_pathmnist_yaml(
        ypath,
        seed=seed,
        checkpoint_dir=str(project_root / "experiments" / "phase2_tmp"),
        log_dir=str(project_root / "logs" / "phase2"),
        save_every_n_epochs=999,
        overrides=None,
    )
    if cfg["training"]["loss"].get("class_weights") is None:
        tl, _, _ = create_dataloaders(cfg)
        cfg["training"]["loss"]["class_weights"] = compute_class_weights(tl.dataset).tolist()
    return cfg


def get_pathmnist_loaders(
    project_root: Path,
    yaml_path: Optional[Path] = None,
    seed: int = 42,
):
    cfg = build_eval_config(project_root, yaml_path, seed)
    return create_dataloaders(cfg), cfg


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    architecture: str,
    variant: str,
    num_classes: int = 9,
    dropout_rate: float = 0.3,
    pretrained: bool = False,
) -> torch.nn.Module:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "config" in ck and ck["config"] is not None:
        from src.models.model_factory import create_model

        model = create_model(ck["config"])
    else:
        model = ModelFactory.create(
            architecture,
            variant,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            num_channels=3,
            image_size=28,
        )
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list, y_list = [], []
    for bi, (images, labels) in enumerate(tqdm(loader, desc="Collect logits")):
        if max_batches is not None and bi >= max_batches:
            break
        images = images.to(device)
        out = model(images)
        logits = _unpack_logits(out)
        logits_list.append(logits.cpu().numpy())
        y_list.append(labels.numpy().astype(np.int64).ravel())
    return np.concatenate(logits_list, axis=0), np.concatenate(y_list, axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_bin_data(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniform [0,1] confidence bins: bin centers, accuracy per bin, mean confidence per bin, counts.
    """
    confidences = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    correctness = (pred_labels == labels).astype(np.float32)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accs = []
    confs = []
    counts = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        c = int(mask.sum())
        counts.append(c)
        if c == 0:
            accs.append(0.0)
            confs.append(float(bin_centers[i]))
        else:
            accs.append(float(correctness[mask].mean()))
            confs.append(float(confidences[mask].mean()))
    return bin_centers, np.array(accs), np.array(confs), np.array(counts, dtype=np.int64)


def make_human_predictions(
    seed: int,
    targets: np.ndarray,
    human_accuracy: float,
    num_classes: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return simulate_human_multiclass(rng, targets, human_accuracy, num_classes)


def error_reduction(ai_accuracy: float, system_accuracy: float) -> float:
    e_ai = 1.0 - ai_accuracy
    e_sys = 1.0 - system_accuracy
    if e_ai <= 1e-12:
        return float("nan")
    return float((e_ai - e_sys) / e_ai)
