"""Shared Phase 3 helpers: Phase 1 triage thresholds, model checkpoint list."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# (csv_model_name, checkpoint relative to project root, architecture, variant)
PATHMNIST_PHASE3_MODELS: List[Tuple[str, str, str, str]] = [
    ("resnet18", "experiments/resnet18_pathmnist/best_model.pt", "resnet", "18"),
    ("densenet121", "experiments/phase1/densenet121/best_model.pt", "densenet", "121"),
    ("efficientnet_b3", "experiments/phase1/efficientnet_b3/best_model.pt", "efficientnet", "b3"),
    ("vit_tiny", "experiments/phase1/vit_tiny/best_model.pt", "vit", "vit_tiny"),
]

DEFAULT_RESNET_THRESHOLD = 0.4916


def load_phase1_thresholds(project_root: Path, csv_path: Path | None = None) -> Dict[str, float]:
    """Load per-model triage thresholds from Phase 1 model_comparison.csv."""
    path = csv_path or (project_root / "results" / "phase1" / "model_comparison.csv")
    out: Dict[str, float] = {}
    if not path.is_file():
        logger.warning("Missing %s — only default resnet18 threshold available", path)
        return {"resnet18": DEFAULT_RESNET_THRESHOLD}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = row.get("model", "").strip()
            t = row.get("threshold", "").strip()
            if m and t:
                try:
                    out[m] = float(t)
                except ValueError:
                    continue
    if "resnet18" not in out:
        out["resnet18"] = DEFAULT_RESNET_THRESHOLD
    return out
