"""Human simulation for triage evaluation (reproducible RNG)."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def simulate_human_multiclass(
    rng: Generator,
    targets: np.ndarray,
    human_accuracy: float,
    num_classes: int,
) -> np.ndarray:
    """
    One label per sample. Correct with probability human_accuracy; else random class.
    """
    n = len(targets)
    human = np.empty(n, dtype=np.int64)
    correct = rng.random(n) < human_accuracy
    human[correct] = targets[correct]
    wrong_n = int((~correct).sum())
    if wrong_n > 0:
        human[~correct] = rng.integers(0, num_classes, size=wrong_n, endpoint=False)
    return human


def simulate_human_multilabel(
    rng: Generator,
    targets: np.ndarray,
    human_accuracy: float,
) -> np.ndarray:
    """Per-label: with probability human_accuracy match ground truth; else flip label."""
    t = targets.astype(np.float32)
    human = t.copy()
    wrong = rng.random(t.shape) > human_accuracy
    human[wrong] = 1.0 - human[wrong]
    return human
