"""
Monte Carlo Dropout for uncertainty estimation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


def _unpack_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


class MCDropout:
    """
    Monte Carlo Dropout uncertainty estimator

    Args:
        model: PyTorch model with dropout layers
        num_samples: Number of forward passes
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 30,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model
        self.num_samples = num_samples
        self.device = device

    def enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_all_predictions: bool = False,
        multilabel: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates

        Args:
            dataloader: DataLoader for inference
            return_all_predictions: Return all MC samples
            multilabel: If True, sigmoid per label; predictions shape (N, C); uncertainty is
                mean per-label entropy across labels.

        Returns:
            predictions: (N,) class indices for softmax, or (N, C) binary for multilabel
            uncertainties: (N,) scores
            probabilities: Mean probabilities (N, num_classes)
            all_predictions: All MC samples if requested (num_samples, N, num_classes)
        """
        self.model.eval()
        self.enable_dropout()

        all_predictions = []

        for _ in range(self.num_samples):
            batch_preds = []

            with torch.no_grad():
                for images, _ in tqdm(dataloader, desc="MC Dropout sampling"):
                    images = images.to(self.device)
                    logits = _unpack_logits(self.model(images))
                    if multilabel:
                        probs = torch.sigmoid(logits)
                    else:
                        probs = torch.softmax(logits, dim=1)
                    batch_preds.append(probs.cpu().numpy())

            all_predictions.append(np.concatenate(batch_preds, axis=0))

        all_predictions = np.stack(all_predictions, axis=0)
        mean_probs = all_predictions.mean(axis=0)

        if multilabel:
            predictions = (mean_probs > 0.5).astype(np.int64)
            uncertainties = self._compute_multilabel_uncertainty(mean_probs)
        else:
            predictions = mean_probs.argmax(axis=1)
            uncertainties = self._compute_uncertainty(all_predictions)

        if return_all_predictions:
            return predictions, uncertainties, mean_probs, all_predictions
        return predictions, uncertainties, mean_probs

    def _compute_multilabel_uncertainty(self, mean_probs: np.ndarray) -> np.ndarray:
        """Mean binary entropy across labels."""
        p = np.clip(mean_probs, 1e-10, 1 - 1e-10)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return ent.mean(axis=1)

    def _compute_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        mean_probs = predictions.mean(axis=0)
        return -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

    def compute_mutual_information(self, predictions: np.ndarray) -> np.ndarray:
        mean_probs = predictions.mean(axis=0)
        predictive_entropy = -np.sum(
            mean_probs * np.log(mean_probs + 1e-10),
            axis=1
        )
        expected_entropy = -np.mean(
            np.sum(predictions * np.log(predictions + 1e-10), axis=2),
            axis=0
        )
        return predictive_entropy - expected_entropy


MCDropoutUncertainty = MCDropout
