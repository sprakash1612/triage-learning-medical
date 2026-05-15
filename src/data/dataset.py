"""
Custom Dataset classes for enhanced functionality
"""

import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple
import numpy as np


class UncertaintyDataset(Dataset):
    """
    Wrapper dataset that adds uncertainty-related functionality
    Useful for storing predictions and uncertainties during inference
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        predictions: Optional[np.ndarray] = None,
        uncertainties: Optional[np.ndarray] = None
    ):
        self.base_dataset = base_dataset
        self.predictions = predictions
        self.uncertainties = uncertainties
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple:
        item = self.base_dataset[idx]
        
        if self.predictions is not None and self.uncertainties is not None:
            return (*item, self.predictions[idx], self.uncertainties[idx])
        elif self.predictions is not None:
            return (*item, self.predictions[idx])
        else:
            return item
    
    def set_predictions(self, predictions: np.ndarray):
        """Store predictions for all samples"""
        assert len(predictions) == len(self), "Predictions length mismatch"
        self.predictions = predictions
        
    def set_uncertainties(self, uncertainties: np.ndarray):
        """Store uncertainties for all samples"""
        assert len(uncertainties) == len(self), "Uncertainties length mismatch"
        self.uncertainties = uncertainties


class TriageDataset(Dataset):
    """
    Dataset wrapper for triage system evaluation
    Adds deferral decisions and ground truth labels
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        deferral_decisions: Optional[np.ndarray] = None
    ):
        self.base_dataset = base_dataset
        self.deferral_decisions = deferral_decisions
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple:
        item = self.base_dataset[idx]
        
        if self.deferral_decisions is not None:
            return (*item, self.deferral_decisions[idx])
        else:
            return item
    
    def set_deferral_decisions(self, decisions: np.ndarray):
        """Store deferral decisions (0 = AI, 1 = Human)"""
        assert len(decisions) == len(self), "Decisions length mismatch"
        self.deferral_decisions = decisions