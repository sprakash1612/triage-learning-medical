"""
Base model class with uncertainty quantification support
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
import numpy as np


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models
    Provides common functionality for classification with uncertainty

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout in classifier / head where applicable
        dropout_p_mc: Default MC-dropout probability (subclasses may use dedicated layers)
    """

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.5,
        dropout_p_mc: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.dropout_p_mc = dropout_p_mc
        self.dropout_enabled = False
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def enable_dropout(self):
        """Enable dropout for MC Dropout inference"""
        self.dropout_enabled = True
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self):
        """Disable dropout for standard inference"""
        self.dropout_enabled = False
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 30,
        method: str = 'mc_dropout'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation
        
        Args:
            x: Input tensor (B, C, H, W)
            num_samples: Number of forward passes for MC Dropout
            method: 'mc_dropout' or 'temperature_scaling'
        
        Returns:
            predictions: Class predictions (B,)
            uncertainties: Uncertainty scores (B,)
        """
        if method == 'mc_dropout':
            return self._mc_dropout_prediction(x, num_samples)
        elif method == 'temperature_scaling':
            return self._temperature_scaled_prediction(x)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def _mc_dropout_prediction(
        self,
        x: torch.Tensor,
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout prediction"""
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        self.disable_dropout()
        
        # Stack predictions (num_samples, batch_size, num_classes)
        predictions = np.stack(predictions, axis=0)
        
        # Mean prediction
        mean_probs = predictions.mean(axis=0)
        pred_classes = mean_probs.argmax(axis=1)
        
        # Uncertainty as predictive entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        
        return torch.from_numpy(pred_classes), torch.from_numpy(entropy)
    
    def _temperature_scaled_prediction(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Temperature-scaled prediction"""
        with torch.no_grad():
            logits = self.forward(x)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            
            pred_classes = probs.argmax(dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            
        return pred_classes, entropy
    
    def get_num_parameters(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)