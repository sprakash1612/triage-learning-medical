"""
Temperature scaling for model calibration.

Implements post-hoc calibration method to improve confidence estimates
without retraining the model.

Reference:
    Guo et al. (2017). On Calibration of Modern Neural Networks.
    ICML.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature scaling for model calibration.
    
    Applies a learned temperature parameter to scale logits:
        p_calibrated = softmax(logits / T)
    
    where T is optimized on a validation set to minimize negative log-likelihood.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize TemperatureScaling.
        
        Args:
            model: PyTorch model to calibrate
            device: torch device ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.temperature = 1.0
        self.is_fitted = False
        
        logger.info(f"Initialized TemperatureScaling on {self.device}")
    
    def fit_temperature(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        init_temperature: float = 1.0,
    ) -> float:
        """
        Optimize temperature parameter on validation set.
        
        Uses negative log-likelihood (NLL) as the loss function.
        
        Args:
            val_dataloader: Validation data loader
            num_epochs: Number of optimization epochs
            learning_rate: Learning rate for temperature optimization
            init_temperature: Initial temperature value
        
        Returns:
            Optimized temperature value
        
        Example:
            >>> cal = TemperatureScaling(model)
            >>> temp = cal.fit_temperature(val_dataloader)
            >>> print(f"Optimal temperature: {temp:.4f}")
        """
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32, device=self.device)
        )
        
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=learning_rate,
            max_iter=num_epochs,
        )
        
        loss_fn = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            loss = 0.0
            
            self.model.eval()
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Get model output
                    if isinstance(self.model(x), tuple):
                        logits, _ = self.model(x)
                    else:
                        logits = self.model(x)
                    
                    # Scale logits by temperature
                    scaled_logits = logits / self.temperature
                    
                    batch_loss = loss_fn(scaled_logits, y)
                    loss += batch_loss.item()
            
            # Compute NLL (which is what LBFGS will minimize)
            nll_loss = torch.tensor(loss, dtype=torch.float32, device=self.device)
            
            return nll_loss
        
        # Optimize
        try:
            optimizer.step(closure)
        except Exception as e:
            logger.warning(f"Optimization failed: {e}. Using initial temperature.")
        
        self.temperature = float(self.temperature.detach().cpu().numpy())
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted. Optimal temperature: {self.temperature:.4f}")
        
        return self.temperature
    
    def fit_temperature_simple(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        init_temperature: float = 1.0,
    ) -> float:
        """
        Optimize temperature using simple SGD-based approach.
        
        Alternative to LBFGS that may be more stable in some cases.
        
        Args:
            val_dataloader: Validation data loader
            num_epochs: Number of optimization epochs
            learning_rate: Learning rate
            init_temperature: Initial temperature value
        
        Returns:
            Optimized temperature value
        """
        temperature = torch.tensor(
            init_temperature,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        
        optimizer = torch.optim.SGD([temperature], lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        best_temperature = init_temperature
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            self.model.eval()
            for x, y in val_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Get model output
                if isinstance(self.model(x), tuple):
                    logits, _ = self.model(x)
                else:
                    logits = self.model(x)
                
                # Scale logits by temperature
                scaled_logits = logits / temperature.clamp(min=1e-5)
                
                loss = loss_fn(scaled_logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(val_dataloader)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_temperature = float(temperature.detach().cpu().numpy())
        
        self.temperature = best_temperature
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted. Optimal temperature: {self.temperature:.4f}")
        
        return self.temperature
    
    def apply_temperature_scaling(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits of shape (batch_size, num_classes)
        
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def predict_with_temperature(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get calibrated predictions using temperature scaling.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Dictionary containing:
            - 'logits': Temperature-scaled logits
            - 'probs': Calibrated probabilities
            - 'temperature': Applied temperature value
        """
        self.model.eval()
        x = x.to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model(x), tuple):
                logits, _ = self.model(x)
            else:
                logits = self.model(x)
            
            scaled_logits = self.apply_temperature_scaling(logits)
            probs = torch.softmax(scaled_logits, dim=1)
        
        return {
            'logits': scaled_logits,
            'probs': probs,
            'temperature': self.temperature,
        }
    
    def plot_reliability_diagram(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        num_bins: int = 10,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
        """
        Plot reliability diagram before and after temperature scaling.
        
        Args:
            val_dataloader: Validation data loader
            num_bins: Number of confidence bins
            save_path: Path to save figure (optional)
        
        Returns:
            Tuple of (figure, accuracies, confidences)
        """
        self.model.eval()
        
        all_probs_original = []
        all_probs_scaled = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                if isinstance(self.model(x), tuple):
                    logits, _ = self.model(x)
                else:
                    logits = self.model(x)
                
                # Original probabilities
                probs_original = torch.softmax(logits, dim=1)
                all_probs_original.append(probs_original.cpu().numpy())
                
                # Scaled probabilities
                logits_scaled = self.apply_temperature_scaling(logits)
                probs_scaled = torch.softmax(logits_scaled, dim=1)
                all_probs_scaled.append(probs_scaled.cpu().numpy())
                
                all_labels.append(y.cpu().numpy())
        
        all_probs_original = np.concatenate(all_probs_original, axis=0)
        all_probs_scaled = np.concatenate(all_probs_scaled, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute reliability diagram for both
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (probs, title) in enumerate([
            (all_probs_original, "Before Calibration"),
            (all_probs_scaled, "After Temperature Scaling"),
        ]):
            # Get max probability and predicted class
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            
            # Compute accuracy per bin
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            accuracies = []
            counts = []
            
            for i in range(num_bins):
                mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
                if mask.sum() > 0:
                    acc = (predictions[mask] == all_labels[mask]).mean()
                    accuracies.append(acc)
                    counts.append(mask.sum())
                else:
                    accuracies.append(0)
                    counts.append(0)
            
            accuracies = np.array(accuracies)
            counts = np.array(counts)
            
            # Plot
            ax = axes[idx]
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
            
            # Histogram
            ax.bar(bin_centers, accuracies, width=1/num_bins, alpha=0.7, 
                  color='steelblue', edgecolor='black', label='Accuracy')
            
            # Confidence line
            ax.plot(bin_centers, bin_centers, 'r-', label='Confidence', linewidth=2)
            
            ax.set_xlabel('Confidence', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {save_path}")
        
        return fig, accuracies, bin_centers
    
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save temperature scaling to disk.
        
        Args:
            save_path: Path to save temperature value
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'temperature': self.temperature,
            'is_fitted': self.is_fitted,
        }, save_path)
        
        logger.info(f"Temperature scaling saved to {save_path}")
    
    def load(self, load_path: Union[str, Path]) -> None:
        """
        Load temperature scaling from disk.
        
        Args:
            load_path: Path to load temperature value from
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.temperature = checkpoint['temperature']
        self.is_fitted = checkpoint['is_fitted']
        
        logger.info(f"Temperature scaling loaded from {load_path}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TemperatureScaling(\n"
            f"  temperature={self.temperature:.4f},\n"
            f"  is_fitted={self.is_fitted},\n"
            f")"
        )
