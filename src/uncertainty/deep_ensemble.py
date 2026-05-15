"""
Deep Ensemble uncertainty estimation and combination.

Implements ensemble-based uncertainty quantification by training multiple
independent models and combining their predictions.

Reference:
    Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty 
    Estimation using Deep Ensembles. NeurIPS.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from copy import deepcopy
import pickle

logger = logging.getLogger(__name__)


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty estimation.
    
    Combines predictions from multiple independently trained models to estimate
    aleatoric and epistemic uncertainty.
    """
    
    def __init__(
        self,
        model_class: type,
        num_models: int = 5,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Deep Ensemble.
        
        Args:
            model_class: Model class to instantiate
            num_models: Number of models in the ensemble
            device: torch device ('cuda' or 'cpu')
        """
        self.model_class = model_class
        self.num_models = num_models
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models: List[nn.Module] = []
        self.trained_models: List[nn.Module] = []
        
        logger.info(f"Initialized DeepEnsemble with {num_models} models on {self.device}")
    
    def create_ensemble(self, model_kwargs: Dict) -> None:
        """
        Create ensemble of untrained models.
        
        Args:
            model_kwargs: Keyword arguments to pass to model_class constructor
        """
        self.models = []
        
        for i in range(self.num_models):
            model = self.model_class(**model_kwargs)
            model.to(self.device)
            self.models.append(model)
            logger.debug(f"Created ensemble model {i+1}/{self.num_models}")
        
        logger.info(f"Created ensemble with {self.num_models} models")
    
    def train_ensemble(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        loss_fn: Optional[nn.Module] = None,
        num_epochs: int = 10,
        early_stopping_patience: int = 10,
        use_different_seeds: bool = True,
    ) -> List[Dict]:
        """
        Train all models in the ensemble.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer_class: Optimizer class (default: Adam)
            optimizer_kwargs: Keyword arguments for optimizer
            loss_fn: Loss function (default: CrossEntropyLoss)
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            use_different_seeds: Whether to use different random seeds for each model
        
        Returns:
            List of training histories for each model
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 1e-3}
        
        training_histories = []
        
        for model_idx, model in enumerate(self.models):
            logger.info(f"\nTraining ensemble model {model_idx + 1}/{self.num_models}")
            
            # Set different seed for each model if requested
            if use_different_seeds:
                seed = 42 + model_idx
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            history = {"train_loss": [], "val_loss": []}
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (x, y) in enumerate(train_dataloader):
                    x, y = x.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if hasattr(model, 'forward'):
                        output, _ = model(x) if isinstance(model(x), tuple) else (model(x), None)
                    else:
                        output = model(x)
                    
                    loss = loss_fn(output, y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_dataloader)
                history["train_loss"].append(train_loss)
                
                # Validation phase
                if val_dataloader is not None:
                    model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for x, y in val_dataloader:
                            x, y = x.to(self.device), y.to(self.device)
                            
                            if hasattr(model, 'forward'):
                                output, _ = model(x) if isinstance(model(x), tuple) else (model(x), None)
                            else:
                                output = model(x)
                            
                            loss = loss_fn(output, y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_dataloader)
                    history["val_loss"].append(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch+1} "
                            f"(best val_loss: {best_val_loss:.4f})"
                        )
                        break
                    
                    if (epoch + 1) % 5 == 0:
                        logger.debug(
                            f"Epoch {epoch+1}/{num_epochs}, "
                            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                        )
            
            training_histories.append(history)
            self.trained_models.append(model)
        
        logger.info(f"Completed training of {len(self.trained_models)} ensemble models")
        return training_histories
    
    def predict_ensemble(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions from all ensemble models.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            return_all: Whether to return predictions from all models
        
        Returns:
            Dictionary containing:
            - 'mean_logits': Mean logits across models (batch_size, num_classes)
            - 'mean_probs': Mean probabilities across models (batch_size, num_classes)
            - 'all_logits': All logits if return_all=True (num_models, batch_size, num_classes)
            - 'all_probs': All probabilities if return_all=True
        """
        models_to_use = self.trained_models if self.trained_models else self.models
        
        if not models_to_use:
            raise RuntimeError("No models available. Create or train ensemble first.")
        
        all_logits = []
        all_probs = []
        
        x = x.to(self.device)
        
        for model in models_to_use:
            model.eval()
            with torch.no_grad():
                if isinstance(model(x), tuple):
                    logits, _ = model(x)
                else:
                    logits = model(x)
                
                probs = torch.softmax(logits, dim=1)
                all_logits.append(logits)
                all_probs.append(probs)
        
        # Stack predictions
        all_logits = torch.stack(all_logits, dim=0)  # (num_models, batch_size, num_classes)
        all_probs = torch.stack(all_probs, dim=0)
        
        # Compute means
        mean_logits = all_logits.mean(dim=0)
        mean_probs = all_probs.mean(dim=0)
        
        results = {
            'mean_logits': mean_logits,
            'mean_probs': mean_probs,
        }
        
        if return_all:
            results['all_logits'] = all_logits
            results['all_probs'] = all_probs
        
        return results
    
    def compute_ensemble_uncertainty(
        self,
        x: torch.Tensor,
        uncertainty_type: str = "entropy",
    ) -> Dict[str, np.ndarray]:
        """
        Compute uncertainty estimates from ensemble predictions.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            uncertainty_type: One of 'entropy', 'variance', 'mutual_information'
        
        Returns:
            Dictionary containing uncertainty estimates:
            - 'uncertainty': Main uncertainty metric (batch_size,)
            - 'aleatoric': Aleatoric (data) uncertainty (batch_size,)
            - 'epistemic': Epistemic (model) uncertainty (batch_size,)
        """
        predictions = self.predict_ensemble(x, return_all=True)
        mean_probs = predictions['mean_probs']  # (batch_size, num_classes)
        all_probs = predictions['all_probs']  # (num_models, batch_size, num_classes)
        
        results = {}
        
        # Entropy-based uncertainty
        if uncertainty_type == "entropy":
            # Aleatoric uncertainty: mean entropy
            entropy_per_model = -torch.sum(
                all_probs * torch.log(all_probs + 1e-10), dim=2
            )  # (num_models, batch_size)
            aleatoric = entropy_per_model.mean(dim=0)  # (batch_size,)
            
            # Epistemic uncertainty: entropy of mean predictions
            epistemic = -torch.sum(
                mean_probs * torch.log(mean_probs + 1e-10), dim=1
            )  # (batch_size,)
            
            results['uncertainty'] = (aleatoric + epistemic).cpu().numpy()
            results['aleatoric'] = aleatoric.cpu().numpy()
            results['epistemic'] = epistemic.cpu().numpy()
        
        # Variance-based uncertainty
        elif uncertainty_type == "variance":
            # Variance of predictions across models
            variance = torch.var(all_probs, dim=0).sum(dim=1)  # (batch_size,)
            results['uncertainty'] = variance.cpu().numpy()
            results['aleatoric'] = torch.zeros_like(variance).cpu().numpy()
            results['epistemic'] = variance.cpu().numpy()
        
        # Mutual information
        elif uncertainty_type == "mutual_information":
            # MI = H[y] - E[H[y|x]]
            entropy_mean = -torch.sum(
                mean_probs * torch.log(mean_probs + 1e-10), dim=1
            )
            
            entropy_per_model = -torch.sum(
                all_probs * torch.log(all_probs + 1e-10), dim=2
            )
            expected_entropy = entropy_per_model.mean(dim=0)
            
            mutual_info = entropy_mean - expected_entropy
            
            results['uncertainty'] = mutual_info.cpu().numpy()
            results['aleatoric'] = expected_entropy.cpu().numpy()
            results['epistemic'] = mutual_info.cpu().numpy()
        
        else:
            raise ValueError(
                f"uncertainty_type must be one of ['entropy', 'variance', 'mutual_information'], "
                f"got {uncertainty_type}"
            )
        
        return results
    
    def save_ensemble(self, save_dir: Union[str, Path]) -> None:
        """
        Save all ensemble models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models_to_save = self.trained_models if self.trained_models else self.models
        
        for i, model in enumerate(models_to_save):
            model_path = save_dir / f"model_{i}.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved ensemble model {i} to {model_path}")
    
    def load_ensemble(
        self,
        load_dir: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Load ensemble models from disk.
        
        Args:
            load_dir: Directory containing saved models
            device: torch device to load models to
        """
        load_dir = Path(load_dir)
        device = device or self.device
        
        model_files = sorted(load_dir.glob("model_*.pth"))
        
        self.trained_models = []
        
        for model_file in model_files:
            if len(self.models) <= len(self.trained_models):
                break
            
            model = self.models[len(self.trained_models)]
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            self.trained_models.append(model)
            logger.info(f"Loaded ensemble model from {model_file}")
    
    def __len__(self) -> int:
        """Return number of models in ensemble."""
        return self.num_models
    
    def __repr__(self) -> str:
        """String representation."""
        trained_count = len(self.trained_models)
        return f"DeepEnsemble(num_models={self.num_models}, trained={trained_count})"


DeepEnsembleUncertainty = DeepEnsemble
