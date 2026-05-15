"""
Training loop implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _unpack_logits(outputs):
    """Support models that return (logits, features)."""
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


class Trainer:
    """
    Model trainer with support for uncertainty and triage evaluation
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        device: torch device
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        self.criterion.to(device)
        self._multilabel = self._is_multilabel_loss()

        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config"""
        opt_config = self.config['training']['optimizer']
        opt_type = opt_config['type'].lower()

        param_groups_cfg = opt_config.get('param_groups')
        if param_groups_cfg and isinstance(param_groups_cfg, list):
            groups = []
            for g in param_groups_cfg:
                entry = {'params': g['params'], 'lr': float(g['lr'])}
                if 'weight_decay' in g:
                    entry['weight_decay'] = float(g['weight_decay'])
                else:
                    entry['weight_decay'] = float(opt_config.get('weight_decay', 0.0))
                groups.append(entry)
            betas = tuple(opt_config.get('betas', (0.9, 0.999)))
            if opt_type == 'adam':
                return optim.Adam(groups, betas=betas)
            if opt_type == 'adamw':
                return optim.AdamW(groups, betas=betas)
            if opt_type == 'sgd':
                return optim.SGD(
                    groups,
                    momentum=float(opt_config.get('momentum', 0.9)),
                )
            raise ValueError(f"Unknown optimizer: {opt_type}")

        if opt_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif opt_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                momentum=opt_config['momentum']
            )
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config['type'].lower()
        
        if sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max']
            )
        elif sched_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=sched_config['patience'],
                factor=sched_config['gamma']
            )
        else:
            scheduler = None
        
        return scheduler

    def _is_multilabel_loss(self) -> bool:
        loss_config = self.config['training']['loss']
        loss_type = loss_config['type'].lower()
        return loss_type in ('weighted_binary_cross_entropy', 'bce_with_logits', 'binary_cross_entropy')
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function"""
        loss_config = self.config['training']['loss']
        loss_type = loss_config['type'].lower()
        
        if loss_type == 'cross_entropy':
            weight = loss_config.get('class_weights')
            if weight is not None:
                w = torch.tensor(weight, dtype=torch.float32)
                criterion = nn.CrossEntropyLoss(
                    weight=w,
                    label_smoothing=loss_config.get('label_smoothing', 0.0),
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=loss_config.get('label_smoothing', 0.0)
                )
        elif loss_type in ('weighted_binary_cross_entropy', 'bce_with_logits', 'binary_cross_entropy'):
            num_classes = self.config['dataset']['num_classes']
            lw = loss_config.get('class_weights')
            pw = float(loss_config.get('pos_weight', 1.0))
            if lw is not None:
                pos_w = torch.tensor(lw, dtype=torch.float32) * pw
            else:
                pos_w = torch.full((num_classes,), pw, dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:
            criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = _unpack_logits(outputs)
            if self._multilabel:
                loss = self.criterion(logits, labels.float())
            else:
                loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            avg_loss: Average validation loss
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        all_logits = []
        
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            logits = _unpack_logits(outputs)
            if self._multilabel:
                loss = self.criterion(logits, labels.float())
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
            else:
                loss = self.criterion(logits, labels)
                predictions = logits.argmax(dim=1)
            
            # Collect results
            total_loss += loss.item()
            num_batches += 1
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        all_logits = np.concatenate(all_logits, axis=0)
        
        if self._multilabel:
            # Exact label-set match (strict multi-label accuracy)
            accuracy = (all_predictions == all_labels).all(axis=1).mean()
            metrics = self._multilabel_auc_metrics(all_labels, all_logits)
            metrics['accuracy'] = float(accuracy)
            metrics['loss'] = float(avg_loss)
        else:
            accuracy = (all_predictions == all_labels).mean()
            metrics = {
                'accuracy': float(accuracy),
                'loss': float(avg_loss)
            }
        
        return avg_loss, metrics

    def _multilabel_auc_metrics(self, y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
        """Per-class and mean AUROC for multi-label (skip degenerate columns)."""
        from sklearn.metrics import roc_auc_score
        y_score = 1.0 / (1.0 + np.exp(-y_logits))
        aucs = []
        auc_indices = []
        for c in range(y_true.shape[1]):
            col_y = y_true[:, c]
            if len(np.unique(col_y)) < 2:
                continue
            try:
                aucs.append(roc_auc_score(col_y, y_score[:, c]))
                auc_indices.append(c)
            except ValueError:
                continue
        mean_auc = float(np.mean(aucs)) if aucs else float('nan')
        out: Dict[str, float] = {'auc_macro': mean_auc}
        for c, a in zip(auc_indices, aucs):
            out[f'auc_class_{c}'] = float(a)
        return out
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (overrides config)
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")

        es_cfg = self.config.get('training', {}).get('early_stopping', {})
        es_enable = bool(es_cfg.get('enable', False))
        es_patience = int(es_cfg.get('patience', 5))
        es_min_delta = float(es_cfg.get('min_delta', 0.0))
        monitor = str(es_cfg.get('monitor_metric', 'accuracy')).lower()
        es_mode = str(es_cfg.get('mode', 'max')).lower()
        epochs_no_improve = 0
        best_monitor = float('-inf') if es_mode == 'max' else float('inf')

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log results
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = val_metrics['accuracy']
                self.save_checkpoint('best_model.pt')
                logger.info(f"Saved best model with accuracy: {self.best_val_metric:.4f}")

            save_every = self.config.get('checkpoint', {}).get('save_every_n_epochs')
            if save_every and (epoch + 1) % int(save_every) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
                logger.info(f"Saved periodic checkpoint at epoch {epoch + 1}")

            if es_enable:
                key = 'accuracy'
                if monitor in ('val_accuracy', 'val_acc', 'accuracy'):
                    key = 'accuracy'
                cur = float(val_metrics.get(key, val_metrics['accuracy']))
                improved = (
                    cur > best_monitor + es_min_delta
                    if es_mode == 'max'
                    else cur < best_monitor - es_min_delta
                )
                if improved:
                    best_monitor = cur
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= es_patience:
                    ck_path = Path(self.config['paths']['checkpoint_dir']) / 'best_model.pt'
                    if ck_path.is_file():
                        self.load_checkpoint('best_model.pt')
                        logger.info(
                            "Early stopping after %s epochs without improvement (patience=%s)",
                            epochs_no_improve,
                            es_patience,
                        )
                    break
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint = torch.load(checkpoint_dir / filename, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_metrics = checkpoint['val_metrics']