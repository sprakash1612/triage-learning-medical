"""
Publication-ready visualizations for triage system evaluation.

Includes training curves, confusion matrices, ROC curves, uncertainty
distributions, calibration analysis, and triage performance plots.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


class TriageVisualizations:
    """Static methods for creating publication-ready visualizations."""
    
    # Default color palette
    COLORS = {
        'ai': '#1f77b4',
        'human': '#ff7f0e',
        'system': '#2ca02c',
        'error': '#d62728',
        'correct': '#2ca02c',
    }
    
    @staticmethod
    def plot_training_curves(
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """
        Plot training loss and metrics over epochs.
        
        Args:
            history: Dict mapping metric name → list of values
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size (width, height)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> history = {
            ...     'train_loss': [...],
            ...     'val_loss': [...],
            ...     'train_acc': [...],
            ...     'val_acc': [...],
            ... }
            >>> fig = TriageVisualizations.plot_training_curves(history)
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        if 'train_loss' in history and 'val_loss' in history:
            epochs = np.arange(len(history['train_loss']))
            axes[0].plot(epochs, history['train_loss'], 'o-', label='Train', alpha=0.7)
            axes[0].plot(epochs, history['val_loss'], 's-', label='Val', alpha=0.7)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_acc' in history and 'val_acc' in history:
            epochs = np.arange(len(history['train_acc']))
            axes[1].plot(epochs, history['train_acc'], 'o-', label='Train', alpha=0.7)
            axes[1].plot(epochs, history['val_acc'], 's-', label='Val', alpha=0.7)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy')
            axes[1].set_ylim([0, 1])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
        normalize: bool = False,
    ) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm: (n_classes, n_classes) confusion matrix
            class_names: Optional list of class names
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size
            normalize: Whether to normalize by true class (row)
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize if requested
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            text_values = cm_normalized
        else:
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            text_values = cm
        
        # Labels
        n_classes = len(cm)
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                if normalize:
                    text = f"{text_values[i, j]:.2f}"
                else:
                    text = f"{int(cm[i, j])}"
                
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if text_values[i, j] > 0.5 else 'black',
                       fontsize=10)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_roc_curve(
        targets: np.ndarray,
        scores: np.ndarray,
        class_name: str = "Class",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> Tuple[plt.Figure, float]:
        """
        Plot ROC curve and compute AUC.
        
        Args:
            targets: Binary labels (0/1)
            scores: Prediction scores (0-1)
            class_name: Name of positive class
            save_path: Optional path to save
            figsize: Figure size
        
        Returns:
            (Figure, AUC) tuple
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, 'o-', linewidth=2, label=f'{class_name} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve - {class_name}', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig, roc_auc
    
    @staticmethod
    def plot_uncertainty_distribution(
        uncertainties: np.ndarray,
        errors: np.ndarray,
        title: str = "Uncertainty Distribution",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5),
        bins: int = 30,
    ) -> plt.Figure:
        """
        Plot uncertainty distributions for correct vs. incorrect predictions.
        
        Args:
            uncertainties: Uncertainty estimates
            errors: Binary error indicators (1=error)
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size
            bins: Number of histogram bins
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        correct_unc = uncertainties[errors == 0]
        error_unc = uncertainties[errors == 1]
        
        # Histograms
        axes[0].hist(correct_unc, bins=bins, alpha=0.6, label='Correct', 
                    color=TriageVisualizations.COLORS['correct'])
        axes[0].hist(error_unc, bins=bins, alpha=0.6, label='Error',
                    color=TriageVisualizations.COLORS['error'])
        axes[0].set_xlabel('Uncertainty', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Histogram', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # KDE / Density
        try:
            from scipy.stats import gaussian_kde
            
            if len(correct_unc) > 1:
                kde_correct = gaussian_kde(correct_unc)
                x_range = np.linspace(uncertainties.min(), uncertainties.max(), 200)
                axes[1].plot(x_range, kde_correct(x_range), label='Correct',
                           color=TriageVisualizations.COLORS['correct'], linewidth=2)
            
            if len(error_unc) > 1:
                kde_error = gaussian_kde(error_unc)
                x_range = np.linspace(uncertainties.min(), uncertainties.max(), 200)
                axes[1].plot(x_range, kde_error(x_range), label='Error',
                           color=TriageVisualizations.COLORS['error'], linewidth=2)
            
            axes[1].set_xlabel('Uncertainty', fontsize=11)
            axes[1].set_ylabel('Density', fontsize=11)
            axes[1].set_title('Kernel Density', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        except Exception as e:
            logger.warning(f"Could not plot KDE: {e}")
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Uncertainty distribution saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_triage_performance(
        results: List[Dict[str, float]],
        title: str = "Triage Performance",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot automation rate vs. system accuracy trade-off.
        
        Args:
            results: List of dicts with 'automation_rate', 'system_accuracy', 'cost_total'
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        automation_rates = [r['automation_rate'] for r in results]
        accuracies = [r['system_accuracy'] for r in results]
        costs = [r['cost_total'] for r in results]
        
        # Automation vs. Accuracy
        axes[0].plot(automation_rates, accuracies, 'o-', linewidth=2, markersize=6,
                    color=TriageVisualizations.COLORS['system'])
        axes[0].set_xlabel('Automation Rate', fontsize=11)
        axes[0].set_ylabel('System Accuracy', fontsize=11)
        axes[0].set_title('Automation vs. Accuracy', fontsize=12)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        
        # Automation vs. Cost
        axes[1].plot(automation_rates, costs, 's-', linewidth=2, markersize=6,
                    color=TriageVisualizations.COLORS['system'])
        axes[1].set_xlabel('Automation Rate', fontsize=11)
        axes[1].set_ylabel('Total Cost', fontsize=11)
        axes[1].set_title('Automation vs. Cost', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Triage performance plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_calibration_analysis(
        accuracies: np.ndarray,
        confidences: np.ndarray,
        title: str = "Calibration Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Tuple[plt.Figure, float]:
        """
        Plot reliability diagram and compute ECE.
        
        Args:
            accuracies: Per-bin accuracy (fraction correct)
            confidences: Per-bin average confidence
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size
        
        Returns:
            (Figure, ECE) tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute ECE
        ece = np.abs(accuracies - confidences).mean()
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
        
        # Model calibration
        bin_size = 1.0 / len(confidences)
        ax.bar(confidences, accuracies - confidences, width=bin_size*0.8, alpha=0.5,
              color=TriageVisualizations.COLORS['system'], label='Model')
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(f'{title} (ECE={ece:.4f})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration analysis saved to {save_path}")
        
        return fig, ece
    
    @staticmethod
    def plot_attention_maps(
        images: np.ndarray,
        attention_maps: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray = None,
        save_path: Optional[str] = None,
        max_samples: int = 6,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Visualize ViT attention maps overlaid on images.
        
        Args:
            images: (n_samples, H, W, C) or (n_samples, H, W) image data
            attention_maps: (n_samples, H_attn, W_attn) attention maps
            predictions: (n_samples,) Predicted classes
            targets: (n_samples,) Ground truth classes (optional)
            save_path: Optional path to save
            max_samples: Maximum samples to display
            figsize: Figure size (auto-computed if None)
        
        Returns:
            matplotlib Figure
        """
        n_show = min(max_samples, len(images))
        
        if figsize is None:
            figsize = (3*n_show, 6 if targets is not None else 3)
        
        n_rows = 2 if targets is not None else 1
        fig, axes = plt.subplots(n_rows, n_show, figsize=figsize)
        
        if n_show == 1:
            axes = axes.reshape(n_rows, 1)
        
        for i in range(n_show):
            # Image
            if images[i].ndim == 3 and images[i].shape[2] == 3:
                # RGB
                axes[0, i].imshow(images[i])
            else:
                # Grayscale
                axes[0, i].imshow(images[i], cmap='gray')
            
            # Attention overlay
            attention_resized = attention_maps[i]
            if attention_resized.shape != images[i].shape[:2]:
                import cv2
                attention_resized = cv2.resize(
                    attention_resized,
                    (images[i].shape[1], images[i].shape[0])
                )
            
            axes[0, i].imshow(attention_resized, alpha=0.4, cmap='hot')
            
            pred_text = f"Pred: {predictions[i]}"
            if targets is not None:
                pred_text += f" (True: {targets[i]})"
                correct = "✓" if predictions[i] == targets[i] else "✗"
                axes[0, i].set_title(f"{pred_text} {correct}", fontsize=10)
            else:
                axes[0, i].set_title(pred_text, fontsize=10)
            
            axes[0, i].axis('off')
            
            # Attention map only
            if targets is not None:
                axes[1, i].imshow(attention_resized, cmap='hot')
                axes[1, i].axis('off')
        
        fig.suptitle('ViT Attention Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention maps saved to {save_path}")
        
        return fig
    
    @staticmethod
    def create_results_dashboard(
        history: Dict[str, List],
        final_metrics: Dict[str, float],
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
    ) -> plt.Figure:
        """
        Create comprehensive results dashboard with multiple subplots.
        
        Args:
            history: Training history
            final_metrics: Dictionary of final metrics
            confusion_matrix: Final confusion matrix
            class_names: List of class names
            save_path: Optional path to save
            figsize: Figure size
        
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)
        
        # Training loss
        ax1 = fig.add_subplot(gs[0, 0])
        if 'train_loss' in history and 'val_loss' in history:
            ax1.plot(history['train_loss'], label='Train')
            ax1.plot(history['val_loss'], label='Val')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Training accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        if 'train_acc' in history and 'val_acc' in history:
            ax2.plot(history['train_acc'], label='Train')
            ax2.plot(history['val_acc'], label='Val')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy')
            ax2.set_ylim([0, 1])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Metrics bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        metrics_dict = {k: v for k, v in final_metrics.items() if isinstance(v, (int, float))}
        metrics_keys = list(metrics_dict.keys())[:10]  # Top 10 metrics
        metrics_values = [metrics_dict[k] for k in metrics_keys]
        ax3.barh(metrics_keys, metrics_values, color=TriageVisualizations.COLORS['system'])
        ax3.set_xlabel('Value')
        ax3.set_title('Final Metrics')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Confusion matrix
        ax4 = fig.add_subplot(gs[1, :])
        im = ax4.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        n_classes = len(confusion_matrix)
        ax4.set_xticks(np.arange(n_classes))
        ax4.set_yticks(np.arange(n_classes))
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.set_yticklabels(class_names)
        
        for i in range(n_classes):
            for j in range(n_classes):
                ax4.text(j, i, f"{int(confusion_matrix[i, j])}",
                       ha='center', va='center',
                       color='white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black')
        
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        ax4.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax4)
        
        fig.suptitle('Results Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results dashboard saved to {save_path}")
        
        return fig
