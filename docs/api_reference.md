# API Reference Documentation

## Quick Start Guide

### Installation
```bash
git clone <repository-url>
cd triage-learning-medical-diagnosis
pip install -r requirements.txt
```

### Training a Model
```bash
# Download data
python scripts/download_data.py --all

# Train model
python scripts/train.py \
  --config configs/chestmnist_config.yaml \
  --dataset chestmnist \
  --output experiments/exp_001/

# Evaluate model
python scripts/evaluate.py \
  --model-path experiments/exp_001/best_model.pt \
  --dataset chestmnist

# Perform triage analysis
python scripts/triage_analysis.py \
  --model-path experiments/exp_001/best_model.pt \
  --dataset chestmnist

# Generate final report
python scripts/generate_report.py \
  --experiment-dir experiments/exp_001/
```

---

## Module APIs

### Data Processing (`src.data`)

#### `MedMNISTLoader`
```python
from src.data.medmnist_loader import MedMNISTLoader

loader = MedMNISTLoader(
    dataset_name: str,      # "pathmnist", "chestmnist", "dermamnist"
    download: bool = True,  # Auto-download if missing
    data_dir: str = "./data"
)

# Loading datasets
train_data = loader.load_train()   # Returns numpy arrays (images, labels)
val_data = loader.load_val()
test_data = loader.load_test()

# Data statistics
info = loader.get_dataset_info()  # {name, num_classes, num_samples, image_size}
```

**Returns**:
- `train_data`: Tuple[np.ndarray, np.ndarray] - (images, labels)
- `images`: Shape (N, H, W, C), dtype uint8, range [0, 255]
- `labels`: Shape (N,) for single-label, (N, C) for multi-label

---

#### `Dataset`
```python
from src.data.dataset import MedMNISTDataset
import torch

dataset = MedMNISTDataset(
    images: np.ndarray,           # (N, H, W, C)
    labels: np.ndarray,           # (N,) or (N, C)
    split: str = "train",         # "train", "val", "test"
    augmentation: Optional[dict] = None,
    preprocessing: Optional[dict] = None
)

# PyTorch iteration
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
for images, labels in dataloader:
    # images: (B, C, H, W) float32, normalized
    # labels: (B,) or (B, C) depending on task
    pass

# Access single sample
image, label = dataset[0]
```

**Methods**:
- `__len__()`: Returns number of samples
- `__getitem__(idx)`: Returns (image, label) pair
- `get_class_names()`: Returns list of class names
- `get_class_distribution()`: Returns class balance statistics

---

#### `Preprocessor`
```python
from src.data.preprocessing import Preprocessor

preprocessor = Preprocessor(
    dataset: str = "pathmnist",       # Dataset name
    normalization: str = "imagenet",  # "imagenet", "medical", "none"
    dtype: torch.dtype = torch.float32
)

# Preprocess image
image_tensor = preprocessor.preprocess(image_np)

# Reverse preprocessing (for visualization)
image_np = preprocessor.inverse_transform(image_tensor)

# Get normalization statistics
mean, std = preprocessor.get_normalization_stats()
```

**Normalization Types**:
- `"imagenet"`: ImageNet statistics [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- `"medical"`: Medical imaging [0.5], std [0.5]
- `"none"`: Scale to [0, 1] range only

---

#### `create_dataloader`
```python
from src.data.dataloader import create_dataloader

dataloader = create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    stratified: bool = True  # Maintain class distribution
)
```

**Returns**: `torch.utils.data.DataLoader` instance

---

### Model Architecture (`src.models`)

#### `ModelFactory`
```python
from src.models.model_factory import ModelFactory

# Create model
model = ModelFactory.create(
    architecture: str,          # "resnet", "efficientnet", "densenet", "vit"
    variant: str,              # "resnet50", "b3", "densenet121", "vit_base"
    num_classes: int = 9,      # Output classes
    pretrained: bool = True,   # ImageNet pretraining
    dropout_rate: float = 0.3,
    device: str = "cuda"
)

# Model properties
num_params = model.get_num_parameters()
input_size = model.get_input_size()  # (3, 224, 224) or similar
```

**Supported Architectures**:
```
Architecture    Variants                          Best For
ResNet          resnet18, resnet34, resnet50,    Baseline, proven
                resnet101, resnet152              
EfficientNet    b0, b1, b2, b3, b4, b5, b6, b7  Efficiency, medical
DenseNet        densenet121, densenet169,        Parameter efficiency
                densenet201, densenet264
ViT             vit_tiny, vit_small, vit_base,  Color features,
                vit_large                        global context
```

---

#### `BaseModel`
```python
from src.models.base_model import BaseModel
import torch

model = BaseModel(...)
model.to("cuda")
model.eval()

# Forward pass
with torch.no_grad():
    output = model(images)  # (B, num_classes)
    logits = output['logits']
    features = output['features']  # For classification layer

# Feature extraction
features = model.extract_features(images)  # Shape: (B, feature_dim)

# Uncertainty estimation
uncertainty = model.get_uncertainty(images)  # Dictionary of uncertainty measures

# Model saving
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

**Output Format**:
```python
{
    'logits': torch.Tensor,         # (B, num_classes) raw model output
    'features': torch.Tensor,       # (B, feature_dim) internal representation
    'probabilities': torch.Tensor,  # (B, num_classes) softmax
    'predictions': torch.Tensor     # (B,) argmax class indices
}
```

---

### Training (`src.training`)

#### `Trainer`
```python
from src.training.trainer import Trainer
from src.models.model_factory import ModelFactory

# Create model, optimizer, scheduler
model = ModelFactory.create("efficientnet", "b3", num_classes=14)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
loss_fn = torch.nn.CrossEntropyLoss()

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device="cuda",
    max_epochs=100,
    checkpoint_dir="./checkpoints"
)

# Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    early_stopping_patience=15
)

# Access training history
train_losses = history['train_loss']  # List[float]
val_accuracies = history['val_accuracy']  # List[float]
```

**Methods**:
- `fit()`: Training loop
- `evaluate()`: Evaluate on dataset
- `predict()`: Get predictions
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load model state

---

#### `EarlyStopping`
```python
from src.training.early_stopping import EarlyStopping

early_stopping = EarlyStopping(
    patience: int = 15,          # Epochs without improvement
    min_delta: float = 1e-4,     # Minimum improvement threshold
    metric: str = "val_loss",    # Metric to monitor
    mode: str = "min"            # "min" or "max"
)

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    if early_stopping(val_loss):
        print("Early stopping triggered")
        break

# Restore best weights
model.load_state_dict(early_stopping.best_model_state)
```

---

#### `OptimizerFactory`
```python
from src.training.optimizer import OptimizerFactory

optimizer = OptimizerFactory.create(
    name: str,                  # "adam", "adamw", "sgd", "radam"
    params: Iterator,           # model.parameters()
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    momentum: float = 0.9       # For SGD
)
```

---

#### `SchedulerFactory`
```python
from src.training.scheduler import SchedulerFactory

scheduler = SchedulerFactory.create(
    name: str,                  # "cosine", "step", "exponential"
    optimizer: Optimizer,
    T_max: int = 100,          # Max epochs (for cosine)
    step_size: int = 30,       # Decay interval (for step)
    gamma: float = 0.1         # Decay factor
)
```

---

#### `LossFunctionFactory`
```python
from src.training.loss_functions import LossFunctionFactory

loss_fn = LossFunctionFactory.create(
    name: str,                  # "cross_entropy", "focal", "dice", etc.
    num_classes: int = 9,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0          # For focal loss
)

loss = loss_fn(logits, labels)
```

---

### Uncertainty Quantification (`src.uncertainty`)

#### `MCDropoutUncertainty`
```python
from src.uncertainty.mc_dropout import MCDropoutUncertainty
import torch

model.train()  # Enable dropout
mc_dropout = MCDropoutUncertainty(
    model=model,
    num_samples: int = 10,
    device: str = "cuda"
)

# Get stochastic predictions
with torch.no_grad():
    predictions, uncertainty = mc_dropout.predict(images)

# Output
predictions:    # (B, num_classes) mean softmax probabilities
uncertainty:    {
    'entropy': (B,),            # Shannon entropy
    'variance': (B,),           # Variance across samples
    'std': (B,),                # Standard deviation
    'confidence': (B,),         # 1 - entropy (normalized)
    'aleatoric': (B,),          # Aleatoric uncertainty estimate
    'epistemic': (B,)           # Epistemic uncertainty estimate
}
```

---

#### `DeepEnsembleUncertainty`
```python
from src.uncertainty.deep_ensemble import DeepEnsembleUncertainty

ensemble = DeepEnsembleUncertainty(num_models=5)

# Add trained models
for seed in range(5):
    model = train_model_with_seed(seed)
    ensemble.add_model(model)

# Get ensemble predictions
predictions, uncertainty = ensemble.predict(images)

# Output
predictions:    # (B, num_classes) mean probability
uncertainty:    {
    'disagreement': (B,),       # Disagreement metric
    'variance': (B,),
    'entropy': (B,),
    'confidence': (B,)
}
```

---

#### `TemperatureScaling`
```python
from src.uncertainty.temperature_scaling import TemperatureScaling

ts = TemperatureScaling(device="cuda")

# Fit on validation set
ts.fit(
    val_logits: torch.Tensor,  # (N, num_classes)
    val_labels: torch.Tensor   # (N,)
)

# Apply to test logits
calibrated_probs = ts.transform(test_logits)

# Get temperature parameter
temperature = ts.temperature
```

---

#### `CalibrationMetrics`
```python
from src.uncertainty.calibration import CalibrationMetrics

calibrator = CalibrationMetrics()

# Expected Calibration Error
ece = calibrator.compute_ece(
    probabilities: torch.Tensor,  # (N, num_classes)
    labels: torch.Tensor,         # (N,)
    n_bins: int = 10
)

# Maximum Calibration Error
mce = calibrator.compute_mce(probabilities, labels)

# Brier Score
brier = calibrator.compute_brier_score(probabilities, labels)

# Calibration curve
bin_accs, bin_confs, bin_counts = calibrator.calibration_curve(
    probabilities, labels, n_bins=10
)
```

---

### Triage System (`src.triage`)

#### `TriagePolicy`
```python
from src.triage.triage_policy import TriagePolicy

policy = TriagePolicy(
    uncertainty_threshold: float = 0.5,  # Defer if uncertainty > threshold
    metric: str = "entropy",             # "entropy", "variance", "confidence"
    ai_error_cost: float = 100,
    human_review_cost: float = 2
)

# Make triage decision
defer_flags = policy.make_decisions(
    predictions: torch.Tensor,   # (N, num_classes)
    uncertainties: torch.Tensor  # (N,)
)

# Output: (N,) boolean array, True = defer to human

# Optimize threshold
optimal_threshold = policy.optimize_threshold(
    predictions=predictions,
    uncertainties=uncertainties,
    labels=labels,
    human_accuracy=0.92
)
```

---

#### `DeferralStrategyFactory`
```python
from src.triage.deferral_strategies import DeferralStrategyFactory

# Strategy 1: Threshold-based
strategy = DeferralStrategyFactory.create(
    name="threshold",
    params={'threshold': 0.5, 'metric': 'entropy'}
)

# Strategy 2: Budget-constrained
strategy = DeferralStrategyFactory.create(
    name="budget_constrained",
    params={'deferral_budget': 0.2}  # Defer 20% of samples
)

# Strategy 3: Risk-based
strategy = DeferralStrategyFactory.create(
    name="risk_based",
    params={
        'ai_error_cost': 100,
        'human_review_cost': 2,
        'human_accuracy': 0.92
    }
)

# Make decisions
decisions = strategy.make_decisions(predictions, uncertainties)
# Output: (N,) boolean array
```

**Strategy Types**:
1. `"threshold"`: Defer if uncertainty > threshold
2. `"budget_constrained"`: Defer N% of samples
3. `"coverage_based"`: Target coverage vs accuracy tradeoff
4. `"risk_based"`: Optimize cost-benefit
5. `"selective_prediction"`: Maximize success rate

---

#### `HumanSimulator`
```python
from src.triage.human_simulator import HumanSimulator

simulator = HumanSimulator(
    accuracy: float = 0.92,      # Human expert accuracy
    random_state: int = 42
)

# Simulate human decisions on deferred samples
human_decisions = simulator.simulate_decisions(
    labels: torch.Tensor,         # Ground truth labels
    defer_mask: torch.Tensor      # Which samples are deferred
)

# Output: (N,) predictions from human expert
```

---

#### `CollaborationMetrics`
```python
from src.triage.collaboration_metrics import CollaborationMetrics

metrics = CollaborationMetrics()

# Compute system performance (AI + Human)
result = metrics.compute_system_accuracy(
    ai_predictions: np.ndarray,
    human_predictions: np.ndarray,
    defer_mask: np.ndarray,
    labels: np.ndarray,
    ai_error_cost: float = 100,
    human_review_cost: float = 2
)

# Output
result = {
    'ai_accuracy': float,           # Accuracy on AI samples
    'human_accuracy': float,        # Accuracy on deferred samples
    'system_accuracy': float,       # Overall system accuracy
    'deferral_rate': float,         # % of deferred samples
    'automation_rate': float,       # % handled by AI
    'total_cost': float,            # Cost-benefit score
    'coverage': float               # Samples with predictions
}
```

---

### Evaluation (`src.evaluation`)

#### `UncertaintyMetrics`
```python
from src.evaluation.uncertainty_metrics import UncertaintyMetrics

metrics = UncertaintyMetrics()

# AUROC of uncertainty (higher uncertainty → higher error)
auroc = metrics.uncertainty_auroc(
    uncertainties: np.ndarray,    # (N,)
    errors: np.ndarray            # (N,) boolean, True = error
)

# Spearman correlation
corr, p_value = metrics.spearman_correlation(uncertainties, errors)

# Risk-coverage curve
risks, coverages = metrics.risk_coverage_curve(uncertainties, errors)

# Selective prediction metrics
coverage, acc = metrics.selective_prediction(
    uncertainties, labels, confidences, coverage_levels=[0.8, 0.9, 0.95]
)
```

---

#### `TriageEvaluator`
```python
from src.evaluation.triage_evaluator import TriageEvaluator

evaluator = TriageEvaluator(
    predictions: np.ndarray,      # (N, num_classes)
    uncertainties: np.ndarray,    # (N,)
    labels: np.ndarray,           # (N,)
    human_accuracy: float = 0.92
)

# Sweep thresholds
results = evaluator.sweep_thresholds(
    thresholds=np.linspace(0, 1, 20),
    metric="entropy"
)

# Returns: List[{threshold, deferral_rate, automation_rate, 
#                 ai_accuracy, system_accuracy, total_cost}]

# Compare strategies
strategy_results = evaluator.compare_strategies(
    strategies=['threshold', 'budget_constrained', 'risk_based'],
    human_accuracy=0.92
)

# Generate report
report = evaluator.generate_report(output_format="html")
```

---

#### `Visualizer`
```python
from src.evaluation.visualization import Visualizer

viz = Visualizer()

# Plot training curves
fig = viz.plot_training_curves(
    history: dict  # {train_loss, val_loss, val_accuracy}
)

# Plot confusion matrix
fig = viz.plot_confusion_matrix(y_true, y_pred)

# Plot ROC/PR curves
fig = viz.plot_roc_pr_curves(y_true, y_probs)

# Plot uncertainty distribution
fig = viz.plot_uncertainty_distribution(uncertainties, errors)

# Plot calibration curve
fig = viz.plot_calibration_curve(probabilities, labels)

# Create dashboard
fig = viz.create_dashboard(
    history=history,
    y_true=y_true,
    y_pred=y_pred,
    uncertainties=uncertainties,
    probabilities=probabilities
)

plt.show()
```

---

### Utilities (`src.utils`)

#### `ConfigParser`
```python
from src.utils.config_parser import ConfigParser

# Load YAML config
config = ConfigParser.load_yaml("configs/chestmnist_config.yaml")

# Access nested values
model_name = config.get("model.architecture")
batch_size = config.get("training.batch_size", default=64)

# Convert to dict
config_dict = config.to_dict()

# Save config
ConfigParser.save_yaml(config, "output/config.yaml")
```

---

#### `Logger`
```python
from src.utils.logger import get_logger

logger = get_logger(__name__, log_level="DEBUG")

logger.debug("Debug message")
logger.info("Important milestone")
logger.warning("Potential issue")
logger.error("Error occurred", exc_info=True)
```

---

#### `DeviceManager`
```python
from src.utils.device_manager import DeviceManager

device = DeviceManager.get_device()  # "cuda" or "cpu"
num_gpus = DeviceManager.num_gpus()

# Distribute model to device
model = model.to(device)
```

---

#### `Checkpoint Manager`
```python
from src.utils.checkpoint import CheckpointManager

checkpoint_mgr = CheckpointManager(save_dir="./checkpoints")

# Save checkpoint
checkpoint_mgr.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={'val_accuracy': 0.92}
)

# Load checkpoint
data = checkpoint_mgr.load_best()
model.load_state_dict(data['model_state_dict'])
epoch = data['epoch']
```

---

#### `Reproducibility`
```python
from src.utils.reproducibility import set_seed

# Set all random seeds
set_seed(42)

# Enables deterministic behavior across platforms
```

---

## CLI Usage Examples

### Training
```bash
python scripts/train.py \
  --config configs/chestmnist_config.yaml \
  --dataset chestmnist \
  --output experiments/exp_001/ \
  --seed 42 \
  --device cuda \
  --num-workers 4
```

### Evaluation
```bash
python scripts/evaluate.py \
  --model-path experiments/exp_001/best_model.pt \
  --config experiments/exp_001/config.yaml \
  --dataset chestmnist \
  --output-dir results/
```

### Uncertainty Estimation
```bash
python scripts/uncertainty_estimation.py \
  --model-path experiments/exp_001/best_model.pt \
  --dataset chestmnist \
  --num-samples 10 \
  --output results/uncertainty.json
```

### Triage Analysis
```bash
python scripts/triage_analysis.py \
  --model-path experiments/exp_001/best_model.pt \
  --dataset chestmnist \
  --human-accuracy 0.92 \
  --num-thresholds 20 \
  --output results/triage_analysis.html
```

### Report Generation
```bash
python scripts/generate_report.py \
  --experiment-dir experiments/exp_001/ \
  --output-format html \
  --output results/final_report.html
```

---

## Configuration YAML Schema

### Full Configuration Structure
```yaml
# Dataset configuration
dataset:
  name: "chestmnist"                    # pathmnist, chestmnist, dermamnist
  num_classes: 14
  task: "multi-label"                   # multi-class, multi-label
  class_names: [disease1, disease2, ...]
  human_accuracy: 0.92

# Model configuration
model:
  architecture: "efficientnet"          # resnet, efficientnet, densenet, vit
  variant: "b3"
  pretrained: true
  dropout_rate: 0.3
  num_classes: 14

# Training configuration
training:
  max_epochs: 100
  batch_size: 64
  learning_rate: 0.001
  num_workers: 4
  weight_decay: 1e-4
  gradient_clip_norm: 1.0

# Optimizer & Scheduler
optimizer:
  name: "adamw"
  warmup_epochs: 5

scheduler:
  name: "cosine"
  T_max: 100

# Loss function
loss:
  name: "weighted_cross_entropy"
  class_weights: [1.0, 1.5, 2.0, ...]

# Early stopping
early_stopping:
  patience: 15
  min_delta: 1e-4

# Uncertainty methods
uncertainty:
  mc_dropout:
    enabled: true
    num_samples: 10
  temperature_scaling:
    enabled: true

# Triage policy
triage:
  uncertainty_metric: "entropy"
  ai_error_cost: 100
  human_review_cost: 2

# Augmentation
augmentation:
  level: "standard"  # light, standard, aggressive

# Data preprocessing
preprocessing:
  normalization: "imagenet"  # imagenet, medical, none
```

---

## Common Workflows

### Workflow 1: Full Training Pipeline
```python
from src.data.medmnist_loader import MedMNISTLoader
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.training.optimizer import OptimizerFactory
from src.utils.config_parser import ConfigParser

# Load config
config = ConfigParser.load_yaml("configs/chestmnist_config.yaml")

# Load data
loader = MedMNISTLoader(config.get("dataset.name"))
train_data = loader.load_train()
val_data = loader.load_val()

# Create dataloaders
train_loader = create_dataloader(train_data, batch_size=64)
val_loader = create_dataloader(val_data, batch_size=64)

# Create model
model = ModelFactory.create(
    architecture=config.get("model.architecture"),
    variant=config.get("model.variant"),
    num_classes=config.get("model.num_classes"),
    pretrained=True
)

# Create optimizer
optimizer = OptimizerFactory.create(
    name=config.get("optimizer.name"),
    params=model.parameters(),
    learning_rate=config.get("training.learning_rate")
)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=torch.nn.CrossEntropyLoss(),
    device="cuda",
    max_epochs=config.get("training.max_epochs")
)

# Train
history = trainer.fit(train_loader, val_loader)
```

### Workflow 2: Uncertainty + Triage
```python
from src.uncertainty.mc_dropout import MCDropoutUncertainty
from src.triage.triage_policy import TriagePolicy

# Load trained model
model = load_model("best_model.pt")

# Get uncertainty
mc_dropout = MCDropoutUncertainty(model, num_samples=10)
predictions, uncertainty = mc_dropout.predict(test_images)

# Make triage decision
policy = TriagePolicy(
    uncertainty_threshold=0.5,
    ai_error_cost=100,
    human_review_cost=2
)
defer_mask = policy.make_decisions(predictions, uncertainty['entropy'])

# System prediction
system_preds = combine_ai_human(
    ai_predictions=predictions,
    human_predictions=human_decisions,
    defer_mask=defer_mask
)

# Evaluate
metrics = compute_system_accuracy(system_preds, labels)
```

### Workflow 3: Hyperparameter Tuning
```python
from src.training.trainer import Trainer
from src.models.model_factory import ModelFactory
import itertools

# Grid search
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [32, 64, 128]
dropout_rates = [0.2, 0.3, 0.4]

results = []

for lr, bs, dropout in itertools.product(learning_rates, batch_sizes, dropout_rates):
    model = ModelFactory.create(..., dropout_rate=dropout)
    trainer = Trainer(..., learning_rate=lr, batch_size=bs)
    history = trainer.fit(train_loader, val_loader)
    
    best_val_acc = max(history['val_accuracy'])
    results.append({
        'lr': lr,
        'batch_size': bs,
        'dropout': dropout,
        'best_accuracy': best_val_acc
    })

# Find best parameters
best = max(results, key=lambda x: x['best_accuracy'])
```

---

## Error Handling

### Common Errors and Solutions

```python
# Out of Memory
try:
    trainer.fit(train_loader, val_loader)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Reduce batch size or model size")
        # Retry with batch_size=32

# Data Loading Error
try:
    loader = MedMNISTLoader("pathmnist", download=True)
except Exception as e:
    print(f"Download failed: {e}")
    # Check internet connection, disk space

# Model Convergence
if best_accuracy < 0.7:
    print("Model not converging")
    # Increase learning rate, adjust scheduler
    # Check data normalization, augmentation

# NaN Loss
if np.isnan(loss.item()):
    print("Loss became NaN")
    # Reduce learning rate, enable gradient clipping
    # Check data for invalid values
```

---

## Performance Optimization Tips

1. **Data Loading**: Use `num_workers > 0` and `pin_memory=True`
2. **Model Inference**: Use `torch.no_grad()` and batch processing
3. **Training**: Enable mixed precision (`torch.cuda.amp`)
4. **Memory**: Use gradient accumulation for larger effective batch sizes
5. **Validation**: Skip expensive operations (MC Dropout) on validation

---

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Timm Models: https://github.com/rwightman/pytorch-image-models
- MedMNIST: https://medmnist.com/
- TorchVision: https://pytorch.org/vision/stable/

---

## Support & Troubleshooting

For issues or questions:
1. Check this documentation
2. Review example scripts in `/scripts/`
3. Check test files for usage examples
4. Refer to original papers for technical details
