# System Architecture Documentation

## Overview

The Triage Learning System is a comprehensive framework for implementing human-AI collaborative diagnosis in medical imaging. It combines deep learning uncertainty quantification with intelligent triage policies to optimize diagnostic workflows.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        End-User Applications                         │
│  (Clinical Dashboard, Web Interface, Mobile App)                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↑       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Triage & Decision Modules                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Triage     │  │  Deferral    │  │  Confidence  │              │
│  │   Policy     │  │  Strategies  │  │  Metrics     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              ↑       ↓
┌─────────────────────────────────────────────────────────────────────┐
│               Uncertainty & Evaluation Modules                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ MC Dropout   │  │   Ensemble   │  │ Temperature  │              │
│  │              │  │   Learning   │  │  Scaling     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│  ┌──────────────────────────────────────────────────┐               │
│  │       Uncertainty & Calibration Metrics           │               │
│  │  (AUROC, Spearman, Risk-Coverage, ECE/MCE)      │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                              ↑       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Model Training Module                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Trainer    │  │    Loss      │  │   Early      │              │
│  │              │  │  Functions   │  │  Stopping    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│  ┌──────────────────────────────────────────────────┐               │
│  │     Model Architectures & Factory                 │               │
│  │  (ResNet, DenseNet, EfficientNet, ViT)          │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                              ↑       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Processing Module                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │    Dataset   │  │  Augmentation│  │ Preprocessing│              │
│  │   Loading    │  │              │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│  ┌──────────────────────────────────────────────────┐               │
│  │     MedMNIST: PathMNIST, ChestMNIST, DermaMNIST │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                              ↑       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Layer (Raw Datasets)                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Organization

### 1. **Data Processing (`src/data/`)**

Handles all data ingestion, preprocessing, and augmentation for medical images.

#### Key Components:
- **`dataset.py`**: Base dataset class with standard PyTorch interface
  - Supports multiple medical imaging datasets
  - Returns images (3D tensors) and labels
  - Configurable preprocessing pipeline

- **`medmnist_loader.py`**: MedMNIST-specific loader
  - Downloads datasets from MedMNIST repository
  - Handles multiple datasets (PathMNIST, ChestMNIST, DermaMNIST)
  - Validates checksums and data integrity

- **`augmentation.py`**: Medical imaging augmentation
  - Dataset-aware augmentation (grayscale vs RGB)
  - Conservative augmentation for chest X-rays
  - Color-preserving augmentation for dermatology
  - Class-specific augmentation levels

- **`preprocessing.py`**: Image normalization and standardization
  - ImageNet normalization (mean/std)
  - Medical imaging specific normalization
  - Data type conversions (uint8 ↔ float32)

- **`dataloader.py`**: PyTorch DataLoader wrapper
  - Configurable batch sizes and workers
  - Stratified sampling for imbalanced datasets
  - Memory-efficient data loading

#### Data Flow:
```
Raw MNIST → Download & Verify → Load → Preprocess → Augment → Batch → Model
```

---

### 2. **Model Architecture (`src/models/`)**

Implements multiple CNN and transformer architectures with uncertainty support.

#### Key Components:
- **`base_model.py`**: Abstract base class for all models
  - Standardized forward pass interface
  - Feature extraction capability
  - Uncertainty quantification hooks
  - Model saving/loading utilities

- **`resnet.py`**: ResNet variants (18, 34, 50, 101, 152)
  - Transfer learning from ImageNet
  - Bottleneck and basic blocks
  - Feature extraction at multiple depths

- **`efficientnet.py`**: EfficientNet variants (B0-B7)
  - Mobile-friendly architectures
  - Compound scaling (depth, width, resolution)
  - Recommended for medical imaging

- **`densenet.py`**: DenseNet variants (121, 169, 201, 264)
  - Dense connections for feature reuse
  - Parameter-efficient training
  - Good for limited data scenarios

- **`vit.py`**: Vision Transformer (ViT)
  - Patch-based input processing
  - Multi-head self-attention
  - Recommended for color-rich images (dermatology)

- **`model_factory.py`**: Factory pattern for model instantiation
  - Unified model creation interface
  - Pretrained weight loading
  - Device management

#### Model Selection Criteria:
| Architecture | Best For | Memory | Speed | Accuracy |
|---|---|---|---|---|
| ResNet | General baseline | Medium | Fast | Good |
| EfficientNet | Mobile/efficient | Low | Fast | Excellent |
| DenseNet | Parameter efficiency | Low | Medium | Good |
| ViT | Color features | High | Slow | Excellent |

---

### 3. **Training Module (`src/training/`)**

Handles model training, optimization, and regularization.

#### Key Components:
- **`trainer.py`**: Main training orchestrator
  - Epoch-based training loop
  - Validation and metric computation
  - Checkpoint management
  - Device management (CPU/GPU/Multi-GPU)

- **`optimizer.py`**: Optimizer factory
  - Support: Adam, AdamW, SGD, RAdam
  - Learning rate scheduling
  - Gradient clipping and accumulation
  - Warmup scheduling

- **`scheduler.py`**: Learning rate schedulers
  - Cosine annealing with warmup
  - Step decay
  - Exponential decay
  - Polynomial decay

- **`loss_functions.py`**: Specialized loss functions
  - Cross-entropy (with label smoothing)
  - Focal loss (for imbalanced data)
  - Dice loss (segmentation-inspired)
  - Symmetric cross-entropy

- **`early_stopping.py`**: Training convergence control
  - Patience-based early stopping
  - Multiple metric monitoring
  - Best weight restoration
  - Warm-up period support

#### Training Pipeline:
```
Data Loader → Forward Pass → Loss Computation → Backward Pass 
  → Gradient Update → Scheduler Step → Validation → Early Stopping Check
```

---

### 4. **Uncertainty Quantification (`src/uncertainty/`)**

Estimates model confidence and calibration.

#### Key Components:
- **`mc_dropout.py`**: Monte Carlo Dropout
  - Stochastic forward passes (10-20 samples)
  - Entropy as uncertainty measure
  - Variance across samples
  - Low computational overhead

- **`deep_ensemble.py`**: Deep Ensemble
  - Multiple model training/inference
  - Disagreement-based uncertainty
  - Highest quality uncertainty estimates
  - Higher computational cost

- **`entropy.py`**: Entropy-based confidence metrics
  - Shannon entropy from softmax
  - Normalized entropy
  - Confidence scores

- **`temperature_scaling.py`**: Post-hoc calibration
  - Temperature parameter optimization
  - Calibration curve fitting
  - Probability distribution adjustment

- **`calibration.py`**: Calibration metrics
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Brier score
  - Calibration curve computation

#### Uncertainty Methods Comparison:
| Method | Cost | Quality | Speed |
|---|---|---|---|
| MC Dropout | Low | Medium | Fast |
| Deep Ensemble | High | Highest | Slow |
| Temperature Scaling | Very Low | Medium | Very Fast |

---

### 5. **Triage System (`src/triage/`)**

Implements human-AI collaboration strategies.

#### Key Components:
- **`triage_policy.py`**: Uncertainty-based deferral
  - Threshold-based deferral to human experts
  - Configurable uncertainty metrics
  - Cost-benefit optimization
  - Decision boundary calibration

- **`deferral_strategies.py`**: Multiple triage strategies
  1. **Threshold**: Defer if uncertainty > threshold
  2. **Budget Constrained**: Defer N% of samples
  3. **Coverage-based**: Maintain coverage vs accuracy tradeoff
  4. **Risk-based**: Weight by error costs
  5. **Selective Prediction**: Maximize success rate

- **`human_simulator.py`**: Human expert model
  - Simulates human accuracy (configurable per dataset)
  - Cost modeling (review time, error rates)
  - Training data simulation

- **`collaboration_metrics.py`**: System performance evaluation
  - AI accuracy on confident samples
  - Human accuracy on deferred samples
  - System accuracy (AI + Human)
  - Cost-benefit analysis

#### Triage Decision Flow:
```
Model Prediction + Uncertainty → Compare with Threshold 
  → Defer to Human OR Accept AI Decision 
  → Compute System Accuracy = (AI correct + Human correct) / Total
```

---

### 6. **Evaluation Module (`src/evaluation/`)**

Comprehensive model and system evaluation.

#### Key Components:
- **`metrics.py`**: Standard evaluation metrics
  - Classification: Accuracy, Precision, Recall, F1
  - Ranking: AUROC, AUPRC
  - Calibration: ECE, MCE
  - Per-class metrics

- **`uncertainty_metrics.py`**: Uncertainty quality assessment
  - Risk-Coverage curves
  - Rejection curves
  - AUROC of uncertainty
  - Spearman correlation with error
  - Selective prediction metrics

- **`triage_evaluator.py`**: End-to-end system evaluation
  - Threshold sweeping
  - Strategy comparison
  - Confusion matrix generation
  - Cost-benefit curves
  - HTML report generation

- **`visualization.py`**: Diagnostic visualizations
  - Training curves (loss, accuracy)
  - Confusion matrices
  - ROC/PR curves
  - Uncertainty distributions
  - Calibration plots
  - Cost-benefit curves
  - System dashboard

---

### 7. **Utilities (`src/utils/`)**

Supporting infrastructure and utilities.

#### Key Components:
- **`device_manager.py`**: GPU/CPU device handling
- **`logger.py`**: Logging and experiment tracking
- **`checkpoint.py`**: Model persistence and loading
- **`config_parser.py`**: YAML configuration parsing
- **`reproducibility.py`**: Seed setting and determinism

---

### 8. **Executable Scripts (`scripts/`)**

End-to-end pipelines for common tasks.

#### Key Scripts:
- **`download_data.py`**: Dataset acquisition and verification
- **`train.py`**: Model training orchestration
- **`evaluate.py`**: Model evaluation on test set
- **`uncertainty_estimation.py`**: MC Dropout inference
- **`triage_analysis.py`**: Threshold optimization and strategy comparison
- **`generate_report.py`**: Comprehensive result reporting

---

## Data Flow Architecture

### Training Pipeline
```
Configuration (YAML) 
    ↓
Data Loading (MedMNIST)
    ↓
Preprocessing & Augmentation
    ↓
Model Creation (Factory)
    ↓
Optimizer & Scheduler Setup
    ↓
Training Loop:
  - Forward Pass
  - Loss Computation
  - Backward Pass
  - Gradient Update
  - Scheduler Step
    ↓
Validation Loop:
  - Inference on Val Set
  - Metric Computation
  - Early Stopping Check
    ↓
Checkpoint Saving
    ↓
Final Model Weights
```

### Inference & Triage Pipeline
```
Test Image
    ↓
Preprocessing
    ↓
Model Inference (Single Pass)
    ↓
Uncertainty Estimation (MC Dropout / Ensemble)
    ↓
Confidence Score Computation
    ↓
Triage Decision:
  - Compare with Threshold
  - Defer to Human or Accept
    ↓
System Prediction:
  - AI + Human Accuracy
    ↓
Cost-Benefit Analysis
```

---

## Design Patterns

### 1. **Factory Pattern**
Used for creating optimizers, schedulers, losses, models, and strategies:
```python
# Model factory
model = ModelFactory.create(
    architecture="efficientnet",
    variant="b3",
    num_classes=14,
    pretrained=True
)

# Optimizer factory
optimizer = OptimizerFactory.create(
    name="adamw",
    params=model.parameters(),
    learning_rate=0.001
)
```

### 2. **Strategy Pattern**
Used for deferral strategies and uncertainty methods:
```python
# Different triage strategies
strategy = ThresholdDeferralStrategy(threshold=0.5)
strategy = BudgetConstrainedStrategy(deferral_budget=0.2)
strategy = RiskBasedStrategy(ai_error_cost=100)

# Different uncertainty methods
uncertainty = MCDropoutUncertainty(num_samples=10)
uncertainty = DeepEnsembleUncertainty(num_models=5)
```

### 3. **Template Method Pattern**
Base model class defines training/inference interface, subclasses implement details:
```python
class BaseModel(nn.Module):
    def forward(self, x) -> dict:
        # Standard interface
        features = self.extract_features(x)
        logits = self.classifier(features)
        return {"logits": logits, "features": features}
```

### 4. **Configuration as Code**
YAML-based configuration with three levels:
```yaml
base_config.yaml          # Common settings
chestmnist_config.yaml    # Dataset-specific
training_config.yaml      # Training presets
```

---

## Extension Points

### Adding a New Model Architecture
1. Create `src/models/your_model.py` inheriting from `BaseModel`
2. Implement `forward()`, `extract_features()`, and `get_uncertainty()`
3. Register in `ModelFactory.create()`
4. Add configuration in `your_dataset_config.yaml`

### Adding a New Triage Strategy
1. Create class inheriting from `DeferralStrategy`
2. Implement `make_decisions()` and `optimize()`
3. Register in strategy factory
4. Test with `test_triage.py`

### Adding a New Dataset
1. Create loader in `src/data/medmnist_loader.py`
2. Create dataset-specific config: `configs/newdataset_config.yaml`
3. Update data downloader
4. Add preprocessing if needed

### Adding Evaluation Metrics
1. Add method to `src/evaluation/metrics.py`
2. Integrate with `triage_evaluator.py`
3. Add visualization in `visualization.py`
4. Document in API reference

---

## Scalability Considerations

### Single GPU
- Batch size: 32-128
- Max epochs: 100-200
- Supported models: All

### Multi-GPU (Data Parallel)
- Use `torch.nn.DataParallel`
- Batch size: 128-512
- Effective learning rate: scale with GPUs

### Distributed Training
- Consider `torch.nn.parallel.DistributedDataParallel`
- Synchronization batch norm
- Gradient accumulation for larger effective batch

### Memory Optimization
- Gradient checkpointing
- Mixed precision training (O1, O2)
- Smaller batch sizes for large models (ViT)

---

## Reproducibility & Configuration Management

### Seed Management
- Python seed: `reproducibility.set_seed(42)`
- NumPy seed
- PyTorch seed
- CUDA determinism

### Configuration Hierarchy
```
Default Settings
    ↓
base_config.yaml (common)
    ↓
{dataset}_config.yaml (dataset-specific)
    ↓
CLI Arguments (override)
    ↓
Final Configuration
```

### Experiment Tracking
- Experiment folder: `experiments/exp_001_baseline/`
- Contains: config.yaml, checkpoints/, logs/, results/
- CSV tracking: `experiments/experiment_tracker.csv`

---

## Performance Optimization

### Model Inference
- Use evaluation mode (`model.eval()`)
- Disable gradients (`torch.no_grad()`)
- Batch inference for throughput
- Model quantization (optional)

### Data Loading
- Multiple workers: 4-8
- Pin memory: True for GPU
- Prefetch with DataLoader

### Training
- Mixed precision training (reduces memory 50%)
- Gradient accumulation (effective batch size)
- Learning rate warmup (first 5-10 epochs)

---

## Error Handling & Debugging

### Logging Levels
- DEBUG: Detailed training progress
- INFO: Key milestones
- WARNING: Potential issues
- ERROR: Critical failures

### Common Issues
1. **OOM (Out of Memory)**: Reduce batch size, use gradient accumulation
2. **NaN Loss**: Check learning rate, gradient clipping, data normalization
3. **Poor Accuracy**: Verify data loading, check loss function, increase epochs
4. **Uncertainty not changing**: Verify MC Dropout is enabled, increase samples

### Validation Checks
- Data loading test
- Model forward pass test
- Loss computation test
- Gradient flow test
- Checkpoint save/load test

---

## Version Control & Reproducibility

### Code Snapshots
- Save code at experiment start
- Enables exact reproduction
- Tracks model version

### Configuration Archiving
- Each experiment stores its config
- Enables parameter review
- Supports result analysis

### Metric Tracking
- Per-epoch metrics (CSV)
- Tensorboard logs
- Final results JSON
- HTML report generation

---

## Integration Points

### External Tools
- **Tensorboard**: Real-time training visualization
- **Weights & Biases**: Experiment tracking and collaboration
- **Kaggle**: Dataset download and competition submission
- **WeasyPrint**: PDF report generation

### API Exposure
- All components have CLI interfaces
- Configuration-driven execution
- Extensible logging and monitoring
- Standard PyTorch model format

---

## Performance Benchmarks

### Training Time (Single GPU, 100 epochs)
- ResNet50: ~2 hours
- EfficientNet-B3: ~4 hours
- ViT-Base: ~8 hours

### Inference Time (per sample, GPU)
- ResNet50: ~5ms
- EfficientNet-B3: ~8ms
- ViT-Base: ~15ms
- MC Dropout (10 samples): 50-150ms

### Memory Requirements
- ResNet50: 2GB
- EfficientNet-B3: 3GB
- ViT-Base: 5GB
- Batch size 64: ~2x model size

---

## Future Extensions

1. **Multi-task Learning**: Disease detection + severity prediction
2. **Continual Learning**: Update models with new data
3. **Explainability**: Attention maps, LIME, SHAP
4. **Federated Learning**: Privacy-preserving training
5. **Real-time Pipeline**: Production deployment with monitoring
6. **Cost-sensitive Learning**: Integrate medical error costs into training
7. **Domain Adaptation**: Handle different imaging protocols/devices
