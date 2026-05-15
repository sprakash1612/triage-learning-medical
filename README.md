# Triage Learning: Medical Diagnosis via Human-AI Collaboration

A PyTorch framework for uncertainty-aware medical image classification that intelligently routes cases between AI and human experts to optimize diagnostic accuracy.

## Key Results (PathMNIST Histopathology)

| Metric | Value | Improvement |
|--------|-------|-------------|
| **System Accuracy** | 97.35% | +8.31% vs baseline |
| **Error Reduction** | 75.8% | 787 → 190 errors |
| **Automation Rate** | 63.9% | 36.1% human review |
| **Cost Savings** | $46,200 | 1,150% ROI |
| **Uncertainty-Error Correlation** | 0.498 | p < 0.001 |

---

## Installation & Quick Start

```bash
# Setup
pip install -r requirements.txt
python scripts/download_data.py --dataset pathmnist

# Train ResNet18 on PathMNIST
python scripts/train.py --config configs/pathmnist_config.yaml

# Optimize triage threshold
python scripts/triage_analysis.py --model-path experiments/pathmnist_triage/best_model.pt

# Generate report
python scripts/generate_report.py --experiment-dir experiments/pathmnist_triage/
```

---

## System Architecture

### Core Pipeline

```
Raw Images
    ↓
[Data Augmentation] → Class-balanced sampling, anatomy-aware transforms
    ↓
[ResNet18 + Dropout] → Classification + stochastic passes
    ↓
[MC Dropout × 10] → Predictive entropy, variance, confidence
    ↓
[Threshold Policy] → Route by uncertainty threshold τ
    ↓
[AI + Human] → Combined prediction & cost analysis
```

### Key Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `data/` | Dataset loaders & preprocessing | `MedMNISTLoader`, `AugmentationPipeline` |
| `models/` | Architecture implementations | `ResNet18`, `EfficientNet`, `DenseNet`, `ViT` |
| `training/` | Training pipeline | `Trainer`, `EarlyStopping`, `WeightedCrossEntropy` |
| `uncertainty/` | Uncertainty quantification | `MCDropoutUncertainty`, `DeepEnsemble`, `TemperatureScaling` |
| `triage/` | Deferral policies & collaboration | `TriagePolicy`, `HumanSimulator`, `CollaborationMetrics` |
| `evaluation/` | Metrics & analysis | `TriageEvaluator`, `CostAnalyzer`, `Visualizer` |

---

## Methodology: Threshold-Based Triage

### 1. Uncertainty via Monte Carlo Dropout
For input $x$, make $T=10$ stochastic forward passes with dropout enabled:

$$\mathbf{p}_t = \text{softmax}(f(x; \theta, m_t)) \quad \text{for } t=1,...,T$$

Compute predictive distribution statistics:
- **Entropy** (main signal): $H(\bar{\mathbf{p}}) = -\sum_c \bar{p}_c \log \bar{p}_c$
- **Variance** (disagreement): $\text{Var}(p_c) = \frac{1}{T}\sum_t(p_{t,c} - \bar{p}_c)^2$

### 2. Deferral Policy
Route cases by uncertainty threshold:
$$d_i = \mathbb{I}(u_i > \tau)$$

Where:
- $d_i = 1$ → defer to human expert
- $d_i = 0$ → use AI prediction
- $\tau$ = learnable threshold

### 3. System Accuracy
Combine AI and human performance:

$$\text{Acc}_{\text{sys}} = (1-DR) \cdot \text{Acc}_{\text{AI}} + DR \cdot \text{Acc}_h$$

Where:
- $DR$ = deferral rate = 36.1% (optimal)
- $\text{Acc}_{\text{AI}}$ = AI accuracy on automated cases = 98.56%
- $\text{Acc}_h$ = human expert accuracy = 95% (assumed)

### 4. Cost Function
Optimize for either accuracy or cost:

$$C_{\text{total}} = c_{\text{err}} \cdot N_{\text{err}}^{\text{AI}} + c_{\text{rev}} \cdot N_{\text{defer}}$$

Where $c_{\text{err}} = \$100$ (error cost), $c_{\text{rev}} = \$2$ (review cost)

---

## Code Examples

### Train Model with Weighted Loss
```python
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.data.medmnist_loader import MedMNISTLoader

# Load dataset with class weights
loader = MedMNISTLoader(dataset="pathmnist")
train_loader, val_loader = loader.get_dataloaders(batch_size=64)

# Create ResNet18 with dropout
model = ModelFactory.create(
    architecture="resnet",
    variant="18",
    num_classes=9,
    pretrained=True,
    dropout_p=0.3
)

# Train with weighted cross-entropy
trainer = Trainer(
    model=model,
    optimizer="adamw",
    loss_fn="weighted_cross_entropy",
    device="cuda"
)

history = trainer.fit(
    train_loader, 
    val_loader, 
    epochs=100,
    early_stopping_patience=15
)
```

### Uncertainty Quantification
```python
from src.uncertainty.mc_dropout import MCDropoutUncertainty

# Enable uncertainty estimation
mc_dropout = MCDropoutUncertainty(model, num_samples=10)

# Get predictions with uncertainty
predictions, uncertainty = mc_dropout.predict_batch(test_images)

# Uncertainty contains: entropy, variance, confidence
entropy = uncertainty['entropy']  # Main signal for triage
variance = uncertainty['variance']
confidence = uncertainty['confidence']

# Validate uncertainty correlates with errors
from scipy.stats import pointbiserialr
correlation, pval = pointbiserialr(is_error, entropy)
print(f"Uncertainty-Error Correlation: {correlation:.3f} (p={pval:.3e})")
```

### Threshold Optimization
```python
from src.triage.triage_policy import TriagePolicy
from src.evaluation.triage_evaluator import TriageEvaluator

# Sweep 50 thresholds to find optimal
thresholds = np.linspace(0, entropy.max(), 50)
results = []

for tau in thresholds:
    # Create policy
    policy = TriagePolicy(
        uncertainty_threshold=tau,
        ai_error_cost=100,
        human_review_cost=2
    )
    
    # Make deferral decisions
    defer_mask = policy.make_decisions(predictions, entropy)
    
    # Evaluate system (with simulated human at 95% accuracy)
    system_acc = (
        (1 - defer_mask.mean()) * accuracy[~defer_mask] +
        defer_mask.mean() * 0.95  # human accuracy
    )
    
    results.append({
        'threshold': tau,
        'deferral_rate': defer_mask.mean(),
        'system_accuracy': system_acc,
        'cost': cost_function(predictions, defer_mask)
    })

# Find optimal threshold
df = pd.DataFrame(results)
optimal_idx = df['system_accuracy'].idxmax()
optimal_threshold = df.loc[optimal_idx, 'threshold']
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Achieves: {df.loc[optimal_idx, 'system_accuracy']:.4f} accuracy")
print(f"            {df.loc[optimal_idx, 'deferral_rate']:.1%} deferral rate")
```

### Human-AI Collaboration
```python
from src.triage.human_simulator import HumanSimulator

# Simulate human expert performance
human = HumanSimulator(accuracy=0.95)

# Make predictions on deferred cases
human_predictions = human.predict(
    test_images[defer_mask],
    true_labels[defer_mask]
)

# Combine AI and human predictions
system_predictions = predictions.copy()
system_predictions[defer_mask] = human_predictions

# Calculate combined system metrics
from sklearn.metrics import accuracy_score, f1_score

system_acc = accuracy_score(true_labels, system_predictions)
print(f"Combined System Accuracy: {system_acc:.4f}")  # 97.35%

# Cost analysis
errors = (system_predictions != true_labels).sum()
cost = 100 * errors + 2 * defer_mask.sum()
print(f"Total Cost: ${cost:,.0f}")
print(f"Cost per Case: ${cost / len(test_images):.2f}")
```

---

## Performance Characteristics

### Computational Requirements

| Metric | Value |
|--------|-------|
| **Training** (ResNet18 on PathMNIST, V100) | ~2 hours |
| **Inference** per image (5 forward passes) | 25 ms |
| **MC Dropout** (10 passes) | 50 ms |
| **Threshold sweep** (50 thresholds) | ~1 hour |

### Memory Usage (Batch Size 64)

| Model | GPU Memory |
|-------|-----------|
| ResNet18 | 1.8 GB |
| EfficientNet-B3 | 2.5 GB |
| Vision Transformer | 4.2 GB |

---

## Experimental Validation

### Baseline Performance
- **Accuracy**: 89.04% (6,393/7,180 correct)
- **Error Rate**: 10.96% (787 errors)
- **Balanced Accuracy**: 86.41% (handles 195:1 imbalance)

### Uncertainty Quality
- **Point-Biserial Correlation** (uncertainty vs error): 0.498 ✓
- **Spearman Rank Correlation**: 0.413 ✓
- **AUROC** (predicting errors): 0.78 ✓

### System Performance at Optimal Threshold (τ=0.4916)
- **System Accuracy**: 97.35% (+8.31%)
- **Correct Diagnoses**: 6,990 (+597)
- **Remaining Errors**: 190 (-75.8%)
- **Automation**: 63.9% (4,594/7,180 cases)
- **Human Reviews**: 36.1% (2,590/7,180 cases)
- **Cost Savings**: $46,200 vs AI-only

### Robustness to Human Accuracy

| Human Accuracy | System Accuracy | Improvement |
|---|---|---|
| 80% | 92.63% | +3.59% |
| 90% | 95.97% | +6.94% |
| **95%** | **97.35%** | **+8.31%** ✓ |
| 99% | 98.75% | +9.71% |

System remains beneficial even with human accuracy below AI baseline.

---

## Testing & Validation

Run comprehensive test suite:

```bash
# All tests (190+ test cases)
python -m pytest tests/ -v

# Specific module tests
python -m pytest tests/test_uncertainty.py -v     # MC Dropout, ensembles
python -m pytest tests/test_triage.py -v          # Deferral policies
python -m pytest tests/test_models.py -v          # All architectures

# Coverage analysis
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Configuration

Configuration hierarchy (CLI args override YAML):

```yaml
# configs/pathmnist_config.yaml
dataset:
  name: "pathmnist"
  num_classes: 9
  imbalance_ratio: 195.0

model:
  architecture: "resnet"
  variant: "18"
  pretrained: true
  dropout_p: 0.3

training:
  max_epochs: 100
  batch_size: 64
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 1e-4
  warmup_epochs: 5

uncertainty:
  method: "mc_dropout"
  num_samples: 10
  tau: 0.4916  # optimal threshold

triage:
  human_accuracy: 0.95
  ai_error_cost: 100
  human_review_cost: 2
```

---

## Next Steps

1. **Review the report** (`report/main.pdf`) for detailed methodology
2. **Run quick start** to train on PathMNIST (~2 hours)
3. **Analyze results** via `scripts/triage_analysis.py`
4. **Extend to your dataset** using same pipeline
5. **Read documentation**:
   - `docs/architecture.md` - system design (15 min)
   - `docs/data_documentation.md` - datasets & preprocessing (15 min)
   - `docs/model_documentation.md` - model selection (20 min)
   - `docs/api_reference.md` - complete API (25 min)

---

## Citation

If you use this framework in research, please cite the associated technical report:

```bibtex
@misc{triage_learning_2025,
  title={Triage Learning: Human-AI Collaboration in Medical Diagnosis},
  author={Prakash, Shanu and Kumar, Gyanendra},
  year={2025},
  note={M.Tech Project Report, IIT Patna}
}
```
