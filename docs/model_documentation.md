# Model Architecture Documentation

## Overview

The system supports multiple deep learning architectures optimized for medical image classification. Each model inherits from `BaseModel`, implementing a standardized interface for training, inference, and uncertainty quantification.

---

## Supported Architectures

### 1. ResNet: Residual Networks

**Architecture Type**: Convolutional Neural Network with skip connections

**Variants**:
- ResNet-18: 18 layers, 11M parameters
- ResNet-34: 34 layers, 22M parameters
- ResNet-50: 50 layers, 26M parameters
- ResNet-101: 101 layers, 45M parameters
- ResNet-152: 152 layers, 60M parameters

**Key Features**:
- Skip connections enabling very deep networks
- Batch normalization after each convolution
- Residual blocks: `y = F(x) + x`
- Downsampling at specific layers
- Global average pooling before classifier

**Computational Complexity**:
| Variant | FLOPs | Memory | Inference Time |
|---------|-------|--------|-----------------|
| ResNet-18 | 1.8B | 180MB | ~3ms |
| ResNet-50 | 4.1B | 280MB | ~5ms |
| ResNet-101 | 7.8B | 420MB | ~8ms |

**Best For**:
- General baseline classification
- Limited compute resources (smaller variants)
- Well-established pretrained weights
- Mixed medical imaging tasks

**Pretrained Weights**:
- ImageNet weights: Excellent for transfer learning
- Medical imaging transfer: Good generalization
- Recommended fine-tuning: Layer-wise learning rate decay

**Configuration**:
```yaml
architecture: "resnet"
variant: "resnet50"
pretrained: true
num_classes: 9  # PathMNIST
dropout_rate: 0.3
```

**Architecture Diagram**:
```
Input (3, 224, 224)
  ↓ Conv 7×7, stride 2
(64, 112, 112)
  ↓ MaxPool
(64, 56, 56)
  ↓ ResBlock 64×4  (skip connections)
(256, 56, 56)
  ↓ ResBlock 128×6
(512, 28, 28)
  ↓ ResBlock 256×3
(1024, 14, 14)
  ↓ ResBlock 512×3
(2048, 7, 7)
  ↓ Global Average Pool
(2048,)
  ↓ FC 1000 (ImageNet) → FC num_classes
Output logits
```

---

### 2. EfficientNet: Efficient Scaling

**Architecture Type**: MobileInvertedResidual blocks with compound scaling

**Variants**:
- EfficientNet-B0: Mobile baseline, 5.3M parameters
- EfficientNet-B1: 7.8M parameters
- EfficientNet-B2: 9.2M parameters
- EfficientNet-B3: 12M parameters
- EfficientNet-B4: 19M parameters
- EfficientNet-B5: 30M parameters
- EfficientNet-B6: 43M parameters
- EfficientNet-B7: 66M parameters

**Key Features**:
- Compound scaling: depth × width × resolution
- Mobile inverted bottleneck blocks (MBConv)
- Depth-wise separable convolutions (efficient)
- Squeeze-and-excitation (SE) attention blocks
- Swish activation function

**Computational Complexity**:
| Variant | FLOPs | Memory | Inference Time |
|---------|-------|--------|-----------------|
| B0 | 390M | 86MB | ~2ms |
| B3 | 1.9B | 200MB | ~8ms |
| B7 | 37B | 550MB | ~80ms |

**Best For**:
- **ChestMNIST**: Recommended B3 variant
- Medical imaging with tight compute budgets
- Mobile or edge deployment
- Efficiency-accuracy tradeoff

**Pretrained Weights**:
- ImageNet weights: Highly effective
- Medical imaging: B3 recommended
- Transfer learning: Excellent across domains

**Configuration**:
```yaml
architecture: "efficientnet"
variant: "b3"
pretrained: true
num_classes: 14  # ChestMNIST
dropout_rate: 0.3
```

**Architecture Advantages**:
- 15-20% better efficiency than ResNet
- Scaling law: increase depth, width, or resolution
- Highly tuned hyperparameters (dropout, stochastic depth)
- SE blocks improve feature reweighting

**Training Tips**:
- Use data augmentation (RandAugment, AutoAugment)
- Longer training: 300-400 epochs
- Large batch sizes: 128-512
- Learning rate warmup: 5 epochs

---

### 3. DenseNet: Densely Connected Networks

**Architecture Type**: Dense connections between all layers

**Variants**:
- DenseNet-121: 121 layers, 7.9M parameters
- DenseNet-169: 169 layers, 14M parameters
- DenseNet-201: 201 layers, 20M parameters
- DenseNet-264: 264 layers, 32M parameters

**Key Features**:
- Dense connections: each layer connected to all previous
- Concatenation instead of addition: `[x, f(x), f(f(x)), ...]`
- Transition blocks: 1×1 conv + 2×2 pool for downsampling
- Bottleneck layers: 1×1 conv before 3×3 conv
- Global average pooling

**Computational Complexity**:
| Variant | FLOPs | Memory | Inference Time |
|---------|-------|--------|-----------------|
| DenseNet-121 | 3.0B | 200MB | ~5ms |
| DenseNet-201 | 5.6B | 320MB | ~10ms |

**Best For**:
- Parameter efficiency (fewer parameters than ResNet)
- Feature reuse and propagation
- Medical imaging with limited training data
- Improved gradient flow

**Pretrained Weights**:
- ImageNet weights: Strong performance
- Medical imaging: Good transfer learning
- Fine-tuning: Effective with moderate learning rates

**Configuration**:
```yaml
architecture: "densenet"
variant: "densenet121"
pretrained: true
num_classes: 7  # DermaMNIST
dropout_rate: 0.2
```

**Architecture Advantages**:
- Feature reuse reduces parameters 20-30%
- Vanishing gradient problem reduced
- Better training efficiency
- Good for small datasets

**Growth Rate**:
- Controls feature concatenation
- Default: 32 (32 new features per layer)
- Higher growth rate → more parameters but better accuracy
- Lower growth rate → compressed, parameter efficient

---

### 4. Vision Transformer: Self-Attention Architecture

**Architecture Type**: Pure transformer-based architecture (no convolutions)

**Variants**:
- ViT-Base: 12 layers, 86M parameters
- ViT-Large: 24 layers, 305M parameters
- ViT-Huge: 32 layers, 632M parameters

**Key Features**:
- Patch embedding: Divide image into 16×16 patches
- Linear projection to embedding space
- Positional embeddings (learnable)
- Multi-head self-attention (12 heads)
- Feed-forward blocks (4× hidden dimension)
- Layer normalization (pre-activation)
- CLS token for classification

**Computational Complexity**:
| Variant | FLOPs | Memory | Inference Time |
|---------|-------|--------|-----------------|
| ViT-Base | 17.6B | 680MB | ~15ms |
| ViT-Large | 61B | 1800MB | ~50ms |

**Best For**:
- **DermaMNIST**: Recommended for color-rich images
- Large datasets (ViT tends to overfit on small data)
- Capturing global dependencies
- Interpretability via attention maps

**Pretrained Weights**:
- ImageNet-21k pretrained (superior for ViT)
- Recommended: Use pretrained ViT-Base
- Transfer learning: Excellent cross-domain
- Fine-tuning: Moderate learning rates, longer warmup

**Configuration**:
```yaml
architecture: "vit"
variant: "vit_base"
pretrained: true
num_classes: 7  # DermaMNIST
dropout_rate: 0.1
patch_size: 16
```

**Architecture Details**:
```
Input image (3, 224, 224)
  ↓ Patch embedding (16×16 → 196 patches)
  ↓ Linear projection (768 dim)
(196 + 1, 768)  # + 1 for CLS token
  ↓ Positional embedding (learnable)
(197, 768)
  ↓ Transformer block × 12:
    - Layer norm
    - Multi-head attention (12 heads)
    - Layer norm
    - Feed-forward (3072 dim)
(197, 768)
  ↓ Layer norm
  ↓ CLS token → FC layer
Output logits
```

**Advantages**:
- Global receptive field from first layer
- Excellent for color/texture features
- Strong transfer learning
- Good for large datasets

**Disadvantages**:
- High memory requirements
- Slow inference compared to CNNs
- Needs large datasets or strong regularization
- Less data-efficient than CNNs with same parameters

**Training Tips**:
- Longer warmup: 10-20 epochs
- Smaller learning rates: 1e-4 to 5e-4
- Larger batch sizes: 256-512
- Stochastic depth or drop path regularization
- Training time: 2-3x longer than EfficientNet

---

## Transfer Learning & Fine-tuning

### Strategy 1: Frozen Backbone (Fastest)
```python
model = ModelFactory.create(
    architecture="efficientnet",
    variant="b3",
    pretrained=True
)

# Freeze all backbone parameters
for param in model.backbone.parameters():
    param.requires_grad = False

# Only train classifier head
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.01
)
```

**When to use**: Limited data (< 5K samples), quick experiments
**Expected accuracy**: Good (ImageNet pretraining is powerful)
**Training time**: ~5 minutes
**Pros**: Fast, stable training
**Cons**: Limited adaptation to medical domain

### Strategy 2: Layer-wise Learning Rate Decay (Recommended)
```python
# Lower learning rates for earlier layers
param_groups = [
    {'params': model.backbone.layer4.parameters(), 'lr': 1e-4},
    {'params': model.backbone.layer3.parameters(), 'lr': 1e-5},
    {'params': model.backbone.layer2.parameters(), 'lr': 1e-6},
    {'params': model.backbone.layer1.parameters(), 'lr': 1e-6},
    {'params': model.classifier.parameters(), 'lr': 1e-3},
]

optimizer = torch.optim.AdamW(param_groups)
```

**When to use**: Standard medical imaging fine-tuning
**Expected accuracy**: Excellent
**Training time**: ~30 minutes
**Pros**: Good adaptation, stable convergence
**Cons**: More tuning required

### Strategy 3: Gradual Unfreezing
```python
# Start with frozen backbone
freeze_epochs = [1, 10, 20]  # Unfreeze at these epochs

for epoch in range(max_epochs):
    if epoch in freeze_epochs:
        unfreeze_layers(model, layers_to_unfreeze=2)
    
    train_epoch(model, train_loader)
    validate(model, val_loader)
```

**When to use**: Large differences between ImageNet and medical domain
**Expected accuracy**: Excellent with stability
**Training time**: ~40 minutes
**Pros**: Gradual domain adaptation, prevents catastrophic forgetting

### Strategy 4: Full Fine-tuning
```python
model = ModelFactory.create(
    architecture="efficientnet",
    variant="b3",
    pretrained=True
)

# All parameters trainable, smaller learning rate
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)
```

**When to use**: Large datasets (> 50K samples), strong GPU resources
**Expected accuracy**: Highest possible
**Training time**: ~2-3 hours
**Pros**: Maximum model adaptation
**Cons**: Risk of overfitting, longer training

---

## Model Selection Criteria

### For PathMNIST (Tissue Classification)
```
Imbalance: Severe (195:1)
Size: Large (90K samples)
Colors: RGB, cellular level details

Recommended 1st: ResNet-50 (proven baseline)
Recommended 2nd: EfficientNet-B3 (better efficiency)
Recommended 3rd: DenseNet-121 (parameter efficient)

Loss function: Focal Loss (gamma=2.0)
Fine-tuning: Layer-wise learning rate decay
Training time: 2-3 hours (GPU)
```

### For ChestMNIST (X-ray Classification)
```
Imbalance: Moderate (30:1)
Size: Large (112K samples)
Colors: Grayscale, multi-label

Recommended 1st: EfficientNet-B3 (good balance)
Recommended 2nd: ResNet-50 (stable baseline)
Recommended 3rd: DenseNet-121 (efficient)

Loss function: Weighted Binary Cross-Entropy
Fine-tuning: Layer-wise learning rate decay
Training time: 2-4 hours (GPU)
Data augmentation: Conservative (no aggressive rotation)
```

### For DermaMNIST (Skin Lesion Classification)
```
Imbalance: Moderate (18:1)
Size: Small (10K samples)
Colors: RGB, color critical for diagnosis

Recommended 1st: Vision Transformer (color features)
Recommended 2nd: EfficientNet-B3 (good tradeoff)
Recommended 3rd: DenseNet-201 (parameter efficient)

Loss function: Focal Loss (gamma=2.5)
Fine-tuning: Gradual unfreezing
Training time: 1-3 hours (GPU)
Data augmentation: Standard with color preservation
Regularization: Stochastic depth, dropout
```

---

## Feature Extraction & Analysis

### Extracting Features

```python
from src.models.model_factory import ModelFactory

# Create model
model = ModelFactory.create("efficientnet", "b3", num_classes=14)

# Load trained weights
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract features (before classifier)
image = torch.randn(1, 3, 28, 28)
features = model.extract_features(image)  # Shape: (1, 1536)
```

**Feature dimensions**:
- ResNet-50: 2048
- EfficientNet-B3: 1536
- DenseNet-121: 1024
- ViT-Base: 768

### Visualization & Analysis

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import TSNE

# Extract features for all samples
features = []
labels = []
for images, batch_labels in test_loader:
    with torch.no_grad():
        batch_features = model.extract_features(images)
    features.append(batch_features.numpy())
    labels.append(batch_labels.numpy())

features = np.concatenate(features)
labels = np.concatenate(labels)

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20')
plt.colorbar()
plt.title('t-SNE of Model Features')
plt.show()
```

---

## Model Ensemble Methods

### Deep Ensemble
```python
from src.uncertainty.deep_ensemble import DeepEnsembleUncertainty

# Create 5 models with different random seeds
ensemble = DeepEnsembleUncertainty(num_models=5)

# Train each model
for i in range(5):
    model = ModelFactory.create("efficientnet", "b3")
    train_model(model, train_loader)
    ensemble.add_model(model)

# Inference
predictions, uncertainty = ensemble.predict(test_image)
```

**Uncertainty estimates**:
- Mean prediction: Average across models
- Variance: Disagreement measure
- Entropy: Model uncertainty
- Aleatoric: Approximate with dropout
- Epistemic: Variance across ensemble

---

## Uncertainty Quantification

### MC Dropout
```python
from src.uncertainty.mc_dropout import MCDropoutUncertainty

# Create model with dropout
model = ModelFactory.create("efficientnet", "b3", dropout_rate=0.3)

# Enable dropout at inference
mc_dropout = MCDropoutUncertainty(model, num_samples=10)

# Get stochastic predictions
predictions, uncertainty = mc_dropout.predict(image)
```

### Temperature Scaling
```python
from src.uncertainty.temperature_scaling import TemperatureScaling

# Calibrate on validation set
ts = TemperatureScaling()
ts.fit(val_logits, val_labels)

# Apply to test predictions
calibrated_probs = ts.transform(test_logits)
```

---

## Model Evaluation Metrics

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Single-label metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
```

### Uncertainty Quality Metrics
```python
from src.evaluation.uncertainty_metrics import UncertaintyMetrics

metrics = UncertaintyMetrics()

# Risk-coverage curve
risks, coverages = metrics.risk_coverage_curve(
    uncertainties, errors
)

# AUROC of uncertainty
auroc = metrics.uncertainty_auroc(uncertainties, errors)

# Spearman correlation with errors
corr, p_value = metrics.spearman_correlation(
    uncertainties, errors
)
```

### Calibration Metrics
```python
# Expected Calibration Error
ece = metrics.compute_ece(probabilities, labels, n_bins=10)

# Maximum Calibration Error
mce = metrics.compute_mce(probabilities, labels, n_bins=10)

# Brier Score
brier = metrics.compute_brier_score(probabilities, labels)
```

---

## Model Checkpointing & Persistence

### Saving Model
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_metric': best_metric,
    'config': config,
}

torch.save(checkpoint, 'best_model.pt')
```

### Loading Model
```python
# Load architecture
model = ModelFactory.create("efficientnet", "b3", num_classes=14)

# Load weights
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Metrics
epoch = checkpoint['epoch']
best_metric = checkpoint['best_metric']
```

---

## Inference Optimization

### Batch Inference
```python
model.eval()
with torch.no_grad():
    # Process in batches for efficiency
    predictions = []
    for images in test_loader:
        outputs = model(images)
        predictions.append(outputs.cpu())

predictions = torch.cat(predictions)
```

### Mixed Precision Inference
```python
with torch.cuda.amp.autocast():
    outputs = model(images)
    # ~2x faster, same accuracy
```

### Model Quantization
```python
# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
# ~4x smaller, 2-3x faster on CPU
```

---

## Troubleshooting & Best Practices

### Common Issues

1. **Model Not Improving**
   - Check learning rate (too high: divergence, too low: slow convergence)
   - Verify data preprocessing (normalization, augmentation)
   - Check batch size (too small: noisy, too large: memory)
   - Increase training duration

2. **Out of Memory (OOM)**
   - Reduce batch size (32 → 16)
   - Use gradient accumulation
   - Enable mixed precision training
   - Use smaller model variant

3. **Poor Generalization (Overfitting)**
   - Increase regularization (dropout, weight decay)
   - Use data augmentation
   - Early stopping
   - Reduce model size

4. **Unstable Training (NaN Loss)**
   - Check gradient clipping
   - Reduce learning rate
   - Verify data normalization
   - Check for invalid input values

### Best Practices

1. **Start Simple**: ResNet-50 or EfficientNet-B3 first
2. **Use Pretrained Weights**: ImageNet pretraining is powerful
3. **Monitor Metrics**: Track accuracy, loss, calibration
4. **Validate Regularly**: Check on validation set every epoch
5. **Save Best Weights**: Based on validation metric
6. **Use Early Stopping**: Prevent overfitting
7. **Layer-wise Fine-tuning**: Different learning rates for layers
8. **Data Augmentation**: Essential for small datasets
9. **Ensemble Models**: Multiple seeds for robustness
10. **Document Hyperparameters**: Save config with each experiment

---

## Performance Benchmarks

### Training Time (ImageNet Pretrained)
```
Single GPU (NVIDIA V100):
ResNet-50:        ~2 hours
EfficientNet-B3:  ~4 hours
DenseNet-121:     ~3 hours
ViT-Base:         ~8 hours
```

### Inference Speed (1 sample)
```
GPU (NVIDIA A100):
ResNet-50:        ~5ms
EfficientNet-B3:  ~8ms
DenseNet-121:     ~6ms
ViT-Base:         ~15ms

CPU (Intel i7):
ResNet-50:        ~100ms
EfficientNet-B3:  ~150ms
```

### Memory Requirements
```
GPU Memory (batch size 64):
ResNet-50:        2.5GB
EfficientNet-B3:  3.0GB
DenseNet-121:     2.8GB
ViT-Base:         5.0GB
```

---

## References & Resources

- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- EfficientNet: "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le, 2019)
- DenseNet: "Densely Connected Convolutional Networks" (Huang et al., 2017)
- Vision Transformer: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- PyTorch Torchvision: https://pytorch.org/vision/stable/models.html
- Timm Library: https://github.com/rwightman/pytorch-image-models
