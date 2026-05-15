# Data Documentation

## Overview

This project uses the **MedMNIST** dataset collection, which provides pre-processed medical imaging datasets in a standardized format. Three primary datasets are supported:

1. **PathMNIST**: Tissue image classification
2. **ChestMNIST**: Chest X-ray disease detection
3. **DermaMNIST**: Skin lesion classification

All datasets are automatically downloaded and verified during pipeline initialization.

---

## MedMNIST Datasets

### PathMNIST: Pathology Image Classification

**Purpose**: Binary tissue classification (tumor vs non-tumor)

**Dataset Statistics**:
- Total samples: 89,996
- Train/Val/Test split: 70% / 15% / 15%
- Image size: 28×28 pixels
- Color space: RGB (3 channels)
- Classes: 9 tissue types
- Class names:
  ```
  0. Adipose (fat tissue)
  1. Background
  2. Debris
  3. Lymphocytes
  4. Mitoses
  5. Monocytes
  6. Neutrophils
  7. Nuclei
  8. Red blood cells
  ```

**Class Distribution**:
```
Nuclei:         ~35,000 (39%)
Background:     ~22,000 (24%)
Red blood cells: ~15,000 (17%)
Lymphocytes:    ~8,000 (9%)
Monocytes:      ~4,000 (4%)
Adipose:        ~3,000 (3%)
Neutrophils:    ~2,000 (2%)
Mitoses:        ~500 (<1%)
Debris:         ~200 (<1%)
```

**Characteristics**:
- Highly imbalanced dataset
- Severe class imbalance (Debris: 0.2%, Nuclei: 39%)
- Very small image size (28×28)
- High-resolution cellular features
- Requires careful augmentation (minimal rotation/distortion)

**Recommended Architecture**: EfficientNet or DenseNet
**Recommended Loss**: Focal Loss with gamma=2.0

---

### ChestMNIST: Chest X-ray Disease Detection

**Purpose**: Multi-label chest disease classification from X-rays

**Dataset Statistics**:
- Total samples: 112,120
- Train/Val/Test split: 70% / 15% / 15%
- Image size: 28×28 pixels
- Color space: Grayscale (1 channel)
- Task: Multi-label (samples can have multiple diseases)
- Classes: 14 diseases

**Class Names & Prevalence**:
```
0. Atelectasis              ~25% of samples
1. Cardiomegaly (CRITICAL)  ~15% of samples
2. Effusion (CRITICAL)      ~20% of samples
3. Infiltration             ~30% of samples
4. Mass                     ~8% of samples
5. Nodule                   ~6% of samples
6. Pneumonia (CRITICAL)     ~18% of samples
7. Pneumothorax (CRITICAL)  ~5% of samples
8. Consolidation            ~8% of samples
9. Edema                    ~6% of samples
10. Emphysema               ~4% of samples
11. Fibrosis                ~5% of samples
12. Pleural Thickening      ~4% of samples
13. Hernia                  ~<1% of samples
```

**Multi-label Distribution**:
- Single label only: ~10%
- Two labels: ~30%
- Three labels: ~35%
- Four+ labels: ~25%

**Characteristics**:
- Multi-label classification (not mutually exclusive)
- Severe class imbalance (Hernia: <1%, Infiltration: 30%)
- Grayscale images (medical X-rays)
- Clinically critical diseases: Cardiomegaly, Effusion, Pneumothorax
- Requires weighted loss for imbalance
- Higher stakes for classification errors

**Critical Classes** (high clinical impact):
- Class 1: Cardiomegaly (enlarged heart, risk of heart failure)
- Class 2: Effusion (fluid in pleural cavity, respiratory distress)
- Class 7: Pneumothorax (collapsed lung, life-threatening)

**Recommended Architecture**: EfficientNet-B3 (good efficiency/accuracy)
**Recommended Loss**: Weighted Binary Cross-Entropy with class weights
**Recommended Augmentation**: Conservative (no aggressive rotation/flip for X-rays)

---

### DermaMNIST: Skin Lesion Classification

**Purpose**: Multi-class skin lesion classification (melanoma detection priority)

**Dataset Statistics**:
- Total samples: 10,015
- Train/Val/Test split: 70% / 15% / 15%
- Image size: 28×28 pixels
- Color space: RGB (3 channels)
- Classes: 7 skin lesion types
- Small dataset (limited samples)

**Class Names & Prevalence**:
```
0. Melanoma (CRITICAL)                    ~8% of samples
1. Melanocytic Nevus                      ~50% of samples
2. Basal Cell Carcinoma (CRITICAL)        ~13% of samples
3. Actinic Keratosis / Bowen Disease      ~13% of samples
4. Benign Keratosis                       ~10% of samples
5. Dermatofibroma                         ~3% of samples
6. Vascular Lesion                        ~3% of samples
```

**Class Distribution**:
```
Melanocytic Nevus:     ~5,000 (50%)
Basal Cell Ca.:        ~1,300 (13%)
Actinic Keratosis:     ~1,300 (13%)
Benign Keratosis:      ~1,000 (10%)
Melanoma:              ~800 (8%)
Dermatofibroma:        ~300 (3%)
Vascular Lesion:       ~300 (3%)
```

**Characteristics**:
- Small dataset (only 10K samples)
- Moderate class imbalance
- Rich color information (RGB)
- Cancer detection critical (Melanoma, BCC)
- High clinical stakes
- Requires data augmentation
- Resistant to aggressive color augmentation

**Critical Classes** (cancer/precancer):
- Class 0: Melanoma (deadliest skin cancer, high mortality if missed)
- Class 2: Basal Cell Carcinoma (most common skin cancer)

**Human Expert Baseline**:
- Dermatologist accuracy: ~89%
- Experienced: ~91%
- Novice: ~75%

**Recommended Architecture**: Vision Transformer (good for color features)
**Recommended Loss**: Focal Loss (severe class imbalance)
**Recommended Augmentation**: Standard with color preservation
**Cost Model**: Very conservative (ai_error_cost=500, human_review=5)

---

## Data Preprocessing Pipeline

### 1. Loading

```python
from src.data.medmnist_loader import MedMNISTLoader

loader = MedMNISTLoader(dataset_name="pathmnist", download=True)
train_data = loader.load_train()
val_data = loader.load_val()
test_data = loader.load_test()
```

**Features**:
- Automatic download from MedMNIST repository
- Checksum verification for data integrity
- Caching to avoid re-downloads
- Support for all three datasets

### 2. Preprocessing

```python
from src.data.preprocessing import Preprocessor

preprocessor = Preprocessor(dataset="chestmnist")
normalized_image = preprocessor.preprocess(image)
```

**Steps**:
1. Convert dtype if needed (uint8 → float32)
2. Normalize to [0, 1] range
3. Standardize with ImageNet statistics (or medical imaging defaults)
4. Reshape to standard format (H, W, C)

**Normalization Constants**:
```
ImageNet (default):
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]

Grayscale Medical:
  mean = [0.5]
  std = [0.5]
```

### 3. Augmentation

Augmentation is dataset-aware to preserve medical imaging characteristics.

#### PathMNIST Augmentation (Standard)
```yaml
augmentation:
  level: "standard"
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation: 15  # degrees
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
  sharpness_range: [0.8, 1.2]
```

#### ChestMNIST Augmentation (Conservative)
```yaml
augmentation:
  level: "light"
  horizontal_flip: 0.3  # Limited flipping for anatomical correctness
  vertical_flip: 0.0    # No vertical flip (preserves anatomy)
  rotation: 10          # Limited rotation
  color_jitter:
    brightness: 0.1     # Minimal brightness changes
    contrast: 0.1
  gaussian_blur: 0.1    # Light blur to preserve details
```

**Rationale**: X-ray anatomy is directional; excessive augmentation loses diagnostic value.

#### DermaMNIST Augmentation (Standard with Color Preservation)
```yaml
augmentation:
  level: "standard"
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation: 20
  color_jitter:
    hue: 0.05           # Preserve color (critical for melanoma detection)
    saturation: 0.1
    brightness: 0.1
    contrast: 0.1
  elastic_transform: false  # Don't distort lesion shapes
```

**Rationale**: Color is critical for skin lesion classification; limit aggressive augmentation.

### 4. Batching & DataLoader

```python
from src.data.dataloader import create_dataloader

train_loader = create_dataloader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

**Features**:
- Stratified sampling (maintains class distribution)
- Imbalance handling (oversampling minority classes)
- Memory-efficient data loading
- Configurable worker threads

---

## Data Validation & Quality Assurance

### 1. Shape Validation
```python
assert image.shape == (H, W, C)  # e.g., (28, 28, 3) for PathMNIST
assert label.shape == (num_classes,) or label.ndim == 0
```

### 2. Value Range Validation
```python
# After normalization: [-1, 1] or [0, 1]
assert image.min() >= -1.0
assert image.max() <= 1.0
```

### 3. Label Validity
```python
# Multi-class: single label per sample
assert (label == 0).sum() + (label == 1).sum() == 1

# Multi-label: multiple labels possible
assert (label >= 0).all() and (label <= 1).all()
```

### 4. Data Integrity
```python
# No NaN or Inf values
assert not np.isnan(image).any()
assert not np.isinf(image).any()

# Class distribution consistency
train_dist = compute_class_distribution(train_set)
val_dist = compute_class_distribution(val_set)
assert correlation(train_dist, val_dist) > 0.95  # Should be similar
```

### 5. Data Leakage Detection
```python
# Verify no sample appears in multiple splits
train_ids = set(train_set.indices)
val_ids = set(val_set.indices)
test_ids = set(test_set.indices)
assert len(train_ids & val_ids) == 0  # No overlap
assert len(train_ids & test_ids) == 0
```

---

## Class Imbalance Handling

### Strategy 1: Weighted Loss

Assign higher weights to minority classes:
```python
class_weights = compute_class_weights(dataset)
# ChestMNIST example:
# [1.5, 2.0, 1.5, 1.3, 1.8, ...]  # Higher weights for rare classes
```

**When to use**: Simple, effective for moderate imbalance

### Strategy 2: Focal Loss

Down-weight easy examples, focus on hard negatives:
```python
focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
```

**When to use**: Severe imbalance (Hernia: <1%, Infiltration: 30%)

### Strategy 3: Oversampling Minority

Duplicate minority class samples:
```python
oversampler = RandomOverSampler(sampling_strategy=0.5)
train_data = oversampler.fit_resample(train_data)
```

**When to use**: Very small minority classes

### Strategy 4: Cost-Sensitive Classification

Assign costs to different errors:
```python
# Hernia misclassification: high cost
# Common disease misclassification: lower cost
cost_matrix = [[0, 1, 2, ...],    # Cost of misclassifying as each class
               [1, 0, 1, ...],
               ...]
```

**When to use**: Medical domain with different error costs

---

## Data Statistics & Analysis

### PathMNIST Statistics
```
Total samples: 89,996
Train: 62,997 (70%)
Val:   13,499 (15%)
Test:  13,500 (15%)

Class distribution (Train):
  Nuclei:         39.0%
  Background:     24.3%
  Red blood cells: 17.1%
  Lymphocytes:     8.9%
  Monocytes:       4.1%
  Adipose:         3.2%
  Neutrophils:     2.3%
  Mitoses:         0.6%
  Debris:          0.2%

Imbalance ratio: 195:1 (Nuclei:Debris)
```

### ChestMNIST Statistics
```
Total samples: 112,120
Train: 78,484 (70%)
Val:   16,818 (15%)
Test:  16,818 (15%)

Multi-label distribution:
  1 disease:  10%
  2 diseases: 30%
  3 diseases: 35%
  4+ diseases: 25%

Top 5 diseases:
  Infiltration: 30.1%
  Atelectasis:  25.3%
  Effusion:     20.1%
  Pneumonia:    18.2%
  Cardiomegaly: 15.0%

Imbalance ratio: 30:1 (Infiltration:Hernia)
```

### DermaMNIST Statistics
```
Total samples: 10,015
Train: 7,010 (70%)
Val:   1,503 (15%)
Test:  1,502 (15%)

Class distribution (Train):
  Melanocytic Nevus: 49.6%
  Basal Cell Ca.:    12.9%
  Actinic Keratosis: 12.9%
  Benign Keratosis:  10.4%
  Melanoma:          8.2%
  Dermatofibroma:    3.2%
  Vascular Lesion:   2.8%

Imbalance ratio: 18:1 (Nevus:Vascular)
Cancer prevalence: 20.3% (Melanoma + BCC)
```

---

## Data Storage & Organization

```
data/
├── raw/
│   ├── pathmnist.npz          # Raw compressed dataset
│   ├── chestmnist.npz
│   └── dermamnist.npz
├── processed/
│   ├── pathmnist/
│   │   ├── train_images.pt
│   │   ├── train_labels.pt
│   │   ├── val_images.pt
│   │   ├── val_labels.pt
│   │   ├── test_images.pt
│   │   └── test_labels.pt
│   ├── chestmnist/
│   │   └── [similar structure]
│   └── dermamnist/
│       └── [similar structure]
└── splits/
    ├── pathmnist_split.csv    # Train/val/test indices
    ├── chestmnist_split.csv
    └── dermamnist_split.csv
```

**File Sizes** (approximate):
- PathMNIST raw: 65 MB
- ChestMNIST raw: 280 MB
- DermaMNIST raw: 40 MB
- Total: 385 MB (uncompressed ~2 GB)

---

## Known Issues & Limitations

### 1. Size Limitation
**Issue**: All images are 28×28 pixels (very small)
**Impact**: Limited fine-grained feature extraction
**Mitigation**: Use upsampling, consider full-resolution MedMNIST+ datasets

### 2. Class Imbalance
**Issue**: Severe imbalance in all datasets
**Impact**: Model may ignore minority classes
**Mitigation**: Use weighted loss, focal loss, or data augmentation

### 3. Multi-label Complexity (ChestMNIST)
**Issue**: Multiple diseases per sample, not mutually exclusive
**Impact**: Complex training and evaluation
**Mitigation**: Use binary cross-entropy, per-label metrics

### 4. Limited Diversity
**Issue**: Datasets are pre-processed, standardized, potentially missing edge cases
**Impact**: Model may not generalize to real clinical data
**Mitigation**: Validate on diverse external datasets

### 5. No Patient Context
**Issue**: Individual images without patient history, demographics
**Impact**: Cannot leverage temporal or patient-level patterns
**Mitigation**: Train on image level, aggregate predictions if needed

---

## Data Download & Verification

### Automatic Download
```bash
python scripts/download_data.py --all
```

### Manual Download
```python
from src.data.medmnist_loader import MedMNISTLoader

loader = MedMNISTLoader(dataset_name="pathmnist", download=True)
# Automatically downloads and verifies
```

### Verification
```bash
python scripts/download_data.py --verify-checksum
```

**Checksums**:
- PathMNIST: Provided by MedMNIST team
- ChestMNIST: Provided by MedMNIST team
- DermaMNIST: Provided by MedMNIST team

---

## Data Privacy & Ethics

### Patient Privacy
- MedMNIST datasets are de-identified
- No patient identifiers included
- Images are standardized and compressed
- Suitable for research/development

### Usage Rights
- Published research datasets
- Citation required in publications
- Check original paper for licensing

### Responsible AI
- Be aware of potential dataset biases
- Test for demographic disparities
- Consider fairness across subgroups
- Document limitations in reports

---

## References & Additional Resources

### MedMNIST Paper
- Dataset: https://medmnist.com/
- Paper: "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification"
- Available: arXiv, ISBI 2021

### Dataset Citations
```bibtex
@article{yang2021medmnist,
  title={MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and others},
  journal={arXiv preprint arXiv:2110.06465},
  year={2021}
}
```

### Related Datasets
- ImageNet: General visual classification baseline
- CIFAR-10: Small image classification reference
- NIH ChestX-ray14: Real clinical X-rays (larger, noisier)

---

## Troubleshooting

### Download Issues
```bash
# Clear cache and retry
rm -rf ~/.medmnist/
python scripts/download_data.py --all
```

### Checksum Mismatch
```bash
# Download may be incomplete or corrupted
python scripts/download_data.py --verify-checksum --force-redownload
```

### Memory Issues
```python
# Use smaller batch size
dataloader = create_dataloader(batch_size=32)  # Instead of 64

# Use streaming instead of loading all at once
# Implement custom data loading for very large datasets
```

### Class Imbalance Problems
```python
# Use stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

# Use weighted sampler
sampler = WeightedRandomSampler(weights=class_weights)

# Use focal loss instead of cross-entropy
loss = FocalLoss(gamma=2.0)
```
