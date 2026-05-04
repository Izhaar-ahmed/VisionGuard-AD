# 🛡️ VisionGuard-AD

**Industrial-Grade Unsupervised Visual Anomaly Detection on Apple Silicon**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-28%2F28-brightgreen.svg)](tests/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-black.svg)](https://support.apple.com/en-us/HT211814)

VisionGuard-AD is a production-ready anomaly detection system for **industrial quality control**. It implements state-of-the-art unsupervised methods (**PatchCore** + **FastFlow**) to detect manufacturing defects — scratches, holes, contamination, structural anomalies — without requiring labeled defect data for training.

> **Key Innovation**: Train on only "good" samples → detect any anomaly at pixel-level precision.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      VisionGuard-AD Pipeline                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────┐│
│  │  Input   │──▶│   Feature    │──▶│   Anomaly    │──▶│ Output ││
│  │  Image   │   │  Extractor   │   │   Scoring    │   │ Report ││
│  └─────────┘   │              │   │              │   └────────┘│
│                 │ • ResNet18   │   │ • PatchCore  │             │
│                 │ • WRN-50-2 ★ │   │   (kNN)      │   ┌────────┐│
│                 │ • ResNet50   │   │ • FastFlow   │──▶│Heatmap ││
│                 │ • EfficientB4│   │   (NF)       │   │Overlay ││
│                 │ • ViT-B/16   │   │              │   └────────┘│
│                 └──────────────┘   └──────────────┘             │
│                       │                    │                     │
│                 ┌─────▼────────────────────▼──────┐             │
│                 │       Evaluation Engine          │             │
│                 │  AUROC · AP · PRO · F1 · ROC    │             │
│                 └─────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results — Real MVTec-AD

### Backbone Comparison on Carpet Category (127 test images)

| Backbone | Image AUROC ↑ | Pixel AUROC ↑ | PRO Score ↑ | F1 Score ↑ | Inference Speed | Memory Bank |
|----------|:------------:|:------------:|:----------:|:---------:|:--------------:|:----------:|
| ResNet-18 | 96.90% | 97.35% | 89.01% | 94.95% | **12.0 img/s** | 19,757 × 384 |
| **WideResNet-50-2** ★ | **98.38%** | **97.85%** | **90.92%** | **96.97%** | 5.2 img/s | 19,757 × 1,536 |
| ViT-B/16 | 96.54% | 96.33% | 83.23% | 95.52% | 7.7 img/s | 4,940 × 1,536 |

> ★ **WideResNet-50-2 is the recommended backbone** — best AUROC and PRO scores, matching the original PatchCore paper.

### Key Findings

| Finding | Details |
|---------|---------|
| **Best Accuracy** | WideResNet-50-2: 98.38% Image AUROC, 97.85% Pixel AUROC |
| **Best Speed** | ResNet-18: 12 img/s on CPU (83ms/image) |
| **Best Localization** | WideResNet-50-2: 90.92% PRO Score |
| **ViT Observation** | Lower spatial resolution (14×14 vs 28×28 patches) limits pixel-level precision for textures |

### Sample Visualizations

The evaluation pipeline generates 4-panel comparison grids:

| Panel | Description |
|-------|-------------|
| **Original** | Input image from the test set |
| **Ground Truth** | Pixel-level defect mask (from MVTec-AD annotations) |
| **Heatmap** | Model's anomaly map with JET colormap (blue=normal, red=anomalous) |
| **Prediction** | Binary defect mask after threshold-based segmentation |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/VisionGuard-AD.git
cd VisionGuard-AD
pip install -r requirements.txt

# For ViT backbone support
pip install timm
```

### 2. Download MVTec-AD Dataset

**Option A — Official Website (Recommended)**
1. Register at https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Download `mvtec_anomaly_detection.tar.xz` (4.9 GB)
3. Extract:
```bash
mkdir -p data/mvtec
tar xf ~/Downloads/mvtec_anomaly_detection.tar.xz -C ./data/mvtec
```

**Option B — HuggingFace** (requires free account + token)
```bash
python scripts/download_mvtec.py --method huggingface --token YOUR_HF_TOKEN
```

**Option C — Synthetic Data** (for pipeline testing)
```bash
python scripts/download_mvtec.py --method synthetic
```

### 3. Train

```bash
# Recommended: WideResNet-50 backbone
python train.py --method patchcore --category carpet --backbone wide_resnet50

# Lightweight: ResNet-18 (faster training)
python train.py --method patchcore --category carpet --backbone resnet18

# ViT backbone (requires timm)
python train.py --method patchcore --category carpet --backbone vit_b16

# FastFlow (normalizing flow)
python train.py --method fastflow --category carpet --num_epochs 100
```

### 4. Evaluate

```bash
python evaluate.py --method patchcore --category carpet \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --backbone wide_resnet50 --visualize
```

### 5. Inference

```bash
# Single image
python inference.py --method patchcore \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --input ./data/mvtec/carpet/test/scratch/001.png \
    --threshold 0.37 --output result.jpg

# Folder batch
python inference.py --input ./data/mvtec/carpet/test/scratch/ \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --output ./results/

# Live webcam
python inference.py --input 0 \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --threshold 0.37
```

### 6. Dashboard

```bash
streamlit run app/app.py
```

---

## 🔬 Methods

### PatchCore (CVPR 2022)

PatchCore detects anomalies by comparing test image patches against a memory bank of normal patches:

1. **Feature Extraction** — Extract multi-scale features from a pretrained CNN/ViT backbone (layers 2 & 3)
2. **Patch Embedding** — Convert feature maps to local patch embeddings via neighborhood aggregation (3×3 AvgPool)
3. **Coreset Subsampling** — Reduce the memory bank using greedy k-center coreset selection (10% ratio)
4. **Anomaly Scoring** — For each test patch, compute distance to nearest neighbor in memory bank
5. **Anomaly Map** — Spatial anomaly scores reshaped to image resolution + Gaussian smoothing

> **Advantages**: No training required (single forward pass), state-of-the-art accuracy, interpretable heatmaps.

### FastFlow (ICCV 2021)

FastFlow uses normalizing flows to model the distribution of normal features:

1. **Feature Extraction** — Same backbone as PatchCore
2. **2D Normalizing Flow** — Affine coupling layers transform normal features to a Gaussian distribution
3. **Anomaly Scoring** — Negative log-likelihood under the learned distribution
4. **Training** — Minimize NLL loss on normal training data (requires gradient-based optimization)

> **Advantages**: Explicit density estimation, compact model, no memory bank storage.

---

## 📈 Evaluation Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| **Image AUROC** | Image | Area under ROC curve for binary normal/defect classification |
| **Pixel AUROC** | Pixel | AUROC computed per-pixel across all defective images |
| **PRO Score** | Region | Per-Region Overlap — weighted by connected component, up to FPR=0.3 |
| **AP** | Image | Average Precision (area under Precision-Recall curve) |
| **F1 Score** | Image | Harmonic mean of precision and recall at optimal threshold |

### Threshold Optimization

The system supports multiple threshold selection criteria:
- **F1** — Maximizes F1 score (default)
- **Youden's J** — Maximizes sensitivity + specificity - 1
- **Cost-based** — Custom cost matrix for FP/FN trade-offs

---

## 🍎 Apple Silicon Optimization

VisionGuard-AD is specifically optimized for M1/M2/M3 Macs:

| Optimization | Detail |
|-------------|--------|
| **MPS Backend** | Auto-detects `torch.backends.mps` for GPU acceleration |
| **Chunked Distance** | `torch.cdist` computed in 2048-patch chunks to prevent unified memory overflow |
| **pin_memory=False** | DataLoader configured for unified memory architecture |
| **num_workers=4** | Tuned for M1 efficiency core count |
| **faiss-cpu** | CPU-based approximate nearest neighbor (no CUDA dependency) |

---

## 📁 Project Structure

```
VisionGuard-AD/
├── models/
│   ├── backbones/
│   │   └── feature_extractor.py    # Multi-backbone feature extraction (5 architectures)
│   ├── patchcore/
│   │   ├── patchcore.py            # PatchCore: fit, predict, save/load
│   │   └── coreset_sampler.py      # Greedy k-center coreset selection
│   └── fastflow/
│       └── fastflow.py             # FastFlow: 2D normalizing flow
├── data/
│   ├── mvtec_dataset.py            # MVTec-AD dataset (15 categories)
│   └── visa_dataset.py             # VisA dataset (12 categories)
├── anomaly_map/
│   └── anomaly_map_generator.py    # Normalize, smooth, overlay, comparison grids
├── metrics/
│   └── evaluator.py                # AUROC, AP, PRO score, report generation
├── threshold/
│   └── threshold_optimizer.py      # F1/Youden/cost threshold, ROC/PR plots
├── app/
│   ├── app.py                      # Streamlit dashboard (4 pages)
│   └── utils.py                    # Dashboard utilities
├── train.py                        # Unified training CLI
├── evaluate.py                     # Full evaluation with metrics + visualizations
├── inference.py                    # Single image, folder, webcam inference
├── benchmark.py                    # All-categories benchmark
├── scripts/
│   ├── download_mvtec.py           # MVTec-AD download (HF, official, synthetic)
│   ├── generate_synthetic_data.py  # Synthetic data for testing
│   └── run_all_categories.sh       # Shell automation
├── configs/
│   ├── patchcore_config.yaml       # PatchCore hyperparameters
│   └── fastflow_config.yaml        # FastFlow hyperparameters
├── tests/
│   ├── test_dataset.py             # 8 dataset tests
│   ├── test_patchcore.py           # 9 model tests
│   └── test_metrics.py             # 11 metrics tests
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_visualization.ipynb
│   └── 03_results_analysis.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🧪 Testing

```bash
# Run all 28 tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_patchcore.py -v    # Model tests
python -m pytest tests/test_metrics.py -v      # Metrics tests
python -m pytest tests/test_dataset.py -v      # Dataset tests
```

All tests pass without requiring the actual MVTec-AD dataset (uses synthetic fixtures).

---

## ⚙️ Configuration

### PatchCore (configs/patchcore_config.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `wide_resnet50` | Feature extractor backbone |
| `coreset_ratio` | `0.1` | Memory bank reduction ratio (10%) |
| `num_neighbors` | `9` | k for kNN anomaly scoring |
| `sigma` | `4.0` | Gaussian smoothing for anomaly map |
| `img_size` | `224` | Input image resolution |

### FastFlow (configs/fastflow_config.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `wide_resnet50` | Feature extractor backbone |
| `flow_steps` | `8` | Number of coupling layers |
| `lr` | `1e-4` | Learning rate |
| `num_epochs` | `100` | Training epochs |

---

## 📖 References

1. **PatchCore**: Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
2. **FastFlow**: Yu et al., "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows" (2021)
3. **MVTec-AD**: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" (CVPR 2019)
4. **PRO Metric**: Bergmann et al., "Uninformed Students: Student-Teacher Anomaly Detection" (CVPR 2020)

---

## 📝 License

This project is licensed under the MIT License. The MVTec-AD dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

<p align="center">
  <strong>Built for industrial quality control on Apple Silicon 🍎</strong><br>
  <em>VisionGuard-AD — See defects before they ship.</em>
</p>
