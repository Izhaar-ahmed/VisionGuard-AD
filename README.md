# VisionGuard-AD

**A Unified Anomaly Detection Framework with Multi-Backbone Comparative Analysis on MVTec-AD**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-28%2F28-brightgreen.svg)](tests/)

---

## What Problem Does This Solve?

Most anomaly detection implementations are **single-backbone, single-method demos** that report numbers from the paper without verifying them. VisionGuard-AD is different:

**Our core contribution is a rigorous, apples-to-apples backbone comparison framework.** We run the exact same PatchCore pipeline — same coreset ratio, same preprocessing, same evaluation metrics — across three fundamentally different feature extractors (ResNet-18, WideResNet-50-2, ViT-B/16) on real MVTec-AD data, and analyze *why* each backbone succeeds or fails on texture vs. object categories.

Key findings that you won't get from the original paper:
- **ViT-B/16 underperforms CNNs on texture categories** (83.23% vs 90.92% PRO on carpet) because its 14×14 patch grid loses fine-grained spatial information that textures require.
- **ResNet-18 achieves 97.35% pixel AUROC** — within 0.5% of WideResNet-50 — despite being 6× smaller, making it viable for edge deployment where WRN-50 is impractical.
- **WideResNet-50-2 remains the best all-rounder** (98.38% image AUROC) but at 4× the memory cost and 2.3× slower inference than ResNet-18.

This framework is designed so any new backbone can be plugged in and benchmarked under identical conditions.

---

## Benchmark Results — Real MVTec-AD (Carpet, 127 Test Images)

All three backbones were trained and evaluated under identical conditions: same coreset ratio (10%), same k=9 neighbors, same Gaussian sigma (4.0), same 224×224 input resolution.

| Backbone | Image AUROC | Pixel AUROC | PRO Score | F1 Score | AP | Speed | Memory Bank |
|----------|:-----------:|:-----------:|:---------:|:--------:|:--:|:-----:|:-----------:|
| ResNet-18 | 96.90% | 97.35% | 89.01% | 94.95% | 99.16% | **12.0 img/s** | 19,757 × 384 |
| **WideResNet-50-2** | **98.38%** | **97.85%** | **90.92%** | **96.97%** | **99.56%** | 5.2 img/s | 19,757 × 1,536 |
| ViT-B/16 | 96.54% | 96.33% | 83.23% | 95.52% | 99.10% | 7.7 img/s | 4,940 × 1,536 |

**Why ViT underperforms here:** ViT-B/16 tokenizes the image into a 14×14 grid of patches (196 tokens). After removing the CLS token, the spatial resolution available for pixel-level anomaly localization is 14×14 — compared to 28×28 for ResNet-18's layer2 features. For texture categories like carpet, where defects can be as small as a few pixels (thin scratches, subtle color shifts), this 4× reduction in spatial density directly hurts the PRO score. ViT would likely outperform on object categories (bottle, transistor) where global context matters more than local texture.

---

## Failure Analysis — Where It Breaks and Why

Understanding limitations is as important as reporting successes:

| Failure Mode | Observed Behavior | Root Cause |
|-------------|-------------------|------------|
| **False positives on "good" textures** | Some normal carpet samples score 0.516–0.528 (close to threshold 0.39) | Natural texture variation in carpet creates feature distributions that overlap with subtle defects. The model sees unusual-but-normal weave patterns as anomalous. |
| **Pixel AP is low (~47-51%)** despite high pixel AUROC (~97%) | AUROC is high but AP is mediocre | Severe class imbalance at pixel level — defective pixels are <5% of total pixels. AUROC is robust to imbalance; AP is not. This is a known limitation of all patch-based methods on MVTec. |
| **ViT spatial resolution** | PRO drops from 90.92% (WRN50) to 83.23% (ViT) | 14×14 feature grid cannot localize small defects. Interpolating back to 224×224 creates blurry anomaly maps that fail per-region overlap. |
| **Coreset sampling is slow on CPU** | ~10 min for 252 images | Greedy k-center requires O(N×k) distance computations. The random projection approximation helps for >50k patches but the bottleneck remains sequential argmax operations. |
| **Single-category training** | Each model is trained per-category | PatchCore by design learns a "normal" distribution for one category. A universal anomaly detector would require a different architecture (e.g., UniAD). |
| **No temporal modeling** | Live camera mode treats each frame independently | Manufacturing defects that evolve over time (progressive wear, gradual contamination) are not captured. Each frame is scored in isolation. |

---

## How It Works — Technical Deep Dive

### The Core Idea

In manufacturing quality control, you have thousands of images of "good" products but very few (or zero) images of defects. Traditional supervised classification fails because you can't train on defect classes you've never seen.

**Unsupervised anomaly detection** solves this by learning what "normal" looks like, then flagging anything that deviates. VisionGuard-AD implements **PatchCore** (CVPR 2022), which works at the *patch level* — it doesn't just say "this image is defective," it shows you exactly *where* the defect is, pixel by pixel.

### PatchCore Pipeline (Step by Step)

**Step 1: Feature Extraction**

We take a pretrained CNN (e.g., WideResNet-50-2 trained on ImageNet) and remove its classification head. We hook into intermediate layers (layer2 and layer3) to capture mid-level features — these contain texture and structural information without being too abstract.

For a 224×224 input image:
- layer2 outputs a 28×28×512 feature map (512 channels, 28×28 spatial grid)
- layer3 outputs a 14×14×1024 feature map

We upsample layer3 to match layer2's spatial size and concatenate them, giving a 28×28×1536 feature tensor. Each spatial position represents one "patch" of the original image.

**Step 2: Patch Embedding with Neighborhood Aggregation**

Raw features at a single spatial position only capture a small receptive field. We apply 3×3 average pooling to each feature map before concatenation. This means each patch embedding incorporates information from its 3×3 neighborhood, making it more robust to small spatial shifts.

The result: 28×28 = 784 patch embeddings per image, each of dimension 1536 (for WRN-50). All embeddings are L2-normalized.

**Step 3: Memory Bank Construction**

For 252 training images, we get 252 × 784 = 197,568 patch embeddings. Storing all of them would make nearest-neighbor search slow and memory-intensive.

**Coreset subsampling** reduces this to 10% (19,757 patches) while preserving coverage. The algorithm is greedy k-center:
1. Pick a random patch as the first coreset point
2. Find the patch that is *farthest* from any selected coreset point
3. Add it to the coreset
4. Repeat until we have enough points

This guarantees that every patch in the full set has a coreset representative within a bounded distance. For sets >50,000 patches, we first project embeddings to 128 dimensions via random projection (Johnson-Lindenstrauss lemma) to speed up distance computation.

**Step 4: Anomaly Scoring (Inference)**

For a test image:
1. Extract patch embeddings (same as training)
2. For each of the 784 patches, compute L2 distance to its nearest neighbor in the memory bank
3. **Image-level score** = maximum patch distance (the most anomalous patch determines the image score)
4. **Anomaly map** = reshape the 784 distances back to 28×28, bilinearly upsample to 224×224, then apply Gaussian smoothing (σ=4.0) for a clean heatmap

**Step 5: Threshold-Based Decision**

The anomaly score is compared against an optimized threshold. The threshold optimizer supports three strategies:
- **F1 optimization**: Maximizes harmonic mean of precision and recall
- **Youden's J**: Maximizes sensitivity + specificity − 1 (optimal point on ROC curve)
- **Cost-based**: Minimizes business cost = cost_FP × FP + cost_FN × FN (default: shipping a defective product costs 10× more than falsely rejecting a good one)

### Feature Extractor Architecture

The feature extractor supports 5 backbone architectures, all loaded with pretrained ImageNet weights:

| Backbone | Params | Feature Layers | Patch Embedding Dim | Spatial Grid |
|----------|--------|---------------|--------------------:|:------------:|
| ResNet-18 | 11M | layer2 (128) + layer3 (256) | 384 | 28×28 |
| ResNet-50 | 25M | layer2 (512) + layer3 (1024) | 1536 | 28×28 |
| WideResNet-50-2 | 69M | layer2 (512) + layer3 (1024) | 1536 | 28×28 |
| EfficientNet-B4 | 19M | features[3] + features[5] | varies | varies |
| ViT-B/16 | 86M | block_6 (768) + block_9 (768) | 1536 | 14×14 |

For CNNs, we register forward hooks on the target layers to capture intermediate activations without modifying the model. For ViT, we hook into transformer blocks 6 and 9, extract patch tokens (removing the CLS token), and reshape them from (B, 196, 768) back to (B, 768, 14, 14) spatial format so they're compatible with the same PatchEmbedding module used by CNNs.

### FastFlow (Alternative Method)

VisionGuard-AD also implements FastFlow, a normalizing flow approach:
- Instead of a memory bank, it trains a stack of 8 affine coupling layers (RealNVP-style) to map normal features to a standard Gaussian N(0,I)
- At inference, anomalous features land in low-probability regions of the Gaussian, yielding high negative log-likelihood scores
- Each coupling layer splits channels in half; one half parameterizes scale and translation for the other half via a small CNN
- Scale parameters are clamped (tanh with clamp=2.0) for numerical stability

FastFlow requires gradient-based training (unlike PatchCore which is gradient-free), using Adam optimizer with cosine annealing LR schedule and early stopping.

> **Note:** FastFlow is implemented and tested but we have not run a full MVTec benchmark with it. The benchmark results above are PatchCore only.

### Evaluation Metrics

**Image-Level Metrics:**
- **AUROC (Area Under ROC Curve):** Threshold-independent measure of classification quality. 100% means perfect separation of normal and anomalous images at some threshold.
- **AP (Average Precision):** Area under the Precision-Recall curve. More sensitive to class imbalance than AUROC.
- **F1 Score:** Harmonic mean of precision and recall at the optimal threshold.

**Pixel-Level Metrics:**
- **Pixel AUROC:** Same as image AUROC but computed per-pixel across all test images. Every pixel's anomaly score is treated as an independent prediction.
- **Pixel AP:** Per-pixel average precision.
- **PRO Score (Per-Region Overlap):** The most demanding metric. It finds connected components in the ground truth mask, computes overlap with predictions for each region separately, and averages. This prevents a single large defect from dominating the score. We integrate the PRO-vs-FPR curve from 0 to 0.3 FPR and normalize.

### Anomaly Map Visualization

The AnomalyMapGenerator provides:
- **Normalization:** min-max, z-score, or percentile-based scaling to [0,1]
- **Gaussian smoothing:** Configurable sigma for noise reduction
- **Heatmap overlay:** JET colormap blended with original image (alpha=0.4), with green contours around detected defect regions
- **4-panel comparison grid:** Original | Ground Truth | Heatmap | Prediction mask — for side-by-side evaluation

### Dataset Support

**MVTec-AD** (15 categories, 5354 images total):
- 5 texture categories: carpet, grid, leather, tile, wood
- 10 object categories: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper
- Training: only "good" images (no defects)
- Testing: mix of good + multiple defect types with pixel-level masks
- The dataset loader handles train/val splitting (10% holdout), ImageNet normalization, and data augmentation (random flip, rotation, color jitter for training)

### Threshold Optimization

The ThresholdOptimizer is a production-critical component with four modes:
1. **F1 optimization** — sweeps precision-recall curve to find threshold maximizing F1
2. **Youden's J** — finds the ROC operating point maximizing TPR−FPR
3. **PR balance** — finds where precision equals recall
4. **Cost-based analysis** — minimizes total business cost given customizable FP/FN cost ratios (default: FN costs 10× more than FP, because shipping a defective product is worse than over-rejecting)

It also generates ROC curves with optimal operating points marked, PR curves with AP annotation, and sensitivity analysis DataFrames showing metrics across all thresholds.

---

## Project Structure (4,595 lines of Python)

```
VisionGuard-AD/
├── models/
│   ├── backbones/
│   │   └── feature_extractor.py    # 405 lines — 5 backbones, forward hooks, ViT reshape, PatchEmbedding
│   ├── patchcore/
│   │   ├── patchcore.py            # 268 lines — fit, predict, chunked kNN, save/load
│   │   └── coreset_sampler.py      # 112 lines — greedy k-center, random projection, chunked cdist
│   └── fastflow/
│       └── fastflow.py             # 320 lines — AffineCouplingLayer, NLL loss, anomaly map
├── data/
│   └── mvtec_dataset.py            # 336 lines — 15 categories, train/val/test splits, augmentation
├── anomaly_map/
│   └── anomaly_map_generator.py    # 263 lines — normalize, smooth, overlay, comparison grid
├── metrics/
│   └── evaluator.py                # 121 lines — AUROC, AP, PRO score, JSON/MD reports
├── threshold/
│   └── threshold_optimizer.py      # 189 lines — F1/Youden/cost optimization, ROC/PR plots
├── app/
│   ├── app.py                      # 350 lines — Streamlit dashboard (4 pages)
│   └── utils.py                    # 110 lines — model loading, inference helpers
├── train.py                        # 268 lines — unified CLI for PatchCore + FastFlow
├── evaluate.py                     # 215 lines — full evaluation pipeline
├── inference.py                    # 279 lines — single image, batch folder, webcam
├── benchmark.py                    # 190 lines — multi-category benchmark runner
├── scripts/
│   ├── download_mvtec.py           # 286 lines — official URL, HuggingFace, synthetic fallback
│   └── generate_synthetic_data.py  # 100 lines — synthetic test data for CI
├── tests/                          # 360 lines — 28 tests (dataset, model, metrics)
├── configs/                        # YAML configs for PatchCore and FastFlow
└── notebooks/                      # Jupyter notebooks for exploration
```

---

## Quick Start

```bash
# Install
git clone https://github.com/Izhaar-ahmed/VisionGuard-AD.git
cd VisionGuard-AD
pip install -r requirements.txt
pip install timm  # for ViT backbone

# Download MVTec-AD (register at mvtec.com, download tar.xz)
mkdir -p data/mvtec
tar xf ~/Downloads/mvtec_anomaly_detection.tar.xz -C ./data/mvtec

# Train (WideResNet-50 recommended)
python train.py --method patchcore --category carpet --backbone wide_resnet50

# Evaluate
python evaluate.py --method patchcore --category carpet \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --backbone wide_resnet50 --visualize

# Single image inference
python inference.py --method patchcore \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --input ./data/mvtec/carpet/test/scratch/001.png \
    --threshold 0.37

# Launch dashboard
pip install streamlit plotly
streamlit run app/app.py
```

---

## Testing

```bash
python -m pytest tests/ -v  # 28/28 tests pass
```

Tests use synthetic fixtures — no MVTec download required. Coverage includes:
- **test_dataset.py** (8 tests): dataset loading, splits, transforms, statistics
- **test_patchcore.py** (9 tests): feature extraction, memory bank, predict, save/load
- **test_metrics.py** (11 tests): AUROC, AP, PRO score, threshold optimization, report generation

---

## Configuration

All hyperparameters are in `configs/patchcore_config.yaml`:

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `backbone` | wide_resnet50 | Feature extractor architecture |
| `coreset_ratio` | 0.1 | Memory bank size (10% of all patches) |
| `num_neighbors` | 9 | k for kNN scoring |
| `sigma` | 4.0 | Gaussian smoothing for anomaly maps |
| `img_size` | 224 | Input resolution |
| `cost_fn` | 10.0 | Cost of missing a defect (for cost-based threshold) |
| `cost_fp` | 1.0 | Cost of false rejection |

---

## References

1. Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
2. Yu et al., "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows" (2021)
3. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" (CVPR 2019)

---

> **Environment note:** Developed and tested on Apple Silicon M1. The codebase auto-detects MPS/CUDA/CPU via `torch.backends.mps.is_available()`. DataLoader uses `pin_memory=False` for unified memory, and distance computations are chunked (2048 patches) to prevent OOM on shared GPU memory.
