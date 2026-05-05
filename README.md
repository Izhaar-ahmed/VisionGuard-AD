# VisionGuard-AD

**Industrial Anomaly Detection System with Multi-Backbone Comparative Analysis**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Tests](https://img.shields.io/badge/Tests-Passing-success)

---

## 🚀 TL;DR

VisionGuard-AD is an unsupervised industrial anomaly detection system that learns "normal" from defect-free images and detects defects using PatchCore and FastFlow.

- Achieves **98.38% Image AUROC** on MVTec-AD (carpet)
- Supports **multiple backbones (ResNet, WideResNet, ViT)** for fair comparison
- Includes **robustness testing** under real-world degradations
- Enables **incremental updates** without retraining
- Provides **end-to-end pipeline**: training → evaluation → inference → dashboard

### 🔑 Key Takeaways

- WideResNet-50-2 gives best accuracy but is slower and memory-heavy
- ResNet-18 offers strong performance with much higher speed (edge-friendly)
- ViT struggles with small texture defects due to lower spatial resolution (14×14)

### 💡 Why This Matters

- Reduces manual inspection cost in manufacturing
- Detects unseen defects without labeled data
- Adapts to real-world conditions via incremental updates
- Provides actionable insights through failure analysis

---

## The Problem We Are Solving

In manufacturing and quality control, factories produce thousands of products daily. Every product must be inspected for defects — scratches, dents, contamination, structural damage. Traditionally, human inspectors do this manually, but humans are slow, inconsistent, and miss subtle defects when fatigued.

The challenge is: **you have thousands of images of good products, but very few (or zero) images of defective ones.** You cannot train a standard image classifier because there are no defect examples to learn from. New defect types can appear at any time — a classifier trained on "scratch" defects would miss "hole" defects entirely.

**Unsupervised anomaly detection** solves this by learning what "normal" looks like from defect-free images only, then flagging anything that deviates from that learned normal distribution. VisionGuard-AD implements this approach using state-of-the-art methods and provides a complete pipeline from training to deployment.

---

## 🛠️ Key Contributions

Most anomaly detection projects implement one model with one backbone and report numbers. We went further:

1. **Built a unified framework** where you can swap backbones (ResNet-18, WideResNet-50-2, ViT-B/16) and get fair, apples-to-apples comparisons under identical conditions — same preprocessing, same coreset ratio, same evaluation metrics.

2. **Ran a robustness study** testing the model against 18 real-world degradation scenarios (noise, blur, lighting changes, JPEG compression, shadows). This answers the question "would this actually work in a factory?" with measured data, not assumptions.

3. **Built incremental memory bank updates** so the model can adapt to changing production conditions (new materials, lighting drift) without full retraining — just a 30-second coreset re-sampling.

4. **Performed false alarm analysis** identifying that 100% of false positives on carpet triggered on image border regions, providing a specific, actionable fix for production deployment.

These are not theoretical additions. Every finding came from running experiments on real MVTec-AD data and analyzing the results.

---

## Methods We Implemented

### PatchCore (Primary Method)

PatchCore (CVPR 2022, "Towards Total Recall in Industrial Anomaly Detection") is our primary anomaly detection method. Here is exactly how it works:

**Training Phase (No Gradients Needed):**

1. **Feature Extraction:** We take a pretrained ImageNet CNN (e.g., WideResNet-50-2), freeze all weights, and hook into intermediate layers (layer2 and layer3). For a 224×224 input image, layer2 produces a 28×28 spatial grid with 512 channels, and layer3 produces 14×14 with 1024 channels. We upsample layer3 to 28×28 and concatenate, giving 28×28×1536.

2. **Patch Embedding:** Each of the 784 spatial positions (28×28) becomes one "patch embedding" of dimension 1536. We apply 3×3 average pooling before concatenation so each patch captures its local neighborhood context, not just a single pixel's features. All embeddings are L2-normalized.

3. **Memory Bank:** For 252 training images, we get 252 × 784 = 197,568 patch embeddings. We store these as the "memory bank" — a lookup table of what normal texture/structure looks like at every local patch position.

4. **Coreset Subsampling:** 197,568 patches is too many for fast nearest-neighbor search. We run greedy k-center coreset subsampling to select 10% (19,757 patches) that maximally cover the full set. The algorithm picks the point farthest from all previously selected points, guaranteeing bounded coverage. For sets >50,000 patches, we first project to 128 dimensions via random projection (Johnson-Lindenstrauss lemma) to speed up distance computation.

**Inference Phase:**

1. Extract 784 patch embeddings from the test image (same process as training)
2. For each test patch, compute L2 distance to its nearest neighbor in the memory bank
3. **Image-level score** = maximum patch distance (the single most anomalous patch determines the verdict)
4. **Anomaly map** = reshape 784 distances to 28×28, bilinearly upsample to 224×224, apply Gaussian smoothing (σ=4.0)
5. Compare score against optimized threshold → NORMAL or DEFECT decision

**Why PatchCore works well:** It operates at the patch level, so it can detect and localize tiny defects that image-level methods would miss. It requires zero training iterations — just one forward pass through all training images to build the memory bank. And the coreset ensures the memory bank stays compact enough for real-time inference.

### FastFlow (Alternative Method)

We also implemented FastFlow ("Unsupervised Anomaly Detection via 2D Normalizing Flows"), which takes a different approach:

- Instead of storing normal features, it trains a normalizing flow (stack of 8 affine coupling layers) to map normal features to a standard Gaussian distribution N(0,I)
- At inference, anomalous features produce low-likelihood outputs (high negative log-likelihood), which become the anomaly score
- Each coupling layer uses the RealNVP architecture: split channels in half, one half parameterizes scale and translation for the other half through a small CNN
- Scale parameters are clamped via tanh (clamp=2.0) for numerical stability
- Training uses Adam optimizer with cosine annealing LR and early stopping

FastFlow requires gradient-based training unlike PatchCore, and we implemented it fully but focused our benchmark experiments on PatchCore.

---

## Backbones We Tested

We tested three fundamentally different feature extractor architectures:

| Backbone | Type | Parameters | Feature Layers | Embedding Dim | Spatial Grid |
|----------|------|-----------|---------------|:-------------:|:------------:|
| ResNet-18 | CNN | 11M | layer2 (128) + layer3 (256) | 384 | 28×28 |
| WideResNet-50-2 | CNN | 69M | layer2 (512) + layer3 (1024) | 1536 | 28×28 |
| ViT-B/16 | Transformer | 86M | block_6 (768) + block_9 (768) | 1536 | 14×14 |

**How we handle different architectures:** For CNNs, we register PyTorch forward hooks on the target layers to capture intermediate activations without modifying the model architecture. For ViT-B/16, we hook into transformer blocks 6 and 9, extract the patch token sequence (removing the CLS token), and reshape from (batch, 196, 768) to (batch, 768, 14, 14) spatial format. This allows the same PatchEmbedding module to work with both CNNs and transformers.

---

## Benchmark Results — MVTec-AD Carpet Category (127 Test Images)

All three backbones were trained and evaluated under identical conditions: coreset ratio 10%, k=9 nearest neighbors, Gaussian sigma 4.0, 224×224 input resolution, CPU inference.

| Backbone | Image AUROC | Pixel AUROC | PRO Score | F1 Score | AP | Inference Speed | Memory Bank Size |
|----------|:-----------:|:-----------:|:---------:|:--------:|:--:|:--------------:|:----------------:|
| ResNet-18 | 96.90% | 97.35% | 89.01% | 94.95% | 99.16% | **12.0 img/s** | 19,757 × 384 |
| **WideResNet-50-2** | **98.38%** | **97.85%** | **90.92%** | **96.97%** | **99.56%** | 5.2 img/s | 19,757 × 1,536 |
| ViT-B/16 | 96.54% | 96.33% | 83.23% | 95.52% | 99.10% | 7.7 img/s | 4,940 × 1,536 |

**What these metrics mean:**
- **Image AUROC:** Can the model tell defective images from normal images? 98.38% means near-perfect classification.
- **Pixel AUROC:** Can the model locate which pixels are defective? 97.85% means excellent localization.
- **PRO Score:** Per-Region Overlap — the hardest metric. It evaluates each defect region separately (so a tiny scratch counts as much as a large hole). 90.92% means the model finds most defects regardless of size.
- **F1 Score:** Balance between precision (how many flagged products are actually defective) and recall (how many defective products are caught).
- **AP:** Average Precision — area under the Precision-Recall curve.

**Key findings from our comparison:**

1. **WideResNet-50-2 wins overall** with the highest scores on every metric, but at the cost of 4× more memory and 2.3× slower inference than ResNet-18.

2. **ResNet-18 is surprisingly competitive** — within 1.5% of WRN-50 on image AUROC and within 0.5% on pixel AUROC, despite being 6× smaller. This makes it viable for edge deployment where model size matters.

3. **ViT-B/16 underperforms on texture categories.** Its PRO score drops to 83.23% (vs 90.92% for WRN-50) because ViT tokenizes the image into a 14×14 grid — compared to 28×28 for CNNs. This 4× lower spatial resolution makes it harder to localize small texture defects like thin scratches. ViT would likely perform better on object categories where global shape context matters more.

---

## Robustness Study — Testing Under Real-World Degradations

We tested the trained WideResNet-50 model against 6 types of image degradation that simulate real factory conditions. The model was **not retrained** — we applied degradations to the test images and measured how AUROC changed.

| Degradation | Severity | Image AUROC | AUROC Drop |
|-------------|----------|:-----------:|:----------:|
| None (baseline) | — | 98.38% | — |
| Gaussian Noise | σ=15 | 98.67% | +0.3% |
| Gaussian Noise | σ=25 | 97.76% | -0.6% |
| Gaussian Noise | σ=40 | 96.86% | -1.5% |
| Motion Blur | 3×3 | 98.41% | +0.0% |
| Motion Blur | 5×5 | 98.59% | +0.2% |
| Motion Blur | 9×9 | 98.81% | +0.4% |
| Brightness ×0.5 | Low light | 98.34% | -0.0% |
| Brightness ×0.7 | Dim | 98.52% | +0.1% |
| Brightness ×1.3 | Bright | 98.77% | +0.4% |
| Brightness ×1.5 | Very bright | 98.99% | +0.6% |
| JPEG Quality 80 | Mild | 98.34% | -0.0% |
| JPEG Quality 50 | Medium | 98.45% | +0.1% |
| JPEG Quality 20 | Heavy | 97.91% | -0.5% |
| Gaussian Blur | σ=1 | 98.63% | +0.3% |
| Gaussian Blur | σ=2 | 99.21% | +0.8% |
| Gaussian Blur | σ=4 | 97.47% | -0.9% |
| **Random Shadow** | **30% coverage** | **47.80%** | **-50.6%** |

**Insights & Findings:**

- **The model is remarkably robust** to noise, blur, brightness changes, and JPEG compression. Even extreme Gaussian noise (σ=40) only drops AUROC by 1.5%. This means minor camera quality variations in production are safe.

- **Motion blur actually improves performance** (+0.4% at 9×9 kernel). The backbone features are somewhat invariant to blur, and blur suppresses high-frequency texture noise that the model sometimes misinterprets as anomalies.

- **Gaussian blur σ=2 gives the highest AUROC (99.21%)** — better than the baseline. This means the raw test images contain high-frequency noise that hurts performance. A simple preprocessing blur step could improve production accuracy.

- **Random shadow is catastrophic (47.8%).** Occluding 30% of the image completely breaks the model because the darkened region creates out-of-distribution features scored as anomalous. **In production, uniform lighting is non-negotiable.**

**Deployment recommendation:** The model is safe for production under noise, blur, brightness ±50%, and JPEG quality ≥20. The only critical requirement is consistent, shadow-free lighting.

---

## Incremental Memory Bank Update

**The problem:** PatchCore's memory bank is static after training. But in production, conditions change — new material batches, seasonal lighting drift, camera aging. The model gradually becomes less accurate.

**Our solution:** We built `update_memory_bank.py` that merges new normal samples into the existing memory bank without full retraining:

1. Take 20 new "good" images under changed conditions (e.g., darker lighting)
2. Extract patch embeddings using the same frozen backbone
3. Concatenate with existing memory bank (19,757 + 15,680 = 35,437 patches)
4. Re-run greedy k-center coreset → 3,544 representative patches
5. Replace memory bank — done in ~30 seconds

| Scenario | Image AUROC |
|----------|:-----------:|
| Original model → standard test | 98.38% |
| Original model → brightness-shifted test | 98.12% |
| Updated model → standard test | 97.55% |
| Updated model → brightness-shifted test | 98.12% |

The update maintains performance on the shifted domain while only slightly reducing performance on the original domain (98.38% → 97.55%). This means the model can adapt to new conditions in 30 seconds instead of the 10-minute full retraining cycle.

---

## False Alarm Analysis

**The problem:** Knowing overall AUROC is not enough for production. You need to understand exactly which images the model gets wrong and why.

**What we found on carpet (WideResNet-50, threshold 0.3729):**

- False positives: 3 out of 28 normal images incorrectly flagged
- False negatives: 3 out of 99 defective images missed
- **100% of false positives (3/3) triggered on image border regions**

**Root cause:** Backbone features near crop boundaries have incomplete neighborhood context from the 3×3 average pooling step. This creates slightly different features near edges compared to interior patches, pushing border patches closer to the anomaly threshold.

**What this means for deployment:** In a production camera setup where the product is centered in frame with consistent framing, this border artifact would not occur. The false positives are a dataset preprocessing issue, not a fundamental model weakness.

---

## Failure Analysis — Where It Breaks and Why

| Failure Mode | What We Observed | Root Cause |
|-------------|-----------------|------------|
| Border false positives | 3/28 normal images flagged, all at borders | Incomplete 3×3 neighborhood at image edges |
| Low pixel AP (~47-51%) | AUROC is 97%+ but AP is mediocre | Severe pixel-level class imbalance (<5% defective pixels). AUROC handles imbalance; AP does not |
| ViT spatial resolution | PRO drops from 90.92% to 83.23% | 14×14 feature grid cannot localize small texture defects |
| Shadow sensitivity | AUROC drops to 47.8% | Darkened regions create OOD features scored as anomalous |
| Single-category training | Must train separate model per product | PatchCore learns one "normal" distribution. Universal detection would need a different architecture |
| No temporal modeling | Each frame scored independently | Progressive defects (wear, gradual contamination) are not captured |

---

## Evaluation Metrics We Implemented

**Image-Level:**
- **AUROC:** Area Under ROC Curve — threshold-independent classification quality. We compute this using scikit-learn's `roc_auc_score`.
- **Average Precision:** Area under Precision-Recall curve — more sensitive to class imbalance than AUROC.
- **F1 / Precision / Recall:** At the F1-optimized threshold.

**Pixel-Level:**
- **Pixel AUROC:** Every pixel's anomaly score treated as independent prediction against ground truth mask.
- **Pixel AP:** Per-pixel average precision.
- **PRO Score (Per-Region Overlap):** Finds connected components in ground truth, computes overlap per region, averages. Integrates PRO-vs-FPR curve from 0 to 0.3 FPR and normalizes. This prevents large defects from dominating the score.

**Threshold Optimization (4 strategies):**
1. **F1 optimization** — maximize F1 score across all thresholds
2. **Youden's J** — maximize TPR − FPR (optimal ROC operating point)
3. **Precision-Recall balance** — find where precision equals recall
4. **Cost-based** — minimize cost_FP × FP + cost_FN × FN (default: missing a defect costs 10× more than over-rejecting)

---

## Anomaly Map Visualization

The `AnomalyMapGenerator` module provides:
- **Normalization:** min-max, z-score, or percentile-based scaling to [0,1]
- **Gaussian smoothing:** Configurable sigma for spatial noise reduction
- **Heatmap overlay:** JET colormap blended onto original image with green contours around detected defect regions
- **4-panel comparison grid:** Original | Ground Truth Mask | Heatmap Overlay | Predicted Binary Mask

---

## Dataset

We used **MVTec-AD** (MVTec Anomaly Detection), the standard benchmark for unsupervised anomaly detection:
- 15 categories: 5 textures (carpet, grid, leather, tile, wood) + 10 objects (bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper)
- 5,354 total images
- Training set: only "good" (defect-free) images
- Test set: mix of good + multiple defect types with pixel-level ground truth masks

Our dataset loader (`data/mvtec_dataset.py`) handles: automatic train/val splitting (10% holdout), ImageNet normalization, data augmentation (random flip, rotation ±5°, color jitter for training), and proper mask loading with the MVTec naming convention.

---

## Project Structure

```
VisionGuard-AD/
├── models/
│   ├── backbones/feature_extractor.py   # Multi-backbone feature extractor (5 architectures)
│   ├── patchcore/patchcore.py           # PatchCore: memory bank + kNN scoring
│   ├── patchcore/coreset_sampler.py     # Greedy k-center coreset subsampling
│   └── fastflow/fastflow.py            # FastFlow: normalizing flow anomaly detection
├── data/mvtec_dataset.py                # MVTec-AD dataset loader (15 categories)
├── anomaly_map/anomaly_map_generator.py # Heatmap overlay, comparison grids
├── metrics/evaluator.py                 # AUROC, AP, PRO score, report generation
├── threshold/threshold_optimizer.py     # F1/Youden/cost-based threshold optimization
├── app/app.py                           # Streamlit dashboard (inference, tuner, benchmark, camera)
├── robustness_study.py                  # Degradation robustness testing (6 types, 18 levels)
├── update_memory_bank.py                # Incremental memory bank adaptation
├── false_alarm_analyzer.py              # Spatial false positive/negative analysis
├── train.py                             # Training CLI for PatchCore and FastFlow
├── evaluate.py                          # Full evaluation pipeline
├── inference.py                         # Single image, batch folder, and webcam inference
├── benchmark.py                         # Multi-category benchmark runner
├── tests/                               # 28 unit tests (dataset, model, metrics)
├── configs/                             # YAML configuration files
└── scripts/                             # Download and data generation utilities
```

---

## How to Run

```bash
# Install
git clone https://github.com/Izhaar-ahmed/VisionGuard-AD.git
cd VisionGuard-AD
pip install -r requirements.txt
pip install timm

# Download MVTec-AD dataset
mkdir -p data/mvtec
tar xf mvtec_anomaly_detection.tar.xz -C ./data/mvtec

# Train
python train.py --method patchcore --category carpet --backbone wide_resnet50

# Evaluate
python evaluate.py --method patchcore --category carpet \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --backbone wide_resnet50 --visualize

# Inference on single image
python inference.py --method patchcore \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --input ./data/mvtec/carpet/test/scratch/001.png --threshold 0.37

# Robustness study
python robustness_study.py --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --backbone wide_resnet50 --category carpet

# Incremental update
python update_memory_bank.py --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --backbone wide_resnet50 --num_new_images 20

# False alarm analysis
python false_alarm_analyzer.py --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --backbone wide_resnet50 --category carpet

# Dashboard
streamlit run app/app.py

# Tests
python -m pytest tests/ -v
```

---

## References

1. Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022) — PatchCore
2. Yu et al., "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows" (2021)
3. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" (CVPR 2019)
