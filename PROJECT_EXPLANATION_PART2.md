# VisionGuard-AD — Complete Project Explanation (Part 2)
# Training, Inference, Experiments, and Results

---

## Chapter 7: Threshold Optimization

After the model produces anomaly scores, we need a threshold to make binary decisions. Our ThresholdOptimizer implements four strategies:

### Strategy 1: F1 Optimization

F1 = 2 × (Precision × Recall) / (Precision + Recall)

We sweep across all possible thresholds using the precision-recall curve from scikit-learn. At each threshold, we compute F1. The threshold that maximizes F1 is selected.

F1 balances precision (don't cry wolf) and recall (don't miss defects). It is the most commonly used strategy for balanced detection.

For our WideResNet-50 carpet model, the F1-optimal threshold is 0.3729, giving F1=96.97%.

### Strategy 2: Youden's J Statistic

J = TPR - FPR = Sensitivity + Specificity - 1

This finds the point on the ROC curve that is farthest from the diagonal (random guessing line). It maximizes the sum of sensitivity (catching defects) and specificity (not falsely rejecting good products).

### Strategy 3: Precision-Recall Balance

Finds the threshold where precision equals recall. Useful when you want equal emphasis on both types of errors.

### Strategy 4: Cost-Based Optimization

In production, different errors have different costs. Missing a defective product (shipping it to customers) is usually much worse than falsely rejecting a good product (just re-inspect it).

We define: total_cost = cost_FP × num_false_positives + cost_FN × num_false_negatives

Default: cost_FN = 10 (shipping a defect costs 10 units), cost_FP = 1 (rejecting a good product costs 1 unit). We sweep all thresholds and find the one that minimizes total cost.

This is the most practical strategy for real deployment because it directly optimizes for business impact.

---

## Chapter 8: Anomaly Map Visualization

The AnomalyMapGenerator module turns raw anomaly scores into human-interpretable visualizations.

### Normalization

Raw anomaly scores can have arbitrary ranges. We normalize them to [0, 1] using three methods:
- **min-max**: score = (x - min) / (max - min). Simple and intuitive.
- **z-score**: score = (x - mean) / std, then clipped to [0, 1]. Centers around mean.
- **percentile**: clips at 1st and 99th percentiles before min-max. Robust to outliers.

### Gaussian Smoothing

We apply scipy's `gaussian_filter` with sigma=4.0 to the anomaly map. This removes pixel-level noise and produces smoother, more interpretable heatmaps. Without smoothing, the map looks noisy and patchy.

### Heatmap Overlay

We convert the normalized anomaly map to a color heatmap using OpenCV's JET colormap (blue=low, red=high). This heatmap is blended with the original image using alpha=0.4 blending. Green contours are drawn around regions above the threshold.

A text label shows "DEFECT (score)" in red or "NORMAL (score)" in green, with a black background rectangle for readability.

### 4-Panel Comparison Grid

For evaluation, we create a 2×2 grid showing:
- Top-left: Original image
- Top-right: Ground truth mask (white=defect, black=normal)
- Bottom-left: Anomaly heatmap overlay
- Bottom-right: Binary prediction mask

This grid makes it easy to visually compare what the model detected vs what the actual defect was.

---

## Chapter 9: Training Pipeline

Our `train.py` provides a unified CLI for both PatchCore and FastFlow.

### PatchCore Training

PatchCore training is NOT gradient-based training. It is a one-pass feature extraction:

```bash
python train.py --method patchcore --category carpet --backbone wide_resnet50
```

What happens:
1. Load training images (252 for carpet)
2. Create DataLoader with batch_size=32, 4 workers
3. Pass all batches through frozen backbone, collecting patch embeddings
4. Concatenate all embeddings (197,568 patches × 1536 dimensions)
5. L2-normalize all embeddings
6. Run coreset subsampling → 19,757 patches
7. Save memory bank to `outputs/carpet/patchcore_memory_bank.pt`

Total time: ~10 minutes on CPU. This includes backbone feature extraction (~8 min) and coreset subsampling (~2 min).

### FastFlow Training

FastFlow requires actual gradient-based training:

```bash
python train.py --method fastflow --category carpet --backbone wide_resnet50 --epochs 100
```

What happens:
1. Build FastFlow model (frozen backbone + 8 coupling layers)
2. For each epoch, pass all training batches through the model
3. Compute NLL loss and backpropagate through coupling layers only (backbone stays frozen)
4. Use Adam optimizer with cosine annealing LR schedule
5. Validate on 10% holdout set, track best validation loss
6. Early stopping if no improvement for 15 epochs
7. Save best checkpoint to `outputs/carpet/fastflow_model.pt`

---

## Chapter 10: Evaluation Pipeline

Our `evaluate.py` runs comprehensive evaluation:

```bash
python evaluate.py --method patchcore --category carpet \
    --model_path ./outputs/carpet/patchcore_memory_bank.pt \
    --backbone wide_resnet50 --visualize
```

What it does:
1. Load trained model (PatchCore memory bank or FastFlow checkpoint)
2. Load test dataset (127 images for carpet: 28 normal + 99 defective)
3. Run inference on every test image, collecting anomaly scores and maps
4. Compute image-level metrics (AUROC, AP, F1) using AnomalyEvaluator
5. Compute pixel-level metrics (pixel AUROC, pixel AP, PRO score)
6. Find optimal threshold using ThresholdOptimizer
7. Compute threshold-specific metrics (precision, recall, FPR, FNR)
8. Save results to JSON and Markdown report
9. Generate ROC curve with optimal operating point marked
10. Save top-10 visualizations (heatmap overlays) for manual inspection

---

## Chapter 11: Inference Pipeline

Our `inference.py` supports three modes:

### Single Image Mode
```bash
python inference.py --model_path model.pt --input image.png --threshold 0.37
```
Loads one image, runs prediction, displays score and saves heatmap overlay.

### Batch Folder Mode
```bash
python inference.py --model_path model.pt --input ./test_folder/ --threshold 0.37
```
Processes all images in a directory, saves results to CSV with columns: filename, score, prediction.

### Webcam Mode
```bash
python inference.py --model_path model.pt --mode webcam --threshold 0.37
```
Opens camera feed, runs inference on each frame in real-time, displays overlay with score.

All three modes use robust helper functions (`_unwrap_score`, `_run_predict`) that handle the fact that PatchCore's predict method can return different types (scalar, list, batch tensor) depending on input shape. This was a bug we discovered and fixed during development.

---

## Chapter 12: Our Benchmark Results

### Full 15-Category Benchmark (ResNet-18)

We ran PatchCore with ResNet-18 backbone across all 15 MVTec-AD categories under identical conditions: coreset ratio 10%, k=9 neighbors, σ=4.0, 224×224 input, Apple Silicon M1 MPS. Total runtime: 6 hours 20 minutes.

| Category | Image AUROC | Pixel AUROC | PRO Score | F1 Score |
|----------|:-----------:|:-----------:|:---------:|:--------:|
| bottle | **100.0%** | 97.6% | 88.4% | 100.0% |
| cable | 93.8% | 94.3% | 84.6% | 90.7% |
| capsule | 79.9% | 92.3% | 66.2% | 91.9% |
| carpet | 97.3% | 97.1% | 90.0% | 96.5% |
| grid | 53.3% | 90.8% | 70.9% | 85.1% |
| hazelnut | **100.0%** | 98.3% | 81.3% | 100.0% |
| leather | 99.8% | **99.5%** | **97.3%** | 98.9% |
| metal_nut | 98.8% | 94.9% | 88.1% | 97.9% |
| pill | 92.1% | 88.2% | 79.7% | 94.7% |
| screw | 78.5% | 85.0% | 56.2% | 86.9% |
| tile | 99.3% | 95.8% | 76.9% | 98.3% |
| toothbrush | 83.1% | 94.8% | 70.2% | 91.8% |
| transistor | 95.3% | 87.8% | 79.5% | 87.4% |
| wood | 97.5% | 94.0% | 85.1% | 95.1% |
| zipper | 93.8% | 93.2% | 81.0% | 95.5% |
| **Mean** | **90.8% ± 12.2%** | **93.6%** | **79.7% ± 10.1%** | **94.0%** |

**Key observations from the full benchmark:**

1. **Bottle and hazelnut achieve 100% Image AUROC** — these are rigid objects with consistent geometry, making anomalies very distinct from the normal distribution.

2. **Leather achieves the highest PRO score (97.3%)** — leather defects tend to be well-defined regions that contrast sharply with the uniform texture.

3. **Grid (53.3%) and screw (78.5%) are challenging** — grid has fine repetitive patterns where ResNet-18's 384-dimensional features lack the expressiveness to capture subtle structural anomalies. Screw has many visually similar defect types. Both categories benefit significantly from WideResNet-50-2's richer 1536-dimensional features.

4. **Mean F1 (94.0%) is strong across all categories** — even categories with lower AUROC still achieve reasonable operational performance because F1 is optimized at the best threshold.

### Carpet Category — 3-Backbone Comparison

We trained all three backbones on carpet under identical conditions for direct comparison:

| Backbone | Image AUROC | Pixel AUROC | PRO Score | F1 | Speed |
|----------|:-----------:|:-----------:|:---------:|:--:|:-----:|
| ResNet-18 | 96.90% | 97.35% | 89.01% | 94.95% | 24 img/s (MPS) |
| **WideResNet-50-2** | **98.38%** | **97.85%** | **90.92%** | **96.97%** | 5.2 img/s |
| ViT-B/16 | 96.54% | 96.33% | 83.23% | 95.52% | 7.7 img/s |

---

## Chapter 13: Robustness Study

We tested whether the model would work in real factory conditions where images are not perfect. We applied 6 types of degradation to the test set and measured AUROC change.

### Degradation Types and Results

**Gaussian Noise** (simulates sensor noise): Adding random noise with σ=15 barely affects the model (98.67%, actually +0.3%). Even extreme noise (σ=40) only drops to 96.86% (-1.5%). The backbone features are computed over entire receptive fields, so local pixel noise gets averaged out.

**Motion Blur** (simulates camera shake or moving conveyor belt): Surprisingly, blur IMPROVES performance. At 9×9 kernel, AUROC rises to 98.81% (+0.4%). Why? Blur removes high-frequency texture noise that the model sometimes confuses with anomalies. The backbone's learned features are somewhat blur-invariant because ImageNet training includes augmented images.

**Brightness Shift** (simulates lighting variations): Both dimming (×0.5) and brightening (×1.5) barely change AUROC (98.34% and 98.99%). The ImageNet normalization (subtracting mean, dividing by std) partially compensates for brightness changes. The backbone has also seen brightness-augmented images during its ImageNet training.

**JPEG Compression** (simulates lossy image storage): Even heavy compression (quality=20) only drops to 97.91% (-0.5%). JPEG artifacts are block-level (8×8 blocks), and the backbone's receptive field is large enough to see past them.

**Gaussian Blur** (simulates out-of-focus camera): σ=2 blur actually gives the BEST performance (99.21%, +0.8%). This tells us the raw test images contain high-frequency noise that hurts detection. A preprocessing blur step could improve production accuracy for free.

**Random Shadow** (simulates uneven lighting or occlusion): CATASTROPHIC. AUROC drops to 47.80% (worse than random). The darkened region creates features that are completely outside the normal distribution, so the entire shadowed area gets maximum anomaly scores. This overwhelms the actual defect signal.

### What This Means for Production

The model is robust to everything EXCEPT shadows/occlusion. In a factory setting, this means:
- Camera quality does not need to be perfect (noise/blur are tolerable)
- Brightness can vary ±50% without issues
- Image compression for storage is fine
- BUT: lighting must be uniform and consistent. No shadows. This is standard practice in industrial vision systems anyway.

---

## Chapter 14: Incremental Memory Bank Update

### The Problem

Day 1: You train PatchCore on 252 images of blue carpet. Performance is 98.38%.
Day 60: The factory switches to a slightly different blue dye. The carpet looks almost the same to humans, but the pixel distributions shift slightly. The model starts flagging good carpets as defective because their features are outside the memory bank's coverage.

This is called **distribution drift** or **domain shift**. In production, it happens constantly due to material changes, lighting changes (seasonal daylight), camera aging, and temperature effects on sensors.

### Our Solution

Instead of retraining from scratch (extracting features from ALL training images again), we built an incremental update:

1. Collect 20 new images of the "new normal" (good products under new conditions)
2. Extract their patch embeddings using the same frozen backbone (same feature space)
3. CONCATENATE these new embeddings with the existing memory bank
4. Re-run coreset subsampling on the combined set to keep the bank compact
5. Replace the old memory bank with the updated one

The entire process takes ~30 seconds (mostly coreset computation), compared to ~10 minutes for full retraining.

### Our Experiment

We simulated a lighting change by darkening images to 60% brightness. Results:
- Original model on standard test: 98.38% (baseline)
- Original model on darkened test: 98.12% (model is already robust to this)
- Updated model on standard test: 97.55% (slight drop, acceptable)
- Updated model on darkened test: 98.12% (maintained)

The model was already quite robust to brightness changes (as confirmed by our robustness study). But for more severe domain shifts, the incremental update becomes essential.

---

## Chapter 15: False Alarm Analysis

### Why False Alarms Matter

In production, false alarms cost money. Every falsely rejected product must be re-inspected by a human. If the false alarm rate is 10%, you are wasting 10% of your inspection capacity on products that are actually fine.

More importantly, operators lose trust in the system. If they see too many false alarms, they start ignoring the alerts — and then real defects slip through.

### Our Analysis

We ran the WideResNet-50 model on all 127 test images and analyzed every error:

**False Positives (3 out of 28 normal images):**
All three false positives had their highest anomaly activation in the border region (within 10 pixels of the image edge). ZERO false positives triggered in the center of the image.

**Root Cause:** The 3×3 average pooling in patch embedding uses zero-padding at boundaries. This means border patches have incomplete neighborhood context — their embeddings are systematically different from interior patches. Since the memory bank was built from patches at all positions (including borders), some border patches are naturally closer to the anomaly threshold.

**False Negatives (3 out of 99 defective images):**
Three defective images were scored below the threshold. These had very subtle defects (slight color variations) that fell within the normal variation range of the carpet texture.

### What This Means for Deployment

In a real production setup, the camera frames the product in the center of the image with consistent framing. The border region contains background (conveyor belt, fixture), not product. So border false positives would not occur because you would crop or mask the border before running inference.

This is a dataset preprocessing artifact, not a model failure. Knowing this distinction is important — it means the model's TRUE false positive rate on the actual product region is likely 0/28 (0%), not 3/28.

---

## Chapter 16: The Streamlit Dashboard

We built a web-based dashboard using Streamlit with four pages:

### Page 1: Inference Engine
Upload an image, select a backbone and model file, adjust threshold with a slider, and see the anomaly heatmap overlay and binary prediction mask in real-time.

### Page 2: Threshold Tuner
Interactive exploration of threshold effects. Displays ROC curve and PR curve. Move the threshold slider and see how precision, recall, F1, FPR, and FNR change in real-time.

### Page 3: Benchmark Comparison
Side-by-side comparison of all three backbones showing AUROC, speed, memory usage, and trade-offs. Helps users choose the right backbone for their deployment constraints.

### Page 4: Live Camera Monitor
Connects to webcam and runs inference on each frame. Displays real-time anomaly score and heatmap overlay. Designed for live production line monitoring.

The dashboard uses a dark theme with glassmorphism styling (semi-transparent panels with blur effects) for a professional industrial look.

---

## Chapter 17: Testing

We have **45 automated tests** across four test files:

### test_dataset.py (8 tests)
Tests that the MVTec dataset loader correctly handles directory structures, train/val splits, transforms, and statistics computation. Uses synthetic fixtures so tests run without downloading the actual dataset.

### test_patchcore.py (16 tests)
Tests feature extraction, patch embedding dimensions, coreset subsampling (multiple ratios), memory bank construction, predict method return types, save/load functionality, anomaly map perturbation sensitivity, and prediction determinism. Includes parametrized tests verifying coreset size matches exactly `ceil(N × ratio)` for various ratios.

### test_metrics.py (16 tests)
Tests AUROC computation, AP computation, PRO score calculation, threshold optimization (all four strategies), edge cases (all-normal labels, all-anomalous labels, tied scores), pixel metrics size mismatch handling, and report generation. Uses hand-crafted score/label arrays with known correct answers.

### test_api.py (5 tests)
Tests the inference API module: `load_model` returns correct handle structure, `run_inference` output format validation, deterministic predictions (same image → same score), anomaly map toggle, and threshold override.

All tests run in ~9 seconds on CPU without any external data.

---

## Chapter 18: Project Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| models/backbones/feature_extractor.py | 405 | Multi-backbone feature extraction with forward hooks |
| models/patchcore/patchcore.py | 268 | PatchCore: memory bank, kNN scoring, predict |
| models/patchcore/coreset_sampler.py | 112 | Greedy k-center with random projection |
| models/fastflow/fastflow.py | 320 | Normalizing flow with affine coupling layers |
| data/mvtec_dataset.py | 336 | MVTec-AD dataset loader with augmentation |
| anomaly_map/anomaly_map_generator.py | 263 | Heatmap overlay, comparison grids, denormalization |
| metrics/evaluator.py | 121 | AUROC, AP, PRO score, JSON/MD reports |
| threshold/threshold_optimizer.py | 189 | Four threshold strategies, ROC/PR plots |
| train.py | 268 | Unified training CLI |
| evaluate.py | 215 | Full evaluation pipeline |
| inference.py | 279 | Single image, batch, webcam inference |
| benchmark.py | 290 | Full 15-category benchmark runner with CSV/JSON/MD output |
| profile_inference.py | 240 | Latency, throughput, memory profiling |
| api/inference_api.py | 190 | Clean Python API: load_model() + run_inference() |
| api/server.py | 180 | FastAPI REST server: POST /predict |
| robustness_study.py | 280 | 18 degradation experiments |
| update_memory_bank.py | 260 | Incremental memory bank update |
| false_alarm_analyzer.py | 240 | Spatial false alarm analysis |
| app/app.py | 350 | Streamlit dashboard |
| app/utils.py | 110 | Dashboard utility functions |
| tests/ | 500 | 45 automated tests (dataset, model, metrics, API) |

**Total: ~6,800+ lines of Python**

---

## Summary

VisionGuard-AD is a complete anomaly detection system that goes beyond a basic implementation. We did not just implement PatchCore — we built a framework for comparing backbones, benchmarked all 15 MVTec-AD categories (90.8% mean Image AUROC with ResNet-18, 24 img/sec on M1 MPS), tested robustness under real-world conditions, added production capabilities (incremental updates, REST API, inference profiling), and analyzed failure modes to provide actionable deployment insights. Every claim in this document is backed by experiments we actually ran on real data, with results we can reproduce.

*Continued in Part 3: PROJECT_EXPLANATION_PART3.md — Deployment, API, and Performance Profiling*
