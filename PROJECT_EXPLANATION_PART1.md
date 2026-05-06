# VisionGuard-AD — Complete Project Explanation (Part 1)
# Concepts, Methods, and Architecture

---

## Brief Overview

VisionGuard-AD is an industrial anomaly detection system. It looks at images of manufactured products (like carpet, bottles, screws) and decides whether each product is normal or defective. The key challenge: we only have images of GOOD products during training. We never show the model what defects look like. Despite this, the model learns to detect defects it has never seen before.

We implemented two detection methods (PatchCore and FastFlow), tested three different neural network backbones (ResNet-18, WideResNet-50-2, ViT-B/16), and benchmarked across all 15 MVTec-AD categories achieving **90.8% mean Image AUROC** with ResNet-18 and **41.7 ms/image inference latency** on Apple Silicon M1 MPS. Beyond basic implementation, we performed robustness testing (18 degradation scenarios), built an incremental update system, analyzed where the model fails and why, created a REST API (FastAPI), and profiled latency/memory for deployment sizing.

---

## Chapter 1: What is Anomaly Detection?

### The Basic Idea

Imagine you work in a carpet factory. Your job is to check every carpet for defects — scratches, stains, holes, color inconsistencies. You look at 1000 carpets per day. After a few hours, you get tired and start missing things.

A computer vision system can do this job 24/7 without fatigue. But there is a fundamental problem: how do you teach a computer what a "defect" looks like?

### Why Not Regular Classification?

In normal image classification (like recognizing cats vs dogs), you show the model thousands of examples of each class. The model learns features of cats and features of dogs, then classifies new images.

For defect detection, this approach fails for two reasons:

1. **You have very few defect images.** Factories aim for near-zero defect rates. If only 1 in 10,000 products is defective, you might wait months to collect enough defect examples.

2. **Defects are unpredictable.** A new defect type can appear at any time. If you trained a classifier on "scratch" and "stain" defects, it would completely miss a new "hole" defect because it never learned what holes look like.

### The Unsupervised Solution

Instead of learning what defects look like, we learn what NORMAL looks like. Then anything that doesn't match "normal" is flagged as potentially defective.

This is called **unsupervised anomaly detection** because we don't use labeled defect data (no "supervision" about what defects are). We only use normal/good images during training.

The analogy: instead of memorizing every possible disease, a doctor memorizes what a healthy person looks like. Then any deviation from healthy is investigated further.

---

## Chapter 2: How PatchCore Works

PatchCore is our primary detection method, published at CVPR 2022. It is currently one of the best-performing methods for industrial anomaly detection. Here is exactly how it works, step by step.

### Step 1: Feature Extraction

We start with a neural network that was pretrained on ImageNet (a dataset of 1.2 million natural images with 1000 categories like "dog", "car", "flower"). This network has already learned to recognize textures, edges, shapes, and structures from these millions of images. We use this pre-learned knowledge.

We take a pretrained CNN — for example, WideResNet-50-2 — and we FREEZE it. We never update its weights. We don't care about its classification output (the final layer that would say "this is a dog"). Instead, we care about its INTERMEDIATE layers, because these layers contain rich feature representations.

**What are intermediate layers?**

A CNN processes an image through many layers. Early layers detect simple things (edges, colors). Middle layers detect textures and patterns. Late layers detect high-level concepts (faces, objects). For anomaly detection, middle layers are ideal because they capture texture and structural information without being too abstract.

We hook into two specific layers:
- **layer2**: Captures texture-level features. For a 224×224 input image, this layer outputs a 28×28 spatial grid with 512 channels. Think of it as 512 different "texture detectors", each producing a 28×28 map showing where that texture pattern appears.
- **layer3**: Captures structure-level features. Outputs a 14×14 grid with 1024 channels. More abstract than layer2, capturing larger-scale patterns.

We upsample layer3's output from 14×14 to 28×28 (using bilinear interpolation) so both layers have the same spatial size, then concatenate them channel-wise. The result is a 28×28×1536 feature tensor.

**What does this 28×28×1536 tensor mean?**

Think of it as a grid of 28×28 = 784 locations on the image. Each location has a 1536-dimensional vector that describes "what the texture/structure looks like at this specific spot." If we're looking at carpet, one location's vector might encode "smooth blue weave pattern", while another location encodes "rough edge transition."

**How do forward hooks work?**

In PyTorch, you can register a "hook" on any layer. When data passes through that layer during a forward pass, the hook function is called and we can capture the layer's output without modifying the network's architecture. Our code does this:

```python
def hook_fn(module, input, output):
    self.features[layer_name] = output

layer.register_forward_hook(hook_fn)
```

This way, we just run `model(image)` normally, and the hooks automatically capture the intermediate features we need.

### Step 2: Patch Embedding

Each of the 784 spatial positions in our 28×28 grid represents one "patch" of the original image. But a single spatial position only captures a very small area of the image (about 8×8 pixels of the original 224×224 image, due to the network's stride).

To make each patch more robust, we apply **3×3 average pooling** to the feature maps before concatenation. This means each patch's embedding incorporates information from its immediate neighbors (a 3×3 neighborhood). If one pixel's features are slightly noisy, the neighborhood averaging smooths it out.

After pooling and concatenation, each patch has a 1536-dimensional embedding vector (for WideResNet-50). We then **L2-normalize** each embedding — this means we scale each vector to have unit length. This ensures that all patch embeddings live on the surface of a hypersphere, making distance computations more meaningful (we compare directions, not magnitudes).

### Step 3: Building the Memory Bank

During training, we pass ALL normal/good training images through the feature extractor. For the carpet category, there are 252 training images. Each image produces 784 patch embeddings. So we get:

252 images × 784 patches = 197,568 patch embeddings

Each embedding is 1536-dimensional. This collection of 197,568 embeddings is called the **memory bank**. It represents "everything the model has ever seen that is normal." Every possible normal texture pattern, every normal structural arrangement, at every location — it is all stored here.

### Step 4: Coreset Subsampling

197,568 embeddings is a lot. During inference, we need to search through all of them for every test patch. This would be slow. We need to reduce the memory bank while keeping good coverage.

**Greedy k-center coreset subsampling** solves this. The algorithm works like this:

1. Pick one random embedding as the first "coreset point"
2. Compute the distance from EVERY embedding to the nearest coreset point
3. The embedding that is FARTHEST from any coreset point gets added to the coreset
4. Repeat steps 2-3 until we have enough coreset points

With a ratio of 10%, we select 19,757 embeddings out of 197,568.

**Why does this work?** By always picking the farthest point, we guarantee that the coreset "covers" the full embedding space. No normal pattern is too far from a coreset representative. This is much better than random sampling, which might miss rare-but-normal patterns.

**Optimization for large sets:** When we have more than 50,000 embeddings, computing pairwise distances in 1536 dimensions is expensive. We use the **Johnson-Lindenstrauss lemma**: project all embeddings from 1536 dimensions to 128 dimensions using a random matrix. This approximately preserves distances (within a small error bound) but makes computation much faster.

### Step 5: Anomaly Scoring (Inference)

When a new test image comes in:

1. Extract 784 patch embeddings (same process as training)
2. For each test patch, compute the L2 distance to its **nearest neighbor** in the memory bank
3. Each patch now has a distance score. High distance = this patch looks different from anything in the memory bank = potentially anomalous

The **image-level anomaly score** is simply the MAXIMUM patch distance. The logic: if even one small region of the product looks abnormal, the product should be flagged. A carpet with one tiny scratch should fail inspection even though 99% of it looks fine.

The **anomaly map** (pixel-level heatmap) is created by:
1. Taking the 784 distance scores
2. Reshaping them back to a 28×28 grid
3. Bilinearly upsampling to 224×224 (original image size)
4. Applying Gaussian smoothing (σ=4.0) for a cleaner, more interpretable heatmap

The result is a heatmap where bright spots indicate "this area looks anomalous" and dark spots indicate "this area looks normal."

### Step 6: Threshold Decision

The anomaly score is a continuous number. To make a binary NORMAL/DEFECT decision, we compare it against a threshold. If score ≥ threshold, the image is classified as defective.

Choosing the right threshold is critical. Too low → too many false alarms (good products rejected). Too high → defects slip through. Our ThresholdOptimizer supports four strategies, each explained in detail in the threshold section below.

---

## Chapter 3: How FastFlow Works

FastFlow is our alternative detection method, based on normalizing flows. While PatchCore stores examples of normal features, FastFlow learns the DISTRIBUTION of normal features.

### The Core Idea

If we could model the exact probability distribution of normal features, then at inference time we just ask: "how likely is this feature under the normal distribution?" Low probability = anomaly.

FastFlow uses **normalizing flows** to model this distribution. A normalizing flow is a sequence of invertible transformations that maps a complex distribution (normal features) to a simple one (standard Gaussian N(0,I)). If we train this mapping well, then:

- Normal features → mapped to high-probability regions of the Gaussian → low anomaly score
- Anomalous features → mapped to low-probability regions → high anomaly score

### Affine Coupling Layers

FastFlow uses **affine coupling layers** (from the RealNVP paper). Each layer works like this:

1. Split the input channels into two halves: x1 and x2
2. Pass x1 through a small CNN to compute scale (s) and translation (t) parameters
3. Transform x2: z2 = x2 × exp(s) + t
4. The output is [x1, z2] concatenated

The key properties:
- **Invertible**: Given [x1, z2], we can recover [x1, x2] by computing x2 = (z2 - t) × exp(-s)
- **Tractable Jacobian**: The log-determinant of the Jacobian is just sum(s), which we need for the loss function
- **Expressive**: Stacking 8 such layers with alternating split patterns creates a powerful transformation

**Scale clamping**: The scale parameter s is passed through `clamp × tanh(s / clamp)` with clamp=2.0. This prevents numerical instability from very large or very small scale values.

### Training

We train FastFlow with the **negative log-likelihood (NLL) loss**:

loss = mean over pixels of [||z||² / 2 - log_det_jacobian]

This pushes normal features toward the Gaussian center (||z||² → 0) while accounting for the volume change of the transformation (log_det_jacobian).

Training uses Adam optimizer with cosine annealing LR schedule (starting at 0.0001, minimum 1e-6), gradient clipping at 1.0, and early stopping with patience of 15 epochs.

### Inference

At inference, we pass test features through the trained flow and compute:

anomaly_score(pixel) = ||z(pixel)||² / 2

High ||z||² means the feature mapped far from the Gaussian center = low probability = anomaly.

---

## Chapter 4: Feature Extractor — Supporting Multiple Backbones

A backbone is the pretrained neural network we use to extract features. Different backbones have different strengths. Our FeatureExtractor class supports five backbones through a unified interface.

### ResNet-18 (11M parameters)

The smallest backbone we tested. ResNet (Residual Network) uses skip connections that add the input of a block directly to its output. This allows training very deep networks without gradient vanishing problems. ResNet-18 has 18 layers organized in 4 groups. We extract from layer2 (128 channels) and layer3 (256 channels), giving 384-dimensional patch embeddings on a 28×28 grid.

**Tradeoff**: Fast (24 img/s on MPS, 12 img/s on CPU) and compact, but lower feature richness. Achieves 90.8% mean Image AUROC across all 15 MVTec categories.

### WideResNet-50-2 (69M parameters)

A wider version of ResNet-50 where each layer has 2× more channels. More channels = richer feature representations = better anomaly detection. We extract from layer2 (512 channels) and layer3 (1024 channels), giving 1536-dimensional embeddings on a 28×28 grid.

**Tradeoff**: Best accuracy (98.38% AUROC on carpet) but 6× more parameters and 2.3× slower than ResNet-18. Expected to bring the full 15-category mean from 90.8% to ~96–99% based on literature.

### ViT-B/16 (86M parameters, Vision Transformer)

Fundamentally different architecture. Instead of convolutional layers, ViT divides the image into 16×16 pixel patches (14×14 = 196 patches for a 224×224 image), adds positional embeddings, and processes them through transformer blocks with self-attention.

We extract features from transformer blocks 6 and 9 (mid-depth, similar to CNN middle layers). The raw output is (batch, 197, 768) — 196 patch tokens plus 1 CLS token. We remove the CLS token and reshape to (batch, 768, 14, 14) spatial format.

**The critical difference**: ViT produces a 14×14 grid vs CNN's 28×28 grid. This means 4× less spatial resolution for anomaly localization. For texture anomalies (thin scratches on carpet), this resolution loss is significant — which is exactly what our experiments confirmed.

**How we handle ViT in our code**: We use the `timm` library to load pretrained ViT weights. We register hooks on transformer blocks and reshape the sequence output back to spatial format, making it compatible with the same PatchEmbedding module used for CNNs. This unified design means adding a new backbone requires only defining which layers to hook and how to reshape features.

---

## Chapter 5: The MVTec-AD Dataset

MVTec-AD is the standard benchmark for industrial anomaly detection, created by MVTec Software GmbH. It contains 5,354 high-resolution images across 15 categories.

### Categories

**5 Texture categories**: carpet, grid, leather, tile, wood
These are flat surface textures where defects are local pattern disruptions.

**10 Object categories**: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper
These are 3D objects photographed from a fixed viewpoint.

### Structure

For each category:
- **train/good/**: Only normal images (no defects). Carpet has 252 training images.
- **test/good/**: Normal test images (to measure false positive rate)
- **test/<defect_type>/**: Images with specific defect types (e.g., test/scratch/, test/hole/)
- **ground_truth/<defect_type>/**: Pixel-level binary masks showing exactly where each defect is

### Our Dataset Loader

Our `MVTecDataset` class handles all the complexity:
- Automatically discovers categories and defect types from directory structure
- Splits training data into train (90%) and validation (10%) sets using seeded random split for reproducibility
- Applies different transforms for training (augmentation) vs testing (just resize + normalize)
- Training augmentation: random horizontal flip, ±5° rotation, brightness/contrast jitter
- All images are normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Masks are loaded as binary tensors (thresholded at 0.5) with nearest-neighbor interpolation to preserve sharp edges
- Returns 5 items per sample: (image_tensor, mask_tensor, label, defect_type_string, file_path)

---

## Chapter 6: Evaluation Metrics

### Image-Level AUROC (Area Under ROC Curve)

The ROC curve plots True Positive Rate (TPR, how many defective products we catch) against False Positive Rate (FPR, how many good products we wrongly reject) at every possible threshold.

AUROC = area under this curve. It ranges from 0 to 1 (we report as percentage):
- 100% = perfect separation between normal and defective scores
- 50% = random guessing (no better than flipping a coin)
- <50% = worse than random (predictions are inverted)

AUROC is threshold-independent — it tells you how well the model RANKS images (defective above normal) regardless of where you set the threshold.

### Average Precision (AP)

AP is the area under the Precision-Recall curve. Precision = "of all images I flagged, how many are actually defective?" Recall = "of all defective images, how many did I catch?"

AP is more sensitive to class imbalance than AUROC. When most images are normal (which is true in manufacturing), a model that flags everything as defective would get decent AUROC but terrible AP.

### Pixel-Level AUROC and AP

Same metrics but computed per-pixel. Every pixel in every test image is treated as an independent prediction. The anomaly map value at each pixel is compared against the ground truth mask.

This measures localization quality — can the model identify WHERE the defect is, not just WHETHER there is one?

### PRO Score (Per-Region Overlap)

The most demanding metric. Standard pixel AUROC can be dominated by large defects (a big stain contributes thousands of pixels, swamping a tiny scratch). PRO prevents this by:

1. Finding connected components in the ground truth mask (each separate defect region)
2. Computing overlap between prediction and ground truth FOR EACH REGION SEPARATELY
3. Averaging overlap across all regions (so a tiny scratch counts as much as a large stain)
4. Plotting this average overlap against FPR at multiple thresholds
5. Integrating the curve from FPR=0 to FPR=0.3 and normalizing

We implemented this from scratch using scipy's `label()` function to find connected components.

---

*Continued in Part 2: PROJECT_EXPLANATION_PART2.md*
