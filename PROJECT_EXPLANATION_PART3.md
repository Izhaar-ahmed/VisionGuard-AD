# VisionGuard-AD — Complete Project Explanation (Part 3)
# Deployment, API, Performance Profiling, and Full Benchmark Analysis

---

## Chapter 19: Full 15-Category Benchmark — Design and Analysis

### Why a Full Benchmark Matters

Most anomaly detection projects report numbers on 1–3 "easy" categories (bottle, carpet, leather) and claim success. This is misleading because categories vary dramatically in difficulty. A full benchmark forces you to confront where the method actually struggles.

Our benchmark script (`benchmark.py`) trains and evaluates PatchCore on all 15 MVTec-AD categories in a single run. Here is what we built and what we learned.

### Benchmark Infrastructure

The benchmark script was designed for long-running M1 sessions (~6 hours):

**Progress reporting:** Every category shows a clear header `[3/15] CAPSULE`, elapsed time, and ETA estimate. Per-image evaluation progress prints every 25 images. After each category completes, a result box shows all metrics immediately. This is critical because on a CPU/MPS machine, you need to know the run is progressing.

**Resume-friendly:** Per-category results are saved to individual `metrics.json` files immediately after each evaluation. If the process is interrupted at category 10, you have complete results for categories 1–9 and can restart from 10.

**Multiple output formats:**
- `benchmark_results.json` — machine-readable, includes config metadata
- `benchmark_results.csv` — for pandas/Excel analysis
- `benchmark_results.md` — paste directly into README

**Error isolation:** If one category crashes (e.g., missing data), the error is caught, logged, and the benchmark continues with remaining categories.

### What the Numbers Mean

Our ResNet-18 results across 15 categories:

**Mean Image AUROC: 90.8% ± 12.2%**

The high standard deviation (12.2%) tells us performance varies significantly by category. This is not a model flaw — it reflects inherent category difficulty differences:

- **Easy categories (>95%):** bottle (100%), hazelnut (100%), leather (99.8%), tile (99.3%), metal_nut (98.8%), carpet (97.3%), wood (97.5%). These either have rigid geometry (bottle) or strongly distinct textures (leather).

- **Medium categories (85–95%):** cable (93.8%), zipper (93.8%), transistor (95.3%), pill (92.1%). These have more varied normal appearances.

- **Hard categories (<85%):** toothbrush (83.1%), capsule (79.9%), screw (78.5%), grid (53.3%). These require fine-grained features that ResNet-18's 384-dimensional embeddings struggle to capture.

**Why Grid is 53.3%:** Grid has extremely fine, regular patterns. Defects are subtle structural irregularities (bent wires, broken junctions) that differ from normal by very few pixels. ResNet-18's features at layer2/layer3 are too coarse to distinguish these. WideResNet-50-2 with 1536-dimensional features would perform significantly better here — the PatchCore paper reports ~99% on grid with WRN-50.

**Mean Pixel AUROC: 93.6%** — even in categories where image-level detection struggles, the pixel-level anomaly maps are often quite good. This means the model correctly identifies WHERE anomalies are, even when the overall score doesn't cross the threshold.

**Mean PRO Score: 79.7% ± 10.1%** — PRO is the hardest metric because it requires precise per-region overlap. Categories with small or scattered defects (screw PRO: 56.2%) pull this down.

### Comparison to Published Baselines

The original PatchCore paper (Roth et al., CVPR 2022) reports **99.1% mean Image AUROC** using WideResNet-50-2 with optimized coreset ratios and nearest-neighbor counts. Intel's anomalib framework reports ~96–99% with WRN-50 and ~85–95% with ResNet-18.

Our 90.8% with ResNet-18 is consistent with these baselines. The ~9% gap from WRN-50 results comes from:
1. 4× smaller feature dimensionality (384 vs 1536)
2. Lower spatial resolution for object categories
3. Less expressive texture features for fine-grained anomalies

This trade-off is deliberate — ResNet-18 runs at 24 img/sec on MPS vs ~5 img/sec for WRN-50, making it viable for real-time inspection.

---

## Chapter 20: Inference Profiling — Measuring Real Performance

### Why Profiling Matters

Saying "the model is fast" without numbers is meaningless for deployment planning. A factory needs to know:
- Can this run on our edge hardware?
- How many cameras can one machine support?
- What's the worst-case latency (will it keep up with the conveyor)?

Our `profile_inference.py` answers these questions with measured data.

### What We Measure

**Latency (ms/image):** Wall-clock time from image tensor creation to score output. We measure:
- Mean — average performance
- Median — typical performance (unaffected by outliers)
- P95 — 95th percentile, the "almost worst case"
- P99 — 99th percentile, true worst case

**Throughput (images/sec):** 1000 / mean_latency. This directly tells you how many items per second one model instance can inspect.

**Memory:**
- Model load overhead (MB added to process)
- Peak Python heap (via `tracemalloc`)
- Process RSS (via `psutil`)
- Memory bank size in MB

### How We Measure Accurately

**Warmup phase:** The first 5 images are excluded from timing. On MPS (Apple's GPU), the first forward pass triggers JIT compilation of Metal shaders. On CUDA, the first pass triggers CUDA context initialization. Including these would inflate latency.

**Device synchronization:** On MPS and CUDA, GPU operations are asynchronous — `model.predict()` returns before the GPU actually finishes. We call `torch.mps.synchronize()` after each prediction to ensure we measure actual computation time, not just "time to queue the work."

### Our Real Numbers (Apple Silicon M1, 8GB)

PatchCore with ResNet-18, 224×224 resolution, carpet category, 50 images:

| Metric | Value |
|--------|-------|
| Mean latency | **41.7 ms/image** |
| Median latency | 40.8 ms/image |
| P95 latency | 52.6 ms/image |
| P99 latency | 58.4 ms/image |
| Std deviation | 5.9 ms |
| **Throughput** | **24.0 images/sec** |
| Memory bank | 19,757 × 384 (28.9 MB) |

**What this means for deployment:**
- At 24 img/sec, one M1 Mac Mini can inspect 1,440 items per minute
- P99 of 58.4 ms means even in the worst case, each image finishes in under 60 ms
- 28.9 MB memory bank means you could run ~30 category models simultaneously on an 8GB machine
- A typical factory conveyor runs at 5–15 items/second — one M1 can handle 2–5× the conveyor speed with headroom

### JSON Output for Tracking

The profiler saves a complete JSON with all measurements, including method, backbone, device, and resolution — so latency numbers are never taken out of context. This is critical for accurate comparison across configurations.

---

## Chapter 21: The Inference API — Clean Integration Point

### Design Philosophy

Most ML research projects have inference scattered across scripts. Our API module (`api/inference_api.py`) provides a single, clean interface that all consumers share — the FastAPI server, the Streamlit dashboard, and any custom integration code.

### load_model()

```python
handle = load_model(
    method="patchcore",
    backbone="resnet18",
    category="carpet",
    model_path="./outputs/benchmark/carpet/patchcore_memory_bank.pt",
    device="auto",
    threshold=0.5,
)
```

Returns a plain Python dict (not a class) containing:
- `model` — the loaded PatchCore or FastFlow instance
- `transform` — preprocessing pipeline (resize + normalize)
- `amap_gen` — anomaly map normalizer
- `device` — resolved device (mps/cpu/cuda)
- `threshold` — decision threshold
- Metadata: method, backbone, category, img_size

**Why a dict, not a class?** Simplicity. A dict is serializable, inspectable, and doesn't require understanding a class hierarchy. You can print it, modify the threshold on the fly, or pass it across process boundaries without pickle issues.

### run_inference()

```python
result = run_inference(handle, image)
```

Takes a BGR numpy array (standard OpenCV format, compatible with camera feeds) and returns:

```python
{
    "image_score": 0.4231,       # Raw anomaly score
    "is_anomalous": False,       # Binary decision
    "threshold": 0.5,            # Threshold used
    "anomaly_map": np.ndarray,   # (224, 224) normalized heatmap
    "meta": {
        "method": "patchcore",
        "backbone": "resnet18",
        "category": "carpet",
        "device": "mps",
        "img_size": 224,
        "inference_ms": 42.3,    # Actual timing
    }
}
```

Every call includes `inference_ms` in metadata, so you can monitor latency in production without external profiling.

### Why BGR Input?

OpenCV reads images as BGR by default. Camera APIs return BGR. Production pipelines use BGR. By accepting BGR, we avoid the common bug where an engineer forgets to convert RGB↔BGR and gets mysteriously bad results because the color channels are swapped during normalization.

---

## Chapter 22: FastAPI Server — REST Deployment

### Architecture

The server (`api/server.py`) is a thin HTTP wrapper around the inference API:

```
Client → POST /predict (image file) → FastAPI → run_inference() → JSON response
```

Three endpoints:
- `POST /predict` — upload image, get anomaly score
- `GET /health` — liveness check for load balancers
- `GET /model-info` — returns loaded model metadata (backbone, category, memory bank size)

### Usage

```bash
# Start server
python api/server.py \
    --model_path ./outputs/benchmark/carpet/patchcore_memory_bank.pt \
    --backbone resnet18 --category carpet

# Test
curl -X POST http://localhost:8000/predict \
    -F "file=@./data/mvtec/carpet/test/scratch/000.png"
```

Response:
```json
{
    "image_score": 0.4231,
    "is_anomalous": false,
    "threshold": 0.5,
    "filename": "000.png",
    "meta": {
        "method": "patchcore",
        "backbone": "resnet18",
        "inference_ms": 42.3
    }
}
```

**Auto-generated documentation** at `http://localhost:8000/docs` (Swagger UI) — engineers can explore and test the API interactively without reading source code.

### Why FastAPI Over Flask

FastAPI provides:
- Automatic OpenAPI documentation
- Type validation via Pydantic (invalid requests get clear error messages)
- Async support (can handle concurrent requests without blocking)
- Built-in file upload handling via `python-multipart`

For an ML inference server, these are exactly the right features.

---

## Chapter 23: Benchmark Configuration and Reproducibility

### YAML Config

Instead of remembering 10 CLI flags, you can define them once in `configs/benchmark_config.yaml`:

```yaml
benchmark:
  backbone: resnet18
  img_size: 224
  batch_size: 32
  coreset_ratio: 0.1
  num_neighbors: 9
  sigma: 4.0
  device: auto
  num_workers: 4
```

Then: `python benchmark.py --config configs/benchmark_config.yaml`

CLI flags override config values, so you can use the config as a baseline and tweak one parameter: `python benchmark.py --config configs/benchmark_config.yaml --backbone wide_resnet50`

### Output Structure

After a full benchmark run, the output directory contains:

```
outputs/benchmark/
├── benchmark_results.json      # Full results + config metadata
├── benchmark_results.csv       # For pandas/Excel
├── benchmark_results.md        # Paste into README
├── bottle/
│   ├── patchcore_memory_bank.pt  # Trained model (reusable)
│   └── metrics.json              # Per-category results
├── cable/
│   ├── patchcore_memory_bank.pt
│   └── metrics.json
└── ... (15 categories)
```

Every trained model is saved and reusable — you can immediately run `profile_inference.py` or `api/server.py` on any category's model without retraining.

---

## Chapter 24: What This Project Demonstrates

This project demonstrates competencies that matter for computer vision engineering roles:

### Research Understanding
- Implemented PatchCore and FastFlow from papers, not from tutorials
- Understood and implemented PRO score, coreset subsampling, normalizing flows
- Compared architectures (CNN vs Transformer) with measured trade-offs

### Engineering Quality
- 45 automated tests covering models, metrics, API, and datasets
- Modular codebase: swapping backbones, methods, or datasets requires changing one argument
- Clean error handling: benchmark continues past category failures, scripts validate inputs

### Deployment Readiness
- REST API with auto-docs and health checks
- Inference profiling with measured latency, throughput, and memory
- Device-agnostic code: CPU, MPS, CUDA without code changes

### Production Thinking
- Cost-based threshold optimization (not just F1)
- False alarm root cause analysis (border artifacts → actionable fix)
- Incremental adaptation for distribution drift
- Robustness testing against 18 real-world degradation scenarios

### Communication
- Full 15-category benchmark table with honest numbers (including the low ones)
- Comparison to published baselines (anomalib, PatchCore paper)
- Three-part documentation explaining every design decision

---

## Summary

Part 3 covered the deployment and engineering layer built on top of the core anomaly detection system:

1. **Full 15-category benchmark** with real numbers (90.8% mean Image AUROC, ResNet-18, 6h 20min on M1)
2. **Inference profiling** with measured latency (41.7 ms, 24 img/sec on MPS)
3. **Clean Python API** (`load_model`/`run_inference`) as the single source of truth
4. **FastAPI REST server** with auto-docs, health checks, and file upload
5. **Reproducible configuration** via YAML + CLI overrides

Together with Parts 1 (methods and architecture) and 2 (training, evaluation, and experiments), this provides complete documentation of a production-grade anomaly detection system.
