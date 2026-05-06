"""
VisionGuard-AD — Inference Latency & Resource Profiling
=========================================================
Measures wall-clock inference time, throughput, and peak memory usage
for a trained PatchCore or FastFlow model on Apple Silicon M1.

Reports per-image latency (mean, median, p95, p99), throughput (images/sec),
and peak memory consumption.

Usage:
    python profile_inference.py \\
        --model_path ./outputs/benchmark/carpet/patchcore_memory_bank.pt \\
        --backbone resnet18 --category carpet --num_images 50

    # Force CPU (even if MPS is available)
    python profile_inference.py --model_path ... --device cpu
"""

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visionguard.profile")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_test_images(data_root, category, img_size, limit=None):
    """Load test images as tensors."""
    test_dir = Path(data_root) / category / "test"
    transform = get_transform(img_size)
    tensors = []

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        for img_path in sorted(defect_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            img = Image.open(img_path).convert("RGB")
            tensors.append(transform(img))
            if limit and len(tensors) >= limit:
                return tensors
    return tensors


def get_memory_usage_mb():
    """Get current process RSS memory in MB using psutil if available."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Profile inference latency and memory for PatchCore/FastFlow.\n"
                    "Measures per-image timing, throughput, and peak memory on your device.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--method", default="patchcore", choices=["patchcore", "fastflow"],
                        help="Anomaly detection method (default: patchcore)")
    parser.add_argument("--backbone", default="resnet18", help="Backbone architecture (default: resnet18)")
    parser.add_argument("--category", default="carpet", help="MVTec category for test images (default: carpet)")
    parser.add_argument("--data_root", default="./data/mvtec", help="Root directory of MVTec dataset")
    parser.add_argument("--num_images", type=int, default=100, help="Number of test images to profile (default: 100)")
    parser.add_argument("--img_size", type=int, default=224, help="Input image resolution (default: 224)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup images excluded from timing (default: 5)")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda (default: auto)")
    parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian smoothing sigma (default: 4.0)")
    parser.add_argument("--output_dir", default="./outputs/profiling", help="Directory to save profiling results")
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  VisionGuard-AD — Inference Profiler")
    print("═" * 60)
    print(f"  Method:     {args.method}")
    print(f"  Backbone:   {args.backbone}")
    print(f"  Device:     {device}")
    print(f"  Resolution: {args.img_size}×{args.img_size}")
    print(f"  Images:     {args.num_images} (+ {args.warmup} warmup)")
    print("═" * 60)

    # Load model
    print("\n  Loading model...", flush=True)
    mem_before = get_memory_usage_mb()

    if args.method == "patchcore":
        model = PatchCore(backbone=args.backbone, device=str(device), sigma=args.sigma)
        model.load(args.model_path)
        mb_size = model.memory_bank.shape if model.memory_bank is not None else (0, 0)
        mb_memory_mb = (model.memory_bank.nelement() * model.memory_bank.element_size()) / (1024 * 1024) if model.memory_bank is not None else 0
    else:
        model = FastFlow(backbone=args.backbone, device=str(device), img_size=args.img_size)
        model.load(args.model_path)
        model.eval()
        mb_size = (0, 0)
        mb_memory_mb = 0

    mem_after_load = get_memory_usage_mb()
    print(f"  ✓ Model loaded ({mem_after_load - mem_before:.1f} MB added to process)")

    if args.method == "patchcore":
        print(f"  Memory bank: {mb_size[0]:,} × {mb_size[1]} ({mb_memory_mb:.1f} MB)")

    # Load test images
    print(f"\n  Loading test images...", flush=True)
    tensors = load_test_images(args.data_root, args.category, args.img_size, limit=args.num_images + args.warmup)
    actual_images = min(len(tensors), args.num_images + args.warmup)
    print(f"  ✓ Loaded {actual_images} images")

    if actual_images <= args.warmup:
        print("  ERROR: Not enough images for profiling (need more than warmup count)")
        sys.exit(1)

    # Start memory tracking
    tracemalloc.start()
    mem_before_inference = get_memory_usage_mb()

    # ── Warmup ─────────────────────────────────────────────────
    print(f"\n  Warmup ({args.warmup} images)...", flush=True)
    for i in range(min(args.warmup, len(tensors))):
        tensor = tensors[i].unsqueeze(0).to(device)
        with torch.no_grad():
            if args.method == "patchcore":
                model.predict(tensor)
            else:
                model.get_anomaly_map(tensor)
    print(f"  ✓ Warmup done")

    # ── Timed inference ────────────────────────────────────────
    n_profile = min(args.num_images, len(tensors) - args.warmup)
    print(f"\n  Profiling {n_profile} images...", flush=True)

    latencies_ms = []
    for i in range(n_profile):
        tensor = tensors[args.warmup + i].unsqueeze(0).to(device)

        t_start = time.perf_counter()
        with torch.no_grad():
            if args.method == "patchcore":
                model.predict(tensor)
            else:
                model.get_anomaly_map(tensor)

        # Sync device for accurate timing
        if device.type == "mps":
            torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None
        elif device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies_ms.append(elapsed_ms)

        if (i + 1) % 25 == 0 or (i + 1) == n_profile:
            print(f"    {i + 1}/{n_profile} images profiled "
                  f"(avg so far: {np.mean(latencies_ms):.1f} ms/img)", flush=True)

    # ── Memory stats ───────────────────────────────────────────
    mem_after_inference = get_memory_usage_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Compute statistics ─────────────────────────────────────
    latencies = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))
    std_ms = float(np.std(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    throughput = 1000.0 / mean_ms  # images/sec

    results = {
        "method": args.method,
        "backbone": args.backbone,
        "device": str(device),
        "img_size": args.img_size,
        "num_images_profiled": n_profile,
        "warmup_images": args.warmup,
        "category": args.category,
        "latency_ms": {
            "mean": round(mean_ms, 2),
            "median": round(median_ms, 2),
            "std": round(std_ms, 2),
            "p95": round(p95_ms, 2),
            "p99": round(p99_ms, 2),
            "min": round(min_ms, 2),
            "max": round(max_ms, 2),
        },
        "throughput_imgs_per_sec": round(throughput, 2),
        "memory": {
            "model_load_mb": round(mem_after_load - mem_before, 1),
            "peak_python_heap_mb": round(peak / (1024 * 1024), 1),
            "process_rss_mb": round(mem_after_inference, 1),
            "memory_bank_mb": round(mb_memory_mb, 2),
        },
    }

    if args.method == "patchcore":
        results["memory_bank_shape"] = list(mb_size)

    # ── Print summary ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PROFILING RESULTS")
    print("═" * 60)
    print(f"  Method:     {args.method} ({args.backbone})")
    print(f"  Device:     {device}")
    print(f"  Resolution: {args.img_size}×{args.img_size}")
    print(f"  Images:     {n_profile}")
    print(f"")
    print(f"  ┌─ Latency ─────────────────────────────────┐")
    print(f"  │  Mean:    {mean_ms:>8.2f} ms/image              │")
    print(f"  │  Median:  {median_ms:>8.2f} ms/image              │")
    print(f"  │  P95:     {p95_ms:>8.2f} ms/image              │")
    print(f"  │  P99:     {p99_ms:>8.2f} ms/image              │")
    print(f"  │  Std:     {std_ms:>8.2f} ms                    │")
    print(f"  ├─ Throughput ────────────────────────────────┤")
    print(f"  │  {throughput:>6.1f} images/sec                      │")
    print(f"  ├─ Memory ───────────────────────────────────┤")
    print(f"  │  Model load:    {results['memory']['model_load_mb']:>6.1f} MB              │")
    print(f"  │  Peak heap:     {results['memory']['peak_python_heap_mb']:>6.1f} MB              │")
    print(f"  │  Process RSS:   {results['memory']['process_rss_mb']:>6.1f} MB              │")
    if args.method == "patchcore":
        print(f"  │  Memory bank:   {results['memory']['memory_bank_mb']:>6.2f} MB              │")
    print(f"  └────────────────────────────────────────────┘")
    print("═" * 60)

    # ── Save ───────────────────────────────────────────────────
    output_file = output_dir / "profiling_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
