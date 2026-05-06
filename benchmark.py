"""
VisionGuard-AD — Full 15-Category MVTec Benchmark
=====================================================
Trains and evaluates PatchCore on all 15 MVTec-AD categories,
producing per-category metrics and aggregate summary tables.

Optimized for Apple Silicon M1: defaults to ResNet-18 backbone,
reasonable batch sizes, and progress reporting throughout.

Usage:
    # Full benchmark (all 15 categories, ResNet-18)
    python benchmark.py --data_root ./data/mvtec

    # Subset of categories
    python benchmark.py --categories carpet bottle screw

    # Different backbone
    python benchmark.py --backbone wide_resnet50

    # With config file
    python benchmark.py --config configs/benchmark_config.yaml
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mvtec_dataset import MVTecDataset, MVTEC_CATEGORIES, get_dataloader
from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow
from anomaly_map.anomaly_map_generator import AnomalyMapGenerator
from metrics.evaluator import AnomalyEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("visionguard.benchmark")


def _unwrap_score(val):
    """Unwrap score to Python float."""
    if isinstance(val, (list, tuple)):
        val = val[0]
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().item()
    if isinstance(val, np.ndarray):
        val = float(val.flat[0])
    return float(val)


def _unwrap_amap(val):
    """Unwrap anomaly map to 2D numpy array."""
    if isinstance(val, (list, tuple)):
        val = val[0]
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        while val.ndim > 2:
            val = val.squeeze(0) if val.shape[0] == 1 else val[0]
    return np.asarray(val, dtype=np.float64)


def run_single_category(method, category, args, device, cat_idx, total_cats):
    """Train and evaluate on a single category. Returns metrics dict."""
    header = f"[{cat_idx}/{total_cats}] {category.upper()}"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")

    train_ds = MVTecDataset(args.data_root, category, "train", img_size=args.img_size)
    test_ds = MVTecDataset(args.data_root, category, "test", img_size=args.img_size)
    train_loader = get_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    n_train = len(train_ds)
    n_test = len(test_ds)
    print(f"  Train images: {n_train}  |  Test images: {n_test}")

    # ── Train ──────────────────────────────────────────────────
    print(f"  Training PatchCore ({args.backbone})...", flush=True)
    train_start = time.time()

    if method == "patchcore":
        model = PatchCore(
            backbone=args.backbone,
            coreset_ratio=args.coreset_ratio,
            num_neighbors=args.num_neighbors,
            device=str(device),
            sigma=args.sigma,
        )
        model.fit(train_loader)
        save_dir = os.path.join(args.output_dir, category)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "patchcore_memory_bank.pt")
        model.save(save_path)
    else:
        model = FastFlow(backbone=args.backbone, device=str(device), img_size=args.img_size)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        model.train()
        for epoch in range(args.num_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                loss, _ = model(batch[0])
                loss.backward()
                optimizer.step()
        model.eval()

    train_time = time.time() - train_start
    print(f"  ✓ Training done in {train_time:.1f}s")

    # ── Evaluate ───────────────────────────────────────────────
    print(f"  Evaluating on {n_test} test images...", flush=True)
    eval_start = time.time()

    scores, labels, amaps, gt_masks = [], [], [], []
    amap_gen = AnomalyMapGenerator()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, masks, lbls = batch[0].to(device), batch[1], batch[2]
            if method == "patchcore":
                raw_score, raw_amap = model.predict(images)
            else:
                raw_score, raw_amap = model.get_anomaly_map(images)

            scores.append(_unwrap_score(raw_score))
            amaps.append(_unwrap_amap(raw_amap))
            labels.extend(lbls.numpy().tolist())
            gt_masks.extend([m.squeeze().numpy() for m in masks])

            # Progress every 25 images
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == n_test:
                print(f"    Evaluated {batch_idx + 1}/{n_test} images", flush=True)

    eval_time = time.time() - eval_start

    # ── Metrics ────────────────────────────────────────────────
    evaluator = AnomalyEvaluator()
    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    img_metrics = evaluator.compute_image_level_metrics(scores_arr, labels_arr)

    norm_maps = [amap_gen.normalize(m) for m in amaps]
    defect_maps = [m for m, l in zip(norm_maps, labels) if l == 1]
    defect_gts = [g for g, l in zip(gt_masks, labels) if l == 1]

    if defect_maps:
        pix_metrics = evaluator.compute_pixel_level_metrics(defect_maps, defect_gts)
        pro = evaluator.compute_pro_score(defect_maps, defect_gts)
    else:
        pix_metrics = {"pixel_auroc": 0.0}
        pro = 0.0

    # ── Clean up ───────────────────────────────────────────────
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

    result = {
        "category": category,
        "image_auroc": img_metrics["image_auroc"],
        "pixel_auroc": pix_metrics.get("pixel_auroc", 0.0),
        "pro_score": pro,
        "image_f1": img_metrics.get("image_f1", 0.0),
        "image_ap": img_metrics.get("image_ap", 0.0),
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        "n_train": n_train,
        "n_test": n_test,
    }

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  {category.upper():<10} RESULTS                    │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  Image AUROC:  {result['image_auroc']:>7.2f}%                  │")
    print(f"  │  Pixel AUROC:  {result['pixel_auroc']:>7.2f}%                  │")
    print(f"  │  PRO Score:    {result['pro_score']:>7.2f}%                  │")
    print(f"  │  F1 Score:     {result['image_f1']:>7.2f}%                  │")
    print(f"  │  Train: {result['train_time_s']:>5.1f}s  Eval: {result['eval_time_s']:>5.1f}s          │")
    print(f"  └──────────────────────────────────────────┘")

    return result


def format_table(results):
    """Format results as Markdown table."""
    lines = [
        "| Category | Image-AUROC | Pixel-AUROC | PRO Score | F1 Score | Train Time | Eval Time |",
        "|----------|:-----------:|:-----------:|:---------:|:--------:|:----------:|:---------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['category']:<12} | {r['image_auroc']:>7.1f}% "
            f"| {r['pixel_auroc']:>7.1f}% "
            f"| {r['pro_score']:>6.1f}% "
            f"| {r['image_f1']:>5.1f}% "
            f"| {r['train_time_s']:>7.1f}s "
            f"| {r['eval_time_s']:>6.1f}s |"
        )

    mean_img = np.mean([r["image_auroc"] for r in results])
    mean_pix = np.mean([r["pixel_auroc"] for r in results])
    mean_pro = np.mean([r["pro_score"] for r in results])
    mean_f1 = np.mean([r["image_f1"] for r in results])
    std_img = np.std([r["image_auroc"] for r in results])
    std_pro = np.std([r["pro_score"] for r in results])

    lines.append(
        f"| **Mean** | **{mean_img:>5.1f}%** "
        f"| **{mean_pix:>5.1f}%** "
        f"| **{mean_pro:>4.1f}%** "
        f"| **{mean_f1:>3.1f}%** "
        f"| — | — |"
    )
    return "\n".join(lines), {
        "mean_image_auroc": round(mean_img, 2),
        "mean_pixel_auroc": round(mean_pix, 2),
        "mean_pro_score": round(mean_pro, 2),
        "mean_f1": round(mean_f1, 2),
        "std_image_auroc": round(std_img, 2),
        "std_pro_score": round(std_pro, 2),
    }


def save_csv(results, path):
    """Save results to CSV file."""
    if not results:
        return
    keys = results[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard-AD Full 15-Category Benchmark\n"
                    "Trains and evaluates PatchCore on all MVTec-AD categories,\n"
                    "producing per-category metrics and aggregate summary tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method", default="patchcore", choices=["patchcore", "fastflow"],
        help="Anomaly detection method (default: patchcore)",
    )
    parser.add_argument(
        "--data_root", default="./data/mvtec",
        help="Root directory of MVTec dataset (default: ./data/mvtec)",
    )
    parser.add_argument(
        "--output_dir", default="./outputs/benchmark",
        help="Directory to save all benchmark outputs (default: ./outputs/benchmark)",
    )
    parser.add_argument(
        "--backbone", default="resnet18",
        help="Feature extractor backbone (default: resnet18). Options: resnet18, wide_resnet50, resnet50, vit_b16",
    )
    parser.add_argument("--img_size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction (default: 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--coreset_ratio", type=float, default=0.1, help="Coreset subsampling ratio (default: 0.1)")
    parser.add_argument("--num_neighbors", type=int, default=9, help="k-NN neighbors for scoring (default: 9)")
    parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian smoothing sigma (default: 4.0)")
    parser.add_argument("--num_epochs", type=int, default=50, help="FastFlow training epochs (default: 50)")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, mps, cuda (default: auto)")
    parser.add_argument(
        "--categories", nargs="*", default=None,
        help="Subset of categories to run (default: all 15). E.g., --categories carpet bottle screw",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (overrides CLI defaults)")
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if cfg:
            for section in cfg.values():
                if isinstance(section, dict):
                    for k, v in section.items():
                        if hasattr(args, k):
                            setattr(args, k, v)

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    categories = args.categories or MVTEC_CATEGORIES
    total = len(categories)

    # ── Banner ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  VisionGuard-AD — Full MVTec Benchmark")
    print("═" * 60)
    print(f"  Method:     {args.method}")
    print(f"  Backbone:   {args.backbone}")
    print(f"  Device:     {device}")
    print(f"  Categories: {total} ({', '.join(categories)})")
    print(f"  Image size: {args.img_size}×{args.img_size}")
    print(f"  Coreset:    {args.coreset_ratio * 100:.0f}%")
    print(f"  k-NN:       {args.num_neighbors}")
    print("═" * 60)

    overall_start = time.time()
    results = []

    for idx, cat in enumerate(categories, 1):
        print(f"\n{'─' * 60}")
        print(f"  PROGRESS: {idx}/{total} categories "
              f"({(idx-1)/total*100:.0f}% complete)")
        if results:
            elapsed = time.time() - overall_start
            avg_per_cat = elapsed / len(results)
            remaining = avg_per_cat * (total - len(results))
            print(f"  Elapsed: {timedelta(seconds=int(elapsed))} "
                  f"| ETA: {timedelta(seconds=int(remaining))}")
        print(f"{'─' * 60}")

        try:
            r = run_single_category(args.method, cat, args, device, idx, total)
            results.append(r)

            # Save per-category results immediately (resume-friendly)
            cat_json = os.path.join(args.output_dir, cat, "metrics.json")
            os.makedirs(os.path.dirname(cat_json), exist_ok=True)
            with open(cat_json, "w") as f:
                json.dump(r, f, indent=2)

        except Exception as e:
            logger.error(f"FAILED on {cat}: {e}", exc_info=True)
            results.append({
                "category": cat, "image_auroc": 0, "pixel_auroc": 0,
                "pro_score": 0, "image_f1": 0, "image_ap": 0,
                "train_time_s": 0, "eval_time_s": 0,
                "n_train": 0, "n_test": 0, "error": str(e),
            })

    total_time = time.time() - overall_start

    # ── Summary ────────────────────────────────────────────────
    table_str, summary_stats = format_table(results)

    print("\n\n" + "═" * 60)
    print("  BENCHMARK COMPLETE")
    print("═" * 60)
    print(f"\n{table_str}\n")
    print(f"  Total time: {timedelta(seconds=int(total_time))}")
    print(f"  Mean Image AUROC: {summary_stats['mean_image_auroc']:.2f}% ± {summary_stats['std_image_auroc']:.2f}%")
    print(f"  Mean PRO Score:   {summary_stats['mean_pro_score']:.2f}% ± {summary_stats['std_pro_score']:.2f}%")
    print("═" * 60)

    # ── Save outputs ───────────────────────────────────────────
    # JSON
    output_data = {
        "config": {
            "method": args.method,
            "backbone": args.backbone,
            "img_size": args.img_size,
            "coreset_ratio": args.coreset_ratio,
            "num_neighbors": args.num_neighbors,
            "sigma": args.sigma,
            "device": str(device),
        },
        "summary": summary_stats,
        "total_time_s": round(total_time, 1),
        "per_category": results,
    }
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(output_data, f, indent=2)

    # CSV
    save_csv(results, os.path.join(args.output_dir, "benchmark_results.csv"))

    # Markdown
    with open(os.path.join(args.output_dir, "benchmark_results.md"), "w") as f:
        f.write(f"# VisionGuard-AD Benchmark — {args.method} ({args.backbone})\n\n")
        f.write(f"**Config:** img_size={args.img_size}, coreset_ratio={args.coreset_ratio}, "
                f"k={args.num_neighbors}, σ={args.sigma}, device={device}\n\n")
        f.write(table_str + "\n\n")
        f.write(f"**Total time:** {timedelta(seconds=int(total_time))}\n")

    logger.info(f"Benchmark complete. Results saved to {args.output_dir}")
    print(f"\n  Results saved to: {args.output_dir}/")
    print(f"    ├── benchmark_results.json")
    print(f"    ├── benchmark_results.csv")
    print(f"    ├── benchmark_results.md")
    print(f"    └── <category>/metrics.json (per-category)\n")


if __name__ == "__main__":
    main()
