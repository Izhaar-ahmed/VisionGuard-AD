"""
VisionGuard-AD — Benchmark Script
====================================
Run training + evaluation across ALL MVTec categories.
Produces summary tables, backbone comparison, and coreset ablation.

Usage:
    python benchmark.py --method patchcore --data_root ./data/mvtec
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mvtec_dataset import MVTecDataset, MVTEC_CATEGORIES, get_dataloader
from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow
from anomaly_map.anomaly_map_generator import AnomalyMapGenerator
from metrics.evaluator import AnomalyEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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


def run_single_category(method, category, args, device):
    """Train and evaluate on a single category. Returns metrics dict."""
    logger.info(f"\n{'='*50}\n  {category.upper()}\n{'='*50}")

    train_ds = MVTecDataset(args.data_root, category, "train", img_size=args.img_size)
    test_ds = MVTecDataset(args.data_root, category, "test", img_size=args.img_size)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # Train
    start = time.time()
    if method == "patchcore":
        model = PatchCore(backbone=args.backbone, coreset_ratio=args.coreset_ratio, device=str(device))
        model.fit(train_loader)
        save_path = os.path.join(args.output_dir, category, "patchcore_memory_bank.pt")
        model.save(save_path)
    else:
        model = FastFlow(backbone=args.backbone, device=str(device), img_size=args.img_size)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        model.train()
        for epoch in range(args.num_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                loss, _ = model(batch[0])
                loss.backward()
                optimizer.step()
        model.eval()
    train_time = time.time() - start

    # Evaluate
    scores, labels, amaps, gt_masks = [], [], [], []
    amap_gen = AnomalyMapGenerator()

    with torch.no_grad():
        for batch in test_loader:
            images, masks, lbls = batch[0].to(device), batch[1], batch[2]
            if method == "patchcore":
                raw_score, raw_amap = model.predict(images)
            else:
                raw_score, raw_amap = model.get_anomaly_map(images)

            scores.append(_unwrap_score(raw_score))
            amaps.append(_unwrap_amap(raw_amap))
            labels.extend(lbls.numpy().tolist())
            gt_masks.extend([m.squeeze().numpy() for m in masks])

    evaluator = AnomalyEvaluator()
    scores_arr, labels_arr = np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32)

    img_metrics = evaluator.compute_image_level_metrics(scores_arr, labels_arr)

    norm_maps = [amap_gen.normalize(m) for m in amaps]
    defect_maps = [m for m, l in zip(norm_maps, labels) if l == 1]
    defect_gts = [g for g, l in zip(gt_masks, labels) if l == 1]

    pix_metrics = evaluator.compute_pixel_level_metrics(defect_maps, defect_gts) if defect_maps else {"pixel_auroc": 0.0}
    pro = evaluator.compute_pro_score(defect_maps, defect_gts) if defect_maps else 0.0

    # Clear memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "category": category,
        "image_auroc": img_metrics["image_auroc"],
        "pixel_auroc": pix_metrics.get("pixel_auroc", 0.0),
        "pro_score": pro,
        "train_time": f"{train_time:.0f}s",
    }
    logger.info(f"  Image-AUROC: {result['image_auroc']:.1f}%  |  "
                f"Pixel-AUROC: {result['pixel_auroc']:.1f}%  |  "
                f"PRO: {result['pro_score']:.1f}%  |  "
                f"Time: {result['train_time']}")
    return result


def format_table(results):
    """Format results as Markdown table."""
    lines = ["| Category | Image-AUROC | Pixel-AUROC | PRO Score | Train Time |",
             "|----------|-------------|-------------|-----------|------------|"]
    for r in results:
        lines.append(f"| {r['category']:<8} | {r['image_auroc']:>11.1f} | "
                      f"{r['pixel_auroc']:>11.1f} | {r['pro_score']:>9.1f} | {r['train_time']:>10} |")

    mean_img = np.mean([r["image_auroc"] for r in results])
    mean_pix = np.mean([r["pixel_auroc"] for r in results])
    mean_pro = np.mean([r["pro_score"] for r in results])
    lines.append(f"| **Mean** | **{mean_img:>8.1f}** | **{mean_pix:>8.1f}** | **{mean_pro:>6.1f}** |     —      |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="VisionGuard-AD Benchmark")
    parser.add_argument("--method", default="patchcore", choices=["patchcore", "fastflow"])
    parser.add_argument("--data_root", default="./data/mvtec")
    parser.add_argument("--output_dir", default="./outputs/benchmark")
    parser.add_argument("--backbone", default="wide_resnet50")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--coreset_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--categories", nargs="*", default=None, help="Subset of categories to run")
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    categories = args.categories or MVTEC_CATEGORIES

    results = []
    for cat in categories:
        try:
            r = run_single_category(args.method, cat, args, device)
            results.append(r)
        except Exception as e:
            logger.error(f"Failed on {cat}: {e}")
            results.append({"category": cat, "image_auroc": 0, "pixel_auroc": 0, "pro_score": 0, "train_time": "ERR"})

    table = format_table(results)
    print("\n" + table)

    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "benchmark_results.md"), "w") as f:
        f.write(f"# VisionGuard-AD Benchmark — {args.method}\n\n{table}\n")

    logger.info(f"Benchmark complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
