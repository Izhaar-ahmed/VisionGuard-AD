"""
VisionGuard-AD — Evaluation Script
====================================
Run inference on test set, compute all metrics, generate reports.

Usage:
    python evaluate.py --method patchcore --category carpet \
        --model_path ./outputs/carpet/patchcore_memory_bank.pt \
        --visualize --num_visualizations 20
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mvtec_dataset import MVTecDataset, get_dataloader
from data.visa_dataset import VisADataset
from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow
from anomaly_map.anomaly_map_generator import AnomalyMapGenerator
from metrics.evaluator import AnomalyEvaluator
from threshold.threshold_optimizer import ThresholdOptimizer


def _unwrap_score(val):
    """Unwrap score to a Python float, regardless of input type."""
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


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "evaluate.log"), mode="w"),
        ],
    )
    return logging.getLogger("visionguard.evaluate")


def main():
    parser = argparse.ArgumentParser(description="VisionGuard-AD Evaluation")
    parser.add_argument("--method", type=str, default="patchcore", choices=["patchcore", "fastflow"])
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--data_root", type=str, default="./data/mvtec")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--backbone", type=str, default="wide_resnet50")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_visualizations", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=4.0)
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.join(args.output_dir, args.category)
    logger = setup_logging(output_dir)
    logger.info(f"Evaluating {args.method} on {args.category}, device={device}")

    # Load model
    if args.method == "patchcore":
        model = PatchCore(backbone=args.backbone, device=str(device), sigma=args.sigma)
        model.load(args.model_path)
    else:
        model = FastFlow(backbone=args.backbone, device=str(device), img_size=args.img_size)
        model.load(args.model_path)
        model.eval()

    # Load test dataset
    DatasetClass = VisADataset if args.dataset == "visa" else MVTecDataset
    test_dataset = DatasetClass(
        root_dir=args.data_root, category=args.category,
        split="test", img_size=args.img_size,
    )
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Run inference
    all_scores, all_labels, all_maps, all_gt_masks = [], [], [], []
    all_images, all_paths, all_defect_types = [], [], []

    logger.info(f"Running inference on {len(test_dataset)} test images...")
    start = time.time()

    for batch in test_loader:
        images, masks, labels, defect_types, paths = batch
        images = images.to(device)

        with torch.no_grad():
            if args.method == "patchcore":
                raw_score, raw_amap = model.predict(images)
            else:
                raw_score, raw_amap = model.get_anomaly_map(images)

        # Robust unwrapping — handle list, tensor, scalar
        all_scores.append(_unwrap_score(raw_score))
        all_maps.append(_unwrap_amap(raw_amap))
        all_labels.extend(labels.numpy().tolist())
        all_gt_masks.extend([m.squeeze().numpy() for m in masks])
        all_images.extend([img for img in images.cpu()])
        all_paths.extend(paths)
        all_defect_types.extend(defect_types)

    inference_time = time.time() - start
    logger.info(f"Inference completed in {inference_time:.1f}s ({len(test_dataset)/max(inference_time,0.01):.1f} img/s)")

    # Compute metrics
    evaluator = AnomalyEvaluator()
    amap_gen = AnomalyMapGenerator()

    # Normalize anomaly maps
    norm_maps = [amap_gen.normalize(m, method="minmax") for m in all_maps]

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int32)

    img_metrics = evaluator.compute_image_level_metrics(scores_arr, labels_arr)
    logger.info(f"Image-level metrics: {img_metrics}")

    # Pixel-level metrics (only on images with defects)
    defect_maps = [m for m, l in zip(norm_maps, all_labels) if l == 1]
    defect_gts = [g for g, l in zip(all_gt_masks, all_labels) if l == 1]

    if defect_maps:
        pix_metrics = evaluator.compute_pixel_level_metrics(defect_maps, defect_gts)
        pro_score = evaluator.compute_pro_score(defect_maps, defect_gts)
        pix_metrics["pro_score"] = pro_score
        logger.info(f"Pixel-level metrics: {pix_metrics}")
    else:
        pix_metrics = {"pixel_auroc": 0.0, "pixel_ap": 0.0, "pro_score": 0.0}

    # Threshold optimization
    thresh_opt = ThresholdOptimizer()
    opt_thresh, thresh_metrics = thresh_opt.find_optimal_threshold(scores_arr, labels_arr, criterion="f1")
    logger.info(f"Optimal threshold: {opt_thresh:.6f}, metrics: {thresh_metrics}")

    # Save curves
    thresh_opt.plot_roc_curve(scores_arr, labels_arr, save_path=os.path.join(output_dir, "roc_curve.png"))
    thresh_opt.plot_precision_recall_curve(scores_arr, labels_arr, save_path=os.path.join(output_dir, "pr_curve.png"))

    # Combine results
    results = {**img_metrics, **pix_metrics, **thresh_metrics, "inference_time_s": round(inference_time, 2)}

    # Generate report
    evaluator.generate_report(results, output_dir, category=args.category)

    # Visualizations
    if args.visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        n_vis = min(args.num_visualizations, len(test_dataset))
        defect_idx = [i for i, l in enumerate(all_labels) if l == 1]
        normal_idx = [i for i, l in enumerate(all_labels) if l == 0]
        vis_idx = random.sample(defect_idx, min(n_vis // 2, len(defect_idx)))
        vis_idx += random.sample(normal_idx, min(n_vis - len(vis_idx), len(normal_idx)))

        for i in vis_idx:
            orig = amap_gen.denormalize_image(all_images[i])
            grid = amap_gen.create_comparison_grid(
                orig, norm_maps[i], all_gt_masks[i],
                score=all_scores[i], threshold=opt_thresh,
            )
            fname = f"{i:04d}_{all_defect_types[i]}_{all_scores[i]:.3f}.png"
            cv2.imwrite(os.path.join(vis_dir, fname), grid)

        logger.info(f"Saved {len(vis_idx)} visualizations to {vis_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {args.category.upper()}")
    print("=" * 60)
    print(f"  Image AUROC:  {img_metrics['image_auroc']:.2f}%")
    print(f"  Pixel AUROC:  {pix_metrics['pixel_auroc']:.2f}%")
    print(f"  PRO Score:    {pix_metrics['pro_score']:.2f}%")
    print(f"  F1 Score:     {img_metrics['image_f1']:.2f}%")
    print(f"  Threshold:    {opt_thresh:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
