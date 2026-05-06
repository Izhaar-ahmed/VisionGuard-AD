"""
VisionGuard-AD — False Alarm Analyzer
=======================================
Analyzes false positives and false negatives to understand WHERE and WHY
the model fails. Clusters false alarms by spatial location to find
systematic patterns (e.g., border artifacts, texture regions).

Usage:
    python false_alarm_analyzer.py --model_path ./outputs_wrn/carpet/patchcore_memory_bank.pt \
        --backbone wide_resnet50 --category carpet --data_root ./data/mvtec
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from models.patchcore.patchcore import PatchCore
from anomaly_map.anomaly_map_generator import AnomalyMapGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visionguard.false_alarm")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def analyze_anomaly_map_location(amap: np.ndarray, threshold: float, border_px: int = 10):
    """
    Analyze where anomaly activations occur in the image.
    Returns region classification: 'border', 'center', 'corner', 'edge'.
    """
    h, w = amap.shape
    binary = (amap > threshold).astype(np.uint8)
    total_active = binary.sum()
    
    if total_active == 0:
        return "none", 0.0, {}
    
    # Border mask (10px from all edges)
    border_mask = np.zeros_like(binary)
    border_mask[:border_px, :] = 1
    border_mask[-border_px:, :] = 1
    border_mask[:, :border_px] = 1
    border_mask[:, -border_px:] = 1
    
    # Center region (middle 50%)
    ch, cw = h // 4, w // 4
    center_mask = np.zeros_like(binary)
    center_mask[ch:h-ch, cw:w-cw] = 1
    
    border_active = (binary * border_mask).sum()
    center_active = (binary * center_mask).sum()
    
    border_frac = border_active / max(total_active, 1)
    center_frac = center_active / max(total_active, 1)
    
    if border_frac > 0.5:
        primary_region = "border"
    elif center_frac > 0.5:
        primary_region = "center"
    else:
        primary_region = "distributed"
    
    stats = {
        "total_active_pixels": int(total_active),
        "border_fraction": round(float(border_frac), 3),
        "center_fraction": round(float(center_frac), 3),
        "primary_region": primary_region,
    }
    
    return primary_region, float(border_frac), stats


def apply_border_mask(amap: np.ndarray, border_px: int = 10) -> np.ndarray:
    """Zero out anomaly scores within border_px of image edges."""
    masked = amap.copy()
    masked[:border_px, :] = 0
    masked[-border_px:, :] = 0
    masked[:, :border_px] = 0
    masked[:, -border_px:] = 0
    return masked


def main():
    parser = argparse.ArgumentParser(
        description="False alarm analysis for PatchCore anomaly detection.\n"
                    "Analyzes false positives and false negatives by spatial location,\n"
                    "identifies border artifacts, and tests border-mask mitigation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Path to trained PatchCore memory bank (.pt)")
    parser.add_argument("--backbone", default="wide_resnet50", help="Backbone used during training (default: wide_resnet50)")
    parser.add_argument("--category", default="carpet", help="MVTec category to analyze (default: carpet)")
    parser.add_argument("--data_root", default="./data/mvtec", help="Root directory of MVTec dataset")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, mps, or cuda (default: auto)")
    parser.add_argument("--border_px", type=int, default=10, help="Border width in pixels for spatial analysis (default: 10)")
    parser.add_argument("--output_dir", default="./outputs/false_alarm", help="Directory to save results and visualizations")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Load model
    model = PatchCore(backbone=args.backbone, device=args.device)
    model.load(args.model_path)
    gen = AnomalyMapGenerator()
    
    # Load test set
    test_dir = Path(args.data_root) / args.category / "test"
    
    all_scores, all_labels = [], []
    all_scores_masked, all_amaps = [], []
    false_positives = []
    false_negatives = []
    
    logger.info("Running inference on full test set...")
    
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_normal = defect_dir.name == "good"
        label = 0 if is_normal else 1
        
        for img_path in sorted(defect_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img)
            score, amap = model.predict(tensor.to(model.device))
            score = float(score) if not isinstance(score, (list, tuple)) else float(score[0])
            
            # Normalize amap
            amap_norm = gen.normalize(amap, method="minmax")
            
            # Also compute score with border mask
            amap_masked = apply_border_mask(amap_norm, border_px=args.border_px)
            score_masked = float(amap_masked.max())
            
            all_scores.append(score)
            all_scores_masked.append(score_masked)
            all_labels.append(label)
            all_amaps.append(amap_norm)
    
    # Find optimal threshold
    from threshold.threshold_optimizer import ThresholdOptimizer
    optimizer = ThresholdOptimizer()
    threshold, metrics = optimizer.find_optimal_threshold(all_scores, all_labels, criterion="f1")
    
    logger.info(f"Optimal threshold: {threshold:.4f}")
    
    # Classify false positives and false negatives
    border_triggered_fp = 0
    fp_regions = defaultdict(int)
    
    idx = 0
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_normal = defect_dir.name == "good"
        label = 0 if is_normal else 1
        
        for img_path in sorted(defect_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            
            score = all_scores[idx]
            pred = 1 if score >= threshold else 0
            amap_norm = all_amaps[idx]
            
            if pred == 1 and label == 0:  # False Positive
                region, border_frac, stats = analyze_anomaly_map_location(
                    amap_norm, threshold=0.5, border_px=args.border_px
                )
                fp_regions[region] += 1
                if region == "border":
                    border_triggered_fp += 1
                
                false_positives.append({
                    "image": str(img_path),
                    "score": round(score, 4),
                    "region": region,
                    "border_fraction": round(border_frac, 3),
                    "stats": stats,
                })
                
                # Save visualization
                img_cv = cv2.imread(str(img_path))
                if img_cv is not None:
                    img_resized = cv2.resize(img_cv, (224, 224))
                    overlay = gen.overlay_on_image(img_resized, amap_norm, alpha=0.4,
                                                   threshold=0.5, score=score)
                    cv2.imwrite(str(viz_dir / f"FP_{img_path.stem}.png"), overlay)
            
            elif pred == 0 and label == 1:  # False Negative
                false_negatives.append({
                    "image": str(img_path),
                    "score": round(score, 4),
                    "defect_type": defect_dir.name,
                })
            
            idx += 1
    
    # Compute AUROC with and without border mask
    auroc_original = roc_auc_score(all_labels, all_scores) * 100
    auroc_masked = roc_auc_score(all_labels, all_scores_masked) * 100
    
    # Count FP with masked scores
    fp_masked = sum(1 for s, l in zip(all_scores_masked, all_labels)
                    if s >= threshold and l == 0)
    fp_original = len(false_positives)
    
    total_normal = sum(1 for l in all_labels if l == 0)
    total_defective = sum(1 for l in all_labels if l == 1)
    
    # Results
    results = {
        "total_test_images": len(all_labels),
        "total_normal": total_normal,
        "total_defective": total_defective,
        "threshold": round(threshold, 4),
        "false_positives": fp_original,
        "false_negatives": len(false_negatives),
        "border_triggered_fp": border_triggered_fp,
        "border_triggered_pct": round(border_triggered_fp / max(fp_original, 1) * 100, 1),
        "fp_region_breakdown": dict(fp_regions),
        "auroc_original": round(auroc_original, 2),
        "auroc_with_border_mask": round(auroc_masked, 2),
        "auroc_drop_from_mask": round(auroc_masked - auroc_original, 2),
        "fp_after_border_mask": fp_masked,
        "fp_reduction_pct": round((1 - fp_masked / max(fp_original, 1)) * 100, 1),
    }
    
    with open(output_dir / "false_alarm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "false_alarm_details.json", "w") as f:
        json.dump({"false_positives": false_positives, "false_negatives": false_negatives}, f, indent=2)
    
    # Generate report
    report = f"""# False Alarm Analysis — {args.category.title()} ({args.backbone})

## Summary

- **Total test images:** {len(all_labels)} ({total_normal} normal, {total_defective} defective)
- **Threshold:** {threshold:.4f} (F1-optimized)
- **False positives:** {fp_original}/{total_normal} normal images incorrectly flagged
- **False negatives:** {len(false_negatives)}/{total_defective} defective images missed

## False Positive Spatial Analysis

| Region | Count | Percentage |
|--------|:-----:|:----------:|
"""
    for region, count in sorted(fp_regions.items(), key=lambda x: -x[1]):
        pct = count / max(fp_original, 1) * 100
        report += f"| {region} | {count} | {pct:.0f}% |\n"
    
    report += f"""
## Border Mask Fix

Applying a {args.border_px}px border mask to anomaly maps:

| Metric | Without Mask | With Mask | Change |
|--------|:-----------:|:---------:|:------:|
| Image AUROC | {auroc_original:.2f}% | {auroc_masked:.2f}% | {auroc_masked - auroc_original:+.2f}% |
| False Positives | {fp_original} | {fp_masked} | -{fp_original - fp_masked} ({results['fp_reduction_pct']:.0f}% reduction) |

## Key Finding

> **{border_triggered_fp}/{fp_original} false positives ({results['border_triggered_pct']:.0f}%) triggered on image border regions.**
> Backbone features near crop boundaries are less representative of the true texture.
> A simple {args.border_px}px border mask reduces false positives by {results['fp_reduction_pct']:.0f}%
> with only {abs(auroc_masked - auroc_original):.2f}% AUROC change.
"""
    
    if false_negatives:
        report += "\n## Missed Defects (False Negatives)\n\n"
        report += "| Image | Score | Defect Type |\n|-------|:-----:|:-----------:|\n"
        for fn in false_negatives:
            report += f"| {Path(fn['image']).name} | {fn['score']:.4f} | {fn['defect_type']} |\n"
    
    with open(output_dir / "false_alarm_report.md", "w") as f:
        f.write(report)
    
    # Print summary
    print("\n" + "="*70)
    print("  FALSE ALARM ANALYSIS")
    print("="*70)
    print(f"  False Positives:  {fp_original}/{total_normal} normal images")
    print(f"  False Negatives:  {len(false_negatives)}/{total_defective} defective images")
    print(f"  Border-triggered: {border_triggered_fp}/{fp_original} ({results['border_triggered_pct']:.0f}%)")
    print(f"  AUROC original:   {auroc_original:.2f}%")
    print(f"  AUROC w/ mask:    {auroc_masked:.2f}%")
    print(f"  FP after mask:    {fp_masked} ({results['fp_reduction_pct']:.0f}% reduction)")
    print("="*70)
    
    logger.info(f"Full report saved to {output_dir}")


if __name__ == "__main__":
    main()
