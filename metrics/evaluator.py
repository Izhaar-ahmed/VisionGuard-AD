"""
VisionGuard-AD — Anomaly Evaluator
====================================
Image-level: AUROC, AP, F1 | Pixel-level: AUROC, AP | PRO score
Report generation in JSON + Markdown.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from scipy.ndimage import label as scipy_label
from sklearn.metrics import (
    auc, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)

logger = logging.getLogger("visionguard.metrics")


class AnomalyEvaluator:
    """Comprehensive evaluation for anomaly detection models."""

    def compute_image_level_metrics(self, scores, labels):
        """Compute AUROC, AP, F1, Precision, Recall at optimal threshold."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        if len(np.unique(labels)) < 2:
            return {"image_auroc": 0.0, "image_ap": 0.0, "image_f1": 0.0,
                    "image_precision": 0.0, "image_recall": 0.0, "optimal_threshold": 0.0}

        auroc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        precs, recs, threshs = precision_recall_curve(labels, scores)
        f1s = 2 * (precs * recs) / (precs + recs + 1e-8)
        best_idx = np.argmax(f1s)
        opt_thresh = float(threshs[best_idx]) if best_idx < len(threshs) else 0.5
        preds = (scores >= opt_thresh).astype(np.int32)

        return {
            "image_auroc": round(auroc * 100, 2),
            "image_ap": round(ap * 100, 2),
            "image_f1": round(f1_score(labels, preds, zero_division=0) * 100, 2),
            "image_precision": round(precision_score(labels, preds, zero_division=0) * 100, 2),
            "image_recall": round(recall_score(labels, preds, zero_division=0) * 100, 2),
            "optimal_threshold": round(opt_thresh, 6),
        }

    def compute_pixel_level_metrics(self, anomaly_maps, gt_masks):
        """Compute pixel-AUROC and pixel-AP."""
        import cv2
        all_scores, all_labels = [], []
        for amap, gt in zip(anomaly_maps, gt_masks):
            if amap.shape != gt.shape:
                amap = cv2.resize(amap, (gt.shape[1], gt.shape[0]))
            all_scores.append(amap.flatten())
            all_labels.append((gt.flatten() > 0.5).astype(np.int32))
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        if len(np.unique(all_labels)) < 2:
            return {"pixel_auroc": 0.0, "pixel_ap": 0.0}
        return {
            "pixel_auroc": round(roc_auc_score(all_labels, all_scores) * 100, 2),
            "pixel_ap": round(average_precision_score(all_labels, all_scores) * 100, 2),
        }

    def compute_pro_score(self, anomaly_maps, gt_masks, integration_limit=0.3, num_thresholds=200):
        """
        Per-Region Overlap score — equal weight to all defect regions.
        Integrates PRO vs FPR curve from 0 to integration_limit.
        """
        import cv2
        all_s = np.concatenate([a.flatten() for a in anomaly_maps])
        thresholds = np.linspace(all_s.max(), all_s.min(), num_thresholds)
        pro_vals, fpr_vals = [], []

        for thresh in thresholds:
            overlaps, total_fp, total_norm = [], 0, 0
            for amap, gt in zip(anomaly_maps, gt_masks):
                if amap.shape != gt.shape:
                    amap = cv2.resize(amap, (gt.shape[1], gt.shape[0]))
                pred = (amap >= thresh).astype(np.uint8)
                gt_bin = (gt > 0.5).astype(np.uint8)
                labeled_gt, n_regions = scipy_label(gt_bin)
                for rid in range(1, n_regions + 1):
                    rmask = (labeled_gt == rid).astype(np.uint8)
                    if rmask.sum() == 0:
                        continue
                    overlaps.append(float((pred * rmask).sum()) / float(rmask.sum()))
                norm_mask = (gt_bin == 0).astype(np.uint8)
                total_fp += int((pred * norm_mask).sum())
                total_norm += int(norm_mask.sum())
            pro_vals.append(np.mean(overlaps) if overlaps else 0.0)
            fpr_vals.append(total_fp / max(total_norm, 1))

        pro_vals, fpr_vals = np.array(pro_vals, dtype=np.float64), np.array(fpr_vals, dtype=np.float64)
        idx = np.argsort(fpr_vals)
        fpr_s, pro_s = fpr_vals[idx], pro_vals[idx]
        valid = fpr_s <= integration_limit
        if valid.sum() < 2:
            return 0.0
        return round(float(auc(fpr_s[valid], pro_s[valid])) / integration_limit * 100, 2)

    def generate_report(self, results_dict, save_path, category=""):
        """Save results as JSON and Markdown table."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "eval_results.json", "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        with open(save_dir / "eval_results.md", "w") as f:
            f.write(f"# Evaluation Results — {category}\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in results_dict.items():
                if isinstance(v, float):
                    f.write(f"| {k} | {v:.2f} |\n")
                else:
                    f.write(f"| {k} | {v} |\n")
        logger.info(f"Report saved to {save_dir}")
