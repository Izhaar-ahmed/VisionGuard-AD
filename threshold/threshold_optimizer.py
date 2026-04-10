"""
VisionGuard-AD — Threshold Optimizer
======================================
Production-critical component for finding optimal decision thresholds.
Supports F1, Youden's J, PR-balance, and cost-based optimization.
"""

import logging
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

logger = logging.getLogger("visionguard.threshold")


class ThresholdOptimizer:
    """
    Finds optimal anomaly decision thresholds for production deployment.
    Supports multiple optimization criteria and cost-based analysis.
    """

    def find_optimal_threshold(self, scores, labels, criterion="f1"):
        """
        Find optimal threshold by maximizing the given criterion.

        Args:
            scores: (N,) anomaly scores.
            labels: (N,) binary labels.
            criterion: 'f1', 'youden', or 'precision_recall_balance'.

        Returns:
            (optimal_threshold, metrics_at_threshold)
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)

        if criterion == "f1":
            return self._optimize_f1(scores, labels)
        elif criterion == "youden":
            return self._optimize_youden(scores, labels)
        elif criterion == "precision_recall_balance":
            return self._optimize_pr_balance(scores, labels)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def _optimize_f1(self, scores, labels):
        precs, recs, threshs = precision_recall_curve(labels, scores)
        f1s = 2 * (precs * recs) / (precs + recs + 1e-8)
        best = np.argmax(f1s)
        thresh = float(threshs[best]) if best < len(threshs) else float(scores.mean())
        return thresh, self._metrics_at_threshold(scores, labels, thresh)

    def _optimize_youden(self, scores, labels):
        fpr, tpr, threshs = roc_curve(labels, scores)
        j = tpr - fpr  # Youden's J = sensitivity + specificity - 1
        best = np.argmax(j)
        thresh = float(threshs[best]) if best < len(threshs) else float(scores.mean())
        return thresh, self._metrics_at_threshold(scores, labels, thresh)

    def _optimize_pr_balance(self, scores, labels):
        precs, recs, threshs = precision_recall_curve(labels, scores)
        diff = np.abs(precs - recs)
        best = np.argmin(diff)
        thresh = float(threshs[best]) if best < len(threshs) else float(scores.mean())
        return thresh, self._metrics_at_threshold(scores, labels, thresh)

    def _metrics_at_threshold(self, scores, labels, threshold):
        preds = (scores >= threshold).astype(np.int32)
        return {
            "threshold": round(threshold, 6),
            "f1": round(f1_score(labels, preds, zero_division=0), 4),
            "precision": round(precision_score(labels, preds, zero_division=0), 4),
            "recall": round(recall_score(labels, preds, zero_division=0), 4),
            "accuracy": round(accuracy_score(labels, preds), 4),
            "fpr": round(((preds == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1), 4),
            "fnr": round(((preds == 0) & (labels == 1)).sum() / max((labels == 1).sum(), 1), 4),
        }

    def plot_roc_curve(self, scores, labels, save_path=None):
        """Plot ROC curve with AUC and optimal operating point."""
        scores, labels = np.asarray(scores), np.asarray(labels)
        fpr, tpr, threshs = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Optimal point (Youden's J)
        j = tpr - fpr
        best = np.argmax(j)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.plot(fpr[best], tpr[best], "ro", markersize=10, label=f"Optimal (J={j[best]:.3f})")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")
        plt.close(fig)
        return fig

    def plot_precision_recall_curve(self, scores, labels, save_path=None):
        """Plot Precision-Recall curve with AP annotation."""
        scores, labels = np.asarray(scores), np.asarray(labels)
        precs, recs, _ = precision_recall_curve(labels, scores)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(labels, scores)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(recs, precs, "b-", linewidth=2, label=f"PR (AP = {ap:.4f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def sensitivity_analysis(self, scores, labels, threshold_range=None, steps=100):
        """
        Sweep thresholds and compute metrics at each.
        Returns pandas DataFrame for inspection.
        """
        scores, labels = np.asarray(scores), np.asarray(labels)
        if threshold_range is None:
            threshold_range = (scores.min(), scores.max())
        thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)

        rows = []
        for t in thresholds:
            m = self._metrics_at_threshold(scores, labels, t)
            rows.append(m)

        df = pd.DataFrame(rows)
        return df

    def production_threshold_report(self, scores, labels, cost_fp=1.0, cost_fn=10.0):
        """
        Find threshold minimizing business cost: cost_fp*FP + cost_fn*FN.

        Args:
            scores: Anomaly scores.
            labels: Ground truth labels.
            cost_fp: Cost of falsely rejecting a good product.
            cost_fn: Cost of shipping a defective product.

        Returns:
            (optimal_threshold, expected_cost, metrics_dict)
        """
        scores, labels = np.asarray(scores), np.asarray(labels)
        thresholds = np.linspace(scores.min(), scores.max(), 1000)

        best_thresh, best_cost = 0.0, float("inf")
        for t in thresholds:
            preds = (scores >= t).astype(np.int32)
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            cost = cost_fp * fp + cost_fn * fn
            if cost < best_cost:
                best_cost = cost
                best_thresh = t

        metrics = self._metrics_at_threshold(scores, labels, best_thresh)
        metrics["expected_cost"] = round(float(best_cost), 2)
        metrics["cost_fp"] = cost_fp
        metrics["cost_fn"] = cost_fn

        logger.info(
            f"Production threshold: {best_thresh:.4f} "
            f"(cost={best_cost:.2f}, FP×{cost_fp} + FN×{cost_fn})"
        )
        return best_thresh, best_cost, metrics
