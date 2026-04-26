"""
VisionGuard-AD — Metrics Tests
================================
Test metric computations on synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.evaluator import AnomalyEvaluator
from threshold.threshold_optimizer import ThresholdOptimizer


class TestAnomalyEvaluator:
    """Tests for AnomalyEvaluator class."""

    def setup_method(self):
        self.evaluator = AnomalyEvaluator()

    def test_perfect_image_metrics(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
        labels = np.array([0, 0, 0, 1, 1, 1])
        metrics = self.evaluator.compute_image_level_metrics(scores, labels)
        assert metrics["image_auroc"] == 100.0
        assert metrics["image_f1"] > 90.0

    def test_random_image_metrics(self):
        np.random.seed(42)
        scores = np.random.rand(100)
        labels = np.random.randint(0, 2, 100)
        metrics = self.evaluator.compute_image_level_metrics(scores, labels)
        assert 0 <= metrics["image_auroc"] <= 100
        assert 0 <= metrics["image_f1"] <= 100

    def test_single_class_handling(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])  # All normal
        metrics = self.evaluator.compute_image_level_metrics(scores, labels)
        assert metrics["image_auroc"] == 0.0  # Undefined, returns 0

    def test_pixel_metrics(self):
        # Perfect pixel prediction
        amap = np.zeros((100, 100))
        amap[40:60, 40:60] = 1.0  # Predicted defect region
        gt = np.zeros((100, 100))
        gt[40:60, 40:60] = 1.0  # Ground truth defect

        metrics = self.evaluator.compute_pixel_level_metrics([amap], [gt])
        assert metrics["pixel_auroc"] == 100.0

    def test_pro_score_perfect(self):
        # Use continuous anomaly scores (gradient) for meaningful threshold sweep
        amap = np.zeros((100, 100), dtype=np.float64)
        gt = np.zeros((100, 100), dtype=np.float64)
        gt[20:40, 20:40] = 1.0
        # Create strong signal in defect region with gradient falloff
        for i in range(100):
            for j in range(100):
                if 20 <= i < 40 and 20 <= j < 40:
                    amap[i, j] = 0.9 + 0.1 * np.random.rand()
                else:
                    amap[i, j] = 0.05 * np.random.rand()

        pro = self.evaluator.compute_pro_score([amap], [gt], num_thresholds=50)
        assert pro > 20.0, f"PRO score should be meaningful for near-perfect prediction, got {pro}"

    def test_report_generation(self, tmp_path):
        results = {"image_auroc": 98.5, "pixel_auroc": 97.2, "pro_score": 93.1}
        self.evaluator.generate_report(results, str(tmp_path), category="carpet")
        assert (tmp_path / "eval_results.json").exists()
        assert (tmp_path / "eval_results.md").exists()


class TestThresholdOptimizer:
    """Tests for ThresholdOptimizer class."""

    def setup_method(self):
        self.optimizer = ThresholdOptimizer()

    def test_f1_optimization(self):
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        thresh, metrics = self.optimizer.find_optimal_threshold(scores, labels, "f1")
        assert 0.3 <= thresh <= 0.7
        assert metrics["f1"] > 0.9

    def test_youden_optimization(self):
        scores = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        thresh, metrics = self.optimizer.find_optimal_threshold(scores, labels, "youden")
        assert 0.2 <= thresh <= 0.8

    def test_sensitivity_analysis(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        df = self.optimizer.sensitivity_analysis(scores, labels, steps=10)
        assert len(df) == 10
        assert "f1" in df.columns
        assert "precision" in df.columns

    def test_production_threshold(self):
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        thresh, cost, metrics = self.optimizer.production_threshold_report(
            scores, labels, cost_fp=1.0, cost_fn=10.0
        )
        assert thresh > 0
        assert cost >= 0
        assert "expected_cost" in metrics

    def test_roc_curve_plot(self, tmp_path):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        self.optimizer.plot_roc_curve(scores, labels, save_path=str(tmp_path / "roc.png"))
        assert (tmp_path / "roc.png").exists()
