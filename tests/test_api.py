"""
VisionGuard-AD — API Tests
=============================
Tests for the inference API module (load_model + run_inference).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.patchcore.patchcore import PatchCore
from models.backbones.feature_extractor import FeatureExtractor


class TestInferenceAPI:
    """Tests for api/inference_api.py."""

    def test_load_model_returns_handle(self, tmp_path):
        """load_model should return a dict with all required keys."""
        from api.inference_api import load_model

        # Create a dummy model to load
        model = PatchCore(backbone="resnet18", coreset_ratio=0.5, device="cpu")
        feat_dims = model.feature_extractor.get_feature_dims()
        total_dim = sum(feat_dims.values())
        model.memory_bank = torch.randn(50, total_dim)
        model.spatial_shape = (28, 28)
        save_path = str(tmp_path / "test_model.pt")
        model.save(save_path)

        handle = load_model(
            method="patchcore",
            backbone="resnet18",
            category="carpet",
            model_path=save_path,
            device="cpu",
        )

        required_keys = {"model", "method", "backbone", "category", "device",
                         "transform", "amap_gen", "threshold", "img_size"}
        assert required_keys.issubset(handle.keys()), \
            f"Missing keys: {required_keys - handle.keys()}"

    def test_run_inference_output_format(self, tmp_path):
        """run_inference should return dict with correct fields and types."""
        from api.inference_api import load_model, run_inference

        model = PatchCore(backbone="resnet18", coreset_ratio=0.5, device="cpu")
        feat_dims = model.feature_extractor.get_feature_dims()
        total_dim = sum(feat_dims.values())
        model.memory_bank = torch.randn(50, total_dim)
        model.spatial_shape = (28, 28)
        save_path = str(tmp_path / "test_model.pt")
        model.save(save_path)

        handle = load_model(
            method="patchcore", backbone="resnet18",
            category="carpet", model_path=save_path, device="cpu",
        )

        # Create a dummy BGR image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = run_inference(handle, image)

        assert isinstance(result["image_score"], float)
        assert isinstance(result["is_anomalous"], bool)
        assert isinstance(result["threshold"], float)
        assert isinstance(result["anomaly_map"], np.ndarray)
        assert result["anomaly_map"].ndim == 2
        assert isinstance(result["meta"], dict)
        assert "inference_ms" in result["meta"]
        assert "backbone" in result["meta"]

    def test_run_inference_deterministic(self, tmp_path):
        """Same image should give same score twice."""
        from api.inference_api import load_model, run_inference

        model = PatchCore(backbone="resnet18", coreset_ratio=0.5, device="cpu")
        feat_dims = model.feature_extractor.get_feature_dims()
        total_dim = sum(feat_dims.values())
        model.memory_bank = torch.randn(50, total_dim)
        model.spatial_shape = (28, 28)
        save_path = str(tmp_path / "test_model.pt")
        model.save(save_path)

        handle = load_model(
            method="patchcore", backbone="resnet18",
            category="carpet", model_path=save_path, device="cpu",
        )

        np.random.seed(42)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        r1 = run_inference(handle, image)
        r2 = run_inference(handle, image)
        assert abs(r1["image_score"] - r2["image_score"]) < 1e-6, \
            "Same image should produce same score"

    def test_run_inference_no_anomaly_map(self, tmp_path):
        """When return_anomaly_map=False, anomaly_map should be None."""
        from api.inference_api import load_model, run_inference

        model = PatchCore(backbone="resnet18", coreset_ratio=0.5, device="cpu")
        feat_dims = model.feature_extractor.get_feature_dims()
        total_dim = sum(feat_dims.values())
        model.memory_bank = torch.randn(50, total_dim)
        model.spatial_shape = (28, 28)
        save_path = str(tmp_path / "test_model.pt")
        model.save(save_path)

        handle = load_model(
            method="patchcore", backbone="resnet18",
            category="carpet", model_path=save_path, device="cpu",
        )

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = run_inference(handle, image, return_anomaly_map=False)
        assert result["anomaly_map"] is None

    def test_threshold_override(self, tmp_path):
        """Custom threshold should be reflected in result."""
        from api.inference_api import load_model, run_inference

        model = PatchCore(backbone="resnet18", coreset_ratio=0.5, device="cpu")
        feat_dims = model.feature_extractor.get_feature_dims()
        total_dim = sum(feat_dims.values())
        model.memory_bank = torch.randn(50, total_dim)
        model.spatial_shape = (28, 28)
        save_path = str(tmp_path / "test_model.pt")
        model.save(save_path)

        handle = load_model(
            method="patchcore", backbone="resnet18",
            category="carpet", model_path=save_path, device="cpu",
        )

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = run_inference(handle, image, threshold=999.0)
        assert result["threshold"] == 999.0
        assert result["is_anomalous"] is False  # Score should be << 999
