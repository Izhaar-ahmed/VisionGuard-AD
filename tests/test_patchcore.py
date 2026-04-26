"""
VisionGuard-AD — PatchCore Tests
==================================
Test feature extractor, coreset sampler, and PatchCore model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbones.feature_extractor import FeatureExtractor, PatchEmbedding
from models.patchcore.coreset_sampler import CoresetSampler
from models.patchcore.patchcore import PatchCore


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_resnet18_output_shapes(self):
        fe = FeatureExtractor(backbone_name="resnet18", device="cpu")
        dummy = torch.randn(2, 3, 224, 224)
        features = fe(dummy)

        assert "layer2" in features
        assert "layer3" in features
        assert features["layer2"].shape[0] == 2  # Batch size preserved
        assert features["layer2"].dim() == 4  # (B, C, H, W)

    def test_feature_dims(self):
        fe = FeatureExtractor(backbone_name="resnet18", device="cpu")
        dims = fe.get_feature_dims()
        assert isinstance(dims, dict)
        assert "layer2" in dims
        assert "layer3" in dims
        assert dims["layer2"] > 0
        assert dims["layer3"] > 0

    def test_frozen_parameters(self):
        fe = FeatureExtractor(backbone_name="resnet18", device="cpu")
        for param in fe.backbone.parameters():
            assert not param.requires_grad, "All backbone params should be frozen"


class TestPatchEmbedding:
    """Tests for PatchEmbedding class."""

    def test_output_shape(self):
        pe = PatchEmbedding(patch_size=3)
        features = {
            "layer2": torch.randn(2, 128, 28, 28),
            "layer3": torch.randn(2, 256, 14, 14),
        }
        embeddings, (B, H, W) = pe(features, return_spatial_shape=True)

        assert B == 2
        assert embeddings.shape[0] == B * H * W
        assert embeddings.shape[1] == 128 + 256  # Concatenated channels


class TestCoresetSampler:
    """Tests for CoresetSampler class."""

    def test_reduces_size(self):
        sampler = CoresetSampler(ratio=0.1, device="cpu")
        embeddings = torch.randn(1000, 128)
        result = sampler.run(embeddings)
        assert result.shape[0] == 100  # 10% of 1000
        assert result.shape[1] == 128

    def test_ratio_one_returns_all(self):
        sampler = CoresetSampler(ratio=1.0, device="cpu")
        embeddings = torch.randn(100, 64)
        result = sampler.run(embeddings)
        assert result.shape[0] == 100

    def test_small_dataset(self):
        sampler = CoresetSampler(ratio=0.5, device="cpu")
        embeddings = torch.randn(10, 32)
        result = sampler.run(embeddings)
        assert result.shape[0] == 5


class TestPatchCore:
    """Tests for PatchCore model."""

    def test_predict_returns_correct_types(self):
        model = PatchCore(backbone="resnet18", coreset_ratio=0.5, device="cpu")

        # Determine actual feature dimension from the extractor
        feat_dims = model.feature_extractor.get_feature_dims()
        total_dim = sum(feat_dims.values())

        # Create dummy memory bank with correct dimension
        model.memory_bank = torch.randn(100, total_dim)

        # Use 3D input (single image, not batch) so predict returns scalar
        dummy_image = torch.randn(3, 224, 224)

        score, amap = model.predict(dummy_image)
        assert isinstance(score, float), f"Expected float, got {type(score)}: {score}"
        assert isinstance(amap, np.ndarray)
        assert amap.shape == (224, 224)

    def test_save_load(self, tmp_path):
        model = PatchCore(backbone="resnet18", coreset_ratio=0.5, device="cpu")
        feat_dims = model.feature_extractor.get_feature_dims()
        total_dim = sum(feat_dims.values())
        model.memory_bank = torch.randn(50, total_dim)
        model.spatial_shape = (14, 14)

        save_path = str(tmp_path / "test_model.pt")
        model.save(save_path)

        model2 = PatchCore(backbone="resnet18", device="cpu")
        model2.load(save_path)

        assert model2.memory_bank is not None
        assert model2.memory_bank.shape == (50, total_dim)
