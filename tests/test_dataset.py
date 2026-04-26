"""
VisionGuard-AD — Dataset Tests
================================
Test MVTec and VisA dataset classes load correctly and return right shapes.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mvtec_dataset import MVTecDataset, MVTEC_CATEGORIES, get_dataloader


def create_dummy_mvtec(root, category="carpet"):
    """Create minimal dummy MVTec directory structure for testing."""
    # Train
    train_dir = Path(root) / category / "train" / "good"
    train_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(train_dir / f"{i:03d}.png")

    # Test — good
    test_good = Path(root) / category / "test" / "good"
    test_good.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(test_good / f"{i:03d}.png")

    # Test — defect
    test_defect = Path(root) / category / "test" / "scratch"
    test_defect.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(root) / category / "ground_truth" / "scratch"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(test_defect / f"{i:03d}.png")
        mask = Image.fromarray(np.random.randint(0, 1, (100, 100), dtype=np.uint8) * 255)
        mask.save(gt_dir / f"{i:03d}_mask.png")


class TestMVTecDataset:
    """Tests for MVTecDataset class."""

    def test_train_split_shapes(self, tmp_path):
        create_dummy_mvtec(tmp_path, "carpet")
        ds = MVTecDataset(str(tmp_path), "carpet", split="train", img_size=224)
        assert len(ds) > 0

        img, mask, label, defect_type, path = ds[0]
        assert img.shape == (3, 224, 224), f"Expected (3,224,224), got {img.shape}"
        assert mask.shape == (1, 224, 224), f"Expected (1,224,224), got {mask.shape}"
        assert label == 0
        assert defect_type == "good"
        assert isinstance(path, str)

    def test_test_split_shapes(self, tmp_path):
        create_dummy_mvtec(tmp_path, "carpet")
        ds = MVTecDataset(str(tmp_path), "carpet", split="test", img_size=224)
        assert len(ds) == 10  # 5 good + 5 defect

        # Check defective sample
        defect_indices = [i for i in range(len(ds)) if ds.labels[i] == 1]
        assert len(defect_indices) == 5

        img, mask, label, dt, path = ds[defect_indices[0]]
        assert label == 1
        assert dt == "scratch"

    def test_val_split(self, tmp_path):
        create_dummy_mvtec(tmp_path, "carpet")
        ds_train = MVTecDataset(str(tmp_path), "carpet", split="train", img_size=224)
        ds_val = MVTecDataset(str(tmp_path), "carpet", split="val", img_size=224)
        assert len(ds_train) + len(ds_val) == 10  # Total training images

    def test_statistics(self, tmp_path):
        create_dummy_mvtec(tmp_path, "carpet")
        ds = MVTecDataset(str(tmp_path), "carpet", split="test", img_size=224)
        stats = ds.get_statistics()
        assert stats["total_images"] == 10
        assert stats["normal_count"] == 5
        assert stats["defect_count"] == 5
        assert "good" in stats["defect_type_breakdown"]
        assert "scratch" in stats["defect_type_breakdown"]

    def test_dataloader(self, tmp_path):
        create_dummy_mvtec(tmp_path, "carpet")
        ds = MVTecDataset(str(tmp_path), "carpet", split="train", img_size=224)
        loader = get_dataloader(ds, batch_size=4, shuffle=True, num_workers=0)

        batch = next(iter(loader))
        images, masks, labels, defect_types, paths = batch
        assert images.shape[0] <= 4
        assert images.shape[1:] == (3, 224, 224)

    def test_invalid_category(self, tmp_path):
        with pytest.raises(AssertionError):
            MVTecDataset(str(tmp_path), "invalid_category", split="train")

    def test_invalid_split(self, tmp_path):
        with pytest.raises(AssertionError):
            MVTecDataset(str(tmp_path), "carpet", split="invalid")

    def test_mask_all_zeros_for_normal(self, tmp_path):
        create_dummy_mvtec(tmp_path, "carpet")
        ds = MVTecDataset(str(tmp_path), "carpet", split="train", img_size=224)
        _, mask, _, _, _ = ds[0]
        assert mask.sum() == 0, "Train masks should be all zeros"
