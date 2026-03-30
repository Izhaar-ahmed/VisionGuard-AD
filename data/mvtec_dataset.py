"""
VisionGuard-AD — MVTec Anomaly Detection Dataset
=================================================

PyTorch Dataset for MVTec-AD (15 categories).
Directory structure expected:
    mvtec/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   └── ...
    │   └── ground_truth/
    │       ├── broken_large/
    │       └── ...
    ├── cable/
    │   └── ...
    └── ...
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logger = logging.getLogger("visionguard.data.mvtec")

# All 15 MVTec-AD categories
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Texture categories (useful for category-specific processing)
TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECT_CATEGORIES = [c for c in MVTEC_CATEGORIES if c not in TEXTURE_CATEGORIES]

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for the MVTec Anomaly Detection benchmark.

    Supports train, test, and val splits. Training data contains ONLY normal
    (good) images. Test data contains both normal and defective images with
    pixel-level ground truth masks.

    Args:
        root_dir: Path to the MVTec dataset root directory.
        category: One of the 15 MVTec categories.
        split: 'train', 'test', or 'val' (val uses 10% of train).
        transform: Optional custom transform. If None, default transforms are applied.
        mask_transform: Optional custom mask transform.
        img_size: Target image size (default 224).
        val_ratio: Fraction of training data to hold out for validation (default 0.1).
        seed: Random seed for train/val split reproducibility.
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        assert category in MVTEC_CATEGORIES, (
            f"Invalid category '{category}'. Must be one of {MVTEC_CATEGORIES}"
        )
        assert split in ("train", "test", "val"), (
            f"Invalid split '{split}'. Must be 'train', 'test', or 'val'."
        )

        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.img_size = img_size
        self.val_ratio = val_ratio
        self.seed = seed

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()

        if mask_transform is not None:
            self.mask_transform = mask_transform
        else:
            self.mask_transform = self._default_mask_transform()

        # Build file lists
        self.image_paths: List[str] = []
        self.mask_paths: List[Optional[str]] = []
        self.labels: List[int] = []
        self.defect_types: List[str] = []

        self._load_dataset()

        logger.info(
            f"MVTecDataset initialized: category={category}, split={split}, "
            f"images={len(self)}, img_size={img_size}"
        )

    def _default_transform(self) -> transforms.Compose:
        """Build default transforms based on split type."""
        if self.split in ("train", "val"):
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:  # test
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def _default_mask_transform(self) -> transforms.Compose:
        """Build default mask transforms (resize only, no normalization)."""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_dataset(self):
        """Scan directory structure and populate file lists."""
        category_dir = self.root_dir / self.category

        if self.split in ("train", "val"):
            self._load_train_val(category_dir)
        else:
            self._load_test(category_dir)

    def _load_train_val(self, category_dir: Path):
        """Load training data (only 'good' images) and optionally split for validation."""
        train_dir = category_dir / "train" / "good"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {train_dir}. "
                f"Please download MVTec-AD dataset first."
            )

        # Collect all good training images
        all_images = sorted([
            str(p) for p in train_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ])

        if len(all_images) == 0:
            raise RuntimeError(f"No images found in {train_dir}")

        # Split into train and val
        rng = random.Random(self.seed)
        indices = list(range(len(all_images)))
        rng.shuffle(indices)

        val_count = max(1, int(len(all_images) * self.val_ratio))

        if self.split == "val":
            selected_indices = indices[:val_count]
        else:  # train
            selected_indices = indices[val_count:]

        for idx in sorted(selected_indices):
            self.image_paths.append(all_images[idx])
            self.mask_paths.append(None)  # No mask for training
            self.labels.append(0)
            self.defect_types.append("good")

    def _load_test(self, category_dir: Path):
        """Load test data with defect masks."""
        test_dir = category_dir / "test"
        gt_dir = category_dir / "ground_truth"

        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Iterate over each defect type subdirectory
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            is_normal = defect_type == "good"

            for img_path in sorted(defect_dir.iterdir()):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                    continue

                self.image_paths.append(str(img_path))
                self.defect_types.append(defect_type)
                self.labels.append(0 if is_normal else 1)

                # Find corresponding ground truth mask
                if is_normal:
                    self.mask_paths.append(None)
                else:
                    # MVTec mask naming convention: ground_truth/<defect>/<img_name>_mask.png
                    mask_name = img_path.stem + "_mask" + ".png"
                    mask_path = gt_dir / defect_type / mask_name
                    if mask_path.exists():
                        self.mask_paths.append(str(mask_path))
                    else:
                        # Try without _mask suffix
                        mask_path_alt = gt_dir / defect_type / img_path.name
                        if mask_path_alt.exists():
                            self.mask_paths.append(str(mask_path_alt))
                        else:
                            logger.warning(f"Mask not found for {img_path}, using empty mask.")
                            self.mask_paths.append(None)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """
        Returns:
            image: (3, H, W) normalized tensor
            mask: (1, H, W) binary mask (0=normal, 1=defect). All zeros for normal.
            label: 0=normal, 1=anomalous
            defect_type: string label ('good' or defect class name)
            image_path: original file path for traceability
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        defect_type = self.defect_types[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Load or create mask
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            # Binarize: threshold at 0.5
            mask = (mask > 0.5).float()
        else:
            # Normal sample — all-zero mask
            mask = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        return image, mask, label, defect_type, image_path

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with total images, normal/defect counts, defect type
            breakdown, and class imbalance ratio.
        """
        total = len(self)
        normal_count = sum(1 for l in self.labels if l == 0)
        defect_count = total - normal_count

        # Defect type breakdown
        defect_breakdown: Dict[str, int] = {}
        for dt in self.defect_types:
            defect_breakdown[dt] = defect_breakdown.get(dt, 0) + 1

        imbalance_ratio = normal_count / max(defect_count, 1)

        stats = {
            "category": self.category,
            "split": self.split,
            "total_images": total,
            "normal_count": normal_count,
            "defect_count": defect_count,
            "defect_type_breakdown": defect_breakdown,
            "class_imbalance_ratio": round(imbalance_ratio, 2),
        }

        logger.info(f"Dataset statistics: {stats}")
        return stats


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Factory function to create a DataLoader with optimized settings.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer (set False for MPS).
        drop_last: Drop last incomplete batch.

    Returns:
        DataLoader instance.
    """
    # On Apple Silicon (MPS), pin_memory should be False
    # as MPS uses unified memory architecture
    if not torch.cuda.is_available():
        pin_memory = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
