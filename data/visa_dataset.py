"""
VisionGuard-AD — VisA (Visual Anomaly) Dataset
===============================================

PyTorch Dataset for the VisA benchmark (12 categories).
VisA directory structure expected:
    visa/
    ├── candle/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   └── bad/
    │   └── ground_truth/
    │       └── bad/
    ├── capsules/
    │   └── ...
    └── ...

VisA may also come in a CSV-based split format. This implementation
handles both directory-based and CSV-based layouts.
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

logger = logging.getLogger("visionguard.data.visa")

# All 12 VisA categories
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2", "pcb1",
    "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VisADataset(Dataset):
    """
    PyTorch Dataset for the VisA (Visual Anomaly) benchmark.

    Supports train, test, and val splits. Same interface as MVTecDataset
    for drop-in interchangeability.

    Args:
        root_dir: Path to the VisA dataset root directory.
        category: One of the 12 VisA categories.
        split: 'train', 'test', or 'val'.
        transform: Optional custom transform.
        mask_transform: Optional custom mask transform.
        img_size: Target image size (default 224).
        val_ratio: Fraction of training data for validation.
        seed: Random seed for reproducibility.
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
        assert category in VISA_CATEGORIES, (
            f"Invalid VisA category '{category}'. Must be one of {VISA_CATEGORIES}"
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

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()

        if mask_transform is not None:
            self.mask_transform = mask_transform
        else:
            self.mask_transform = self._default_mask_transform()

        self.image_paths: List[str] = []
        self.mask_paths: List[Optional[str]] = []
        self.labels: List[int] = []
        self.defect_types: List[str] = []

        self._load_dataset()

        logger.info(
            f"VisADataset initialized: category={category}, split={split}, "
            f"images={len(self)}, img_size={img_size}"
        )

    def _default_transform(self) -> transforms.Compose:
        """Build default transforms based on split."""
        if self.split in ("train", "val"):
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def _default_mask_transform(self) -> transforms.Compose:
        """Mask transform: resize with nearest interpolation, convert to tensor."""
        return transforms.Compose([
            transforms.Resize(
                (self.img_size, self.img_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def _load_dataset(self):
        """Load dataset from directory structure."""
        category_dir = self.root_dir / self.category

        # Check for CSV-based split first
        csv_path = self.root_dir / "split_csv" / f"{self.category}.csv"
        if csv_path.exists():
            self._load_from_csv(csv_path)
        else:
            # Fall back to directory-based loading
            if self.split in ("train", "val"):
                self._load_train_val(category_dir)
            else:
                self._load_test(category_dir)

    def _load_from_csv(self, csv_path: Path):
        """Load dataset using CSV split file."""
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Map split names
        split_map = {"train": "train", "val": "train", "test": "test"}
        target_split = split_map[self.split]
        df_split = df[df["split"] == target_split].reset_index(drop=True)

        if self.split == "val":
            rng = random.Random(self.seed)
            indices = list(range(len(df_split)))
            rng.shuffle(indices)
            val_count = max(1, int(len(indices) * self.val_ratio))
            selected = sorted(indices[:val_count])
            df_split = df_split.iloc[selected]
        elif self.split == "train":
            rng = random.Random(self.seed)
            indices = list(range(len(df_split)))
            rng.shuffle(indices)
            val_count = max(1, int(len(indices) * self.val_ratio))
            selected = sorted(indices[val_count:])
            df_split = df_split.iloc[selected]

        for _, row in df_split.iterrows():
            img_path = str(self.root_dir / row["image"])
            self.image_paths.append(img_path)

            label = int(row.get("label", 0))
            self.labels.append(label)
            self.defect_types.append("bad" if label == 1 else "good")

            mask_val = row.get("mask", "")
            if label == 1 and isinstance(mask_val, str) and mask_val.strip():
                self.mask_paths.append(str(self.root_dir / mask_val))
            else:
                self.mask_paths.append(None)

    def _load_train_val(self, category_dir: Path):
        """Load training images (good only) with optional val split."""
        train_dir = category_dir / "train" / "good"
        if not train_dir.exists():
            # Try alternate structure
            train_dir = category_dir / "train" / "Normal"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {category_dir / 'train'}. "
                f"Expected subdirectory 'good' or 'Normal'."
            )

        all_images = sorted([
            str(p) for p in train_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".JPG")
        ])

        if len(all_images) == 0:
            raise RuntimeError(f"No images found in {train_dir}")

        rng = random.Random(self.seed)
        indices = list(range(len(all_images)))
        rng.shuffle(indices)
        val_count = max(1, int(len(all_images) * self.val_ratio))

        if self.split == "val":
            selected_indices = indices[:val_count]
        else:
            selected_indices = indices[val_count:]

        for idx in sorted(selected_indices):
            self.image_paths.append(all_images[idx])
            self.mask_paths.append(None)
            self.labels.append(0)
            self.defect_types.append("good")

    def _load_test(self, category_dir: Path):
        """Load test images with ground truth masks."""
        test_dir = category_dir / "test"
        gt_dir = category_dir / "ground_truth"

        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            is_normal = defect_type.lower() in ("good", "normal")

            for img_path in sorted(defect_dir.iterdir()):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".JPG"):
                    continue

                self.image_paths.append(str(img_path))
                self.defect_types.append(defect_type)
                self.labels.append(0 if is_normal else 1)

                if is_normal:
                    self.mask_paths.append(None)
                else:
                    # Try various mask naming conventions
                    mask_candidates = [
                        gt_dir / defect_type / (img_path.stem + "_mask.png"),
                        gt_dir / defect_type / img_path.name,
                        gt_dir / defect_type / (img_path.stem + ".png"),
                    ]
                    mask_found = False
                    for mask_path in mask_candidates:
                        if mask_path.exists():
                            self.mask_paths.append(str(mask_path))
                            mask_found = True
                            break
                    if not mask_found:
                        logger.warning(f"Mask not found for {img_path}")
                        self.mask_paths.append(None)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """
        Returns:
            image: (3, H, W) normalized tensor
            mask: (1, H, W) binary mask
            label: 0=normal, 1=anomalous
            defect_type: string label
            image_path: original file path
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        defect_type = self.defect_types[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        return image, mask, label, defect_type, image_path

    def get_statistics(self) -> Dict[str, Any]:
        """Compute and return dataset statistics."""
        total = len(self)
        normal_count = sum(1 for l in self.labels if l == 0)
        defect_count = total - normal_count

        defect_breakdown: Dict[str, int] = {}
        for dt in self.defect_types:
            defect_breakdown[dt] = defect_breakdown.get(dt, 0) + 1

        return {
            "category": self.category,
            "split": self.split,
            "total_images": total,
            "normal_count": normal_count,
            "defect_count": defect_count,
            "defect_type_breakdown": defect_breakdown,
            "class_imbalance_ratio": round(normal_count / max(defect_count, 1), 2),
        }
