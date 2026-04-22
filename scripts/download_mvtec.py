#!/usr/bin/env python3
"""
VisionGuard-AD — Download MVTec-AD
====================================
Three methods:
  1. HuggingFace (needs HF_TOKEN if rate-limited)
  2. Direct URL (official MVTec mirror — fastest)
  3. Synthetic fallback (for testing pipeline)

Usage:
    python scripts/download_mvtec.py                                  # Default: try HF, fallback synthetic
    python scripts/download_mvtec.py --method direct                  # Download from official URL
    python scripts/download_mvtec.py --method huggingface --token YOUR_TOKEN
    python scripts/download_mvtec.py --method synthetic               # Quick synthetic for testing
    python scripts/download_mvtec.py --category carpet bottle         # Specific categories only
"""

import argparse
import os
import sys
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Official MVTec-AD download URL (4.9 GB)
MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"


def download_direct(output_dir, categories=None):
    """Download MVTec-AD from the official mirror URL."""
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, "mvtec_anomaly_detection.tar.xz")

    if not os.path.exists(archive_path):
        print("=" * 60)
        print("  Downloading MVTec-AD (4.9 GB) from official mirror...")
        print("=" * 60)
        try:
            subprocess.run(
                ["wget", "-c", "--progress=bar:force", "-O", archive_path, MVTEC_URL],
                check=True,
            )
        except FileNotFoundError:
            # wget not available, try curl
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", archive_path, MVTEC_URL],
                check=True,
            )
    else:
        print(f"  Archive already exists: {archive_path}")

    # Extract
    print("\n  Extracting archive...")
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(path=output_dir)
    print("  ✅ Extraction complete")

    _verify(output_dir, categories or MVTEC_CATEGORIES)


def download_huggingface(output_dir, categories=None, token=None):
    """Download from HuggingFace using datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q datasets")
        from datasets import load_dataset

    cats = categories or MVTEC_CATEGORIES
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token

    print("=" * 60)
    print("  MVTec-AD Download via HuggingFace")
    print("=" * 60)

    try:
        print("  Loading dataset from Voxel51/mvtec-ad...")
        ds = load_dataset("Voxel51/mvtec-ad", **kwargs)
    except Exception as e:
        print(f"  ❌ HuggingFace download failed: {e}")
        print("  💡 Tips:")
        print("     1. Get a free token at https://huggingface.co/settings/tokens")
        print("     2. Run: python scripts/download_mvtec.py --method huggingface --token YOUR_TOKEN")
        print("     3. Or use: python scripts/download_mvtec.py --method direct")
        print("     4. Or use synthetic: python scripts/download_mvtec.py --method synthetic")
        return False

    # Process and save
    for split_name in ds:
        print(f"\n  Processing {split_name} split ({len(ds[split_name])} samples)...")
        for i, sample in enumerate(ds[split_name]):
            _save_hf_sample(sample, output_dir, i)
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(ds[split_name])}")

    _verify(output_dir, cats)
    return True


def _save_hf_sample(sample, output_dir, idx):
    """Save a single HuggingFace sample to MVTec directory structure."""
    img = sample.get("image")
    if img is None or not isinstance(img, Image.Image):
        return

    # Determine category, split, and defect type from metadata
    category = sample.get("category", sample.get("label", "unknown"))
    split = sample.get("split", "test")
    defect_type = sample.get("defect_type", sample.get("label_name", "good"))
    is_good = sample.get("is_good", defect_type == "good" or defect_type == "normal")

    if is_good or defect_type in ("good", "normal", "ok"):
        defect_type = "good"

    # Save image
    if split == "train" or defect_type == "good":
        save_dir = Path(output_dir) / category / split / "good"
    else:
        save_dir = Path(output_dir) / category / split / defect_type

    save_dir.mkdir(parents=True, exist_ok=True)
    img.save(save_dir / f"{idx:04d}.png")

    # Save mask
    mask = sample.get("mask", sample.get("segmentation_mask", None))
    if mask is not None and defect_type != "good":
        if isinstance(mask, Image.Image):
            mask_img = mask.convert("L")
        else:
            mask_arr = np.array(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            if mask_arr.max() <= 1:
                mask_arr = (mask_arr * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_arr)

        gt_dir = Path(output_dir) / category / "ground_truth" / defect_type
        gt_dir.mkdir(parents=True, exist_ok=True)
        mask_img.save(gt_dir / f"{idx:04d}_mask.png")


def generate_synthetic(output_dir, categories=None, n_train=30, n_test_good=10, n_test_defect=10):
    """Generate synthetic MVTec-like data for pipeline testing."""
    cats = categories or MVTEC_CATEGORIES
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Generating Synthetic MVTec-AD Data")
    print("=" * 60)

    size = 256
    for category in cats:
        cat_dir = Path(output_dir) / category
        if (cat_dir / "train" / "good").exists() and len(list((cat_dir / "train" / "good").glob("*.png"))) >= n_train:
            print(f"  ⏭️  {category}: already exists")
            continue

        print(f"  Creating: {category}...")

        # Unique base color per category for variety
        np.random.seed(hash(category) % 2**31)
        base = np.random.randint(80, 220, 3)

        # Train
        train_dir = cat_dir / "train" / "good"
        train_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_train):
            img = _make_texture(size, base)
            img.save(train_dir / f"{i:03d}.png")

        # Test/good
        test_good = cat_dir / "test" / "good"
        test_good.mkdir(parents=True, exist_ok=True)
        for i in range(n_test_good):
            img = _make_texture(size, base)
            img.save(test_good / f"{i:03d}.png")

        # Test/scratch + masks
        test_def = cat_dir / "test" / "scratch"
        gt_dir = cat_dir / "ground_truth" / "scratch"
        test_def.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_test_defect):
            img, mask = _make_defect(size, base)
            img.save(test_def / f"{i:03d}.png")
            mask.save(gt_dir / f"{i:03d}_mask.png")

        # Test/hole + masks
        test_hole = cat_dir / "test" / "hole"
        gt_hole = cat_dir / "ground_truth" / "hole"
        test_hole.mkdir(parents=True, exist_ok=True)
        gt_hole.mkdir(parents=True, exist_ok=True)
        for i in range(n_test_defect // 2):
            img, mask = _make_defect(size, base, defect_type="circle")
            img.save(test_hole / f"{i:03d}.png")
            mask.save(gt_hole / f"{i:03d}_mask.png")

        print(f"    ✅ train={n_train}, test_good={n_test_good}, test_defect={n_test_defect + n_test_defect//2}")

    _verify(output_dir, cats)


def _make_texture(size, base_color):
    """Generate normal texture image."""
    noise = np.random.normal(0, 12, (size, size, 3))
    arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(2))


def _make_defect(size, base_color, defect_type="scratch"):
    """Generate defective image with ground truth mask."""
    img = _make_texture(size, base_color)
    mask = Image.new("L", (size, size), 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    if defect_type == "scratch":
        x1, y1 = np.random.randint(40, 100), np.random.randint(40, 100)
        x2, y2 = x1 + np.random.randint(40, 120), y1 + np.random.randint(40, 120)
        w = np.random.randint(3, 8)
        draw_img.line([(x1, y1), (x2, y2)], fill=(30, 30, 30), width=w)
        draw_mask.line([(x1, y1), (x2, y2)], fill=255, width=w + 4)
    else:
        cx, cy = np.random.randint(60, 196), np.random.randint(60, 196)
        r = np.random.randint(10, 30)
        draw_img.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(20, 20, 20))
        draw_mask.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)

    return img, mask


def _verify(output_dir, categories):
    """Verify downloaded data."""
    print(f"\n{'='*60}")
    print("  VERIFICATION")
    print(f"{'='*60}")

    total_train, total_test = 0, 0
    for cat in categories:
        cat_dir = Path(output_dir) / cat
        if not cat_dir.exists():
            print(f"  ❌ {cat}: NOT FOUND")
            continue
        tc = len(list((cat_dir / "train").rglob("*.png"))) if (cat_dir / "train").exists() else 0
        te = len(list((cat_dir / "test").rglob("*.png"))) if (cat_dir / "test").exists() else 0
        gt = len(list((cat_dir / "ground_truth").rglob("*.png"))) if (cat_dir / "ground_truth").exists() else 0
        total_train += tc
        total_test += te
        print(f"  ✅ {cat:15s}: train={tc:4d}, test={te:4d}, masks={gt:4d}")

    print(f"\n  Total: {total_train} train + {total_test} test images")


def main():
    parser = argparse.ArgumentParser(description="Download MVTec-AD Dataset")
    parser.add_argument("--output", default="./data/mvtec")
    parser.add_argument("--category", nargs="*", default=None)
    parser.add_argument("--method", default="synthetic",
                        choices=["direct", "huggingface", "synthetic"],
                        help="direct=official URL (4.9GB), huggingface=HF Hub, synthetic=fake data")
    parser.add_argument("--token", default=None, help="HuggingFace token for rate-limited downloads")
    args = parser.parse_args()

    if args.method == "direct":
        download_direct(args.output, args.category)
    elif args.method == "huggingface":
        download_huggingface(args.output, args.category, args.token)
    else:
        generate_synthetic(args.output, args.category)


if __name__ == "__main__":
    main()
