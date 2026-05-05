"""
VisionGuard-AD — Robustness Study Under Real-World Degradation
================================================================
Tests trained PatchCore models against programmatic image degradations
that simulate real factory conditions: noise, blur, lighting shifts, compression.

Usage:
    python robustness_study.py --model_path ./outputs_wrn/carpet/patchcore_memory_bank.pt \
        --backbone wide_resnet50 --category carpet --data_root ./data/mvtec
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from data.mvtec_dataset import MVTecDataset
from models.patchcore.patchcore import PatchCore
from models.backbones.feature_extractor import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visionguard.robustness")

# ─── Degradation Functions ───

def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise with given standard deviation."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply horizontal motion blur with given kernel size."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_brightness_shift(image: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by factor (simulate lighting changes)."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Compress and decompress via JPEG at given quality level."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded

def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma."""
    ksize = int(sigma * 6) | 1  # Ensure odd kernel size
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def apply_random_shadow(image: np.ndarray, coverage: float = 0.3) -> np.ndarray:
    """Add a dark polygonal shadow over ~coverage fraction of image."""
    h, w = image.shape[:2]
    np.random.seed(42)  # Reproducible shadows
    num_points = 4
    pts = np.array([
        [np.random.randint(0, w), np.random.randint(0, h)]
        for _ in range(num_points)
    ], dtype=np.int32)
    # Scale points to cover ~coverage of image
    center = np.array([w * 0.3, h * 0.3])
    pts = (pts * coverage + center * (1 - coverage)).astype(np.int32)
    
    mask = np.ones_like(image, dtype=np.float32)
    cv2.fillPoly(mask, [pts], (0.3, 0.3, 0.3))  # Darken to 30%
    result = np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    return result


# Full degradation suite
DEGRADATIONS = {
    "None (baseline)":       {"fn": None, "params": [None], "labels": ["—"]},
    "Gaussian Noise":        {"fn": add_gaussian_noise, "params": [15, 25, 40], "labels": ["σ=15", "σ=25", "σ=40"]},
    "Motion Blur":           {"fn": apply_motion_blur, "params": [3, 5, 9], "labels": ["3×3", "5×5", "9×9"]},
    "Brightness Shift":      {"fn": apply_brightness_shift, "params": [0.5, 0.7, 1.3, 1.5], "labels": ["×0.5", "×0.7", "×1.3", "×1.5"]},
    "JPEG Compression":      {"fn": apply_jpeg_compression, "params": [80, 50, 20], "labels": ["Q=80", "Q=50", "Q=20"]},
    "Gaussian Blur":         {"fn": apply_gaussian_blur, "params": [1, 2, 4], "labels": ["σ=1", "σ=2", "σ=4"]},
    "Random Shadow":         {"fn": apply_random_shadow, "params": [0.3], "labels": ["30% coverage"]},
}


def load_test_images_raw(data_root: str, category: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load raw test images (before transforms) and their labels."""
    cat_dir = Path(data_root) / category / "test"
    images, labels = [], []
    
    for defect_dir in sorted(cat_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_normal = defect_dir.name == "good"
        for img_path in sorted(defect_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(0 if is_normal else 1)
    
    return images, labels


def image_to_tensor(image: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """Convert BGR numpy image to normalized tensor."""
    from torchvision import transforms
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img)


def run_study(model: PatchCore, images: List[np.ndarray], labels: List[int],
              device: torch.device) -> List[Dict]:
    """Run all degradations and compute metrics."""
    results = []
    baseline_auroc = None
    
    for deg_name, deg_config in DEGRADATIONS.items():
        fn = deg_config["fn"]
        params = deg_config["params"]
        param_labels = deg_config["labels"]
        
        for param, plabel in zip(params, param_labels):
            # Apply degradation to all images
            if fn is None:
                degraded_images = images  # Baseline
            else:
                degraded_images = [fn(img.copy(), param) for img in images]
            
            # Run inference
            scores = []
            for img in degraded_images:
                tensor = image_to_tensor(img).to(device)
                score, _ = model.predict(tensor)
                scores.append(float(score) if not isinstance(score, (list, tuple)) else float(score[0]))
            
            # Compute AUROC
            if len(set(labels)) < 2:
                auroc = 0.0
            else:
                auroc = roc_auc_score(labels, scores) * 100
            
            if baseline_auroc is None:
                baseline_auroc = auroc
            
            drop = auroc - baseline_auroc
            
            row = {
                "degradation": deg_name,
                "severity": plabel,
                "image_auroc": round(auroc, 2),
                "auroc_drop": round(drop, 2),
                "auroc_drop_pct": f"{drop:+.1f}%",
            }
            results.append(row)
            logger.info(f"{deg_name} [{plabel}]: AUROC={auroc:.2f}% (drop={drop:+.2f}%)")
    
    return results


def plot_results(results: List[Dict], save_path: str):
    """Create degradation impact visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Robustness Study — AUROC Under Real-World Degradations", fontsize=16, fontweight="bold")
    
    deg_groups = {}
    for r in results:
        name = r["degradation"]
        if name == "None (baseline)":
            continue
        if name not in deg_groups:
            deg_groups[name] = {"severities": [], "aurocs": []}
        deg_groups[name]["severities"].append(r["severity"])
        deg_groups[name]["aurocs"].append(r["image_auroc"])
    
    baseline = results[0]["image_auroc"]
    colors = ["#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]
    
    for idx, (name, data) in enumerate(deg_groups.items()):
        ax = axes[idx // 3][idx % 3]
        bars = ax.bar(data["severities"], data["aurocs"], color=colors[idx], alpha=0.8)
        ax.axhline(y=baseline, color="red", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.1f}%)")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Image AUROC (%)")
        ax.set_ylim(max(min(data["aurocs"]) - 10, 50), 100)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, data["aurocs"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Robustness plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Robustness study under real-world degradations")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--backbone", default="wide_resnet50")
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--data_root", default="./data/mvtec")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="./outputs/robustness")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading PatchCore model: {args.model_path}")
    model = PatchCore(backbone=args.backbone, device=args.device)
    model.load(args.model_path)
    
    # Load raw test images
    logger.info(f"Loading test images for {args.category}...")
    images, labels = load_test_images_raw(args.data_root, args.category)
    logger.info(f"Loaded {len(images)} test images ({sum(labels)} defective, {len(labels)-sum(labels)} normal)")
    
    # Run study
    results = run_study(model, images, labels, model.device)
    
    # Save results
    with open(output_dir / "robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate table
    print("\n" + "="*80)
    print("  ROBUSTNESS STUDY RESULTS")
    print("="*80)
    print(f"  {'Degradation':<22} {'Severity':<14} {'Image AUROC':<14} {'AUROC Drop':<12}")
    print("-"*62)
    for r in results:
        print(f"  {r['degradation']:<22} {r['severity']:<14} {r['image_auroc']:<14.2f} {r['auroc_drop_pct']:<12}")
    print("="*80)
    
    # Plot
    plot_results(results, str(output_dir / "robustness_chart.png"))
    
    # Save markdown report
    with open(output_dir / "robustness_report.md", "w") as f:
        f.write("# Robustness Study — Real-World Degradation Analysis\n\n")
        f.write(f"Model: PatchCore ({args.backbone}) trained on {args.category}\n\n")
        f.write("| Degradation | Severity | Image AUROC | AUROC Drop |\n")
        f.write("|-------------|----------|:-----------:|:----------:|\n")
        for r in results:
            f.write(f"| {r['degradation']} | {r['severity']} | {r['image_auroc']:.2f}% | {r['auroc_drop_pct']} |\n")
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
