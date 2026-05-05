"""
VisionGuard-AD — Incremental Memory Bank Update
==================================================
Enables production-grade model adaptation by merging new normal samples
into an existing PatchCore memory bank without full retraining.

Usage:
    python update_memory_bank.py --model_path ./outputs_wrn/carpet/patchcore_memory_bank.pt \
        --new_images ./data/mvtec/carpet/train/good/ --backbone wide_resnet50 \
        --output_path ./outputs_wrn/carpet/patchcore_memory_bank_updated.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from models.patchcore.patchcore import PatchCore
from models.patchcore.coreset_sampler import CoresetSampler
from models.backbones.feature_extractor import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visionguard.update")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_images_as_tensors(image_dir: str, limit: int = None) -> List[torch.Tensor]:
    """Load images from directory and convert to tensors."""
    img_dir = Path(image_dir)
    tensors = []
    
    files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    if limit:
        files = files[:limit]
    
    for f in files:
        img = Image.open(f).convert("RGB")
        tensors.append(TRANSFORM(img))
    
    logger.info(f"Loaded {len(tensors)} images from {image_dir}")
    return tensors


def apply_brightness_to_images(image_dir: str, factor: float, limit: int = None) -> List[torch.Tensor]:
    """Load images, apply brightness shift, then convert to tensors."""
    img_dir = Path(image_dir)
    tensors = []
    
    files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    if limit:
        files = files[:limit]
    
    for f in files:
        img = cv2.imread(str(f))
        brightened = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(brightened, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensors.append(TRANSFORM(pil_img))
    
    return tensors


def extract_embeddings(model: PatchCore, tensors: List[torch.Tensor]) -> torch.Tensor:
    """Extract L2-normalized patch embeddings from image tensors."""
    all_emb = []
    
    with torch.no_grad():
        for tensor in tensors:
            img = tensor.unsqueeze(0).to(model.device)
            features = model.feature_extractor(img)
            embeddings, _ = model.patch_embedding(features, return_spatial_shape=True)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_emb.append(embeddings.cpu())
    
    return torch.cat(all_emb, dim=0)


def update_memory_bank(model: PatchCore, new_embeddings: torch.Tensor,
                       coreset_ratio: float = 0.1) -> torch.Tensor:
    """
    Merge new embeddings with existing memory bank and re-run coreset sampling.
    
    Steps:
        1. Concatenate existing memory bank with new embeddings
        2. Re-run greedy k-center coreset on the combined set
        3. Return updated memory bank
    """
    old_bank = model.memory_bank.cpu()
    logger.info(f"Old memory bank: {old_bank.shape[0]} patches")
    logger.info(f"New embeddings:  {new_embeddings.shape[0]} patches")
    
    # Merge
    combined = torch.cat([old_bank, new_embeddings], dim=0)
    logger.info(f"Combined set:    {combined.shape[0]} patches")
    
    # Re-run coreset
    sampler = CoresetSampler(ratio=coreset_ratio, device="cpu")
    updated_bank = sampler.run(combined)
    
    logger.info(f"Updated bank:    {updated_bank.shape[0]} patches (after coreset)")
    return updated_bank


def evaluate_on_images(model: PatchCore, tensors: List[torch.Tensor],
                       labels: List[int]) -> float:
    """Compute image AUROC on given tensors."""
    from sklearn.metrics import roc_auc_score
    scores = []
    for t in tensors:
        score, _ = model.predict(t.to(model.device))
        scores.append(float(score) if not isinstance(score, (list, tuple)) else float(score[0]))
    
    if len(set(labels)) < 2:
        return 0.0
    return roc_auc_score(labels, scores) * 100


def main():
    parser = argparse.ArgumentParser(description="Incremental memory bank update")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backbone", default="wide_resnet50")
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--data_root", default="./data/mvtec")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_new_images", type=int, default=20)
    parser.add_argument("--brightness_factor", type=float, default=0.6)
    parser.add_argument("--output_dir", default="./outputs/incremental")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading PatchCore model...")
    model = PatchCore(backbone=args.backbone, device=args.device)
    model.load(args.model_path)
    
    train_dir = Path(args.data_root) / args.category / "train" / "good"
    test_dir = Path(args.data_root) / args.category / "test"
    
    # Load standard test set
    logger.info("Loading standard test set...")
    std_test_tensors = []
    std_test_labels = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_normal = defect_dir.name == "good"
        for img_path in sorted(defect_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img = Image.open(img_path).convert("RGB")
                std_test_tensors.append(TRANSFORM(img))
                std_test_labels.append(0 if is_normal else 1)
    
    # Create brightness-shifted test set
    logger.info(f"Creating brightness-shifted test set (factor={args.brightness_factor})...")
    bright_test_tensors = []
    bright_test_labels = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_normal = defect_dir.name == "good"
        for img_path in sorted(defect_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                img = cv2.imread(str(img_path))
                dark = np.clip(img.astype(np.float32) * args.brightness_factor, 0, 255).astype(np.uint8)
                dark_rgb = cv2.cvtColor(dark, cv2.COLOR_BGR2RGB)
                bright_test_tensors.append(TRANSFORM(Image.fromarray(dark_rgb)))
                bright_test_labels.append(0 if is_normal else 1)
    
    # Scenario 1: Original model on standard test
    logger.info("Scenario 1: Original model → standard test")
    auroc_original = evaluate_on_images(model, std_test_tensors, std_test_labels)
    logger.info(f"  AUROC = {auroc_original:.2f}%")
    
    # Scenario 2: Original model on brightness-shifted test (NO update)
    logger.info("Scenario 2: Original model → brightness-shifted test (no update)")
    auroc_bright_no_update = evaluate_on_images(model, bright_test_tensors, bright_test_labels)
    logger.info(f"  AUROC = {auroc_bright_no_update:.2f}%")
    
    # Scenario 3: Update model with brightness-shifted normal images
    logger.info(f"Extracting features from {args.num_new_images} brightness-shifted normal images...")
    bright_train_tensors = apply_brightness_to_images(
        str(train_dir), args.brightness_factor, limit=args.num_new_images
    )
    new_embeddings = extract_embeddings(model, bright_train_tensors)
    
    # Update memory bank
    logger.info("Updating memory bank...")
    t0 = time.time()
    updated_bank = update_memory_bank(model, new_embeddings)
    update_time = time.time() - t0
    
    # Replace memory bank
    model.memory_bank = updated_bank.to(model.device)
    
    # Scenario 3a: Updated model on standard test (should NOT drop)
    logger.info("Scenario 3: Updated model → standard test")
    auroc_std_updated = evaluate_on_images(model, std_test_tensors, std_test_labels)
    logger.info(f"  AUROC = {auroc_std_updated:.2f}%")
    
    # Scenario 3b: Updated model on brightness-shifted test (should IMPROVE)
    logger.info("Scenario 4: Updated model → brightness-shifted test")
    auroc_bright_updated = evaluate_on_images(model, bright_test_tensors, bright_test_labels)
    logger.info(f"  AUROC = {auroc_bright_updated:.2f}%")
    
    # Save updated model
    save_path = output_dir / "patchcore_memory_bank_updated.pt"
    model.save(str(save_path))
    
    # Print results
    results = {
        "original_standard": auroc_original,
        "original_brightness_shifted": auroc_bright_no_update,
        "updated_standard": auroc_std_updated,
        "updated_brightness_shifted": auroc_bright_updated,
        "update_time_s": round(update_time, 1),
        "new_images_added": args.num_new_images,
        "brightness_factor": args.brightness_factor,
    }
    
    with open(output_dir / "incremental_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("  INCREMENTAL MEMORY BANK UPDATE RESULTS")
    print("="*70)
    print(f"  {'Scenario':<48} {'AUROC':>8}")
    print("-"*58)
    print(f"  {'Original model → standard test':<48} {auroc_original:>7.2f}%")
    print(f"  {'Original model → brightness-shifted test':<48} {auroc_bright_no_update:>7.2f}%")
    print(f"  {'Updated model → standard test':<48} {auroc_std_updated:>7.2f}%")
    print(f"  {'Updated model → brightness-shifted test':<48} {auroc_bright_updated:>7.2f}%")
    print(f"\n  Update time: {update_time:.1f}s | New images: {args.num_new_images}")
    print("="*70)
    
    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
