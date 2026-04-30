"""
VisionGuard-AD — Streamlit App Utilities
==========================================
Helper functions for the Streamlit dashboard.
"""

import os
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow
from anomaly_map.anomaly_map_generator import AnomalyMapGenerator

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform(img_size=224):
    """Standard inference transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_model_files(output_dir="./outputs"):
    """Scan output directory for saved model files."""
    models = {}
    output_path = Path(output_dir)
    if not output_path.exists():
        return models

    for cat_dir in output_path.iterdir():
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for f in cat_dir.iterdir():
            if f.suffix == ".pt":
                method = "patchcore" if "patchcore" in f.name else "fastflow"
                key = f"{category} ({method})"
                models[key] = {"path": str(f), "category": category, "method": method}
    return models


def load_model_cached(model_path, method, backbone="wide_resnet50", device="auto", img_size=224):
    """Load a model (designed for Streamlit caching)."""
    dev = get_device(device)
    if method == "patchcore":
        model = PatchCore(backbone=backbone, device=str(dev))
        model.load(model_path)
    else:
        model = FastFlow(backbone=backbone, device=str(dev), img_size=img_size)
        model.load(model_path)
        model.eval()
    return model, dev


def run_inference(model, method, image_pil, device, img_size=224):
    """Run inference on a PIL image. Returns (score, anomaly_map_normalized)."""
    transform = get_inference_transform(img_size)
    tensor = transform(image_pil).unsqueeze(0).to(device)

    amap_gen = AnomalyMapGenerator()

    with torch.no_grad():
        if method == "patchcore":
            score, amap = model.predict(tensor)
        else:
            score, amap = model.get_anomaly_map(tensor)

    amap_norm = amap_gen.normalize(amap, method="minmax")
    return score, amap_norm


def create_overlay_pil(image_pil, anomaly_map, threshold=0.5, score=None):
    """Create overlay visualization and return as PIL Image."""
    amap_gen = AnomalyMapGenerator()

    # Convert PIL to OpenCV BGR
    img_np = np.array(image_pil)
    if len(img_np.shape) == 2:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    overlay = amap_gen.overlay_on_image(img_bgr, anomaly_map, alpha=0.4, threshold=threshold, score=score)

    # Convert back to RGB PIL
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)


def create_binary_mask_pil(anomaly_map, threshold=0.5):
    """Create binary mask and return as PIL Image."""
    amap_gen = AnomalyMapGenerator()
    binary = amap_gen.threshold_map(anomaly_map, threshold)
    return Image.fromarray(binary)
