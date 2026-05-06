"""
VisionGuard-AD — Inference API
================================
Clean Python interface for loading models and running inference.
This is the single source of truth for inference logic — the FastAPI
server and Streamlit dashboard should both use this module.

Usage (Python):
    from api.inference_api import load_model, run_inference
    import cv2

    handle = load_model(
        method="patchcore",
        backbone="resnet18",
        category="carpet",
        model_path="./outputs/benchmark/carpet/patchcore_memory_bank.pt",
    )
    image = cv2.imread("test.png")
    result = run_inference(handle, image)
    print(result["image_score"], result["is_anomalous"])
"""

import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow
from anomaly_map.anomaly_map_generator import AnomalyMapGenerator

logger = logging.getLogger("visionguard.api")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_transform(img_size: int = 224) -> transforms.Compose:
    """Standard inference transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _unwrap_score(val) -> float:
    """Unwrap score to Python float."""
    if isinstance(val, (list, tuple)):
        val = val[0]
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().item()
    if isinstance(val, np.ndarray):
        val = float(val.flat[0])
    return float(val)


def _unwrap_amap(val) -> np.ndarray:
    """Unwrap anomaly map to 2D numpy array."""
    if isinstance(val, (list, tuple)):
        val = val[0]
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        while val.ndim > 2:
            val = val.squeeze(0) if val.shape[0] == 1 else val[0]
    return np.asarray(val, dtype=np.float64)


def load_model(
    method: str = "patchcore",
    backbone: str = "resnet18",
    category: str = "carpet",
    model_path: str = "",
    device: str = "auto",
    img_size: int = 224,
    threshold: float = 0.5,
    sigma: float = 4.0,
) -> Dict[str, Any]:
    """
    Load a trained anomaly detection model and return a handle.

    The handle is a plain dict containing all state needed for inference.
    Pass it to ``run_inference()`` to get predictions.

    Args:
        method: 'patchcore' or 'fastflow'.
        backbone: Backbone architecture name (e.g. 'resnet18', 'wide_resnet50').
        category: MVTec category the model was trained on.
        model_path: Path to the saved model file (.pt).
        device: Device string ('auto', 'cpu', 'mps', 'cuda').
        img_size: Expected input image resolution.
        threshold: Decision threshold for is_anomalous flag.
        sigma: Gaussian smoothing sigma for anomaly maps.

    Returns:
        Model handle dict with keys: model, method, backbone, category,
        device, transform, amap_gen, threshold, img_size.
    """
    resolved_device = get_device(device)
    logger.info(f"Loading {method} model: backbone={backbone}, device={resolved_device}")

    if method == "patchcore":
        model = PatchCore(backbone=backbone, device=str(resolved_device), sigma=sigma)
        model.load(model_path)
    elif method == "fastflow":
        model = FastFlow(backbone=backbone, device=str(resolved_device), img_size=img_size)
        model.load(model_path)
        model.eval()
    else:
        raise ValueError(f"Unknown method '{method}'. Supported: patchcore, fastflow")

    handle = {
        "model": model,
        "method": method,
        "backbone": backbone,
        "category": category,
        "device": resolved_device,
        "transform": _get_transform(img_size),
        "amap_gen": AnomalyMapGenerator(),
        "threshold": threshold,
        "img_size": img_size,
    }

    logger.info(f"Model loaded successfully: {method}/{backbone} for '{category}'")
    return handle


def run_inference(
    model_handle: Dict[str, Any],
    image: np.ndarray,
    return_anomaly_map: bool = True,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run anomaly detection on a single image.

    Args:
        model_handle: Handle returned by ``load_model()``.
        image: Input image as numpy array. Accepts:
            - BGR uint8 (H, W, 3) — standard OpenCV format
            - RGB uint8 (H, W, 3)
            - Grayscale uint8 (H, W)
        return_anomaly_map: If True, include the anomaly heatmap in results.
        threshold: Override the default threshold from model_handle.

    Returns:
        Dict with keys:
            - image_score (float): Raw anomaly score
            - is_anomalous (bool): True if score exceeds threshold
            - threshold (float): Decision threshold used
            - anomaly_map (np.ndarray or None): (H, W) normalized anomaly map
            - meta (dict): Inference metadata (backbone, category, device, ms)
    """
    model = model_handle["model"]
    method = model_handle["method"]
    device = model_handle["device"]
    transform = model_handle["transform"]
    amap_gen = model_handle["amap_gen"]
    thresh = threshold if threshold is not None else model_handle["threshold"]

    # Prepare image
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    # Run inference
    t_start = time.perf_counter()
    with torch.no_grad():
        if method == "patchcore":
            raw_score, raw_amap = model.predict(tensor)
        else:
            raw_score, raw_amap = model.get_anomaly_map(tensor)
    inference_ms = (time.perf_counter() - t_start) * 1000

    score = _unwrap_score(raw_score)

    anomaly_map = None
    if return_anomaly_map:
        anomaly_map = _unwrap_amap(raw_amap)
        anomaly_map = amap_gen.normalize(anomaly_map, method="minmax")

    return {
        "image_score": score,
        "is_anomalous": score > thresh,
        "threshold": thresh,
        "anomaly_map": anomaly_map,
        "meta": {
            "method": method,
            "backbone": model_handle["backbone"],
            "category": model_handle["category"],
            "device": str(device),
            "img_size": model_handle["img_size"],
            "inference_ms": round(inference_ms, 2),
        },
    }
