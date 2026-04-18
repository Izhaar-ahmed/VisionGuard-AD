"""
VisionGuard-AD — Inference Script
===================================
Single image, folder batch, or live webcam inference.

Usage:
    python inference.py --method patchcore --model_path ./outputs/carpet/patchcore_memory_bank.pt \
        --input ./test_image.jpg --threshold 0.42 --output ./result.jpg
    python inference.py --input ./test_folder/ --output ./results/
    python inference.py --input 0   # Webcam live demo
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow
from anomaly_map.anomaly_map_generator import AnomalyMapGenerator
from torchvision import transforms

logger = logging.getLogger("visionguard.inference")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _unwrap(val):
    """Unwrap score/amap from list or tensor to scalar/ndarray."""
    if isinstance(val, (list, tuple)):
        val = val[0]
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray) and val.ndim == 0:
        val = float(val)
    return val


def load_model(args, device):
    """Load trained model from checkpoint."""
    if args.method == "patchcore":
        model = PatchCore(backbone=args.backbone, device=str(device), sigma=args.sigma)
        model.load(args.model_path)
    else:
        model = FastFlow(backbone=args.backbone, device=str(device), img_size=args.img_size)
        model.load(args.model_path)
        model.eval()
    return model


def get_transform(img_size=224):
    """Inference-time image transform."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _run_predict(model, method, tensor):
    """Run model prediction and unwrap results to (float, ndarray)."""
    with torch.no_grad():
        if method == "patchcore":
            score, amap = model.predict(tensor)
        else:
            score, amap = model.get_anomaly_map(tensor)

    score = _unwrap(score)
    amap = _unwrap(amap)

    # Ensure score is float
    if not isinstance(score, float):
        score = float(score)

    # Ensure amap is 2D ndarray
    if isinstance(amap, np.ndarray):
        if amap.ndim == 3:
            amap = amap.squeeze(0) if amap.shape[0] == 1 else amap[:, :, 0]
    else:
        amap = np.array(amap, dtype=np.float64)

    return score, amap


def infer_single(model, image_path, args, device, amap_gen):
    """Run inference on a single image file."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.error(f"Cannot read image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = get_transform(args.img_size)
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    start = time.time()
    score, amap = _run_predict(model, args.method, tensor)
    elapsed = time.time() - start

    # Normalize and create overlay
    amap_norm = amap_gen.normalize(amap, method="minmax")
    overlay = amap_gen.overlay_on_image(
        img_bgr, amap_norm, alpha=0.4, threshold=args.threshold, score=score,
    )

    is_defect = score > args.threshold
    status = "DEFECTIVE" if is_defect else "NORMAL"
    logger.info(f"{image_path}: score={score:.4f}, {status} ({elapsed*1000:.1f}ms)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        cv2.imwrite(args.output, overlay)
        logger.info(f"Result saved to {args.output}")
    else:
        cv2.imshow("VisionGuard-AD Inference", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_folder(model, folder_path, args, device, amap_gen):
    """Run inference on all images in a folder."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted([
        str(p) for p in Path(folder_path).iterdir()
        if p.suffix.lower() in exts
    ])
    logger.info(f"Found {len(image_files)} images in {folder_path}")

    out_dir = args.output or os.path.join(folder_path, "results")
    os.makedirs(out_dir, exist_ok=True)

    transform = get_transform(args.img_size)
    results = []

    for img_path in image_files:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)

        score, amap = _run_predict(model, args.method, tensor)

        amap_norm = amap_gen.normalize(amap, method="minmax")
        overlay = amap_gen.overlay_on_image(
            img_bgr, amap_norm, alpha=0.4, threshold=args.threshold, score=score,
        )

        fname = Path(img_path).stem + "_result.png"
        cv2.imwrite(os.path.join(out_dir, fname), overlay)

        status = "DEFECT" if score > args.threshold else "NORMAL"
        results.append({"file": img_path, "score": round(score, 4), "decision": status})
        logger.info(f"  {Path(img_path).name}: {score:.4f} → {status}")

    # Save results summary
    import json
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_dir}")


def infer_webcam(model, args, device, amap_gen):
    """Live webcam inference with anomaly overlay."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    transform = get_transform(args.img_size)
    ema_score = 0.0
    alpha_ema = 0.3  # Exponential moving average smoothing
    frame_count = 0
    fps_start = time.time()

    logger.info("Webcam started. Press 'q' to quit, 's' to save frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(frame_rgb).unsqueeze(0).to(device)

        score, amap = _run_predict(model, args.method, tensor)

        # EMA smoothing
        ema_score = alpha_ema * score + (1 - alpha_ema) * ema_score

        # Create overlay
        amap_norm = amap_gen.normalize(amap, method="minmax")
        amap_resized = cv2.resize(amap_norm, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.applyColorMap((amap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Decision text
        is_defect = ema_score > args.threshold
        label = f"DEFECT DETECTED ({ema_score:.3f})" if is_defect else f"NORMAL ({ema_score:.3f})"
        color = (0, 0, 255) if is_defect else (0, 255, 0)

        cv2.putText(overlay, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count / max(elapsed, 0.001)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Threshold bar
        bar_w = 300
        bar_h = 20
        bar_x, bar_y = 10, overlay.shape[0] - 40
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill = int(min(ema_score / max(args.threshold * 2, 0.01), 1.0) * bar_w)
        bar_color = (0, 0, 255) if is_defect else (0, 200, 0)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
        thresh_x = int(args.threshold / max(args.threshold * 2, 0.01) * bar_w) + bar_x
        cv2.line(overlay, (thresh_x, bar_y - 5), (thresh_x, bar_y + bar_h + 5), (255, 255, 0), 2)

        cv2.imshow("VisionGuard-AD Live", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_path = f"capture_{int(time.time())}.png"
            cv2.imwrite(save_path, overlay)
            logger.info(f"Frame saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="VisionGuard-AD Inference")
    parser.add_argument("--method", type=str, default="patchcore", choices=["patchcore", "fastflow"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image path, folder, or '0' for webcam")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backbone", type=str, default="wide_resnet50")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = get_device(args.device)
    model = load_model(args, device)
    amap_gen = AnomalyMapGenerator()

    input_path = args.input

    if input_path in ("0", "1", "2"):
        infer_webcam(model, args, device, amap_gen)
    elif os.path.isdir(input_path):
        infer_folder(model, input_path, args, device, amap_gen)
    elif os.path.isfile(input_path):
        infer_single(model, input_path, args, device, amap_gen)
    else:
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
