"""
VisionGuard-AD — Anomaly Map Generator
=======================================

Post-processing utilities for anomaly heatmaps:
- Normalization (minmax, zscore, percentile)
- Gaussian smoothing
- Thresholding to binary mask
- Visualization overlays and comparison grids
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger("visionguard.anomaly_map")

# ImageNet denormalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class AnomalyMapGenerator:
    """
    Post-processes raw anomaly maps into visualizations and binary masks.

    Provides normalization, smoothing, thresholding, overlay generation,
    and side-by-side comparison grids for evaluation and demos.
    """

    def normalize(self, anomaly_map: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize anomaly scores to [0, 1].

        Args:
            anomaly_map: (H, W) float array of raw anomaly scores.
            method: 'minmax', 'zscore', or 'percentile'.

        Returns:
            (H, W) float array normalized to [0, 1].
        """
        amap = anomaly_map.astype(np.float64).copy()

        if method == "minmax":
            vmin, vmax = amap.min(), amap.max()
            if vmax - vmin > 1e-8:
                amap = (amap - vmin) / (vmax - vmin)
            else:
                amap = np.zeros_like(amap)

        elif method == "zscore":
            mean, std = amap.mean(), amap.std()
            if std > 1e-8:
                amap = (amap - mean) / std
            amap = np.clip(amap, 0, 1)

        elif method == "percentile":
            p1 = np.percentile(amap, 1)
            p99 = np.percentile(amap, 99)
            amap = np.clip(amap, p1, p99)
            vmin, vmax = amap.min(), amap.max()
            if vmax - vmin > 1e-8:
                amap = (amap - vmin) / (vmax - vmin)
            else:
                amap = np.zeros_like(amap)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return amap.astype(np.float32)

    def smooth(self, anomaly_map: np.ndarray, sigma: float = 4.0) -> np.ndarray:
        """Apply Gaussian filter for spatial smoothing."""
        return gaussian_filter(anomaly_map, sigma=sigma)

    def threshold_map(self, anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
        """
        Create binary mask: pixels above threshold = defect.

        Args:
            anomaly_map: (H, W) float array, should be normalized to [0, 1].
            threshold: Decision threshold in [0, 1].

        Returns:
            (H, W) uint8 binary mask (0 or 255).
        """
        binary = (anomaly_map > threshold).astype(np.uint8) * 255
        return binary

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: float = 0.4,
        threshold: Optional[float] = None,
        score: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create visualization overlay of anomaly heatmap on original image.

        Steps:
            1. Convert anomaly_map to heatmap using COLORMAP_JET
            2. Blend with original image
            3. Draw contours around defect regions (if threshold given)
            4. Add score text on image

        Args:
            original_image: (H, W, 3) BGR or RGB uint8 image.
            anomaly_map: (H, W) float array (normalized to [0, 1]).
            alpha: Blending factor for overlay.
            threshold: If given, draw contours around regions above threshold.
            score: If given, display anomaly score text.

        Returns:
            (H, W, 3) BGR uint8 image ready for display/save.
        """
        H, W = original_image.shape[:2]

        # Resize anomaly map to match image size
        if anomaly_map.shape != (H, W):
            anomaly_map = cv2.resize(anomaly_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Convert to heatmap
        heatmap_uint8 = (anomaly_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Ensure original is BGR uint8
        if original_image.dtype != np.uint8:
            orig = (original_image * 255).astype(np.uint8)
        else:
            orig = original_image.copy()

        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

        # Blend
        overlay = cv2.addWeighted(orig, 1 - alpha, heatmap_color, alpha, 0)

        # Draw contours if threshold is provided
        if threshold is not None:
            binary = self.threshold_map(anomaly_map, threshold)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Add score text
        if score is not None:
            is_defective = score > (threshold or 0.5)
            label = f"DEFECT ({score:.3f})" if is_defective else f"NORMAL ({score:.3f})"
            color = (0, 0, 255) if is_defective else (0, 255, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, H / 500)
            thickness = max(1, int(H / 300))

            # Background rectangle for text
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(overlay, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
            cv2.putText(overlay, label, (10, 10 + th), font, font_scale, color, thickness)

        return overlay

    def create_comparison_grid(
        self,
        original: np.ndarray,
        anomaly_map: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        prediction_mask: Optional[np.ndarray] = None,
        score: Optional[float] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Create a 2x2 grid showing:
            Top-left:     Original image
            Top-right:    Ground truth mask
            Bottom-left:  Anomaly heatmap overlay
            Bottom-right: Predicted binary mask

        Args:
            original: (H, W, 3) uint8 image.
            anomaly_map: (H, W) float normalized anomaly map.
            gt_mask: (H, W) binary ground truth mask (optional).
            prediction_mask: (H, W) binary predicted mask (optional).
            score: Anomaly score for display.
            threshold: Threshold for generating prediction mask if not given.

        Returns:
            (2*H, 2*W, 3) uint8 combined image.
        """
        H, W = original.shape[:2]

        # Ensure original is uint8 BGR
        if original.dtype != np.uint8:
            orig = (original * 255).astype(np.uint8)
        else:
            orig = original.copy()

        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

        # Top-left: original
        tl = orig.copy()
        cv2.putText(tl, "Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Top-right: ground truth mask
        if gt_mask is not None:
            if gt_mask.shape != (H, W):
                gt_mask = cv2.resize(gt_mask.astype(np.float32), (W, H))
            tr = cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            tr = np.zeros_like(orig)
        cv2.putText(tr, "Ground Truth", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bottom-left: heatmap overlay
        bl = self.overlay_on_image(orig, anomaly_map, alpha=0.4, score=score, threshold=threshold)
        cv2.putText(bl, "Heatmap", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bottom-right: predicted mask
        if prediction_mask is None:
            prediction_mask = self.threshold_map(anomaly_map, threshold)
        if prediction_mask.shape != (H, W):
            prediction_mask = cv2.resize(prediction_mask, (W, H))
        br = cv2.cvtColor(prediction_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(br, "Prediction", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Assemble grid
        top_row = np.hstack([tl, tr])
        bot_row = np.hstack([bl, br])
        grid = np.vstack([top_row, bot_row])

        return grid

    @staticmethod
    def denormalize_image(tensor):
        """
        Convert a normalized image tensor back to uint8 numpy array.

        Args:
            tensor: (3, H, W) or (H, W, 3) tensor normalized with ImageNet stats.

        Returns:
            (H, W, 3) uint8 BGR numpy array.
        """
        if hasattr(tensor, 'numpy'):
            if tensor.dim() == 3 and tensor.shape[0] == 3:
                img = tensor.permute(1, 2, 0).cpu().numpy()
            else:
                img = tensor.cpu().numpy()
        else:
            img = tensor

        # Denormalize
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # RGB → BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
