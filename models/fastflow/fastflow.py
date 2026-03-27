"""
VisionGuard-AD — FastFlow Implementation
=========================================
"FastFlow: Unsupervised Anomaly Detection and Localization
 via 2D Normalizing Flows"

Uses a 2D normalizing flow on top of frozen CNN features for
density-based anomaly scoring. Normal patches map to high-likelihood
regions under N(0,I); anomalies map to low-likelihood regions.

Optimized for Apple Silicon M1 (MPS backend).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from models.backbones.feature_extractor import FeatureExtractor, get_device

logger = logging.getLogger("visionguard.models.fastflow")


class AffineCouplingLayer(nn.Module):
    """
    2D Affine Coupling Layer (RealNVP-style).

    Splits the input channel-wise into two halves. One half passes through
    a small CNN (s_net, t_net) that predicts scale and translation for
    the other half.

    Transform: z2 = x2 * exp(s(x1)) + t(x1)
    Log-det-Jacobian: sum of s(x1)

    Args:
        in_channels: Number of input channels.
        hidden_channels: Hidden channels in the coupling CNN.
        kernel_size: Kernel size for the coupling CNN.
        clamp: Clamping value for scale parameter (numerical stability).
        reverse_mask: If True, swap which half is transformed.
    """

    def __init__(self, in_channels, hidden_channels=256, kernel_size=3,
                 clamp=2.0, reverse_mask=False):
        super().__init__()
        self.clamp = clamp
        self.reverse_mask = reverse_mask

        half_channels = in_channels // 2

        # Scale and translation network
        padding = kernel_size // 2
        self.st_net = nn.Sequential(
            nn.Conv2d(half_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, half_channels * 2, kernel_size, padding=padding),
        )

        # Initialize last layer to zero for stable training start
        nn.init.zeros_(self.st_net[-1].weight)
        nn.init.zeros_(self.st_net[-1].bias)

    def forward(self, x):
        """
        Forward pass (data → latent).
        Returns: (z, log_jac_det)
        """
        if self.reverse_mask:
            x1, x2 = x.chunk(2, dim=1)
            x1, x2 = x2, x1
        else:
            x1, x2 = x.chunk(2, dim=1)

        st = self.st_net(x1)
        s, t = st.chunk(2, dim=1)

        # Clamp scale for numerical stability
        s = self.clamp * torch.tanh(s / self.clamp)

        # Affine transform
        z2 = x2 * torch.exp(s) + t
        log_jac_det = s.sum(dim=(1, 2, 3))  # Per-sample log-det

        if self.reverse_mask:
            z = torch.cat([z2, x1], dim=1)
        else:
            z = torch.cat([x1, z2], dim=1)

        return z, log_jac_det

    def inverse(self, z):
        """Inverse pass (latent → data)."""
        if self.reverse_mask:
            z1, z2 = z.chunk(2, dim=1)
            z1, z2 = z2, z1
        else:
            z1, z2 = z.chunk(2, dim=1)

        st = self.st_net(z1)
        s, t = st.chunk(2, dim=1)
        s = self.clamp * torch.tanh(s / self.clamp)

        x2 = (z2 - t) * torch.exp(-s)

        if self.reverse_mask:
            x = torch.cat([x2, z1], dim=1)
        else:
            x = torch.cat([z1, x2], dim=1)

        return x


class FastFlow(nn.Module):
    """
    FastFlow anomaly detection model.

    Architecture:
        - Frozen backbone as feature encoder
        - Stack of 2D affine coupling layers on the feature map
        - NLL loss pushes normal features to N(0,I) in latent space
        - Anomalies have low likelihood → high anomaly score

    Args:
        backbone: Backbone name for feature extraction.
        layers: Which backbone layers to use.
        flow_steps: Number of coupling layers.
        hidden_channels: Hidden channels in coupling CNNs.
        kernel_size: Kernel size for coupling CNNs.
        clamp: Scale clamping value.
        img_size: Input image size.
        device: Torch device string.
        sigma: Gaussian smoothing sigma for anomaly map.
    """

    def __init__(self, backbone="wide_resnet50", layers=None,
                 flow_steps=8, hidden_channels=256, kernel_size=3,
                 clamp=2.0, img_size=256, device="auto", sigma=4.0):
        super().__init__()
        self.device = get_device(device)
        self.flow_steps = flow_steps
        self.img_size = img_size
        self.sigma = sigma

        # Frozen feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone, layers=layers, device=str(self.device)
        )

        # Determine feature dimensions from extractor
        feat_dims = self.feature_extractor.get_feature_dims()

        # Build separate flow for each feature layer
        self.flows = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        for layer_name, channels in feat_dims.items():
            # Ensure even number of channels for splitting
            if channels % 2 != 0:
                channels = channels + 1

            # Batch normalization before flow
            self.norms[layer_name] = nn.BatchNorm2d(channels, affine=False)

            # Stack of coupling layers with alternating masks
            flow_layers = nn.ModuleList()
            for i in range(flow_steps):
                flow_layers.append(
                    AffineCouplingLayer(
                        in_channels=channels,
                        hidden_channels=hidden_channels,
                        kernel_size=kernel_size,
                        clamp=clamp,
                        reverse_mask=(i % 2 == 1),
                    )
                )
            self.flows[layer_name] = flow_layers

        self._feat_dims = feat_dims
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass for training.

        Args:
            x: (B, 3, H, W) input images.

        Returns:
            total_loss: NLL loss scalar (for training).
            layer_losses: Dict of per-layer losses (for logging).
        """
        x = x.to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(x)

        total_loss = 0.0
        layer_losses = {}

        for layer_name, feat in features.items():
            # Pad channels if odd
            expected_c = list(self._feat_dims.values())[
                list(self._feat_dims.keys()).index(layer_name)
            ]
            if expected_c % 2 != 0:
                feat = F.pad(feat, (0, 0, 0, 0, 0, 1))  # Pad 1 channel

            # Batch normalize
            feat = self.norms[layer_name](feat)

            # Pass through flow
            z = feat
            log_jac_det = 0.0

            for flow_layer in self.flows[layer_name]:
                z, ljd = flow_layer(z)
                log_jac_det = log_jac_det + ljd

            # NLL loss: mean over pixels of [||z||^2/2 - log_jac_det]
            z_squared = 0.5 * z.pow(2).sum(dim=1)  # (B, H, W)
            nll = z_squared.mean(dim=(1, 2)) - log_jac_det / (z.shape[2] * z.shape[3])
            loss = nll.mean()

            total_loss = total_loss + loss
            layer_losses[layer_name] = loss.item()

        return total_loss, layer_losses

    def get_anomaly_map(self, x):
        """
        Compute pixel-level anomaly map at inference time.

        score(i,j) = ||z(i,j)||^2 / 2  (negative log likelihood proxy)
        High score = low probability = anomaly.

        Args:
            x: (B, 3, H, W) or (3, H, W) input tensor.

        Returns:
            anomaly_score: float (image-level, max of map).
            anomaly_map: (H, W) numpy array.
        """
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(x)

        anomaly_maps = []

        for layer_name, feat in features.items():
            expected_c = list(self._feat_dims.values())[
                list(self._feat_dims.keys()).index(layer_name)
            ]
            if expected_c % 2 != 0:
                feat = F.pad(feat, (0, 0, 0, 0, 0, 1))

            feat = self.norms[layer_name](feat)

            z = feat
            for flow_layer in self.flows[layer_name]:
                z, _ = flow_layer(z)

            # Per-pixel anomaly: ||z||^2 / 2 summed over channels
            pixel_score = 0.5 * z.pow(2).sum(dim=1)  # (B, H, W)

            # Upsample to image size
            pixel_score = F.interpolate(
                pixel_score.unsqueeze(1),
                size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(1)  # (B, H, W)

            anomaly_maps.append(pixel_score)

        # Average anomaly maps from all layers
        combined = torch.stack(anomaly_maps, dim=0).mean(dim=0)  # (B, H, W)

        scores = []
        maps = []
        for i in range(combined.shape[0]):
            amap = combined[i].cpu().numpy()
            amap = gaussian_filter(amap, sigma=self.sigma)
            score = float(amap.max())
            scores.append(score)
            maps.append(amap)

        if single:
            return scores[0], maps[0]
        return scores, maps

    def save(self, path):
        """Save model checkpoint."""
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "flow_steps": self.flow_steps,
            "img_size": self.img_size,
            "sigma": self.sigma,
            "feat_dims": self._feat_dims,
        }, path)
        logger.info(f"FastFlow saved to {path}")

    def load(self, path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])
        self.img_size = ckpt.get("img_size", self.img_size)
        self.sigma = ckpt.get("sigma", self.sigma)
        logger.info(f"FastFlow loaded from {path}")
