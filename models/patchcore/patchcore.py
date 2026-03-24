"""
VisionGuard-AD — PatchCore Implementation
==========================================
CVPR 2022: "Towards Total Recall in Industrial Anomaly Detection"

Memory bank of normal patch embeddings + nearest-neighbor anomaly scoring.
No training (gradient-free) — just feature extraction + coreset subsampling.

Optimized for Apple Silicon M1 (MPS backend, chunked distance computation).
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from models.backbones.feature_extractor import FeatureExtractor, PatchEmbedding, get_device
from models.patchcore.coreset_sampler import CoresetSampler

logger = logging.getLogger("visionguard.models.patchcore")


class PatchCore:
    """
    PatchCore anomaly detection model.

    Builds a memory bank of L2-normalized patch embeddings from normal training
    images, then scores test images by computing nearest-neighbor distances
    from their patch embeddings to the memory bank.

    Args:
        backbone: Backbone name (e.g. 'wide_resnet50').
        layers: Feature layers to extract.
        coreset_ratio: Fraction of patches to keep after coreset subsampling.
        num_neighbors: Number of nearest neighbors for scoring.
        patch_size: Neighborhood size for patch aggregation.
        device: Torch device string.
        sigma: Gaussian smoothing sigma for anomaly map.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        layers=None,
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        patch_size: int = 3,
        device: str = "auto",
        sigma: float = 4.0,
    ):
        self.device = get_device(device)
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.sigma = sigma
        self.backbone_name = backbone

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone, layers=layers, device=str(self.device)
        )
        self.patch_embedding = PatchEmbedding(patch_size=patch_size)

        # Memory bank (set during fit)
        self.memory_bank: Optional[torch.Tensor] = None
        self.spatial_shape: Optional[Tuple[int, int]] = None  # (H, W) of feature map

        # Coreset sampler
        self.coreset_sampler = CoresetSampler(
            ratio=coreset_ratio, device=str(self.device)
        )

        logger.info(
            f"PatchCore initialized: backbone={backbone}, "
            f"coreset_ratio={coreset_ratio}, device={self.device}"
        )

    def fit(self, dataloader):
        """
        Build memory bank from normal training images.

        Steps:
            1. Extract patch embeddings from all training images
            2. L2-normalize embeddings
            3. Apply coreset subsampling
            4. Store final coreset as memory bank

        Args:
            dataloader: DataLoader yielding (image, mask, label, ...) tuples.
        """
        logger.info("Building PatchCore memory bank...")
        start_time = time.time()

        all_embeddings = []
        self.feature_extractor.backbone.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch[0].to(self.device)

                # Extract multi-layer features
                features = self.feature_extractor(images)

                # Convert to patch embeddings
                embeddings, (B, H, W) = self.patch_embedding(
                    features, return_spatial_shape=True
                )
                self.spatial_shape = (H, W)

                all_embeddings.append(embeddings.cpu())

        # Concatenate all patch embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        logger.info(
            f"Total patch embeddings: {all_embeddings.shape[0]} "
            f"(dim={all_embeddings.shape[1]})"
        )

        # L2-normalize embeddings
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

        # Apply coreset subsampling
        logger.info(f"Memory bank BEFORE coreset: {all_embeddings.shape[0]} patches")
        self.memory_bank = self.coreset_sampler.run(all_embeddings)
        self.memory_bank = self.memory_bank.to(self.device)
        logger.info(f"Memory bank AFTER coreset: {self.memory_bank.shape[0]} patches")

        elapsed = time.time() - start_time
        logger.info(f"Memory bank built in {elapsed:.1f}s")

    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Compute anomaly score and heatmap for input image(s).

        Steps:
            1. Extract patch embeddings
            2. Compute distance to nearest neighbor in memory bank
            3. Image-level score = max patch distance
            4. Reshape distances to spatial grid → anomaly map
            5. Upsample + Gaussian smooth

        Args:
            image_tensor: (B, 3, H, W) or (3, H, W) input tensor.

        Returns:
            anomaly_score: float (image-level anomaly score)
            anomaly_map: (H_orig, W_orig) numpy array (pixel-level heatmap)
        """
        if self.memory_bank is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        single = image_tensor.dim() == 3
        if single:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        B = image_tensor.shape[0]
        img_size = image_tensor.shape[-1]

        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            embeddings, (_, H, W) = self.patch_embedding(
                features, return_spatial_shape=True
            )
            # L2-normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute distances to nearest neighbor in memory bank (chunked for memory)
        distances = self._compute_nn_distances(embeddings)

        # Reshape to spatial grid: (B, H, W)
        distance_map = distances.reshape(B, H, W)

        # Process each image in batch
        scores = []
        maps = []
        for i in range(B):
            # Image-level score = max patch distance
            score = distance_map[i].max().item()
            scores.append(score)

            # Upsample anomaly map to original image size
            amap = distance_map[i].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            amap = F.interpolate(
                amap, size=(img_size, img_size),
                mode="bilinear", align_corners=False
            )
            amap = amap.squeeze().cpu().numpy()

            # Gaussian smoothing for cleaner heatmap
            amap = gaussian_filter(amap, sigma=self.sigma)
            maps.append(amap)

        if single:
            return scores[0], maps[0]
        return scores, maps

    def _compute_nn_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute nearest-neighbor distances from embeddings to memory bank.
        Uses chunked computation for M1 memory efficiency.

        Args:
            embeddings: (N, D) query patch embeddings.
        Returns:
            (N,) tensor of nearest-neighbor distances.
        """
        N = embeddings.shape[0]
        chunk_size = 2048
        nn_distances = torch.zeros(N, device=self.device)

        # Move memory bank computation to CPU if on MPS to avoid MPS cdist issues
        compute_device = self.device
        mem_bank = self.memory_bank

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = embeddings[start:end]

            # Compute pairwise distances: (chunk_size, memory_bank_size)
            dists = torch.cdist(chunk, mem_bank, p=2)

            # Get distance to nearest neighbor
            nn_dist, _ = dists.min(dim=1)
            nn_distances[start:end] = nn_dist

        return nn_distances

    def save(self, path: str):
        """Save memory bank and configuration to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "spatial_shape": self.spatial_shape,
            "backbone_name": self.backbone_name,
            "coreset_ratio": self.coreset_ratio,
            "num_neighbors": self.num_neighbors,
            "sigma": self.sigma,
            "feature_dims": self.feature_extractor.get_feature_dims(),
        }
        torch.save(save_dict, path)
        logger.info(f"PatchCore model saved to {path}")

    def load(self, path: str):
        """Load memory bank from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.memory_bank = save_dict["memory_bank"].to(self.device)
        self.spatial_shape = save_dict["spatial_shape"]
        self.coreset_ratio = save_dict.get("coreset_ratio", self.coreset_ratio)
        self.num_neighbors = save_dict.get("num_neighbors", self.num_neighbors)
        self.sigma = save_dict.get("sigma", self.sigma)

        logger.info(
            f"PatchCore loaded from {path}: "
            f"memory_bank={self.memory_bank.shape}"
        )
