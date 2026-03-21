"""
VisionGuard-AD — Coreset Subsampling
=====================================
Greedy k-center coreset subsampling for PatchCore memory bank reduction.
Optimized for Apple Silicon M1 with chunked distance computation.
"""

import logging
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger("visionguard.models.coreset")


class CoresetSampler:
    """
    Greedy k-center coreset subsampling.

    Selects representative subset S of size k from full embedding set M
    such that every point in M is close to at least one point in S.

    Args:
        ratio: Fraction of embeddings to keep (0 to 1).
        device: Torch device.
        chunk_size: Chunk size for distance computation (memory management).
        approximate: Use random projection for >50k points.
        projection_dim: Target dim for random projection.
        seed: Random seed.
    """

    def __init__(self, ratio=0.1, device="cpu", chunk_size=4096,
                 approximate=True, projection_dim=128, seed=42):
        assert 0.0 < ratio <= 1.0
        self.ratio = ratio
        self.device = torch.device(device) if isinstance(device, str) else device
        self.chunk_size = chunk_size
        self.approximate = approximate
        self.projection_dim = projection_dim
        self.seed = seed

    def run(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Run coreset subsampling.

        Args:
            embeddings: (N, D) all patch embeddings.
        Returns:
            Subsampled embeddings (k, D).
        """
        N, D = embeddings.shape
        k = max(1, int(np.ceil(N * self.ratio)))

        if k >= N:
            logger.info(f"Ratio selects {k}>={N}, returning all.")
            return embeddings

        logger.info(f"Coreset: {N} -> {k} (ratio={self.ratio}, D={D})")

        if self.approximate and N > 50000:
            logger.info(f"Approximate mode: project D={D} -> {self.projection_dim}")
            indices = self._run_approximate(embeddings, k)
        else:
            indices = self._run_exact(embeddings, k)

        selected = embeddings[indices]
        logger.info(f"Coreset complete: {len(selected)} embeddings selected")
        return selected

    def _run_exact(self, embeddings, k):
        """Exact greedy k-center with chunked distances."""
        N = embeddings.shape[0]
        embeddings = embeddings.to(self.device)
        torch.manual_seed(self.seed)

        first_idx = torch.randint(0, N, (1,)).item()
        selected = [first_idx]
        min_dists = torch.full((N,), float("inf"), device=self.device)
        min_dists = self._update_min_dists(embeddings, embeddings[first_idx:first_idx+1], min_dists)

        for _ in tqdm(range(1, k), desc="Coreset sampling", leave=False):
            next_idx = torch.argmax(min_dists).item()
            selected.append(next_idx)
            min_dists = self._update_min_dists(embeddings, embeddings[next_idx:next_idx+1], min_dists)
            min_dists[next_idx] = 0.0

        return torch.tensor(selected, dtype=torch.long)

    def _run_approximate(self, embeddings, k):
        """Approximate coreset via random projection + exact k-center."""
        N, D = embeddings.shape
        torch.manual_seed(self.seed)
        proj = torch.randn(D, self.projection_dim, device=self.device) / np.sqrt(self.projection_dim)

        projected = torch.zeros(N, self.projection_dim, device=self.device)
        for s in range(0, N, self.chunk_size):
            e = min(s + self.chunk_size, N)
            projected[s:e] = embeddings[s:e].to(self.device) @ proj

        return self._run_exact(projected, k)

    def _update_min_dists(self, embeddings, new_center, current_min):
        """Update min distances with chunked cdist for memory efficiency."""
        N = embeddings.shape[0]
        for s in range(0, N, self.chunk_size):
            e = min(s + self.chunk_size, N)
            dists = torch.cdist(embeddings[s:e], new_center, p=2).squeeze(-1)
            current_min[s:e] = torch.minimum(current_min[s:e], dists)
        return current_min
