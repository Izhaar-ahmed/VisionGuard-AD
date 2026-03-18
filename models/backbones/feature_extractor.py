"""
VisionGuard-AD — Feature Extractor
====================================

Wraps pretrained CNN and ViT backbones to extract intermediate feature maps
for anomaly detection. Uses forward hooks to capture activations from specific
layers without modifying the original model architecture.

Supported backbones:
    - Wide ResNet-50-2 (default, best for PatchCore)
    - ResNet-18 (lightweight)
    - ResNet-50
    - EfficientNet-B4
    - ViT-B/16 (Vision Transformer)

Optimized for Apple Silicon M1 with MPS backend support.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger("visionguard.models.feature_extractor")

# Backbone registry: maps name → (constructor, layer_name_map)
BACKBONE_REGISTRY = {
    "wide_resnet50": "wide_resnet50",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet_b4": "efficientnet_b4",
    "vit_b16": "vit_b16",
}


def get_device(device: str = "auto") -> torch.device:
    """
    Resolve device string to torch.device.
    'auto' → MPS on Apple Silicon, CUDA if available, else CPU.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device)


class FeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from pretrained backbones using forward hooks.

    For CNN backbones (ResNet, WideResNet, EfficientNet):
        - Hooks into layer2 and layer3 (or equivalent mid-level layers)
        - Returns feature maps from BOTH layers
        - Does NOT use the final classification head
        - Model is set to eval() with all parameters frozen

    For ViT-B/16:
        - Extracts patch tokens from intermediate transformer blocks (blocks 6, 9)
        - Reshapes patch tokens back to 2D spatial grid

    Args:
        backbone_name: Name of the backbone architecture.
        layers: List of layer names to extract features from.
        device: Device string ('auto', 'cpu', 'mps', 'cuda').
        pretrained: Whether to load pretrained ImageNet weights.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50",
        layers: Optional[List[str]] = None,
        device: str = "auto",
        pretrained: bool = True,
    ):
        super().__init__()

        assert backbone_name in BACKBONE_REGISTRY, (
            f"Unknown backbone '{backbone_name}'. "
            f"Supported: {list(BACKBONE_REGISTRY.keys())}"
        )

        self.backbone_name = backbone_name
        self.device = get_device(device)
        self.pretrained = pretrained
        self.is_vit = backbone_name == "vit_b16"

        # Default layers if not specified
        if layers is None:
            if self.is_vit:
                layers = ["block_6", "block_9"]
            else:
                layers = ["layer2", "layer3"]
        self.layers = layers

        # Storage for hooked features
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks = []

        # Build backbone and register hooks
        self.backbone = self._build_backbone()
        self.backbone.eval()
        self.backbone.to(self.device)

        # Freeze all parameters — no gradient computation needed
        for param in self.backbone.parameters():
            param.requires_grad = False

        self._register_hooks()

        # Cache feature dimensions
        self._feature_dims = self._compute_feature_dims()

        logger.info(
            f"FeatureExtractor initialized: backbone={backbone_name}, "
            f"layers={layers}, device={self.device}, "
            f"feature_dims={self._feature_dims}"
        )

    def _build_backbone(self) -> nn.Module:
        """Construct and return the pretrained backbone model."""
        weights_arg = "DEFAULT" if self.pretrained else None

        if self.backbone_name == "wide_resnet50":
            model = models.wide_resnet50_2(weights=weights_arg)
            return model

        elif self.backbone_name == "resnet18":
            model = models.resnet18(weights=weights_arg)
            return model

        elif self.backbone_name == "resnet50":
            model = models.resnet50(weights=weights_arg)
            return model

        elif self.backbone_name == "efficientnet_b4":
            model = models.efficientnet_b4(weights=weights_arg)
            return model

        elif self.backbone_name == "vit_b16":
            try:
                import timm
                model = timm.create_model(
                    "vit_base_patch16_224",
                    pretrained=self.pretrained,
                    num_classes=0,  # Remove classification head
                )
                return model
            except ImportError:
                raise ImportError(
                    "timm is required for ViT backbone. "
                    "Install with: pip install timm"
                )

        raise ValueError(f"Unsupported backbone: {self.backbone_name}")

    def _register_hooks(self):
        """Register forward hooks on target layers to capture intermediate features."""
        # Clear existing hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._features = {}

        if self.backbone_name in ("wide_resnet50", "resnet18", "resnet50"):
            self._register_resnet_hooks()
        elif self.backbone_name == "efficientnet_b4":
            self._register_efficientnet_hooks()
        elif self.backbone_name == "vit_b16":
            self._register_vit_hooks()

    def _register_resnet_hooks(self):
        """Register hooks for ResNet-family backbones."""
        layer_map = {
            "layer1": self.backbone.layer1,
            "layer2": self.backbone.layer2,
            "layer3": self.backbone.layer3,
            "layer4": self.backbone.layer4,
        }

        for layer_name in self.layers:
            if layer_name not in layer_map:
                raise ValueError(
                    f"Layer '{layer_name}' not found in ResNet. "
                    f"Available: {list(layer_map.keys())}"
                )
            hook = layer_map[layer_name].register_forward_hook(
                self._make_hook(layer_name)
            )
            self._hooks.append(hook)

    def _register_efficientnet_hooks(self):
        """Register hooks for EfficientNet-B4."""
        # EfficientNet features are in model.features (Sequential of blocks)
        # Map layer2/layer3 to appropriate feature blocks
        # EfficientNet-B4 has 9 feature blocks (0-8)
        # layer2 ≈ features[3] (stride 8), layer3 ≈ features[5] (stride 16)
        efficientnet_layer_map = {
            "layer2": 3,
            "layer3": 5,
            "layer1": 2,
            "layer4": 7,
        }

        for layer_name in self.layers:
            block_idx = efficientnet_layer_map.get(layer_name)
            if block_idx is None:
                raise ValueError(
                    f"Layer '{layer_name}' not mapped for EfficientNet. "
                    f"Available: {list(efficientnet_layer_map.keys())}"
                )
            hook = self.backbone.features[block_idx].register_forward_hook(
                self._make_hook(layer_name)
            )
            self._hooks.append(hook)

    def _register_vit_hooks(self):
        """Register hooks for ViT-B/16 transformer blocks."""
        block_map = {}
        for i, block in enumerate(self.backbone.blocks):
            block_map[f"block_{i}"] = block

        for layer_name in self.layers:
            if layer_name not in block_map:
                available = [f"block_{i}" for i in range(len(self.backbone.blocks))]
                raise ValueError(
                    f"Layer '{layer_name}' not found in ViT. "
                    f"Available: {available}"
                )
            hook = block_map[layer_name].register_forward_hook(
                self._make_hook(layer_name)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Create a forward hook closure that stores layer output."""
        def hook_fn(module, input, output):
            self._features[layer_name] = output
        return hook_fn

    def _compute_feature_dims(self) -> Dict[str, int]:
        """Compute output channel dimensions for each hooked layer using a dummy forward pass."""
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            self.forward(dummy_input)

        dims = {}
        for layer_name, feat in self._features.items():
            if self.is_vit:
                # ViT output shape: (B, num_patches+1, embed_dim) or (B, num_patches, embed_dim)
                dims[layer_name] = feat.shape[-1]
            else:
                # CNN output shape: (B, C, H, W)
                dims[layer_name] = feat.shape[1]

        return dims

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps from the backbone.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary mapping layer names to feature tensors.
            CNN: {layer_name: (B, C, H, W)}
            ViT: {layer_name: (B, C, h, w)} — reshaped to 2D spatial grid
        """
        self._features = {}
        x = x.to(self.device)

        with torch.no_grad():
            _ = self.backbone(x)

        # For ViT: reshape patch tokens to 2D spatial grid
        if self.is_vit:
            processed = {}
            for layer_name, feat in self._features.items():
                # feat shape: (B, num_tokens, embed_dim)
                # Remove CLS token if present
                if hasattr(self.backbone, "num_prefix_tokens"):
                    num_prefix = self.backbone.num_prefix_tokens
                else:
                    num_prefix = 1  # Default: 1 CLS token

                patch_tokens = feat[:, num_prefix:, :]  # (B, num_patches, D)
                B, N, D = patch_tokens.shape
                h = w = int(math.sqrt(N))
                assert h * w == N, f"Cannot reshape {N} patches into square grid"

                # Reshape to (B, D, h, w) for spatial consistency with CNN features
                spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)
                processed[layer_name] = spatial

            return processed

        return dict(self._features)

    def get_feature_dims(self) -> Dict[str, int]:
        """Return channel dimensions for each extracted layer."""
        return dict(self._feature_dims)

    def __del__(self):
        """Clean up hooks on deletion."""
        for hook in self._hooks:
            hook.remove()


class PatchEmbedding(nn.Module):
    """
    Converts feature maps into local patch embeddings.

    This is the critical step that makes PatchCore work at patch level, not image
    level. It extracts local patch neighborhoods using unfold and applies
    neighborhood aggregation (average pooling over a window).

    The resulting patch embeddings capture local texture/structure information
    needed for fine-grained anomaly localization.

    Args:
        patch_size: Neighborhood size for aggregation (default 3 → 3×3 window).
        stride: Stride for patch extraction (default 1).
    """

    def __init__(self, patch_size: int = 3, stride: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        # Average pooling for neighborhood aggregation
        self.pool = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        return_spatial_shape: bool = False,
    ) -> torch.Tensor:
        """
        Convert multi-layer feature maps into concatenated patch embeddings.

        Steps:
        1. Apply neighborhood aggregation (avg pool) to each feature map
        2. Resize all feature maps to the same spatial dimensions (largest)
        3. Concatenate along channel dimension
        4. Reshape to (B*H*W, C) — one embedding per spatial location

        Args:
            features: Dict of {layer_name: (B, C, H, W)} feature maps.
            return_spatial_shape: If True, also return (B, H, W) spatial dims.

        Returns:
            Patch embeddings of shape (B*H*W, C).
            If return_spatial_shape: tuple of (embeddings, (B, H, W)).
        """
        layer_features = []
        target_size = None

        # Find the largest spatial size (from earliest layer = highest resolution)
        for layer_name in sorted(features.keys()):
            feat = features[layer_name]
            if target_size is None or (feat.shape[2] * feat.shape[3]) > (target_size[0] * target_size[1]):
                target_size = (feat.shape[2], feat.shape[3])

        # Process each layer
        for layer_name in sorted(features.keys()):
            feat = features[layer_name]

            # Apply neighborhood aggregation
            feat = self.pool(feat)

            # Resize to target spatial dimensions
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            layer_features.append(feat)

        # Concatenate along channel dimension: (B, C_total, H, W)
        concatenated = torch.cat(layer_features, dim=1)

        B, C, H, W = concatenated.shape

        # Reshape to patch embeddings: (B*H*W, C)
        embeddings = concatenated.permute(0, 2, 3, 1).reshape(B * H * W, C)

        if return_spatial_shape:
            return embeddings, (B, H, W)

        return embeddings
