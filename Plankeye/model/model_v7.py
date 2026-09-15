from __future__ import annotations

"""PlankEye V7 - industrial quadrilateral corner detector.

Main ideas
----------
- Stronger ResNet50 backbone with ImageNet normalization inside the model.
- Frozen backbone BatchNorm statistics for stability with small per-GPU batches.
- FPN + PAN multi-scale neck.
- Dedicated high-resolution detail branch at stride 2.
- CoordConv heads with GroupNorm (stable for small batches).
- Center heatmap + sub-cell offset.
- Dual corner representation:
    1) center-relative corner vectors (primary representation),
    2) absolute normalized corners (auxiliary representation).
- Auxiliary object-size and localization-quality heads.
- Adaptive center Gaussian targets.
- Permutation-invariant quadrilateral matching.
- Geometry supervision without forcing 90-degree angles.
- Polygon NMS for inference.

Label semantics
---------------
Each object is expected as:
    {"cls": 0, "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
where coordinates are normalized to [0, 1].

The corner order may be any cyclic order and either winding direction.
The loss chooses the best cyclic/reversed permutation automatically.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

print("====================Model_V7=====================")

# =============================================================================
# MODEL / TRAINING CONSTANTS
# =============================================================================

MODEL_VERSION = "v7-resnet50-fpn-pan-dual-corners-quality"

IMG_SIZE = 512
PRED_STRIDE = 2
HEATMAP_SIZE = IMG_SIZE // PRED_STRIDE

NUM_CLASSES = 1
NUM_CORNERS = 4

# Center target.
GAUSSIAN_MIN_SIGMA = 1.15
GAUSSIAN_MAX_SIGMA = 5.0
GAUSSIAN_SIZE_DIVISOR = 18.0

# Loss weights. These are a solid starting point, not immutable constants.
LOSS_W_HEATMAP = 1.00
LOSS_W_OFFSET = 1.00
LOSS_W_CORNER_DELTA = 6.00
LOSS_W_CORNER_ABS = 2.00
LOSS_W_RECONSTRUCTION = 4.00
LOSS_W_SIZE = 0.50
LOSS_W_GEOMETRY = 1.50
LOSS_W_DUAL_CONSISTENCY = 0.75
LOSS_W_CENTER_CONSISTENCY = 0.50
LOSS_W_QUALITY = 0.25

# Geometry sub-loss weights.
GEOM_W_AREA = 1.00
GEOM_W_DIRECTION = 0.50
GEOM_W_LENGTH = 0.50
GEOM_W_DIAGONAL = 0.25
GEOM_W_ANGLE = 0.25
GEOM_W_CONVEXITY = 0.25

# The relative corner vectors are normalized to image size.
MAX_CORNER_DELTA = 1.0

# Decoder fusion between the two corner representations.
DEFAULT_RELATIVE_CORNER_BLEND = 0.75

# Quality target sharpness. Lower -> stricter quality score.
QUALITY_TAU = 0.035

# Cyclic permutations + reversed cyclic permutations.
CORNER_PERMUTATIONS = torch.tensor(
    [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [0, 3, 2, 1],
        [3, 2, 1, 0],
        [2, 1, 0, 3],
        [1, 0, 3, 2],
    ],
    dtype=torch.long,
)


# =============================================================================
# SMALL BUILDING BLOCKS
# =============================================================================


def _best_group_count(channels: int, preferred: int = 32) -> int:
    """Pick a GroupNorm group count that exactly divides channels."""
    for groups in (preferred, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            _best_group_count(out_channels),
            out_channels,
        )
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResidualDWBlock(nn.Module):
    """Depthwise-separable residual block with GroupNorm + SE."""

    def __init__(
        self,
        channels: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = int(round(channels * expansion))

        self.dw = ConvGNAct(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
        )
        self.pw1 = ConvGNAct(channels, hidden, kernel_size=1, padding=0)
        self.pw2 = ConvGNAct(
            hidden,
            channels,
            kernel_size=1,
            padding=0,
            activation=False,
        )
        self.se = SqueezeExcitation(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw1(y)
        y = self.pw2(y)
        y = self.se(y)
        y = self.dropout(y)
        return self.act(x + y)


class CoordConv2d(nn.Module):
    """CoordConv with x, y and radial coordinate channels."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 3, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        gr = torch.sqrt(torch.clamp(gx.square() + gy.square(), min=0.0))

        coords = torch.stack([gx, gy, gr], dim=0)
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1)
        return self.conv(torch.cat([x, coords], dim=1))


class PredictionTower(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input = nn.Sequential(
            CoordConv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                _best_group_count(hidden_channels),
                hidden_channels,
            ),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualDWBlock(hidden_channels, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.input(x)))


class WeightedFeatureFusion(nn.Module):
    """Fast normalized non-negative learnable fusion weights."""

    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} features, got {len(features)}."
            )

        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)

        out = features[0] * w[0]
        for i in range(1, len(features)):
            out = out + features[i] * w[i]
        return out


# =============================================================================
# BACKBONE
# =============================================================================


class ResNet50Backbone(nn.Module):
    """ResNet50 feature extractor returning stride 2/4/8/16/32 maps."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.freeze_bn = freeze_bn
        if freeze_bn:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            self._freeze_batchnorm()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s2 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(s2)
        s4 = self.layer1(x)
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)

        return {
            "s2": s2,
            "s4": s4,
            "s8": s8,
            "s16": s16,
            "s32": s32,
        }


# =============================================================================
# FPN + PAN NECK
# =============================================================================


class FPNPANNeck(nn.Module):
    def __init__(
        self,
        out_channels: int = 192,
        highres_channels: int = 160,
    ):
        super().__init__()
        c = out_channels

        # ResNet50 feature channels: 256, 512, 1024, 2048.
        self.lat4 = ConvGNAct(256, c, kernel_size=1, padding=0)
        self.lat8 = ConvGNAct(512, c, kernel_size=1, padding=0)
        self.lat16 = ConvGNAct(1024, c, kernel_size=1, padding=0)
        self.lat32 = ConvGNAct(2048, c, kernel_size=1, padding=0)

        self.ref4 = ResidualDWBlock(c)
        self.ref8 = ResidualDWBlock(c)
        self.ref16 = ResidualDWBlock(c)
        self.ref32 = ResidualDWBlock(c)

        # Bottom-up PAN path.
        self.down4_8 = ConvGNAct(c, c, kernel_size=3, stride=2)
        self.down8_16 = ConvGNAct(c, c, kernel_size=3, stride=2)
        self.down16_32 = ConvGNAct(c, c, kernel_size=3, stride=2)

        self.pan8 = ResidualDWBlock(c)
        self.pan16 = ResidualDWBlock(c)
        self.pan32 = ResidualDWBlock(c)

        # Aggregate every scale back to stride 4.
        self.final_fusion = WeightedFeatureFusion(4)
        self.final_refine = nn.Sequential(
            ResidualDWBlock(c),
            ResidualDWBlock(c),
        )

        # High-resolution stride-2 fusion.
        self.s2_proj = ConvGNAct(64, 64, kernel_size=1, padding=0)
        self.detail_branch = nn.Sequential(
            ConvGNAct(3, 48, kernel_size=3, stride=2),
            ResidualDWBlock(48),
            ConvGNAct(48, 64, kernel_size=3),
            ResidualDWBlock(64),
        )

        self.highres_fuse = nn.Sequential(
            ConvGNAct(c + 64 + 64, highres_channels, kernel_size=3),
            ResidualDWBlock(highres_channels, dropout=0.03),
            ResidualDWBlock(highres_channels, dropout=0.03),
        )

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        raw_image: torch.Tensor,
    ) -> torch.Tensor:
        # Top-down FPN.
        p32 = self.ref32(self.lat32(features["s32"]))

        p16 = self.lat16(features["s16"]) + self._resize_like(p32, features["s16"])
        p16 = self.ref16(p16)

        p8 = self.lat8(features["s8"]) + self._resize_like(p16, features["s8"])
        p8 = self.ref8(p8)

        p4 = self.lat4(features["s4"]) + self._resize_like(p8, features["s4"])
        p4 = self.ref4(p4)

        # Bottom-up PAN.
        n8 = self.pan8(p8 + self.down4_8(p4))
        n16 = self.pan16(p16 + self.down8_16(n8))
        n32 = self.pan32(p32 + self.down16_32(n16))

        # Multi-scale fusion at stride 4.
        fused_s4 = self.final_fusion(
            [
                p4,
                self._resize_like(n8, p4),
                self._resize_like(n16, p4),
                self._resize_like(n32, p4),
            ]
        )
        fused_s4 = self.final_refine(fused_s4)

        # Restore fine details at stride 2.
        fused_s2 = self._resize_like(fused_s4, features["s2"])
        backbone_s2 = self.s2_proj(features["s2"])
        detail_s2 = self.detail_branch(raw_image)

        return self.highres_fuse(
            torch.cat([fused_s2, backbone_s2, detail_s2], dim=1)
        )


# =============================================================================
# MODEL OUTPUT
# =============================================================================


@dataclass
class PlankEyeOutput:
    heatmap_logits: torch.Tensor
    corner_delta_logits: torch.Tensor
    corner_abs_logits: torch.Tensor
    center_offsets: torch.Tensor
    size_logits: torch.Tensor
    quality_logits: torch.Tensor


class PlankEyeV7(nn.Module):
    def __init__(
        self,
        pretrained_backbone: bool = True,
        freeze_backbone_bn: bool = True,
        neck_channels: int = 192,
        highres_channels: int = 160,
        head_channels: int = 128,
    ):
        super().__init__()

        self.backbone = ResNet50Backbone(
            pretrained=pretrained_backbone,
            freeze_bn=freeze_backbone_bn,
        )
        self.neck = FPNPANNeck(
            out_channels=neck_channels,
            highres_channels=highres_channels,
        )

        # Shared high-resolution feature refinement before all task heads.
        self.shared_head = nn.Sequential(
            ResidualDWBlock(highres_channels, dropout=0.03),
            ResidualDWBlock(highres_channels, dropout=0.03),
        )

        self.hmap_head = PredictionTower(
            highres_channels,
            head_channels,
            NUM_CLASSES,
            depth=2,
            dropout=0.02,
        )
        self.corner_delta_head = PredictionTower(
            highres_channels,
            head_channels,
            NUM_CLASSES * NUM_CORNERS * 2,
            depth=3,
            dropout=0.02,
        )
        self.corner_abs_head = PredictionTower(
            highres_channels,
            head_channels,
            NUM_CLASSES * NUM_CORNERS * 2,
            depth=2,
            dropout=0.02,
        )
        self.offset_head = PredictionTower(
            highres_channels,
            head_channels,
            NUM_CLASSES * 2,
            depth=2,
            dropout=0.02,
        )
        self.size_head = PredictionTower(
            highres_channels,
            head_channels,
            NUM_CLASSES * 2,
            depth=2,
            dropout=0.02,
        )
        self.quality_head = PredictionTower(
            highres_channels,
            head_channels,
            NUM_CLASSES,
            depth=2,
            dropout=0.02,
        )

        # ImageNet normalization is done internally so the data loader only
        # needs to provide RGB tensors in [0, 1].
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self._initialize_prediction_heads()

    def _initialize_prediction_heads(self) -> None:
        # Low prior foreground probability (~1%).
        nn.init.constant_(self.hmap_head.out.bias, -4.595)

        # Localization quality prior around 10%.
        nn.init.constant_(self.quality_head.out.bias, -2.197)

        # Conservative size prior sigmoid(-0.85) ~= 0.30.
        nn.init.constant_(self.size_head.out.bias, -0.85)

        # Near-zero regression initializations are easier to optimize.
        for head in (
            self.corner_delta_head,
            self.corner_abs_head,
            self.offset_head,
        ):
            nn.init.normal_(head.out.weight, mean=0.0, std=0.001)
            nn.init.constant_(head.out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> PlankEyeOutput:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Expected input [B,3,H,W], got {tuple(x.shape)}."
            )

        raw = x
        normalized = (x - self.imagenet_mean) / self.imagenet_std

        features = self.backbone(normalized)
        fused = self.shared_head(self.neck(features, raw))

        heatmap_logits = self.hmap_head(fused)
        corner_delta_logits = self.corner_delta_head(fused)
        corner_abs_logits = self.corner_abs_head(fused)

        # Offset relative to cell center, explicitly bounded to [-0.5, 0.5].
        center_offsets = 0.5 * torch.tanh(self.offset_head(fused))

        size_logits = self.size_head(fused)
        quality_logits = self.quality_head(fused)

        return PlankEyeOutput(
            heatmap_logits=heatmap_logits,
            corner_delta_logits=corner_delta_logits,
            corner_abs_logits=corner_abs_logits,
            center_offsets=center_offsets,
            size_logits=size_logits,
            quality_logits=quality_logits,
        )


# Compatibility alias if the training code imports PlankEyeV6.
PlankEyeV6 = PlankEyeV7


# =============================================================================
# TARGET CREATION
# =============================================================================


def draw_gaussian(
    heatmap: torch.Tensor,
    cx_i: int,
    cy_i: int,
    sigma: float,
) -> None:
    h, w = heatmap.shape
    sigma = max(float(sigma), 1e-6)
    radius = max(1, int(np.ceil(3.0 * sigma)))

    x0 = max(0, cx_i - radius)
    x1 = min(w, cx_i + radius + 1)
    y0 = max(0, cy_i - radius)
    y1 = min(h, cy_i + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return

    yy = torch.arange(y0, y1, device=heatmap.device, dtype=heatmap.dtype).view(-1, 1)
    xx = torch.arange(x0, x1, device=heatmap.device, dtype=heatmap.dtype).view(1, -1)

    g = torch.exp(
        -((xx - float(cx_i)).square() + (yy - float(cy_i)).square())
        / (2.0 * sigma * sigma)
    )
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], g)


def _adaptive_gaussian_sigma(
    corners: torch.Tensor,
    h: int,
    w: int,
) -> float:
    x_span = (corners[:, 0].max() - corners[:, 0].min()).clamp(min=0.0)
    y_span = (corners[:, 1].max() - corners[:, 1].min()).clamp(min=0.0)

    obj_w = x_span * float(w)
    obj_h = y_span * float(h)
    geometric_size = torch.sqrt((obj_w * obj_h).clamp(min=1e-6))

    sigma = geometric_size / GAUSSIAN_SIZE_DIVISOR
    sigma = sigma.clamp(GAUSSIAN_MIN_SIGMA, GAUSSIAN_MAX_SIGMA)
    return float(sigma.item())


def build_targets_gpu(
    batch_objects,
    device: torch.device,
    heatmap_size: Tuple[int, int],
):
    bsz = len(batch_objects)
    h, w = int(heatmap_size[0]), int(heatmap_size[1])

    hmap = torch.zeros(bsz, NUM_CLASSES, h, w, device=device)
    corner_abs = torch.zeros(
        bsz,
        NUM_CLASSES,
        NUM_CORNERS,
        2,
        h,
        w,
        device=device,
    )
    corner_delta = torch.zeros_like(corner_abs)
    offset_target = torch.zeros(bsz, NUM_CLASSES, 2, h, w, device=device)
    size_target = torch.zeros(bsz, NUM_CLASSES, 2, h, w, device=device)
    positive_mask = torch.zeros(bsz, NUM_CLASSES, h, w, device=device)

    best_dist = torch.full(
        (bsz, NUM_CLASSES, h, w),
        float("inf"),
        device=device,
    )

    for b, objects in enumerate(batch_objects):
        for obj in objects:
            cls = int(obj["cls"])
            if cls < 0 or cls >= NUM_CLASSES:
                continue

            corners = torch.as_tensor(
                obj["corners"],
                dtype=torch.float32,
                device=device,
            ).reshape(NUM_CORNERS, 2)

            if not torch.isfinite(corners).all():
                continue

            corners = corners.clamp(0.0, 1.0)
            center = corners.mean(dim=0)

            cx = torch.clamp(center[0] * w, 0.0, w - 1e-4)
            cy = torch.clamp(center[1] * h, 0.0, h - 1e-4)
            cx_i = int(torch.floor(cx).item())
            cy_i = int(torch.floor(cy).item())

            sigma = _adaptive_gaussian_sigma(corners, h=h, w=w)
            draw_gaussian(hmap[b, cls], cx_i, cy_i, sigma)

            # Only one regression target can occupy a center cell. Keep the
            # object whose actual center is nearest that cell center.
            dist = (cx - (cx_i + 0.5)).square() + (cy - (cy_i + 0.5)).square()
            if dist >= best_dist[b, cls, cy_i, cx_i]:
                continue

            best_dist[b, cls, cy_i, cx_i] = dist
            positive_mask[b, cls, cy_i, cx_i] = 1.0

            offset_target[b, cls, 0, cy_i, cx_i] = cx - (cx_i + 0.5)
            offset_target[b, cls, 1, cy_i, cx_i] = cy - (cy_i + 0.5)

            corner_abs[b, cls, :, :, cy_i, cx_i] = corners
            corner_delta[b, cls, :, :, cy_i, cx_i] = corners - center.view(1, 2)

            x_min = corners[:, 0].min()
            y_min = corners[:, 1].min()
            x_max = corners[:, 0].max()
            y_max = corners[:, 1].max()
            size_target[b, cls, 0, cy_i, cx_i] = (x_max - x_min).clamp(0.0, 1.0)
            size_target[b, cls, 1, cy_i, cx_i] = (y_max - y_min).clamp(0.0, 1.0)

    return {
        "heatmap": hmap,
        "corner_abs": corner_abs,
        "corner_delta": corner_delta,
        "offset": offset_target,
        "size": size_target,
        "positive_mask": positive_mask,
    }


# =============================================================================
# LOSS HELPERS
# =============================================================================


def focal_heatmap_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable CenterNet-style modified focal loss."""
    pred_prob = torch.sigmoid(pred_logits)

    pos_mask = target.eq(1.0)
    neg_mask = target.lt(1.0)
    neg_weights = (1.0 - target).pow(4)

    pos_loss = F.logsigmoid(pred_logits) * (1.0 - pred_prob).pow(2) * pos_mask
    neg_loss = (
        F.logsigmoid(-pred_logits)
        * pred_prob.pow(2)
        * neg_weights
        * neg_mask
    )

    num_pos = pos_mask.sum().to(dtype=pred_logits.dtype)
    if num_pos.item() > 0:
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    return -neg_loss.sum() / neg_mask.sum().clamp(min=1).to(pred_logits.dtype)


def _reshape_corner_map(
    x: torch.Tensor,
    apply: str,
) -> torch.Tensor:
    b, _, h, w = x.shape
    x = x.view(b, NUM_CLASSES, NUM_CORNERS, 2, h, w)

    if apply == "tanh":
        x = MAX_CORNER_DELTA * torch.tanh(x)
    elif apply == "sigmoid":
        x = torch.sigmoid(x)
    elif apply != "none":
        raise ValueError(f"Unknown transform: {apply}")

    return x


def _positive_corner_tensors(
    pred_delta_logits: torch.Tensor,
    pred_abs_logits: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
):
    mask = targets["positive_mask"].bool()

    pred_delta = _reshape_corner_map(pred_delta_logits, "tanh")
    pred_abs = _reshape_corner_map(pred_abs_logits, "sigmoid")

    if mask.sum().item() == 0:
        return pred_delta, pred_abs, None

    def gather(x: torch.Tensor) -> torch.Tensor:
        # [B,C,4,2,H,W] -> [B,C,H,W,4,2] -> [N,4,2]
        return x.permute(0, 1, 4, 5, 2, 3).contiguous()[mask]

    packed = {
        "pred_delta": gather(pred_delta),
        "pred_abs": gather(pred_abs),
        "target_delta": gather(targets["corner_delta"]),
        "target_abs": gather(targets["corner_abs"]),
    }
    return pred_delta, pred_abs, packed


def _best_dual_permutation(
    pred_delta_pos: torch.Tensor,
    pred_abs_pos: torch.Tensor,
    target_delta_pos: torch.Tensor,
    target_abs_pos: torch.Tensor,
):
    perms = CORNER_PERMUTATIONS.to(pred_delta_pos.device)

    all_total = []
    all_delta_targets = []
    all_abs_targets = []
    all_delta_losses = []
    all_abs_losses = []

    for perm in perms:
        td = target_delta_pos[:, perm, :]
        ta = target_abs_pos[:, perm, :]

        ld = F.smooth_l1_loss(
            pred_delta_pos,
            td,
            reduction="none",
            beta=0.02,
        ).mean(dim=(1, 2))

        la = F.smooth_l1_loss(
            pred_abs_pos,
            ta,
            reduction="none",
            beta=0.02,
        ).mean(dim=(1, 2))

        # Primary relative representation drives the matching more strongly.
        total = 0.75 * ld + 0.25 * la

        all_total.append(total)
        all_delta_targets.append(td)
        all_abs_targets.append(ta)
        all_delta_losses.append(ld)
        all_abs_losses.append(la)

    total_matrix = torch.stack(all_total, dim=1)
    best_idx = total_matrix.argmin(dim=1)
    row = torch.arange(pred_delta_pos.size(0), device=pred_delta_pos.device)

    delta_targets = torch.stack(all_delta_targets, dim=1)[row, best_idx]
    abs_targets = torch.stack(all_abs_targets, dim=1)[row, best_idx]
    delta_losses = torch.stack(all_delta_losses, dim=1)[row, best_idx]
    abs_losses = torch.stack(all_abs_losses, dim=1)[row, best_idx]

    return delta_targets, abs_targets, delta_losses, abs_losses


def _positive_centers_from_offsets(
    center_offsets: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    b, c, h, w = positive_mask.shape
    mask = positive_mask.bool()

    offsets = center_offsets.view(b, c, 2, h, w)
    offsets = offsets.permute(0, 1, 3, 4, 2).contiguous()[mask]

    idx = mask.nonzero(as_tuple=False)
    ys = idx[:, 2].to(dtype=offsets.dtype)
    xs = idx[:, 3].to(dtype=offsets.dtype)

    center_x = (xs + 0.5 + offsets[:, 0]) / float(w)
    center_y = (ys + 0.5 + offsets[:, 1]) / float(h)
    return torch.stack([center_x, center_y], dim=-1)


def _positive_scalar_or_vector_map(
    tensor: torch.Tensor,
    positive_mask: torch.Tensor,
    channels_per_class: int,
) -> torch.Tensor:
    b, c, h, w = positive_mask.shape
    x = tensor.view(b, c, channels_per_class, h, w)
    x = x.permute(0, 1, 3, 4, 2).contiguous()
    return x[positive_mask.bool()]


def polygon_area_torch(points: torch.Tensor) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    return 0.5 * torch.abs(
        torch.sum(
            x * torch.roll(y, -1, dims=-1)
            - y * torch.roll(x, -1, dims=-1),
            dim=-1,
        )
    )


def geometry_loss_components(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Perspective-friendly geometry loss.

    It matches target shape properties rather than forcing right angles or
    parallel edges, which would be incorrect under perspective distortion.
    """
    if pred_points.numel() == 0:
        zero = pred_points.sum() * 0.0
        return {
            "total": zero,
            "area": zero,
            "direction": zero,
            "length": zero,
            "diagonal": zero,
            "angle": zero,
            "convexity": zero,
        }

    pred_area = polygon_area_torch(pred_points)
    target_area = polygon_area_torch(target_points)

    # Relative area supervision prevents large objects from dominating.
    area_norm = target_area.detach().clamp(min=1e-3)
    area_loss = F.smooth_l1_loss(
        pred_area / area_norm,
        target_area / area_norm,
        beta=0.05,
    )

    pred_edges = torch.roll(pred_points, shifts=-1, dims=1) - pred_points
    target_edges = torch.roll(target_points, shifts=-1, dims=1) - target_points

    pred_dirs = F.normalize(pred_edges, dim=-1, eps=1e-6)
    target_dirs = F.normalize(target_edges, dim=-1, eps=1e-6)
    direction_loss = (1.0 - (pred_dirs * target_dirs).sum(dim=-1)).mean()

    pred_lengths = torch.linalg.vector_norm(pred_edges, dim=-1)
    target_lengths = torch.linalg.vector_norm(target_edges, dim=-1)
    length_scale = target_lengths.detach().mean(dim=1, keepdim=True).clamp(min=1e-3)
    length_loss = F.smooth_l1_loss(
        pred_lengths / length_scale,
        target_lengths / length_scale,
        beta=0.05,
    )

    pred_diagonals = torch.stack(
        [
            pred_points[:, 2] - pred_points[:, 0],
            pred_points[:, 3] - pred_points[:, 1],
        ],
        dim=1,
    )
    target_diagonals = torch.stack(
        [
            target_points[:, 2] - target_points[:, 0],
            target_points[:, 3] - target_points[:, 1],
        ],
        dim=1,
    )
    pred_diag_len = torch.linalg.vector_norm(pred_diagonals, dim=-1)
    target_diag_len = torch.linalg.vector_norm(target_diagonals, dim=-1)
    diag_scale = target_diag_len.detach().mean(dim=1, keepdim=True).clamp(min=1e-3)
    diagonal_loss = F.smooth_l1_loss(
        pred_diag_len / diag_scale,
        target_diag_len / diag_scale,
        beta=0.05,
    )

    # Match corner angles to the target instead of forcing 90 degrees.
    pred_prev = F.normalize(
        pred_points - torch.roll(pred_points, shifts=1, dims=1),
        dim=-1,
        eps=1e-6,
    )
    pred_next = F.normalize(
        torch.roll(pred_points, shifts=-1, dims=1) - pred_points,
        dim=-1,
        eps=1e-6,
    )
    target_prev = F.normalize(
        target_points - torch.roll(target_points, shifts=1, dims=1),
        dim=-1,
        eps=1e-6,
    )
    target_next = F.normalize(
        torch.roll(target_points, shifts=-1, dims=1) - target_points,
        dim=-1,
        eps=1e-6,
    )

    pred_cos = (pred_prev * pred_next).sum(dim=-1)
    target_cos = (target_prev * target_next).sum(dim=-1)
    angle_loss = F.smooth_l1_loss(pred_cos, target_cos, beta=0.05)

    # Convexity / winding consistency.
    cross_values = []
    for i in range(NUM_CORNERS):
        a = pred_edges[:, i]
        b = pred_edges[:, (i + 1) % NUM_CORNERS]
        cross_values.append(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    cross_values = torch.stack(cross_values, dim=1)
    target_edges_for_cross = target_edges
    target_cross = []
    for i in range(NUM_CORNERS):
        a = target_edges_for_cross[:, i]
        b = target_edges_for_cross[:, (i + 1) % NUM_CORNERS]
        target_cross.append(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    target_cross = torch.stack(target_cross, dim=1)

    target_orientation = torch.sign(target_cross.sum(dim=1, keepdim=True).detach())
    target_orientation = torch.where(
        target_orientation == 0,
        torch.ones_like(target_orientation),
        target_orientation,
    )
    convexity_loss = F.relu(1e-4 - target_orientation * cross_values).mean()

    total = (
        GEOM_W_AREA * area_loss
        + GEOM_W_DIRECTION * direction_loss
        + GEOM_W_LENGTH * length_loss
        + GEOM_W_DIAGONAL * diagonal_loss
        + GEOM_W_ANGLE * angle_loss
        + GEOM_W_CONVEXITY * convexity_loss
    )

    return {
        "total": total,
        "area": area_loss,
        "direction": direction_loss,
        "length": length_loss,
        "diagonal": diagonal_loss,
        "angle": angle_loss,
        "convexity": convexity_loss,
    }


# =============================================================================
# COMBINED LOSS
# =============================================================================


def combined_loss_v7(
    preds: PlankEyeOutput,
    batch_objects,
    device: torch.device,
):
    targets = build_targets_gpu(
        batch_objects,
        device=device,
        heatmap_size=preds.heatmap_logits.shape[-2:],
    )

    positive_mask = targets["positive_mask"]
    num_positive = int(positive_mask.sum().item())

    loss_hmap = focal_heatmap_loss(preds.heatmap_logits, targets["heatmap"])

    # No positive objects: keep only heatmap learning active.
    if num_positive == 0:
        zero = preds.corner_delta_logits.sum() * 0.0
        total = LOSS_W_HEATMAP * loss_hmap
        logs = {
            "total": float(total.detach().item()),
            "heatmap": float(loss_hmap.detach().item()),
            "offset": 0.0,
            "corner_delta": 0.0,
            "corner_abs": 0.0,
            "reconstruction": 0.0,
            "size": 0.0,
            "geometry": 0.0,
            "dual_consistency": 0.0,
            "center_consistency": 0.0,
            "quality": 0.0,
            "geom_area": 0.0,
            "geom_direction": 0.0,
            "geom_length": 0.0,
            "geom_diagonal": 0.0,
            "geom_angle": 0.0,
            "geom_convexity": 0.0,
            "num_positive": 0,
        }
        return total + zero, logs

    _, _, packed = _positive_corner_tensors(
        preds.corner_delta_logits,
        preds.corner_abs_logits,
        targets,
    )
    assert packed is not None

    (
        best_delta_target,
        best_abs_target,
        delta_loss_per_object,
        abs_loss_per_object,
    ) = _best_dual_permutation(
        packed["pred_delta"],
        packed["pred_abs"],
        packed["target_delta"],
        packed["target_abs"],
    )

    loss_corner_delta = delta_loss_per_object.mean()
    loss_corner_abs = abs_loss_per_object.mean()

    # Center offset loss.
    pred_offset_pos = _positive_scalar_or_vector_map(
        preds.center_offsets,
        positive_mask,
        channels_per_class=2,
    )
    target_offset_pos = _positive_scalar_or_vector_map(
        targets["offset"],
        positive_mask,
        channels_per_class=2,
    )
    loss_offset = F.smooth_l1_loss(
        pred_offset_pos,
        target_offset_pos,
        beta=0.05,
    )

    # Predicted normalized center coordinates.
    pred_center_pos = _positive_centers_from_offsets(
        preds.center_offsets,
        positive_mask,
    )

    # Reconstruct absolute corners from predicted center + relative vectors.
    pred_reconstructed = pred_center_pos.unsqueeze(1) + packed["pred_delta"]
    loss_reconstruction = F.smooth_l1_loss(
        pred_reconstructed,
        best_abs_target,
        beta=0.02,
    )

    # Auxiliary size head.
    pred_size_pos = torch.sigmoid(
        _positive_scalar_or_vector_map(
            preds.size_logits,
            positive_mask,
            channels_per_class=2,
        )
    )
    target_size_pos = _positive_scalar_or_vector_map(
        targets["size"],
        positive_mask,
        channels_per_class=2,
    )
    loss_size = F.smooth_l1_loss(
        pred_size_pos,
        target_size_pos,
        beta=0.03,
    )

    # Dual representation agreement.
    loss_dual_consistency = F.smooth_l1_loss(
        pred_reconstructed,
        packed["pred_abs"],
        beta=0.02,
    )

    # The absolute head centroid should agree with the predicted center.
    abs_centroid = packed["pred_abs"].mean(dim=1)
    loss_center_consistency = F.smooth_l1_loss(
        abs_centroid,
        pred_center_pos,
        beta=0.01,
    )

    # Geometry on a fused prediction reduces dependence on either corner head.
    fused_pred = (
        DEFAULT_RELATIVE_CORNER_BLEND * pred_reconstructed
        + (1.0 - DEFAULT_RELATIVE_CORNER_BLEND) * packed["pred_abs"]
    )
    geom = geometry_loss_components(fused_pred, best_abs_target)
    loss_geometry = geom["total"]

    # Localization quality target from current detached corner error.
    # This teaches the quality head to down-rank detections whose corners are
    # likely inaccurate even if the center heatmap is confident.
    with torch.no_grad():
        corner_error = torch.linalg.vector_norm(
            fused_pred.detach() - best_abs_target,
            dim=-1,
        ).mean(dim=1)
        quality_target = torch.exp(-corner_error / QUALITY_TAU).clamp(0.0, 1.0)

    quality_logits_pos = _positive_scalar_or_vector_map(
        preds.quality_logits,
        positive_mask,
        channels_per_class=1,
    ).squeeze(-1)
    loss_quality = F.binary_cross_entropy_with_logits(
        quality_logits_pos,
        quality_target,
    )

    total = (
        LOSS_W_HEATMAP * loss_hmap
        + LOSS_W_OFFSET * loss_offset
        + LOSS_W_CORNER_DELTA * loss_corner_delta
        + LOSS_W_CORNER_ABS * loss_corner_abs
        + LOSS_W_RECONSTRUCTION * loss_reconstruction
        + LOSS_W_SIZE * loss_size
        + LOSS_W_GEOMETRY * loss_geometry
        + LOSS_W_DUAL_CONSISTENCY * loss_dual_consistency
        + LOSS_W_CENTER_CONSISTENCY * loss_center_consistency
        + LOSS_W_QUALITY * loss_quality
    )

    logs = {
        "total": float(total.detach().item()),
        "heatmap": float(loss_hmap.detach().item()),
        "offset": float(loss_offset.detach().item()),
        "corner_delta": float(loss_corner_delta.detach().item()),
        "corner_abs": float(loss_corner_abs.detach().item()),
        "reconstruction": float(loss_reconstruction.detach().item()),
        "size": float(loss_size.detach().item()),
        "geometry": float(loss_geometry.detach().item()),
        "dual_consistency": float(loss_dual_consistency.detach().item()),
        "center_consistency": float(loss_center_consistency.detach().item()),
        "quality": float(loss_quality.detach().item()),
        "geom_area": float(geom["area"].detach().item()),
        "geom_direction": float(geom["direction"].detach().item()),
        "geom_length": float(geom["length"].detach().item()),
        "geom_diagonal": float(geom["diagonal"].detach().item()),
        "geom_angle": float(geom["angle"].detach().item()),
        "geom_convexity": float(geom["convexity"].detach().item()),
        "num_positive": num_positive,
    }

    return total, logs


# Backward-compatible function name if an old train script imports it.
def combined_loss_v5(preds, batch_objects, device):
    return combined_loss_v7(preds, batch_objects, device)


# =============================================================================
# POLYGON HELPERS + NMS
# =============================================================================


def _order_quad_np(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # Rotate to a stable top-left-ish first point.
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)

    # Prefer TL -> TR -> BR -> BL in image coordinates.
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = np.concatenate([ordered[:1], ordered[:0:-1]], axis=0)

    return ordered


def _polygon_area_np(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _signed_polygon_area_np(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    subject = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)

    if len(subject) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=np.float64)

    clip_pts = clip if _signed_polygon_area_np(clip) >= 0 else clip[::-1]

    def inside(p, a, b):
        return (
            (b[0] - a[0]) * (p[1] - a[1])
            - (b[1] - a[1]) * (p[0] - a[0])
            >= -1e-12
        )

    def intersect(p1, p2, a, b):
        d1 = p2 - p1
        d2 = b - a
        den = d1[0] * d2[1] - d1[1] * d2[0]

        if abs(den) < 1e-12:
            return p2

        t = (
            (a[0] - p1[0]) * d2[1]
            - (a[1] - p1[1]) * d2[0]
        ) / den
        return p1 + t * d1

    output = [p.copy() for p in subject]

    for i in range(len(clip_pts)):
        if not output:
            break

        a = clip_pts[i]
        b = clip_pts[(i + 1) % len(clip_pts)]
        input_list = output
        output = []

        prev = input_list[-1]
        prev_in = inside(prev, a, b)

        for cur in input_list:
            cur_in = inside(cur, a, b)

            if cur_in:
                if not prev_in:
                    output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(intersect(prev, cur, a, b))

            prev = cur
            prev_in = cur_in

    if output:
        return np.asarray(output, dtype=np.float64)
    return np.empty((0, 2), dtype=np.float64)


def _is_convex_quad_np(pts: np.ndarray, eps: float = 1e-10) -> bool:
    pts = _order_quad_np(pts)
    cross = []

    for i in range(4):
        e1 = pts[(i + 1) % 4] - pts[i]
        e2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        cross.append(e1[0] * e2[1] - e1[1] * e2[0])

    pos = all(v > eps for v in cross)
    neg = all(v < -eps for v in cross)
    return pos or neg


def quad_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    a = _order_quad_np(a)
    b = _order_quad_np(b)

    if not _is_convex_quad_np(a) or not _is_convex_quad_np(b):
        return 0.0

    area_a = _polygon_area_np(a)
    area_b = _polygon_area_np(b)

    if area_a <= 1e-12 or area_b <= 1e-12:
        return 0.0

    inter_poly = _clip_polygon(a, b)
    inter = _polygon_area_np(inter_poly)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-12 else 0.0


def _polygon_nms(detections: List[dict], iou_thresh: float) -> List[dict]:
    if iou_thresh <= 0 or len(detections) <= 1:
        return detections

    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: List[dict] = []

    for det in ordered:
        quad = np.asarray(det["corners"], dtype=np.float64)
        if all(
            quad_iou_np(quad, np.asarray(k["corners"], dtype=np.float64)) < iou_thresh
            for k in kept
        ):
            kept.append(det)

    return kept


# =============================================================================
# DECODER
# =============================================================================


@torch.no_grad()
def decode_predictions(
    preds: PlankEyeOutput,
    score_thresh: float = 0.30,
    topk: int = 100,
    nms_iou: float = 0.45,
    min_quad_area: float = 1e-4,
    relative_corner_blend: float = DEFAULT_RELATIVE_CORNER_BLEND,
    quality_power: float = 0.25,
    size_agreement_power: float = 0.15,
):
    hmap = torch.sigmoid(preds.heatmap_logits)
    bsz, classes, h, w = hmap.shape

    # Local-maximum suppression on center heatmap.
    keep_map = hmap.eq(
        F.max_pool2d(
            hmap,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    )
    hmap = hmap * keep_map

    rel_map = _reshape_corner_map(preds.corner_delta_logits, "tanh")
    abs_map = _reshape_corner_map(preds.corner_abs_logits, "sigmoid")
    size_map = torch.sigmoid(
        preds.size_logits.view(bsz, classes, 2, h, w)
    )
    quality_map = torch.sigmoid(preds.quality_logits)
    offset_map = preds.center_offsets.view(bsz, classes, 2, h, w)

    results = []

    for b in range(bsz):
        detections: List[dict] = []

        for c in range(classes):
            flat = hmap[b, c].flatten()
            k = min(topk, flat.numel())
            scores, idx = torch.topk(flat, k)

            valid = scores >= score_thresh
            scores = scores[valid]
            idx = idx[valid]

            for center_score, flat_idx in zip(scores.tolist(), idx.tolist()):
                cy_i = flat_idx // w
                cx_i = flat_idx % w

                off_x = float(offset_map[b, c, 0, cy_i, cx_i].item())
                off_y = float(offset_map[b, c, 1, cy_i, cx_i].item())

                center = torch.tensor(
                    [
                        (cx_i + 0.5 + off_x) / float(w),
                        (cy_i + 0.5 + off_y) / float(h),
                    ],
                    dtype=rel_map.dtype,
                    device=rel_map.device,
                )

                rel_corners = center.view(1, 2) + rel_map[b, c, :, :, cy_i, cx_i]
                abs_corners = abs_map[b, c, :, :, cy_i, cx_i]

                quad = (
                    relative_corner_blend * rel_corners
                    + (1.0 - relative_corner_blend) * abs_corners
                ).clamp(0.0, 1.0)

                quad_np = _order_quad_np(quad.cpu().numpy())

                area = _polygon_area_np(quad_np)
                if area < min_quad_area:
                    continue
                if not _is_convex_quad_np(quad_np):
                    continue

                quality = float(quality_map[b, c, cy_i, cx_i].item())

                pred_size = size_map[b, c, :, cy_i, cx_i]
                decoded_w = float(quad[:, 0].max().item() - quad[:, 0].min().item())
                decoded_h = float(quad[:, 1].max().item() - quad[:, 1].min().item())
                decoded_size = torch.tensor(
                    [decoded_w, decoded_h],
                    device=pred_size.device,
                    dtype=pred_size.dtype,
                )
                size_error = float(torch.abs(pred_size - decoded_size).mean().item())
                size_agreement = float(np.exp(-size_error / 0.15))

                final_score = float(center_score)
                final_score *= max(quality, 1e-6) ** quality_power
                final_score *= max(size_agreement, 1e-6) ** size_agreement_power

                detections.append(
                    {
                        "score": final_score,
                        "center_score": float(center_score),
                        "quality": quality,
                        "size_agreement": size_agreement,
                        "class": c,
                        "corners": quad_np.astype(np.float32),
                        "center": (float(center[0].item()), float(center[1].item())),
                        "predicted_size": (
                            float(pred_size[0].item()),
                            float(pred_size[1].item()),
                        ),
                        "area": area,
                    }
                )

        detections = _polygon_nms(detections, iou_thresh=nms_iou)
        results.append(detections)

    return results


# =============================================================================
# CHECKPOINT METADATA HELPER
# =============================================================================


def model_metadata() -> dict:
    return {
        "model_version": MODEL_VERSION,
        "architecture": "ResNet50 + FPN/PAN + stride-2 detail branch + dual corner heads",
        "num_classes": NUM_CLASSES,
        "num_corners": NUM_CORNERS,
        "prediction_stride": PRED_STRIDE,
        "corner_representation": {
            "primary": "center-relative normalized corner vectors",
            "auxiliary": "absolute normalized corner coordinates",
        },
        "heads": [
            "heatmap",
            "center_offset",
            "corner_delta",
            "corner_absolute",
            "bbox_size",
            "localization_quality",
        ],
        "loss_weights": {
            "heatmap": LOSS_W_HEATMAP,
            "offset": LOSS_W_OFFSET,
            "corner_delta": LOSS_W_CORNER_DELTA,
            "corner_abs": LOSS_W_CORNER_ABS,
            "reconstruction": LOSS_W_RECONSTRUCTION,
            "size": LOSS_W_SIZE,
            "geometry": LOSS_W_GEOMETRY,
            "dual_consistency": LOSS_W_DUAL_CONSISTENCY,
            "center_consistency": LOSS_W_CENTER_CONSISTENCY,
            "quality": LOSS_W_QUALITY,
        },
    }


# =============================================================================
# QUICK SELF TEST
# =============================================================================


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlankEyeV7(pretrained_backbone=False).to(device)
    model.eval()

    x = torch.rand(2, 3, 512, 512, device=device)

    with torch.no_grad():
        out = model(x)

    print("MODEL_VERSION       :", MODEL_VERSION)
    print("heatmap_logits      :", tuple(out.heatmap_logits.shape))
    print("corner_delta_logits :", tuple(out.corner_delta_logits.shape))
    print("corner_abs_logits   :", tuple(out.corner_abs_logits.shape))
    print("center_offsets      :", tuple(out.center_offsets.shape))
    print("size_logits         :", tuple(out.size_logits.shape))
    print("quality_logits      :", tuple(out.quality_logits.shape))
