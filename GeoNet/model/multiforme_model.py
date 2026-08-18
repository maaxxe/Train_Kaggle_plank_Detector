# -*- coding: utf-8 -*-
"""
MultiFormeNet
=============

Architecture adaptée à des pièces de formes arbitraires :

    image
      -> EfficientNetV2-S encoder (ImageNet)
      -> U-Net decoder multi-échelle
      -> tête segmentation binaire pleine résolution
      -> tête heatmap de centres à stride 2

La segmentation décrit la forme complète. Les 32 points finaux ne sont PAS
régressés directement : ils sont extraits de manière déterministe depuis le
contour prédit. Cela permet de changer plus tard 32 -> 64 points sans réentraîner
le réseau.

La heatmap de centres sert à séparer les instances proches/touchantes au besoin.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Blocs réseau
# ============================================================


def _group_norm(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False),
            _group_norm(out_c),
            nn.SiLU(inplace=True),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = ConvGNAct(in_c, out_c, 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            _group_norm(out_c),
        )
        self.skip = (
            nn.Identity()
            if in_c == out_c
            else nn.Conv2d(in_c, out_c, 1, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(y + self.skip(x))


class UpBlock(nn.Module):
    """Upsample x2, concatène le skip encodeur puis affine."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.reduce = ConvGNAct(in_c, out_c, 1)
        self.fuse = ResidualConvBlock(out_c + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


# ============================================================
# Encodeur EfficientNetV2-S
# ============================================================


class EfficientNetV2SEncoder(nn.Module):
    """
    Extrait cinq niveaux :
        s2  : 24 canaux
        s4  : 48 canaux
        s8  : 64 canaux
        s16 : 160 canaux
        s32 : 256 canaux
    """

    channels = {
        "s2": 24,
        "s4": 48,
        "s8": 64,
        "s16": 160,
        "s32": 256,
    }

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = efficientnet_v2_s(weights=weights)
        self.features = base.features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # features[0] : s2, 24
        x = self.features[0](x)
        x = self.features[1](x)
        s2 = x

        # features[2] : s4, 48
        x = self.features[2](x)
        s4 = x

        # features[3] : s8, 64
        x = self.features[3](x)
        s8 = x

        # features[4] puis [5] restent à s16 ; [5] finit à 160 canaux
        x = self.features[4](x)
        x = self.features[5](x)
        s16 = x

        # features[6] : s32, 256
        x = self.features[6](x)
        s32 = x

        return {
            "s2": s2,
            "s4": s4,
            "s8": s8,
            "s16": s16,
            "s32": s32,
        }


# ============================================================
# Modèle principal
# ============================================================


class MultiFormeNet(nn.Module):
    """
    Sorties :
        mask_logits   : (B, 1, H, W)
        center_logits : (B, 1, H/2, W/2)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.encoder = EfficientNetV2SEncoder(pretrained=pretrained)

        self.bottleneck = ResidualConvBlock(256, 256)

        self.up16 = UpBlock(256, 160, 192)
        self.up8 = UpBlock(192, 64, 128)
        self.up4 = UpBlock(128, 48, 96)
        self.up2 = UpBlock(96, 24, 64)

        # Centre à stride 2 : assez précis comme graine d'instance.
        self.center_head = nn.Sequential(
            ResidualConvBlock(64, 64),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.constant_(self.center_head[-1].bias, -2.19)

        # Raffinement pleine résolution. On réinjecte l'image originale
        # normalisée afin de conserver les bords fins que le backbone a pu lisser.
        self.mask_refine = nn.Sequential(
            ConvGNAct(64 + 3, 48, 3),
            ResidualConvBlock(48, 32),
            nn.Conv2d(32, 1, 1),
        )

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]

        feats = self.encoder(x)

        y = self.bottleneck(feats["s32"])
        y = self.up16(y, feats["s16"])
        y = self.up8(y, feats["s8"])
        y = self.up4(y, feats["s4"])
        y = self.up2(y, feats["s2"])

        center_logits = self.center_head(y)

        y_full = F.interpolate(
            y,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        mask_logits = self.mask_refine(torch.cat([y_full, x], dim=1))

        return {
            "mask_logits": mask_logits,
            "center_logits": center_logits,
        }


# ============================================================
# Losses
# ============================================================


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def _soft_boundary(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    max_v = F.max_pool2d(x, kernel_size, stride=1, padding=pad)
    min_v = -F.max_pool2d(-x, kernel_size, stride=1, padding=pad)
    return (max_v - min_v).clamp(0.0, 1.0)


def boundary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_boundary = _soft_boundary(torch.sigmoid(logits))
    gt_boundary = _soft_boundary(target)
    dims = (1, 2, 3)
    inter = (pred_boundary * gt_boundary).sum(dim=dims)
    denom = pred_boundary.sum(dim=dims) + gt_boundary.sum(dim=dims)
    score = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - score).mean()


def center_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Focal loss de type CenterNet pour une heatmap gaussienne."""
    pred = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)

    pos_mask = (target >= 0.999).float()
    neg_mask = 1.0 - pos_mask
    neg_weights = (1.0 - target).pow(beta)

    pos_loss = -torch.log(pred) * (1.0 - pred).pow(alpha) * pos_mask
    neg_loss = -torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg_mask

    n_pos = pos_mask.sum()
    if n_pos > 0:
        return (pos_loss.sum() + neg_loss.sum()) / n_pos
    return neg_loss.sum() / max(1, logits.shape[0])


def multiforme_loss(
    outputs: Dict[str, torch.Tensor],
    mask_target: torch.Tensor,
    center_target: torch.Tensor,
    w_bce: float = 1.0,
    w_dice: float = 1.0,
    w_boundary: float = 0.35,
    w_center: float = 0.50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask_logits = outputs["mask_logits"]
    center_logits = outputs["center_logits"]

    bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
    dsc = dice_loss(mask_logits, mask_target)
    boundary = boundary_dice_loss(mask_logits, mask_target)
    center = center_focal_loss(center_logits, center_target)

    total = (
        w_bce * bce
        + w_dice * dsc
        + w_boundary * boundary
        + w_center * center
    )

    return total, {
        "bce": float(bce.detach()),
        "dice_loss": float(dsc.detach()),
        "boundary": float(boundary.detach()),
        "center": float(center.detach()),
        "total": float(total.detach()),
    }


# ============================================================
# Métriques segmentation
# ============================================================


@torch.no_grad()
def segmentation_metrics(
    mask_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    pred = (torch.sigmoid(mask_logits) >= threshold).float()

    dims = (1, 2, 3)
    inter = (pred * target).sum(dim=dims)
    union = ((pred + target) > 0).float().sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    gt_sum = target.sum(dim=dims)

    iou = (inter + eps) / (union + eps)
    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)

    return {
        "iou": float(iou.mean()),
        "dice": float(dice.mean()),
    }


# ============================================================
# Post-traitement : mask -> instances -> 32 points
# ============================================================


def resample_contour(contour: np.ndarray, n_points: int = 32) -> Optional[np.ndarray]:
    contour = contour.reshape(-1, 2).astype(np.float32)
    if len(contour) < 2:
        return None

    closed = np.vstack([contour, contour[0]])
    segments = np.diff(closed, axis=0)
    lengths = np.sqrt((segments ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])

    if total < 1.0:
        return None

    targets = np.linspace(0.0, total, n_points, endpoint=False)
    output = []

    for t in targets:
        idx = np.searchsorted(cumulative, t, side="right") - 1
        idx = min(max(0, idx), len(segments) - 1)
        seg_len = lengths[idx]
        if seg_len < 1e-6:
            p = closed[idx]
        else:
            alpha = (t - cumulative[idx]) / seg_len
            p = closed[idx] + alpha * segments[idx]
        output.append(p)

    points = np.asarray(output, dtype=np.float32)

    # Orientation constante.
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if area < 0:
        points = points[::-1]

    # P0 = point le plus haut, puis le plus à gauche.
    idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    points = np.roll(points, -idx, axis=0)

    return points


def inside_center(instance_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    binary = (instance_mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return None

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, max_value, _, max_loc = cv2.minMaxLoc(distance)
    if max_value <= 0:
        return None

    return float(max_loc[0]), float(max_loc[1])


def _local_center_peaks(
    center_prob: np.ndarray,
    foreground: np.ndarray,
    threshold: float,
    min_distance: int,
) -> np.ndarray:
    k = max(3, 2 * int(min_distance) + 1)
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(center_prob.astype(np.float32), kernel)
    peaks = (
        (center_prob >= threshold)
        & (center_prob >= dilated - 1e-7)
        & (foreground > 0)
    ).astype(np.uint8)

    # Un plateau de plusieurs pixels doit devenir une seule graine.
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, 8)
    clean = np.zeros_like(peaks)
    for i in range(1, n):
        cx, cy = centroids[i]
        x = int(round(cx))
        y = int(round(cy))
        if 0 <= x < clean.shape[1] and 0 <= y < clean.shape[0]:
            clean[y, x] = 1
    return clean


def split_instances_with_centers(
    binary_mask: np.ndarray,
    center_prob: Optional[np.ndarray] = None,
    center_threshold: float = 0.35,
    center_min_distance: int = 6,
    min_area: int = 80,
) -> List[np.ndarray]:
    """
    Sépare les instances.

    - Si les composantes sont déjà séparées, connectedComponents suffit.
    - Si une composante contient plusieurs pics de centre, watershed piloté par
      ces centres la sépare.
    """
    binary = (binary_mask > 0).astype(np.uint8)
    n_cc, cc = cv2.connectedComponents(binary, 8)
    instances: List[np.ndarray] = []

    if center_prob is None:
        for cid in range(1, n_cc):
            inst = (cc == cid).astype(np.uint8)
            if int(inst.sum()) >= min_area:
                instances.append(inst)
        return instances

    peaks = _local_center_peaks(
        center_prob,
        binary,
        threshold=center_threshold,
        min_distance=center_min_distance,
    )

    for cid in range(1, n_cc):
        component = (cc == cid).astype(np.uint8)
        area = int(component.sum())
        if area < min_area:
            continue

        peak_inside = (peaks * component).astype(np.uint8)
        n_peak, peak_labels = cv2.connectedComponents(peak_inside, 8)
        n_centers = n_peak - 1

        if n_centers <= 1:
            instances.append(component)
            continue

        # Marker-based watershed sur l'inverse de la distance au bord.
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            instances.append(component)
            continue

        surface = 255.0 * (1.0 - dist / (dist.max() + 1e-6))
        surface = surface.astype(np.uint8)
        surface_bgr = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

        # 1 = fond connu ; 0 = intérieur inconnu ; 2... = centres.
        markers = np.ones(component.shape, dtype=np.int32)
        markers[component > 0] = 0

        peak_coords = np.argwhere(peak_inside > 0)
        for marker_id, (py, px) in enumerate(peak_coords, start=2):
            cv2.circle(markers, (int(px), int(py)), 2, int(marker_id), -1)

        cv2.watershed(surface_bgr, markers)

        produced = 0
        for marker_id in range(2, 2 + len(peak_coords)):
            inst = ((markers == marker_id) & (component > 0)).astype(np.uint8)
            if int(inst.sum()) >= min_area:
                instances.append(inst)
                produced += 1

        if produced == 0:
            instances.append(component)

    return instances


def extract_instances(
    mask_prob: np.ndarray,
    center_prob: Optional[np.ndarray] = None,
    mask_threshold: float = 0.50,
    center_threshold: float = 0.35,
    n_points: int = 32,
    min_area: int = 80,
) -> List[Dict[str, np.ndarray]]:
    """
    Convertit les sorties réseau en objets :
        {
            "center": (cx, cy) en pixels,
            "points": (N, 2) en pixels,
            "mask": masque uint8 de l'instance,
            "area": aire en pixels,
        }
    """
    if mask_prob.ndim != 2:
        raise ValueError("mask_prob doit être 2D")

    binary = (mask_prob >= mask_threshold).astype(np.uint8)

    # Nettoyage léger sans détruire les petits détails.
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    if center_prob is not None and center_prob.shape != binary.shape:
        center_prob = cv2.resize(
            center_prob.astype(np.float32),
            (binary.shape[1], binary.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    instance_masks = split_instances_with_centers(
        binary,
        center_prob=center_prob,
        center_threshold=center_threshold,
        min_area=min_area,
    )

    results = []
    for inst in instance_masks:
        contours, _ = cv2.findContours(
            (inst * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue

        points = resample_contour(contour, n_points=n_points)
        center = inside_center(inst)
        if points is None or center is None:
            continue

        results.append(
            {
                "center": np.asarray(center, dtype=np.float32),
                "points": points,
                "mask": inst,
                "area": float(cv2.contourArea(contour)),
            }
        )

    results.sort(key=lambda obj: obj["area"], reverse=True)
    return results


# ============================================================
# Construction
# ============================================================


def build_model(pretrained: bool = True) -> MultiFormeNet:
    return MultiFormeNet(pretrained=pretrained)


if __name__ == "__main__":
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print("mask_logits  :", tuple(out["mask_logits"].shape))
    print("center_logits:", tuple(out["center_logits"].shape))
    print("parameters   :", f"{sum(p.numel() for p in model.parameters()):,}")
