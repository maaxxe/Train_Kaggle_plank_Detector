from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(path: Path):
    import re
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    ]


def read_polygon_label(label_path: Path):
    """
    Format attendu par ligne:
        classe x1 y1 x2 y2 x3 y3 x4 y4

    Coordonnées normalisées dans [0, 1].
    """
    polygons = []

    if not label_path.exists():
        return polygons

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:9]))

        pts = np.asarray(coords, dtype=np.float32).reshape(4, 2)
        pts = np.clip(pts, 0.0, 1.0)

        polygons.append((cls, pts))

    return polygons


def polygons_to_mask(polygons, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    for _, pts_norm in polygons:
        pts = pts_norm.copy()
        pts[:, 0] *= max(width - 1, 1)
        pts[:, 1] *= max(height - 1, 1)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def denorm_image(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (t.clamp(-1, 1) + 1.0) * 0.5


def save_triplet_grid(mask, fake, real, path: Path, max_items: int = 4):
    """
    Sauvegarde une grille:
      ligne 1 : masques
      ligne 2 : images GAN
      ligne 3 : images réelles
    """
    from torchvision.utils import make_grid, save_image

    n = min(max_items, mask.shape[0])

    mask_rgb = mask[:n].repeat(1, 3, 1, 1)
    fake = denorm_image(fake[:n])
    real = denorm_image(real[:n])

    grid = torch.cat([mask_rgb, fake, real], dim=0)
    grid = make_grid(grid, nrow=n, padding=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(path))
