from __future__ import annotations

"""Training script for PlankEye V7.

Designed for:
- one or several GPUs through torchrun / DistributedDataParallel (DDP);
- NVIDIA T4 GPUs with AMP FP16;
- on-the-fly image + quadrilateral augmentation;
- labels formatted as:
      class x1 y1 x2 y2 x3 y3 x4 y4
  with normalized coordinates in [0, 1];
- one or several objects per image;
- robust checkpoints containing model/EMA/optimizer/scheduler/scaler,
  epoch, losses, complete history and training metadata.

Expected project layout:
    project/
      model_v7.py
      train_v7.py
      data_kaggle_2_propre/
        images/
        labels/

Recommended 2-GPU launch:
    torchrun --standalone --nproc_per_node=2 train_v7.py \
        --data data_kaggle_2_propre \
        --output runs/plankeye_v7

Progressive backbone unfreezing (default, 1-based epochs):
- epochs 1-3  : heads/neck only, ResNet50 backbone frozen;
- epochs 4-8  : ResNet50 layer4 trainable;
- epochs 9-15 : ResNet50 layer3 + layer4 trainable;
- epoch 16+   : complete ResNet50 trainable.
Backbone BatchNorm layers stay frozen by default.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from model_v7 import (
    IMG_SIZE,
    MODEL_VERSION,
    NUM_CLASSES,
    PlankEyeV7,
    combined_loss_v7,
    model_metadata,
)

print("====================Train_V7=====================")
# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

CHECKPOINT_VERSION = 3
DEFAULT_SEED = 1337

# OpenCV uses its own thread pool. DataLoader already parallelizes loading.
cv2.setNumThreads(0)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AugmentConfig:
    horizontal_flip: float = 0.50
    vertical_flip: float = 0.05
    rotate90: float = 0.15
    affine_perspective: float = 0.60
    max_rotate_deg: float = 12.0
    scale_min: float = 0.90
    scale_max: float = 1.10
    max_translate_fraction: float = 0.04
    max_perspective_fraction: float = 0.025

    brightness_contrast: float = 0.55
    gamma: float = 0.25
    hsv: float = 0.35
    grayscale: float = 0.04
    gaussian_blur: float = 0.10
    motion_blur: float = 0.08
    gaussian_noise: float = 0.16
    jpeg_artifacts: float = 0.10
    sharpen: float = 0.10
    illumination_gradient: float = 0.18


# =============================================================================
# DISTRIBUTED HELPERS
# =============================================================================


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def all_ranks_finite(value: torch.Tensor, device: torch.device, distributed: bool) -> bool:
    local = torch.tensor(
        1 if torch.isfinite(value.detach()).all() else 0,
        dtype=torch.int32,
        device=device,
    )
    if distributed:
        dist.all_reduce(local, op=dist.ReduceOp.MIN)
    return bool(local.item())


class DistributedEvalSampler(Sampler[int]):
    """Shard validation data without padding/duplicating samples."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        n = len(self.dataset)
        if n <= self.rank:
            return 0
        return (n - 1 - self.rank) // self.world_size + 1


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # torch.initial_seed() is already unique per rank/worker/epoch.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# =============================================================================
# LABEL / GEOMETRY HELPERS
# =============================================================================


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return a stable cyclic TL->TR->BR->BL-like order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)

    # In image coordinates y grows downward. Prefer the second point to be on
    # the right side, giving a stable clockwise-looking order.
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = np.concatenate([ordered[:1], ordered[:0:-1]], axis=0)

    return ordered.astype(np.float32)


def signed_polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(points: np.ndarray) -> float:
    return abs(signed_polygon_area(points))


def is_convex_quad(points: np.ndarray, eps: float = 1e-8) -> bool:
    pts = order_quad(points)
    crosses = []
    for i in range(4):
        a = pts[(i + 1) % 4] - pts[i]
        b = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
    pos = all(v > eps for v in crosses)
    neg = all(v < -eps for v in crosses)
    return pos or neg


def find_label_path(image_path: Path, labels_dir: Path) -> Path | None:
    stem = image_path.stem
    candidates: List[Path] = []

    if stem.startswith("image"):
        suffix = stem[len("image") :]
        candidates.extend(
            [
                labels_dir / f"label{suffix}.txt",
                labels_dir / f"image{suffix}.txt",
            ]
        )

    candidates.extend(
        [
            labels_dir / f"{stem}.txt",
            labels_dir / f"label{stem}.txt",
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def parse_label_file(
    label_path: Path,
    *,
    clamp_tolerance: float = 0.02,
    min_area: float = 1e-5,
) -> List[dict]:
    objects: List[dict] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="latin-1").splitlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 9:
            raise ValueError(
                f"{label_path.name}:{line_number}: expected 9 values, got {len(parts)}"
            )

        try:
            cls = int(float(parts[0]))
            coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
        except ValueError as exc:
            raise ValueError(
                f"{label_path.name}:{line_number}: non-numeric label"
            ) from exc

        if cls < 0 or cls >= NUM_CLASSES:
            raise ValueError(
                f"{label_path.name}:{line_number}: class {cls} outside [0,{NUM_CLASSES - 1}]"
            )

        if not np.isfinite(coords).all():
            raise ValueError(f"{label_path.name}:{line_number}: NaN/Inf coordinates")

        if coords.min() < -clamp_tolerance or coords.max() > 1.0 + clamp_tolerance:
            raise ValueError(
                f"{label_path.name}:{line_number}: normalized coordinates outside [0,1]"
            )

        coords = np.clip(coords, 0.0, 1.0)
        coords = order_quad(coords)

        if polygon_area(coords) < min_area:
            raise ValueError(f"{label_path.name}:{line_number}: degenerate quadrilateral")

        if not is_convex_quad(coords):
            raise ValueError(f"{label_path.name}:{line_number}: non-convex quadrilateral")

        objects.append({"cls": cls, "corners": coords})

    if not objects:
        raise ValueError(f"{label_path.name}: no valid object")

    return objects


# =============================================================================
# DATASET DISCOVERY / SPLIT
# =============================================================================


def dataset_signature(records: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for record in records:
        h.update(record["image"].encode("utf-8"))
        h.update(b"\0")
        h.update(record["label"].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def scan_dataset(data_dir: Path, verbose: bool = True) -> Tuple[List[dict], dict]:
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    records: List[dict] = []
    errors: List[str] = []
    total_objects = 0

    iterator: Iterable[Path] = image_paths
    if verbose:
        iterator = tqdm(image_paths, desc="Dataset scan", dynamic_ncols=True)

    for image_path in iterator:
        label_path = find_label_path(image_path, labels_dir)
        if label_path is None:
            errors.append(f"missing label: {image_path.name}")
            continue

        # Check that OpenCV can decode the image now, not after several epochs.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[0] < 8 or image.shape[1] < 8:
            errors.append(f"unreadable image: {image_path.name}")
            continue

        try:
            objects = parse_label_file(label_path)
        except Exception as exc:
            errors.append(str(exc))
            continue

        total_objects += len(objects)
        records.append(
            {
                "image": str(image_path.relative_to(data_dir)),
                "label": str(label_path.relative_to(data_dir)),
                "objects": len(objects),
            }
        )

    stats = {
        "images_found": len(image_paths),
        "valid_images": len(records),
        "invalid_images": len(errors),
        "total_objects": total_objects,
        "dataset_signature": dataset_signature(records),
        "errors_preview": errors[:100],
    }

    if not records:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"No valid training sample found.\n{preview}")

    return records, stats


def create_or_load_split(
    data_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    rank: int,
    distributed: bool,
) -> Tuple[List[dict], List[dict], dict]:
    split_path = output_dir / "dataset_split.json"

    if rank == 0:
        records, stats = scan_dataset(data_dir, verbose=True)

        if split_path.exists():
            existing = json.loads(split_path.read_text(encoding="utf-8"))
            old_signature = existing.get("dataset_signature")
            new_signature = stats["dataset_signature"]

            if old_signature == new_signature:
                payload = existing
                print(f"Using existing dataset split: {split_path}")
            else:
                print(
                    "Dataset changed since the previous split; creating a new deterministic split."
                )
                payload = _build_split_payload(records, stats, val_ratio, seed)
                split_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        else:
            payload = _build_split_payload(records, stats, val_ratio, seed)
            split_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    if distributed:
        barrier()

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return payload["train"], payload["val"], payload


def _build_split_payload(
    records: Sequence[dict],
    stats: Mapping[str, Any],
    val_ratio: float,
    seed: int,
) -> dict:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        train_records = shuffled
        val_records = shuffled
    else:
        val_count = int(round(len(shuffled) * val_ratio))
        val_count = min(max(1, val_count), len(shuffled) - 1)
        val_records = shuffled[:val_count]
        train_records = shuffled[val_count:]

    return {
        "version": 1,
        "seed": seed,
        "val_ratio": val_ratio,
        "dataset_signature": stats["dataset_signature"],
        "dataset_stats": dict(stats),
        "train": train_records,
        "val": val_records,
    }


# =============================================================================
# AUGMENTATION
# =============================================================================


def normalized_to_pixels(quads: np.ndarray, width: int, height: int) -> np.ndarray:
    out = quads.astype(np.float32).copy()
    out[..., 0] *= float(width - 1)
    out[..., 1] *= float(height - 1)
    return out


def transform_points_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    shape = points.shape
    flat = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(flat, matrix.astype(np.float32))
    return transformed.reshape(shape).astype(np.float32)


def _valid_pixel_quads(quads: np.ndarray, width: int, height: int) -> bool:
    if not np.isfinite(quads).all():
        return False

    if quads[..., 0].min() < 0.0 or quads[..., 0].max() > width - 1:
        return False
    if quads[..., 1].min() < 0.0 or quads[..., 1].max() > height - 1:
        return False

    min_area = max(12.0, width * height * 1e-5)
    for quad in quads:
        ordered = order_quad(quad)
        if polygon_area(ordered) < min_area:
            return False
        if not is_convex_quad(ordered):
            return False
    return True


def horizontal_flip(image: np.ndarray, quads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    image = cv2.flip(image, 1)
    q = quads.copy()
    q[..., 0] = (w - 1) - q[..., 0]
    q = np.stack([order_quad(x) for x in q], axis=0)
    return image, q


def vertical_flip(image: np.ndarray, quads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    image = cv2.flip(image, 0)
    q = quads.copy()
    q[..., 1] = (h - 1) - q[..., 1]
    q = np.stack([order_quad(x) for x in q], axis=0)
    return image, q


def rotate_90(image: np.ndarray, quads: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = int(k) % 4
    if k == 0:
        return image, quads

    old_h, old_w = image.shape[:2]
    q = quads.copy()
    x = q[..., 0].copy()
    y = q[..., 1].copy()

    image = np.rot90(image, k=k).copy()

    if k == 1:  # CCW
        q[..., 0] = y
        q[..., 1] = old_w - 1 - x
    elif k == 2:
        q[..., 0] = old_w - 1 - x
        q[..., 1] = old_h - 1 - y
    else:  # k == 3, clockwise
        q[..., 0] = old_h - 1 - y
        q[..., 1] = x

    q = np.stack([order_quad(quad) for quad in q], axis=0)
    return image, q


def random_affine_perspective(
    image: np.ndarray,
    quads: np.ndarray,
    cfg: AugmentConfig,
    max_attempts: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if min(h, w) < 16:
        return image, quads

    src_corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )

    for _ in range(max_attempts):
        angle = random.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
        scale = random.uniform(cfg.scale_min, cfg.scale_max)
        tx = random.uniform(-cfg.max_translate_fraction, cfg.max_translate_fraction) * w
        ty = random.uniform(-cfg.max_translate_fraction, cfg.max_translate_fraction) * h

        affine = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        affine[0, 2] += tx
        affine[1, 2] += ty
        affine3 = np.eye(3, dtype=np.float32)
        affine3[:2] = affine.astype(np.float32)

        jitter_px = cfg.max_perspective_fraction * min(w, h)
        dst_corners = src_corners + np.random.uniform(
            -jitter_px,
            jitter_px,
            size=(4, 2),
        ).astype(np.float32)
        perspective = cv2.getPerspectiveTransform(src_corners, dst_corners)
        matrix = perspective @ affine3

        transformed_quads = transform_points_homography(quads, matrix)
        if not _valid_pixel_quads(transformed_quads, w, h):
            continue

        warped = cv2.warpPerspective(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        transformed_quads = np.stack(
            [order_quad(quad) for quad in transformed_quads], axis=0
        )
        return warped, transformed_quads

    return image, quads


def random_brightness_contrast(image: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.78, 1.22)
    beta = random.uniform(-28.0, 28.0)
    out = image.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def random_gamma(image: np.ndarray) -> np.ndarray:
    gamma = random.uniform(0.75, 1.35)
    inv_gamma = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv_gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def random_hsv(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-8.0, 8.0)) % 180.0
    hsv[..., 1] *= random.uniform(0.75, 1.25)
    hsv[..., 2] *= random.uniform(0.80, 1.20)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def random_motion_blur(image: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    return cv2.filter2D(image, -1, kernel)


def random_gaussian_noise(image: np.ndarray) -> np.ndarray:
    sigma = random.uniform(2.0, 10.0)
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def random_jpeg_artifacts(image: np.ndarray) -> np.ndarray:
    quality = random.randint(45, 92)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return image
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def random_sharpen(image: np.ndarray) -> np.ndarray:
    sigma = random.uniform(0.7, 1.4)
    amount = random.uniform(0.4, 1.0)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    out = image.astype(np.float32) * (1.0 + amount) - blurred.astype(np.float32) * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def random_illumination_gradient(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    angle = random.uniform(0.0, 2.0 * math.pi)
    gradient = math.cos(angle) * x + math.sin(angle) * y
    strength = random.uniform(-35.0, 35.0)
    bias = gradient[..., None] * strength
    out = image.astype(np.float32) + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_image_and_quads(
    image: np.ndarray,
    quads: np.ndarray,
    cfg: AugmentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    # Geometry first so labels follow exactly the same transform.
    if random.random() < cfg.horizontal_flip:
        image, quads = horizontal_flip(image, quads)

    if random.random() < cfg.vertical_flip:
        image, quads = vertical_flip(image, quads)

    if random.random() < cfg.rotate90:
        image, quads = rotate_90(image, quads, random.choice([1, 2, 3]))

    if random.random() < cfg.affine_perspective:
        image, quads = random_affine_perspective(image, quads, cfg)

    # Photometric transformations do not modify labels.
    if random.random() < cfg.brightness_contrast:
        image = random_brightness_contrast(image)
    if random.random() < cfg.gamma:
        image = random_gamma(image)
    if random.random() < cfg.hsv:
        image = random_hsv(image)
    if random.random() < cfg.grayscale:
        image = random_grayscale(image)
    if random.random() < cfg.gaussian_blur:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)
    if random.random() < cfg.motion_blur:
        image = random_motion_blur(image)
    if random.random() < cfg.gaussian_noise:
        image = random_gaussian_noise(image)
    if random.random() < cfg.jpeg_artifacts:
        image = random_jpeg_artifacts(image)
    if random.random() < cfg.sharpen:
        image = random_sharpen(image)
    if random.random() < cfg.illumination_gradient:
        image = random_illumination_gradient(image)

    return np.ascontiguousarray(image), np.ascontiguousarray(quads)


def letterbox(
    image: np.ndarray,
    quads: np.ndarray,
    size: int,
    random_padding: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = min(size / float(w), size / float(h))

    new_w = max(1, min(size, int(round(w * scale))))
    new_h = max(1, min(size, int(round(h * scale))))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA,
    )

    pad_w = size - new_w
    pad_h = size - new_h

    if random_padding:
        left = random.randint(0, pad_w) if pad_w > 0 else 0
        top = random.randint(0, pad_h) if pad_h > 0 else 0
    else:
        left = pad_w // 2
        top = pad_h // 2

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized

    q = quads.astype(np.float32).copy()
    q[..., 0] = q[..., 0] * scale + left
    q[..., 1] = q[..., 1] * scale + top
    q[..., 0] /= float(size)
    q[..., 1] /= float(size)
    q = np.clip(q, 0.0, 1.0)
    q = np.stack([order_quad(quad) for quad in q], axis=0)

    return canvas, q


# =============================================================================
# DATASET
# =============================================================================


class PlankDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        records: Sequence[dict],
        image_size: int,
        augment: bool,
        augment_cfg: AugmentConfig,
    ):
        self.data_dir = Path(data_dir)
        self.records = list(records)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.augment_cfg = augment_cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.data_dir / record["image"]
        label_path = self.data_dir / record["label"]

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV failed to read {image_path}")

        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        parsed = parse_label_file(label_path)
        classes = [int(obj["cls"]) for obj in parsed]
        quads_norm = np.stack([obj["corners"] for obj in parsed], axis=0).astype(np.float32)
        quads_px = normalized_to_pixels(quads_norm, w, h)

        if self.augment:
            image, quads_px = augment_image_and_quads(
                image,
                quads_px,
                self.augment_cfg,
            )

        image, quads_final = letterbox(
            image,
            quads_px,
            size=self.image_size,
            random_padding=self.augment,
        )

        objects = [
            {"cls": cls, "corners": quad.astype(np.float32)}
            for cls, quad in zip(classes, quads_final)
        ]

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32).div_(255.0)

        meta = {
            "index": index,
            "image": record["image"],
            "label": record["label"],
            "objects": len(objects),
        }
        return tensor, objects, meta


def collate_fn(batch):
    images, objects, metas = zip(*batch)
    return torch.stack(images, dim=0), list(objects), list(metas)


# =============================================================================
# EMA
# =============================================================================


class ModelEMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9998,
        tau: float = 2000.0,
    ):
        self.ema = deepcopy(unwrap_model(model)).eval()
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0

        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def _decay(self) -> float:
        # Gentle early updates, asymptotically approaching self.decay.
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self._decay()
        source = unwrap_model(model).state_dict()
        target = self.ema.state_dict()

        for key, ema_value in target.items():
            source_value = source[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(source_value)

    def state_dict(self) -> dict:
        return {
            "updates": self.updates,
            "decay": self.decay,
            "tau": self.tau,
            "model": self.ema.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.updates = int(state.get("updates", 0))
        self.decay = float(state.get("decay", self.decay))
        self.tau = float(state.get("tau", self.tau))
        model_state = state.get("model", state)
        self.ema.load_state_dict(model_state, strict=True)


# =============================================================================
# OPTIMIZER / SCHEDULER
# =============================================================================


def build_optimizer(
    model: nn.Module,
    lr: float,
    backbone_lr_mult: float,
    weight_decay: float,
) -> AdamW:
    groups: Dict[Tuple[bool, bool], List[nn.Parameter]] = defaultdict(list)

    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue

        is_backbone = name.startswith("backbone.")
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        groups[(is_backbone, no_decay)].append(parameter)

    param_groups = []
    for (is_backbone, no_decay), params in groups.items():
        group_lr = lr * (backbone_lr_mult if is_backbone else 1.0)
        param_groups.append(
            {
                "params": params,
                "lr": group_lr,
                "initial_lr": group_lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "group_name": (
                    "backbone_no_decay"
                    if is_backbone and no_decay
                    else "backbone_decay"
                    if is_backbone
                    else "head_no_decay"
                    if no_decay
                    else "head_decay"
                ),
            }
        )

    return AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # Start at 10% LR and reach 100% at the end of warmup.
            progress = (step + 1) / float(warmup_steps)
            return 0.10 + 0.90 * progress

        if total_steps <= warmup_steps + 1:
            return 1.0

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# =============================================================================
# PROGRESSIVE BACKBONE UNFREEZING
# =============================================================================


BACKBONE_STAGE_FROZEN = "heads_only"
BACKBONE_STAGE_LAYER4 = "layer4"
BACKBONE_STAGE_LAYER34 = "layer3_layer4"
BACKBONE_STAGE_FULL = "full_backbone"


def progressive_unfreeze_enabled(
    args: argparse.Namespace,
    pretrained_backbone: bool,
) -> bool:
    """Enable gradual unfreezing only for a pretrained backbone by default."""
    return bool(pretrained_backbone and not args.no_progressive_unfreeze)


def backbone_stage_for_epoch(
    epoch: int,
    args: argparse.Namespace,
    enabled: bool,
) -> str:
    """Return the desired backbone trainability stage for a zero-based epoch."""
    if not enabled:
        return BACKBONE_STAGE_FULL

    epoch_human = int(epoch) + 1

    if epoch_human < args.unfreeze_layer4_epoch:
        return BACKBONE_STAGE_FROZEN

    if epoch_human < args.unfreeze_layer3_epoch:
        return BACKBONE_STAGE_LAYER4

    if epoch_human < args.unfreeze_all_epoch:
        return BACKBONE_STAGE_LAYER34

    return BACKBONE_STAGE_FULL


def _set_module_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def apply_backbone_stage(
    model: nn.Module,
    stage: str,
    *,
    freeze_backbone_bn: bool,
) -> Dict[str, int | str]:
    """Apply one progressive unfreezing stage to the ResNet50 backbone.

    The optimizer must be built BEFORE the first call to this function so all
    backbone parameters are already present in its parameter groups. Frozen
    parameters then simply have no gradient/update until they are unfrozen.
    """
    base_model = unwrap_model(model)

    if not hasattr(base_model, "backbone"):
        raise AttributeError("PlankEyeV7 model has no 'backbone' attribute.")

    backbone = base_model.backbone

    # Start from a fully frozen backbone, then selectively enable stages.
    _set_module_requires_grad(backbone, False)

    if stage == BACKBONE_STAGE_FROZEN:
        pass

    elif stage == BACKBONE_STAGE_LAYER4:
        _set_module_requires_grad(backbone.layer4, True)

    elif stage == BACKBONE_STAGE_LAYER34:
        _set_module_requires_grad(backbone.layer3, True)
        _set_module_requires_grad(backbone.layer4, True)

    elif stage == BACKBONE_STAGE_FULL:
        _set_module_requires_grad(backbone, True)

    else:
        raise ValueError(f"Unknown backbone stage: {stage!r}")

    # With the small per-GPU batches used on the T4s, keeping pretrained
    # BatchNorm statistics/affine parameters frozen is substantially safer.
    if freeze_backbone_bn:
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    backbone_total = sum(parameter.numel() for parameter in backbone.parameters())
    backbone_trainable = sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )
    model_total = sum(parameter.numel() for parameter in base_model.parameters())
    model_trainable = sum(
        parameter.numel()
        for parameter in base_model.parameters()
        if parameter.requires_grad
    )

    return {
        "stage": stage,
        "backbone_total": int(backbone_total),
        "backbone_trainable": int(backbone_trainable),
        "model_total": int(model_total),
        "model_trainable": int(model_trainable),
    }


def format_trainable_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def print_backbone_stage(
    info: Mapping[str, Any],
    *,
    epoch: int,
    rank: int,
) -> None:
    if rank != 0:
        return

    backbone_trainable = int(info["backbone_trainable"])
    backbone_total = max(1, int(info["backbone_total"]))
    model_trainable = int(info["model_trainable"])
    model_total = max(1, int(info["model_total"]))

    print()
    print("=" * 88)
    print(f"BACKBONE UNFREEZE STAGE - EPOCH {epoch + 1}")
    print("=" * 88)
    print(f"Stage                  : {info['stage']}")
    print(
        "Backbone trainable     : "
        f"{format_trainable_count(backbone_trainable)} / "
        f"{format_trainable_count(backbone_total)} "
        f"({100.0 * backbone_trainable / backbone_total:.1f}%)"
    )
    print(
        "Whole model trainable  : "
        f"{format_trainable_count(model_trainable)} / "
        f"{format_trainable_count(model_total)} "
        f"({100.0 * model_trainable / model_total:.1f}%)"
    )
    print("=" * 88)
    print()


# =============================================================================
# METRIC ACCUMULATION
# =============================================================================


class MetricAccumulator:
    def __init__(self):
        self.weighted_sums: Dict[str, float] = defaultdict(float)
        self.weight = 0.0
        self.num_positive = 0.0
        self.samples = 0.0
        self.batches = 0.0

    def update(self, logs: Mapping[str, Any], batch_size: int) -> None:
        weight = float(batch_size)
        for key, value in logs.items():
            if key == "num_positive":
                self.num_positive += float(value)
            else:
                self.weighted_sums[key] += float(value) * weight
        self.weight += weight
        self.samples += weight
        self.batches += 1.0

    def synchronize(self, device: torch.device, distributed: bool) -> Dict[str, float]:
        keys = sorted(self.weighted_sums.keys())
        values = [self.weight, self.samples, self.batches, self.num_positive]
        values.extend(self.weighted_sums[key] for key in keys)

        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        values = tensor.cpu().tolist()

        weight, samples, batches, positives = values[:4]
        sums = values[4:]

        result = {
            key: sums[i] / max(weight, 1.0)
            for i, key in enumerate(keys)
        }
        result["num_positive"] = positives
        result["samples"] = samples
        result["batches"] = batches
        return result


# =============================================================================
# TRAIN / VALIDATION
# =============================================================================


def move_images(
    images: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    images = images.to(device=device, non_blocking=True)
    if channels_last and images.ndim == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def current_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{index}"))
        result[name] = float(group["lr"])
    return result


def train_one_epoch(
    *,
    model: nn.Module,
    ema: ModelEMA | None,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    rank: int,
    distributed: bool,
    grad_accum: int,
    max_grad_norm: float,
    amp_enabled: bool,
    channels_last: bool,
) -> Tuple[Dict[str, float], int]:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)
    accumulator = MetricAccumulator()
    skipped_nonfinite = 0
    optimizer_steps = 0

    progress = tqdm(
        loader,
        desc=f"Train {epoch + 1:03d}/{total_epochs:03d}",
        dynamic_ncols=True,
        disable=rank != 0,
    )

    accumulation_count = 0

    for batch_index, (images, batch_objects, _metas) in enumerate(progress):
        images = move_images(images, device, channels_last)
        batch_size = int(images.shape[0])

        # DDP no_sync avoids unnecessary gradient synchronization on non-step
        # accumulation micro-batches.
        is_last_batch = batch_index + 1 == len(loader)
        will_step = accumulation_count + 1 >= grad_accum or is_last_batch
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and not will_step
            else _NullContext()
        )

        with sync_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                preds = model(images)
                loss, logs = combined_loss_v7(preds, batch_objects, device)

            finite_everywhere = all_ranks_finite(loss, device, distributed)
            if not finite_everywhere:
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                if rank == 0:
                    progress.write(
                        f"WARNING: non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}; batch skipped."
                    )
                continue

            scaled_loss = loss / float(grad_accum)
            scaler.scale(scaled_loss).backward()
            accumulation_count += 1

        accumulator.update(logs, batch_size=batch_size)

        if will_step and accumulation_count > 0:
            scaler.unscale_(optimizer)

            # If the final accumulation window contains fewer micro-batches
            # than grad_accum, compensate for the earlier division so the
            # effective gradient keeps the same scale.
            if accumulation_count < grad_accum:
                correction = float(grad_accum) / float(accumulation_count)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)

            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )
            else:
                grad_norm = torch.tensor(0.0, device=device)

            grad_finite = all_ranks_finite(grad_norm, device, distributed)
            if grad_finite:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer_steps += 1
                if ema is not None:
                    ema.update(model)
            else:
                skipped_nonfinite += 1
                scaler.update()
                if rank == 0:
                    progress.write(
                        f"WARNING: non-finite gradient norm at epoch {epoch + 1}, batch {batch_index + 1}; step skipped."
                    )

            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0

        if rank == 0:
            lr_values = current_learning_rates(optimizer)
            head_lr = lr_values.get("head_decay", next(iter(lr_values.values())))
            progress.set_postfix(
                loss=f"{float(logs['total']):.4f}",
                corner=f"{float(logs['corner_delta']):.4f}",
                hm=f"{float(logs['heatmap']):.4f}",
                lr=f"{head_lr:.2e}",
            )

    metrics = accumulator.synchronize(device, distributed)
    metrics["skipped_nonfinite"] = float(skipped_nonfinite)
    metrics["optimizer_steps"] = float(optimizer_steps)
    return metrics, optimizer_steps


@torch.no_grad()
def validate_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    rank: int,
    distributed: bool,
    amp_enabled: bool,
    channels_last: bool,
) -> Dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()

    progress = tqdm(
        loader,
        desc=f"Val   {epoch + 1:03d}/{total_epochs:03d}",
        dynamic_ncols=True,
        disable=rank != 0,
    )

    for images, batch_objects, _metas in progress:
        images = move_images(images, device, channels_last)
        batch_size = int(images.shape[0])

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            preds = model(images)
            _loss, logs = combined_loss_v7(preds, batch_objects, device)

        accumulator.update(logs, batch_size=batch_size)

        if rank == 0:
            progress.set_postfix(
                loss=f"{float(logs['total']):.4f}",
                corner=f"{float(logs['corner_delta']):.4f}",
                hm=f"{float(logs['heatmap']):.4f}",
            )

    return accumulator.synchronize(device, distributed)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# =============================================================================
# CHECKPOINTING / HISTORY
# =============================================================================


def git_info(project_dir: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_dir,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def runtime_info(device: torch.device, world_size: int) -> dict:
    gpu_names = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(index))

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "world_size": world_size,
        "device": str(device),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA | None,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    epoch: int,
    args: argparse.Namespace,
    augment_cfg: AugmentConfig,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    history: Sequence[dict],
    best_val_loss: float,
    global_optimizer_step: int,
    backbone_stage: str,
    backbone_trainability: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    git: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_version": MODEL_VERSION,
        "model_metadata": model_metadata(),
        "epoch": int(epoch),
        "epoch_human": int(epoch + 1),
        "epochs_requested": int(args.epochs),
        "global_optimizer_step": int(global_optimizer_step),
        "backbone_stage": str(backbone_stage),
        "backbone_trainability": dict(backbone_trainability),
        "progressive_unfreeze": {
            "enabled": bool(
                not args.no_progressive_unfreeze
                and not args.no_pretrained
            ),
            "unfreeze_layer4_epoch": int(args.unfreeze_layer4_epoch),
            "unfreeze_layer3_epoch": int(args.unfreeze_layer3_epoch),
            "unfreeze_all_epoch": int(args.unfreeze_all_epoch),
            "backbone_bn_frozen": bool(not args.unfreeze_backbone_bn),
        },
        "best_val_loss": float(best_val_loss),
        "train_loss": float(train_metrics.get("total", math.nan)),
        "val_loss": float(val_metrics.get("total", math.nan)),
        "train_metrics": dict(train_metrics),
        "val_metrics": dict(val_metrics),
        "history": list(history),
        "learning_rates": current_learning_rates(optimizer),
        "model_state_dict": unwrap_model(model).state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": vars(args),
        "augmentation": asdict(augment_cfg),
        "dataset": {
            "signature": split_payload.get("dataset_signature"),
            "stats": split_payload.get("dataset_stats"),
            "train_count": len(split_payload.get("train", [])),
            "val_count": len(split_payload.get("val", [])),
            "split_seed": split_payload.get("seed"),
            "val_ratio": split_payload.get("val_ratio"),
        },
        "runtime": dict(runtime),
        "git": dict(git),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def resolve_resume_path(resume: str, output_dir: Path) -> Path | None:
    value = resume.strip()
    if not value:
        return None
    if value.lower() == "auto":
        candidate = output_dir / "last.pt"
        return candidate if candidate.exists() else None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA | None,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
) -> Tuple[int, float, List[dict], int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    ckpt_model_version = checkpoint.get("model_version")
    if ckpt_model_version and ckpt_model_version != MODEL_VERSION:
        raise RuntimeError(
            f"Checkpoint model version mismatch: {ckpt_model_version!r} != {MODEL_VERSION!r}"
        )

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)

    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    elif ema is not None:
        ema.ema.load_state_dict(unwrap_model(model).state_dict(), strict=True)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    history = list(checkpoint.get("history", []))
    global_step = int(checkpoint.get("global_optimizer_step", 0))

    return start_epoch, best_val_loss, history, global_step


def save_history_files(output_dir: Path, history: Sequence[dict]) -> None:
    json_path = output_dir / "history.json"
    json_path.write_text(
        json.dumps(list(history), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not history:
        return

    flat_rows = []
    for item in history:
        row = {
            "epoch": item["epoch"],
            "epoch_human": item["epoch_human"],
            "duration_seconds": item.get("duration_seconds"),
            "backbone_stage": item.get("backbone_stage"),
            "backbone_trainable": item.get("backbone_trainable"),
        }
        for prefix in ("train", "val"):
            for key, value in item.get(prefix, {}).items():
                row[f"{prefix}_{key}"] = value
        for key, value in item.get("learning_rates", {}).items():
            row[f"lr_{key}"] = value
        flat_rows.append(row)

    all_fields = sorted({key for row in flat_rows for key in row.keys()})
    preferred = ["epoch", "epoch_human", "duration_seconds", "backbone_stage", "backbone_trainable"]
    fields = preferred + [f for f in all_fields if f not in preferred]

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)

    # Best-effort plot; training never fails because matplotlib is unavailable.
    try:
        import matplotlib.pyplot as plt

        epochs = [int(item["epoch_human"]) for item in history]
        train_loss = [float(item["train"]["total"]) for item in history]
        val_loss = [float(item["val"]["total"]) for item in history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="train total")
        plt.plot(epochs, val_loss, label="validation total")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PlankEye V7 training")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=150)
        plt.close()
    except Exception:
        pass


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================


def build_model_safely(
    *,
    pretrained_backbone: bool,
    freeze_backbone_bn: bool,
    device: torch.device,
    distributed: bool,
    rank: int,
    channels_last: bool,
) -> nn.Module:
    # Avoid two ranks trying to download ImageNet weights simultaneously.
    if distributed and pretrained_backbone:
        if rank == 0:
            model = PlankEyeV7(
                pretrained_backbone=True,
                freeze_backbone_bn=freeze_backbone_bn,
            )
            model.to(device)
            barrier()
        else:
            barrier()
            model = PlankEyeV7(
                pretrained_backbone=True,
                freeze_backbone_bn=freeze_backbone_bn,
            )
            model.to(device)
    else:
        model = PlankEyeV7(
            pretrained_backbone=pretrained_backbone,
            freeze_backbone_bn=freeze_backbone_bn,
        ).to(device)

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    return model


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PlankEye V7 on normalized 4-corner plank labels."
    )

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/plankeye_v7"))

    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per process/GPU.")
    parser.add_argument("--image-size", type=int, default=IMG_SIZE)
    parser.add_argument("--val-ratio", type=float, default=0.10)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Progressive ResNet50 unfreezing. Epoch numbers are 1-based.
    parser.add_argument(
        "--unfreeze-layer4-epoch",
        type=int,
        default=4,
        help="First epoch where ResNet50 layer4 becomes trainable (1-based).",
    )
    parser.add_argument(
        "--unfreeze-layer3-epoch",
        type=int,
        default=9,
        help="First epoch where ResNet50 layer3 + layer4 become trainable (1-based).",
    )
    parser.add_argument(
        "--unfreeze-all-epoch",
        type=int,
        default=16,
        help="First epoch where the complete ResNet50 backbone becomes trainable (1-based).",
    )
    parser.add_argument(
        "--no-progressive-unfreeze",
        action="store_true",
        help="Train the complete backbone from epoch 1 (disables progressive unfreezing).",
    )
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path or 'auto'.")
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")

    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone-bn", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--ema-tau", type=float, default=2000.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.image_size < 128 or args.image_size % 32 != 0:
        raise ValueError("--image-size should be >=128 and divisible by 32")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1)")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if not (0.0 < args.backbone_lr_mult <= 1.0):
        raise ValueError("--backbone-lr-mult must be in (0,1]")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")

    if args.unfreeze_layer4_epoch < 1:
        raise ValueError("--unfreeze-layer4-epoch must be >= 1")
    if args.unfreeze_layer3_epoch < 1:
        raise ValueError("--unfreeze-layer3-epoch must be >= 1")
    if args.unfreeze_all_epoch < 1:
        raise ValueError("--unfreeze-all-epoch must be >= 1")
    if not (
        args.unfreeze_layer4_epoch
        <= args.unfreeze_layer3_epoch
        <= args.unfreeze_all_epoch
    ):
        raise ValueError(
            "Progressive unfreeze epochs must satisfy: "
            "layer4 <= layer3 <= all."
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()
    validate_args(args)

    distributed, rank, local_rank, world_size, device = setup_distributed()

    try:
        seed_everything(args.seed + rank)

        # Fast fixed-size convolution kernels. Determinism is intentionally not
        # forced because industrial training speed is more useful here.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        args.data = args.data.resolve()
        args.output = args.output.resolve()

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
        if distributed:
            barrier()

        augment_cfg = AugmentConfig()
        amp_enabled = bool(device.type == "cuda" and not args.no_amp)
        channels_last = bool(device.type == "cuda" and not args.no_channels_last)
        pretrained_backbone = not args.no_pretrained
        freeze_backbone_bn = not args.unfreeze_backbone_bn
        progressive_unfreeze = progressive_unfreeze_enabled(
            args,
            pretrained_backbone,
        )

        train_records, val_records, split_payload = create_or_load_split(
            data_dir=args.data,
            output_dir=args.output,
            val_ratio=args.val_ratio,
            seed=args.seed,
            rank=rank,
            distributed=distributed,
        )

        train_dataset = PlankDataset(
            args.data,
            train_records,
            image_size=args.image_size,
            augment=not args.no_augmentation,
            augment_cfg=augment_cfg,
        )
        val_dataset = PlankDataset(
            args.data,
            val_records,
            image_size=args.image_size,
            augment=False,
            augment_cfg=augment_cfg,
        )

        if distributed:
            train_sampler: DistributedSampler | None = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            val_sampler: Sampler[int] | None = DistributedEvalSampler(
                val_dataset,
                rank=rank,
                world_size=world_size,
            )
        else:
            train_sampler = None
            val_sampler = None

        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed + rank)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=loader_generator,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=loader_generator,
            drop_last=False,
        )

        model = build_model_safely(
            pretrained_backbone=pretrained_backbone,
            freeze_backbone_bn=freeze_backbone_bn,
            device=device,
            distributed=distributed,
            rank=rank,
            channels_last=channels_last,
        )

        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                # Progressive freezing toggles requires_grad after DDP has
                # registered hooks for all initially trainable backbone params.
                # Unused-parameter detection keeps reduction correct while some
                # backbone stages are frozen.
                find_unused_parameters=progressive_unfreeze,
                gradient_as_bucket_view=True,
            )

        ema = None
        if not args.no_ema:
            ema = ModelEMA(model, decay=args.ema_decay, tau=args.ema_tau)

        optimizer = build_optimizer(
            model,
            lr=args.lr,
            backbone_lr_mult=args.backbone_lr_mult,
            weight_decay=args.weight_decay,
        )

        updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
        total_scheduler_steps = max(1, updates_per_epoch * args.epochs)
        warmup_steps = int(round(updates_per_epoch * args.warmup_epochs))
        effective_warmup_steps = max(0, min(warmup_steps, total_scheduler_steps - 1))

        scheduler = build_scheduler(
            optimizer,
            total_steps=total_scheduler_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        scaler = create_grad_scaler(amp_enabled)

        start_epoch = 0
        best_val_loss = math.inf
        history: List[dict] = []
        global_optimizer_step = 0
        epochs_without_improvement = 0

        resume_path = resolve_resume_path(args.resume, args.output)
        if resume_path is not None:
            if rank == 0:
                print(f"Loading checkpoint: {resume_path}")
            (
                start_epoch,
                best_val_loss,
                history,
                global_optimizer_step,
            ) = load_checkpoint(
                resume_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )

        # IMPORTANT: the optimizer was intentionally created before freezing.
        # Therefore every backbone parameter already belongs to an optimizer
        # group and can be safely unfrozen later without rebuilding optimizer,
        # scheduler, or checkpoint state.
        current_backbone_stage = backbone_stage_for_epoch(
            start_epoch,
            args,
            progressive_unfreeze,
        )
        backbone_trainability = apply_backbone_stage(
            model,
            current_backbone_stage,
            freeze_backbone_bn=freeze_backbone_bn,
        )

        runtime = runtime_info(device, world_size)
        git = git_info(Path(__file__).resolve().parent)

        if rank == 0:
            effective_batch = args.batch_size * world_size * args.grad_accum
            stats = split_payload.get("dataset_stats", {})

            print()
            print("=" * 88)
            print("PLANKEYE V7 TRAINING")
            print("=" * 88)
            print(f"Model version          : {MODEL_VERSION}")
            print(f"Data                   : {args.data}")
            print(f"Output                 : {args.output}")
            print(f"Train / Val            : {len(train_dataset)} / {len(val_dataset)}")
            print(f"Objects                : {stats.get('total_objects', '?')}")
            print(f"Dataset signature      : {split_payload.get('dataset_signature')}")
            print(f"Image size             : {args.image_size} x {args.image_size}")
            print(f"Epochs                 : {args.epochs}")
            print(f"Batch / GPU            : {args.batch_size}")
            print(f"GPUs / world size      : {world_size}")
            print(f"Gradient accumulation  : {args.grad_accum}")
            print(f"Effective batch        : {effective_batch}")
            print(f"AMP FP16               : {amp_enabled}")
            print(f"Channels last          : {channels_last}")
            print(f"EMA                    : {ema is not None}")
            print(f"Pretrained backbone    : {pretrained_backbone}")
            print(f"Frozen backbone BN     : {freeze_backbone_bn}")
            print(f"Progressive unfreeze   : {progressive_unfreeze}")
            if progressive_unfreeze:
                print(
                    "Unfreeze schedule      : "
                    f"L4@{args.unfreeze_layer4_epoch}, "
                    f"L3+L4@{args.unfreeze_layer3_epoch}, "
                    f"ALL@{args.unfreeze_all_epoch}"
                )
            print(f"Initial backbone stage : {current_backbone_stage}")
            print(
                "Backbone trainable     : "
                f"{format_trainable_count(int(backbone_trainability['backbone_trainable']))} / "
                f"{format_trainable_count(int(backbone_trainability['backbone_total']))}"
            )
            print(f"Base LR                : {args.lr:.3e}")
            print(f"Backbone LR            : {args.lr * args.backbone_lr_mult:.3e}")
            print(f"Warmup steps           : {effective_warmup_steps}")
            print(f"Total optimizer steps  : {total_scheduler_steps}")
            print(f"Device                 : {device}")
            if torch.cuda.is_available():
                for idx, name in enumerate(runtime["gpu_names"]):
                    props = torch.cuda.get_device_properties(idx)
                    vram_gb = props.total_memory / (1024**3)
                    print(f"GPU {idx}                  : {name} ({vram_gb:.1f} GiB)")
            if world_size == 1 and torch.cuda.device_count() > 1:
                print("WARNING: multiple GPUs are visible but DDP is not active. Launch with torchrun.")
            if resume_path is not None:
                print(f"Resume epoch           : {start_epoch + 1}")
            print("=" * 88)
            print()

            error_preview = stats.get("errors_preview", [])
            if error_preview:
                print(f"Ignored invalid samples: {stats.get('invalid_images', len(error_preview))}")
                for error in error_preview[:10]:
                    print(f"  - {error}")
                if len(error_preview) > 10:
                    print("  ...")
                print()

            # Keep exact source code beside every training run.
            project_dir = Path(__file__).resolve().parent
            for filename in ("model_v7.py", "train_v7.py"):
                source = project_dir / filename
                if source.exists():
                    shutil.copy2(source, args.output / filename)

            (args.output / "run_config.json").write_text(
                json.dumps(
                    {
                        "args": vars(args),
                        "augmentation": asdict(augment_cfg),
                        "model": model_metadata(),
                        "runtime": runtime,
                        "git": git,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

        if distributed:
            barrier()

        if start_epoch >= args.epochs:
            if rank == 0:
                print(
                    f"Checkpoint already reached epoch {start_epoch}; requested epochs={args.epochs}. Nothing to train."
                )
            return

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()

            desired_backbone_stage = backbone_stage_for_epoch(
                epoch,
                args,
                progressive_unfreeze,
            )

            if desired_backbone_stage != current_backbone_stage:
                # All ranks execute the exact same transition at the epoch
                # boundary. No optimizer rebuild is needed because all backbone
                # parameters were registered before the first freeze.
                if distributed:
                    barrier()

                current_backbone_stage = desired_backbone_stage
                backbone_trainability = apply_backbone_stage(
                    model,
                    current_backbone_stage,
                    freeze_backbone_bn=freeze_backbone_bn,
                )

                if distributed:
                    barrier()

                print_backbone_stage(
                    backbone_trainability,
                    epoch=epoch,
                    rank=rank,
                )

            train_metrics, optimizer_steps = train_one_epoch(
                model=model,
                ema=ema,
                loader=train_loader,
                sampler=train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
                rank=rank,
                distributed=distributed,
                grad_accum=args.grad_accum,
                max_grad_norm=args.max_grad_norm,
                amp_enabled=amp_enabled,
                channels_last=channels_last,
            )
            global_optimizer_step += optimizer_steps

            validation_model = ema.ema if ema is not None else unwrap_model(model)
            val_metrics = validate_one_epoch(
                model=validation_model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
                rank=rank,
                distributed=distributed,
                amp_enabled=amp_enabled,
                channels_last=channels_last,
            )

            epoch_duration = time.time() - epoch_start
            val_loss = float(val_metrics["total"])
            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            learning_rates = current_learning_rates(optimizer)
            history_item = {
                "epoch": epoch,
                "epoch_human": epoch + 1,
                "duration_seconds": epoch_duration,
                "train": dict(train_metrics),
                "val": dict(val_metrics),
                "learning_rates": learning_rates,
                "best_val_loss": best_val_loss,
                "global_optimizer_step": global_optimizer_step,
                "backbone_stage": current_backbone_stage,
                "backbone_trainable": int(
                    backbone_trainability["backbone_trainable"]
                ),
            }
            history.append(history_item)

            if rank == 0:
                print()
                print("-" * 88)
                print(f"EPOCH {epoch + 1}/{args.epochs}  |  {epoch_duration / 60.0:.2f} min")
                print(
                    f"Train total={train_metrics['total']:.6f}  "
                    f"Val total={val_metrics['total']:.6f}  "
                    f"Best={best_val_loss:.6f}"
                )
                print(
                    f"Train corner_delta={train_metrics.get('corner_delta', 0.0):.6f}  "
                    f"abs={train_metrics.get('corner_abs', 0.0):.6f}  "
                    f"recon={train_metrics.get('reconstruction', 0.0):.6f}  "
                    f"heatmap={train_metrics.get('heatmap', 0.0):.6f}"
                )
                print(
                    f"Val   corner_delta={val_metrics.get('corner_delta', 0.0):.6f}  "
                    f"abs={val_metrics.get('corner_abs', 0.0):.6f}  "
                    f"recon={val_metrics.get('reconstruction', 0.0):.6f}  "
                    f"heatmap={val_metrics.get('heatmap', 0.0):.6f}"
                )
                print(
                    "LR: "
                    + ", ".join(f"{name}={value:.3e}" for name, value in learning_rates.items())
                )
                print(
                    "Backbone stage: "
                    f"{current_backbone_stage} | trainable "
                    f"{format_trainable_count(int(backbone_trainability['backbone_trainable']))} / "
                    f"{format_trainable_count(int(backbone_trainability['backbone_total']))}"
                )
                print("-" * 88)

                save_checkpoint(
                    args.output / "last.pt",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    args=args,
                    augment_cfg=augment_cfg,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    history=history,
                    best_val_loss=best_val_loss,
                    global_optimizer_step=global_optimizer_step,
                    backbone_stage=current_backbone_stage,
                    backbone_trainability=backbone_trainability,
                    split_payload=split_payload,
                    runtime=runtime,
                    git=git,
                )

                if improved:
                    save_checkpoint(
                        args.output / "best.pt",
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        args=args,
                        augment_cfg=augment_cfg,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        history=history,
                        best_val_loss=best_val_loss,
                        global_optimizer_step=global_optimizer_step,
                        backbone_stage=current_backbone_stage,
                        backbone_trainability=backbone_trainability,
                        split_payload=split_payload,
                        runtime=runtime,
                        git=git,
                    )
                    print(f"NEW BEST -> {args.output / 'best.pt'}")

                if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                    save_checkpoint(
                        args.output / f"epoch_{epoch + 1:03d}.pt",
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        args=args,
                        augment_cfg=augment_cfg,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        history=history,
                        best_val_loss=best_val_loss,
                        global_optimizer_step=global_optimizer_step,
                        backbone_stage=current_backbone_stage,
                        backbone_trainability=backbone_trainability,
                        split_payload=split_payload,
                        runtime=runtime,
                        git=git,
                    )

                save_history_files(args.output, history)

            if distributed:
                barrier()

            should_stop = args.patience > 0 and epochs_without_improvement >= args.patience
            stop_tensor = torch.tensor(
                1 if should_stop else 0,
                dtype=torch.int32,
                device=device,
            )
            if distributed:
                dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

            if should_stop:
                if rank == 0:
                    print(
                        f"Early stopping after {epochs_without_improvement} epochs without validation improvement."
                    )
                break

        if rank == 0:
            print()
            print("=" * 88)
            print("TRAINING FINISHED")
            print(f"Best validation loss : {best_val_loss:.6f}")
            print(f"Best checkpoint      : {args.output / 'best.pt'}")
            print(f"Last checkpoint      : {args.output / 'last.pt'}")
            print(f"History JSON         : {args.output / 'history.json'}")
            print(f"History CSV          : {args.output / 'history.csv'}")
            print(f"Loss curve           : {args.output / 'loss_curve.png'}")
            print("=" * 88)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
