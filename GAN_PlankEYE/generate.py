from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from models import UNetGenerator


def polygon_mask(points, size):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def order_clockwise(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)

    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0],
    )

    ordered = points[np.argsort(angles)]

    # Commence par le coin le plus proche du haut-gauche,
    # puis conserve l'ordre circulaire: TL-ish, TR-ish, BR-ish, BL-ish.
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)

    return ordered


def rotated_plank(size, rng):
    """
    Génère une planche sous forme de rectangle orienté,
    légèrement perturbé pour obtenir un quadrilatère réaliste.
    """
    w = rng.uniform(0.22, 0.55) * size
    h = rng.uniform(0.07, 0.18) * size

    # Certaines images contiennent des planches plutôt verticales.
    angle = rng.uniform(-80.0, 80.0)

    margin = max(w, h) * 0.65 + 4
    cx = rng.uniform(margin, size - margin)
    cy = rng.uniform(margin, size - margin)

    rect = ((cx, cy), (w, h), angle)
    pts = cv2.boxPoints(rect).astype(np.float32)

    # Petit bruit sur chaque coin.
    jitter = min(w, h) * 0.04
    pts += np.asarray(
        [
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)]
            for _ in range(4)
        ],
        dtype=np.float32,
    )

    pts[:, 0] = np.clip(pts[:, 0], 2, size - 3)
    pts[:, 1] = np.clip(pts[:, 1], 2, size - 3)

    return order_clockwise(pts)


def overlap_ratio(mask_a, mask_b):
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    area_b = (mask_b > 0).sum()

    if area_b == 0:
        return 1.0

    return float(inter) / float(area_b)


def too_close(existing_mask, new_mask, min_gap):
    if min_gap <= 0:
        return False

    k = 2 * int(min_gap) + 1
    kernel = np.ones((k, k), dtype=np.uint8)

    dilated = cv2.dilate(
        existing_mask,
        kernel,
        iterations=1,
    )

    return bool(
        np.logical_and(dilated > 0, new_mask > 0).any()
    )


def build_layout(
    size,
    plank_count,
    min_gap,
    max_overlap,
    rng,
    max_attempts_per_plank=250,
):
    occupancy = np.zeros((size, size), dtype=np.uint8)
    polygons = []

    for _ in range(plank_count):
        accepted = False

        for _attempt in range(max_attempts_per_plank):
            pts = rotated_plank(size, rng)
            pmask = polygon_mask(pts, size)

            overlap = overlap_ratio(occupancy, pmask)

            if overlap > max_overlap:
                continue

            # Avec max_overlap=0, on applique aussi l'écart minimal.
            if max_overlap <= 0.0 and too_close(
                occupancy,
                pmask,
                min_gap,
            ):
                continue

            occupancy = np.maximum(occupancy, pmask)
            polygons.append(pts)
            accepted = True
            break

        if not accepted:
            return None, None

    return occupancy, polygons


def save_label(path, polygons, size):
    lines = []

    for pts in polygons:
        norm = pts.astype(np.float32).copy()
        norm[:, 0] /= float(size)
        norm[:, 1] /= float(size)
        norm = np.clip(norm, 0.0, 1.0)

        coords = " ".join(
            f"{v:.6f}" for v in norm.reshape(-1)
        )
        lines.append(f"0 {coords}")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def load_generator(checkpoint_path, device, latent_channels, base):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    ckpt_args = checkpoint.get("args", {})

    latent_channels = int(
        ckpt_args.get("latent_channels", latent_channels)
    )
    base = int(
        ckpt_args.get("base", base)
    )

    model = UNetGenerator(
        latent_channels=latent_channels,
        base=base,
    ).to(device)

    model.load_state_dict(checkpoint["generator"])
    model.eval()

    return model, latent_channels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    # Gardé pour compatibilité avec notre ancienne commande.
    parser.add_argument("--prepared", type=str, default="")

    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=512)

    parser.add_argument("--min-planks", type=int, default=1)
    parser.add_argument("--max-planks", type=int, default=7)
    parser.add_argument("--min-gap", type=int, default=8)
    parser.add_argument("--max-overlap", type=float, default=0.0)

    parser.add_argument("--latent-channels", type=int, default=3)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible.")

    device = torch.device("cuda")

    out = Path(args.out)
    image_dir = out / "images"
    label_dir = out / "labels"
    mask_dir = out / "masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    G, latent_channels = load_generator(
        args.checkpoint,
        device,
        args.latent_channels,
        args.base,
    )

    rng = random.Random(args.seed)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(args.seed)

    produced = 0
    failed_layouts = 0

    metadata = []

    while produced < args.count:
        plank_count = rng.randint(
            args.min_planks,
            args.max_planks,
        )

        occupancy, polygons = build_layout(
            size=args.size,
            plank_count=plank_count,
            min_gap=args.min_gap,
            max_overlap=args.max_overlap,
            rng=rng,
        )

        if occupancy is None:
            failed_layouts += 1

            if failed_layouts > args.count * 50:
                raise RuntimeError(
                    "Impossible de construire suffisamment de layouts. "
                    "Réduis min-gap / max-planks ou augmente max-overlap."
                )
            continue

        mask = torch.from_numpy(
            (occupancy.astype(np.float32) / 255.0)[None, None]
        ).to(device)

        noise = torch.randn(
            1,
            latent_channels,
            args.size,
            args.size,
            generator=torch_gen,
            device=device,
        )

        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            fake = G(mask, noise)

        image = (
            (fake[0].float().clamp(-1, 1) + 1.0)
            * 127.5
        )
        image = (
            image.permute(1, 2, 0)
            .cpu()
            .numpy()
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )

        stem = f"gan_{produced:06d}"

        Image.fromarray(image).save(
            image_dir / f"{stem}.jpg",
            quality=95,
        )

        cv2.imwrite(
            str(mask_dir / f"{stem}.png"),
            occupancy,
        )

        save_label(
            label_dir / f"{stem}.txt",
            polygons,
            args.size,
        )

        metadata.append(
            {
                "stem": stem,
                "plank_count": plank_count,
            }
        )

        produced += 1

        if produced == 1 or produced % 50 == 0 or produced == args.count:
            print(
                f"\rGénération: {produced}/{args.count}",
                end="",
                flush=True,
            )

    print()

    (out / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "count": produced,
                "size": args.size,
                "min_planks": args.min_planks,
                "max_planks": args.max_planks,
                "min_gap": args.min_gap,
                "max_overlap": args.max_overlap,
                "seed": args.seed,
                "items": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("GÉNÉRATION TERMINÉE")
    print("=" * 72)
    print(f"Images : {image_dir}")
    print(f"Labels : {label_dir}")
    print(f"Masks  : {mask_dir}")
    print(f"Total  : {produced}")


if __name__ == "__main__":
    main()
