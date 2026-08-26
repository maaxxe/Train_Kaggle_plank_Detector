from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils import IMAGE_EXTENSIONS, natural_key, polygons_to_mask, read_polygon_label


def find_dataset_candidates(root: Path):
    """
    Recherche automatiquement les dossiers:
        .../images
        .../labels
    dans /kaggle/input.
    """
    candidates = []

    for image_dir in root.rglob("images"):
        if not image_dir.is_dir():
            continue

        label_dir = image_dir.parent / "labels"
        if not label_dir.is_dir():
            continue

        images = [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        paired = sum(
            1 for p in images
            if (label_dir / f"{p.stem}.txt").exists()
        )

        if paired:
            candidates.append((paired, image_dir.parent))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates


def prepare(source: Path, out: Path, size: int, overwrite: bool):
    image_dir = source / "images"
    label_dir = source / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Le dossier source doit contenir `images/` et `labels/`.\n"
            f"Source reçue: {source}"
        )

    if overwrite and out.exists():
        shutil.rmtree(out)

    out_images = out / "images"
    out_masks = out / "masks"
    out_labels = out / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )

    kept = 0
    missing_label = 0
    empty_label = 0
    count_distribution = Counter()

    metadata = []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_label += 1
            continue

        polygons = read_polygon_label(label_path)

        if not polygons:
            empty_label += 1
            continue

        image = Image.open(image_path).convert("RGB")
        image = image.resize((size, size), Image.Resampling.LANCZOS)

        mask = polygons_to_mask(polygons, size, size)

        out_image = out_images / f"{image_path.stem}.jpg"
        out_mask = out_masks / f"{image_path.stem}.png"
        out_label = out_labels / f"{image_path.stem}.txt"

        image.save(out_image, quality=95)
        cv2.imwrite(str(out_mask), mask)

        # Les coordonnées sont normalisées: le resize ne les change pas.
        shutil.copy2(label_path, out_label)

        count_distribution[len(polygons)] += 1

        metadata.append(
            {
                "stem": image_path.stem,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "plank_count": len(polygons),
            }
        )

        kept += 1

    info = {
        "source": str(source),
        "output": str(out),
        "size": size,
        "pairs": kept,
        "missing_label": missing_label,
        "empty_label": empty_label,
        "plank_count_distribution": dict(sorted(count_distribution.items())),
        "items": metadata,
    }

    (out / "metadata.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PRÉPARATION TERMINÉE")
    print("=" * 72)
    print(f"Source                : {source}")
    print(f"Sortie                : {out}")
    print(f"Taille                : {size}x{size}")
    print(f"Paires conservées     : {kept}")
    print(f"Sans label            : {missing_label}")
    print(f"Labels vides/invalides: {empty_label}")
    print("Distribution nb planches:")
    for n, c in sorted(count_distribution.items()):
        print(f"  {n}: {c}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--search-root", type=str, default="/kaggle/input")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source:
        source = Path(args.source)
    else:
        candidates = find_dataset_candidates(Path(args.search_root))

        if not candidates:
            raise RuntimeError(
                "Aucun dataset `images/ + labels/` trouvé dans /kaggle/input.\n"
                "Passe explicitement --source /kaggle/input/.../ton_dataset"
            )

        print("Datasets candidats:")
        for paired, path in candidates[:10]:
            print(f"  {paired:5d} paires -> {path}")

        source = candidates[0][1]
        print(f"\nSélection automatique: {source}\n")

    prepare(source, Path(args.out), args.size, args.overwrite)


if __name__ == "__main__":
    main()
