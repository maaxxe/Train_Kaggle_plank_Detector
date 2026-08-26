from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PairedPlankDataset(Dataset):
    def __init__(self, root: str | Path, augment: bool = False):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Dossier images introuvable: {self.image_dir}")

        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        valid = []
        for img in self.images:
            mask = self.mask_dir / f"{img.stem}.png"
            if mask.exists():
                valid.append(img)

        self.images = valid
        self.augment = augment

        if not self.images:
            raise RuntimeError(f"Aucune paire image/masque trouvée dans {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Augmentations géométriques synchronisées.
        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.25:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        image = image * 2.0 - 1.0

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "name": image_path.stem,
        }
