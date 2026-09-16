"""
PlankEye V5 — entraînement LOCAL (sans Kaggle, sans DDP, CPU ou 1 seul GPU).

Adapté depuis le notebook Kaggle `plankeye-v5.ipynb` (cellule "%%writefile
train_plankeye_ddp.py"). Différences principales :

  - plus de kaggle_secrets / clone GitHub / CLI kaggle : le modèle
    (model/model_v5.py) et le dataset sont lus directement sur le disque
    local ;
  - plus de torch.distributed / DDP / lancement 2 process : un seul
    process, un seul device (CPU si pas de GPU CUDA dédié) ;
  - les chemins (dataset, checkpoints) sont paramétrables en ligne de
    commande, avec des valeurs par défaut qui correspondent à ta machine.

Exemple d'utilisation (depuis Plankeye/) :

    python local_train/train_plankeye_v5_local.py

    # test rapide (peu d'images, peu d'epochs, batch réduit) :
    python local_train/train_plankeye_v5_local.py --epochs 2 --batch-size 4 --limit-images 40

Ta machine a un GPU dédié NVIDIA Quadro M2200 (4 Go de VRAM) : le script
l'utilise automatiquement (device CUDA détecté, un seul GPU, pas de DDP).
4 Go de VRAM est peu pour un batch de 16 en 512px : si tu as une erreur
"CUDA out of memory", réduis --batch-size (essaie 8, puis 4). Le Quadro
M2200 est une puce Maxwell : l'AMP (fp16) reste activé pour économiser de
la VRAM mais n'apportera pas forcément de gain de vitesse (pas de tensor
cores sur cette génération).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (gardé pour usage futur / tracé des courbes)
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

# ============================================================
# CHEMINS PAR DÉFAUT — modifie ici si besoin, ou passe les
# arguments --data-dir / --checkpoint-dir / --project-dir en ligne
# de commande.
# ============================================================

DEFAULT_PROJECT_DIR = Path.home() / "Train_Kaggle_plank_Detector" / "Plankeye"
DEFAULT_DATA_DIR = Path("/home/mrobin/MonProjet/augmentation_images_reel/data_kaggle_2")
DEFAULT_CHECKPOINT_DIR = Path("/home/mrobin/MonProjet/Models/models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement local PlankEye V5")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR,
                         help="Racine du projet (contient model/model_v5.py). "
                              f"Défaut : {DEFAULT_PROJECT_DIR}")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="Dossier contenant images/ et labels/. "
                              f"Défaut : {DEFAULT_DATA_DIR}")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                         help="Dossier où écrire best_checkpoint_v5.pt / last_checkpoint_v5.pt. "
                              f"Défaut : {DEFAULT_CHECKPOINT_DIR}")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None,
                         help="Nombre de workers DataLoader (défaut : auto selon CPU, max 4).")
    parser.add_argument("--limit-images", type=int, default=None,
                         help="Ne charger que N images (pour un test rapide du pipeline).")
    parser.add_argument("--device", type=str, default=None,
                         help="Forcer le device ('cpu' ou 'cuda'). Auto-détecté sinon.")
    parser.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction,
                         default=True, help="Reprendre depuis last_checkpoint_v5.pt s'il existe "
                                             "(désactiver avec --no-resume).")
    return parser.parse_args()


ARGS = parse_args()

PROJECT: Path = ARGS.project_dir
MODEL_DIR = PROJECT / "model"
DATA_DIR: Path = ARGS.data_dir
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LOCAL_TRAIN_DIR = PROJECT / "local_train"
LOCAL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR: Path = ARGS.checkpoint_dir
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Dossier model/ introuvable : {MODEL_DIR}\n"
        "Vérifie --project-dir (doit pointer vers Plankeye/)."
    )
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_v5 import (  # noqa: E402
    IMG_SIZE,
    NUM_CLASSES,
    PlankEyeV5,
    combined_loss_v5,
)

try:
    from model_v5 import compute_detection_metrics  # noqa: E402, F401
except ImportError:
    def compute_detection_metrics(dets, gts, num_classes):
        return {"map50": 0.0, "map75": 0.0, "pck4": 0.0, "match_recall": 0.0}

CLASS_NAMES = ["plank"]

if NUM_CLASSES != 1:
    raise RuntimeError(f"Le modèle doit être en 1 classe, NUM_CLASSES={NUM_CLASSES}")

SEED = 48
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

BATCH_SIZE = ARGS.batch_size
VAL_BATCH_MULT = 2
EPOCHS = ARGS.epochs

LR_HEAD = 3.0e-4
LR_BACKBONE = 2.0e-5
LR_MIN_HEAD = 2.0e-6
LR_MIN_BACKBONE = 2.0e-7
HEAD_WARMUP_EPOCHS = 3
BACKBONE_WARMUP_EPOCHS = 3

UNFREEZE_LAST_EPOCH = 8
UNFREEZE_ALL_EPOCH = 20

WEIGHT_DECAY = 2.0e-4
GRAD_CLIP_NORM = 2.0
EMA_DECAY = 0.9998
EARLY_STOP_PATIENCE = 32
EARLY_STOP_MIN_DELTA = 1e-4

ROT_MAX_DEG = 30.0
ROT_MAX_OUT_OF_BOUNDS = 0.015

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_checkpoint_v5.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint_v5.pt"

PLOT_PATH = LOCAL_TRAIN_DIR / "training_curves_v5.png"
SPLIT_MANIFEST_PATH = LOCAL_TRAIN_DIR / "split_v5.json"

RUN_ID = "v5_run"
PRETRAINED_BACKBONE = True

# ============================================================
# Un seul process, un seul device : pas de DDP / torch.distributed.
# ============================================================

if ARGS.device:
    DEVICE = torch.device(ARGS.device)
    DEVICE_TYPE = ARGS.device
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    DEVICE_TYPE = "cuda"
else:
    DEVICE = torch.device("cpu")
    DEVICE_TYPE = "cpu"

AMP_ENABLED = DEVICE_TYPE == "cuda"


def log(*args, **kwargs) -> None:
    print(*args, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reorder_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype(np.float32)


def polygon_area_np(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def read_label(label_path: Path):
    objects = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            try:
                coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            if not np.isfinite(coords).all():
                continue
            pts = coords.reshape(4, 2)
            if (pts < -0.02).any() or (pts > 1.02).any():
                continue
            pts = np.clip(pts, 0.0, 1.0)
            pts = reorder_corners(pts)
            if polygon_area_np(pts) < 1e-06:
                continue
            objects.append({'cls': 0, 'corners': pts})
    return objects


# Fichiers corrompus connus à ignorer
CORRUPTED_IMAGES = {"image4515.jpg"}


def collect_pairs():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise RuntimeError(f'Dataset absent : {IMAGES_DIR} / {LABELS_DIR}')
    pairs = []
    missing = []
    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if img_path.name in CORRUPTED_IMAGES:
            continue
        label_path = LABELS_DIR / f'{img_path.stem}.txt'
        if not label_path.exists():
            match = re.fullmatch(r'image(\d+)', img_path.stem, flags=re.IGNORECASE)
            if match:
                alt = LABELS_DIR / f'label{match.group(1)}.txt'
                if alt.exists():
                    label_path = alt
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing.append(img_path.name)
    if missing:
        raise RuntimeError(f'{len(missing)} image(s) sans label associé.')
    if ARGS.limit_images:
        pairs = pairs[:ARGS.limit_images]
        log(f"⚠️  --limit-images actif : {len(pairs)} paires utilisées seulement.")
    log(f"✅ {len(pairs)} paires chargées ({len(CORRUPTED_IMAGES)} image(s) exclue(s) : {CORRUPTED_IMAGES})")
    return pairs


def _class_signature(label_path: Path) -> Tuple[int, ...]:
    objects = read_label(label_path)
    return (len(objects),)


def _write_split_manifest(train_p, val_p, test_p) -> None:
    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'seed': SEED,
        'dataset_dir': str(DATA_DIR),
        'train': [p[0].name for p in train_p],
        'val':   [p[0].name for p in val_p],
        'test':  [p[0].name for p in test_p]
    }
    tmp = SPLIT_MANIFEST_PATH.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SPLIT_MANIFEST_PATH)


def _try_load_split_manifest(pairs):
    if not SPLIT_MANIFEST_PATH.exists():
        return None
    try:
        payload = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding='utf-8'))
    except Exception:
        return None
    by_name = {p[0].name: p for p in pairs}
    used = set()

    def resolve(names):
        out = []
        for name in names:
            pair = by_name.get(name)
            if pair is not None:
                out.append(pair)
                used.add(name)
        return out

    train_p = resolve(payload.get('train', []))
    val_p = resolve(payload.get('val', []))
    test_p = resolve(payload.get('test', []))
    new_pairs = [p for p in pairs if p[0].name not in used]
    train_p.extend(new_pairs)
    if not val_p or not test_p:
        return None
    return (train_p, val_p, test_p)


def split_pairs(pairs):
    existing = _try_load_split_manifest(pairs)
    if existing is not None:
        return existing

    groups = defaultdict(list)
    for pair in pairs:
        groups[_class_signature(pair[1])].append(pair)

    rng = random.Random(SEED)
    train_p, val_p, test_p = [], [], []

    for signature in sorted(groups, key=str):
        group = sorted(groups[signature])
        rng.shuffle(group)
        n = len(group)
        n_val = int(round(n * VAL_RATIO))
        n_test = int(round(n * TEST_RATIO))
        if n >= 10:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        while n_val + n_test >= n and (n_val > 0 or n_test > 0):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
        val_p.extend(group[:n_val])
        test_p.extend(group[n_val:n_val + n_test])
        train_p.extend(group[n_val + n_test:])

    rng.shuffle(train_p)
    rng.shuffle(val_p)
    rng.shuffle(test_p)

    _write_split_manifest(train_p, val_p, test_p)

    return (train_p, val_p, test_p)


def collate_fn(batch):
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch])


def letterbox_resize(image: Image.Image, target_size: int, objects, return_valid_box=False):
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scale_x = new_w / w
    scale_y = new_h / h
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    resampling = getattr(Image, 'Resampling', Image)
    resized = image.resize((new_w, new_h), resample=resampling.BILINEAR)
    canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    transformed = []
    for obj in objects:
        pts = np.asarray(obj['corners'], dtype=np.float32).copy()
        px = pts[:, 0] * w * scale_x + left
        py = pts[:, 1] * h * scale_y + top
        out = np.stack([px / target_size, py / target_size], axis=1)
        transformed.append({'cls': int(obj['cls']), 'corners': reorder_corners(out)})
    if return_valid_box:
        return (canvas, transformed, (left, top, left + new_w, top + new_h))
    return (canvas, transformed)


class GeometricAug:
    def __init__(self, hflip: float = 0.5, vflip: float = 0.0, rot_deg: float = ROT_MAX_DEG,
                 max_out_of_bounds: float = ROT_MAX_OUT_OF_BOUNDS):
        self.hflip = hflip
        self.vflip = vflip
        self.rot_deg = rot_deg
        self.max_out_of_bounds = max_out_of_bounds

    @staticmethod
    def _rotate_points_pixel(pts_norm: np.ndarray, w: int, h: int, angle_deg: float):
        pts = pts_norm.astype(np.float64).copy()
        x = pts[:, 0] * w
        y = pts[:, 1] * h
        cx, cy = (w / 2.0, h / 2.0)
        dx, dy = (x - cx, y - cy)
        a = math.radians(angle_deg)
        cos_a, sin_a = (math.cos(a), math.sin(a))
        xr = cos_a * dx + sin_a * dy + cx
        yr = -sin_a * dx + cos_a * dy + cy
        return np.stack([xr / w, yr / h], axis=1).astype(np.float32)

    def __call__(self, image: Image.Image, objects):
        w, h = image.size
        out_img = image
        out_objs = [{'cls': o['cls'], 'corners': np.array(o['corners'], copy=True)} for o in objects]
        if random.random() < self.hflip:
            out_img = out_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for obj in out_objs:
                obj['corners'][:, 0] = 1.0 - obj['corners'][:, 0]
        angle = random.uniform(-self.rot_deg, self.rot_deg)
        if abs(angle) < 0.5:
            for obj in out_objs:
                obj['corners'] = reorder_corners(obj['corners'])
            return (out_img, out_objs)
        rotated_objs = []
        reject = False
        for obj in out_objs:
            pts = self._rotate_points_pixel(obj['corners'], w, h, angle)
            overflow = np.clip(-pts, 0, None).sum() + np.clip(pts - 1.0, 0, None).sum()
            if overflow > self.max_out_of_bounds:
                reject = True
                break
            pts = np.clip(pts, 0.0, 1.0)
            rotated_objs.append({'cls': obj['cls'], 'corners': reorder_corners(pts)})
        if reject:
            for obj in out_objs:
                obj['corners'] = reorder_corners(obj['corners'])
            return (out_img, out_objs)
        resampling = getattr(Image, 'Resampling', Image)
        rotated_img = out_img.rotate(angle, resample=resampling.BILINEAR, expand=False, fillcolor=(114, 114, 114))
        return (rotated_img, rotated_objs)


class PlankDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = list(pairs)
        self.augment = augment
        self.geo_aug = GeometricAug() if augment else None
        if augment:
            self.photo_tf = transforms.Compose([
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.035)], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.15, 1.0))], p=0.12)
            ])
        else:
            self.photo_tf = None
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        try:
            with Image.open(img_path) as im:
                image = im.convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self.pairs))

        objects = read_label(label_path)
        if self.geo_aug is not None:
            image, objects = self.geo_aug(image, objects)
        if self.photo_tf is not None:
            image = self.photo_tf(image)
        image, objects, valid_box = letterbox_resize(image, IMG_SIZE, objects, return_valid_box=True)
        tensor = self.to_tensor(image)
        return (tensor, objects)


def make_loaders(train_p, val_p, test_p):
    cpu = os.cpu_count() or 4
    workers = ARGS.workers if ARGS.workers is not None else max(1, min(4, cpu))
    common = {
        "num_workers": workers,
        "pin_memory": False,
        "persistent_workers": False,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker if workers > 0 else None,
        "prefetch_factor": 2 if workers > 0 else None,
    }
    train_ds = PlankDataset(train_p, augment=True)
    val_ds = PlankDataset(val_p, augment=False)
    test_ds = PlankDataset(test_p, augment=False)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * VAL_BATCH_MULT, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * VAL_BATCH_MULT, shuffle=False, **common)

    log(f"📦 DataLoaders prêts : {len(train_p)} images d'entraînement, {len(val_p)} en validation, "
        f"{len(test_p)} en test. (workers={workers})")
    return train_loader, val_loader, test_loader


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        model_state = model.state_dict()
        for key, value in self.ema.state_dict().items():
            src = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)


def set_backbone_trainable(model, epoch):
    """
    Gestion progressive du backbone ResNet18.

    Epoch 1-7  : backbone gele
    Epoch 8-19 : layer3 + layer4 entrainables
    Epoch 20+  : backbone complet entrainable
    """
    backbone = model.backbone

    for param in backbone.parameters():
        param.requires_grad_(False)

    if epoch >= UNFREEZE_LAST_EPOCH:
        for param in backbone.layer3.parameters():
            param.requires_grad_(True)
        for param in backbone.layer4.parameters():
            param.requires_grad_(True)

    if epoch >= UNFREEZE_ALL_EPOCH:
        for param in backbone.parameters():
            param.requires_grad_(True)

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())

    if epoch < UNFREEZE_LAST_EPOCH:
        phase = "GELÉ"
    elif epoch < UNFREEZE_ALL_EPOCH:
        phase = "PARTIEL (layer3+layer4)"
    else:
        phase = "COMPLET"

    log(f"🔓 Backbone epoch {epoch}: {phase} | {trainable:,}/{total:,} paramètres entraînables")


def build_optimizer(model):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": LR_HEAD, "group_name": "head"})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": LR_BACKBONE, "group_name": "backbone"})

    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)


def run_epoch(model, loader, optimizer=None, scaler=None, epoch=None, total_epochs=None, phase="train", ema=None):
    training = optimizer is not None
    active_model = model if training else (ema.ema if ema is not None else model)

    if training:
        model.train()
        if epoch is not None:
            backbone = model.backbone
            if epoch < UNFREEZE_LAST_EPOCH:
                backbone.eval()
            elif epoch < UNFREEZE_ALL_EPOCH:
                backbone.stem.eval()
                backbone.layer1.eval()
                backbone.layer2.eval()
                backbone.layer3.train()
                backbone.layer4.train()
            else:
                backbone.train()
    else:
        active_model.eval()

    totals = defaultdict(float)
    total_images = 0

    bar = tqdm(loader, desc=f"{phase.upper()} {epoch}/{total_epochs}" if epoch else phase.upper())

    for step, (images, batch_objects) in enumerate(bar):
        images = images.to(DEVICE, non_blocking=True)
        bs = images.shape[0]

        autocast_ctx = torch.amp.autocast(
            device_type=DEVICE_TYPE,
            dtype=torch.float16 if DEVICE_TYPE == "cuda" else torch.bfloat16,
            enabled=AMP_ENABLED,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                outputs = active_model(images)
                loss, breakdown = combined_loss_v5(outputs, batch_objects, DEVICE)

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            if ema is not None:
                ema.update(model)
        else:
            with torch.no_grad(), autocast_ctx:
                outputs = active_model(images)
                loss, breakdown = combined_loss_v5(outputs, batch_objects, DEVICE)

        totals["loss"] += float(loss.detach()) * bs
        for k, v in breakdown.items():
            if isinstance(v, (int, float)):
                totals[k] += float(v) * bs
        total_images += bs

    denom = max(total_images, 1)
    return {k: v / denom for k, v in totals.items()}


def save_ckpt(path: Path, model, optimizer, scaler, ema, epoch, histories, best_quality):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "ema": ema.ema.state_dict() if ema else None,
        "histories": histories,
        "best_quality": best_quality,
        "config": {"run_id": RUN_ID, "pretrained_backbone": PRETRAINED_BACKBONE},
    }
    torch.save(payload, path)


def get_checkpoint_start_epoch(path: Path) -> int:
    if not path.exists():
        return 1
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) + 1


def load_ckpt(model, optimizer, scaler, ema, path: Path):
    empty_histories = {"train_loss": [], "val_loss": [], "quality": []}
    if not path.exists():
        return 1, empty_histories, float("-inf")

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    if checkpoint.get("optim"):
        try:
            optimizer.load_state_dict(checkpoint["optim"])
        except (ValueError, RuntimeError) as exc:
            log(f"⚠️ Etat optimizer incompatible avec la nouvelle phase: {exc}")
            log("   → Reprise des poids OK, optimizer réinitialisé pour cette phase.")

    if ema and checkpoint.get("ema"):
        ema.ema.load_state_dict(checkpoint["ema"])

    return (
        int(checkpoint.get("epoch", 0)) + 1,
        checkpoint.get("histories", empty_histories),
        float(checkpoint.get("best_quality", float("-inf"))),
    )


def main():
    set_seed(SEED)

    log("=" * 60)
    log("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT PLANKEYE V5 (LOCAL)")
    log("=" * 60)
    log(f"Device            : {DEVICE} ({DEVICE_TYPE})")
    if DEVICE_TYPE == "cpu":
        log("⚠️  Pas de GPU CUDA détecté → entraînement sur CPU (lent). "
            "Utilise --limit-images/--epochs pour un test rapide, ou entraîne sur Kaggle/Colab.")
    else:
        props = torch.cuda.get_device_properties(DEVICE)
        log(f"GPU               : {props.name} ({props.total_memory / 1024**3:.1f} Go VRAM)")
        log("   Si erreur 'CUDA out of memory' : réduis --batch-size (essaie 8, puis 4).")
    log(f"Projet            : {PROJECT}")
    log(f"Dataset           : {DATA_DIR}")
    log(f"Checkpoints       : {CHECKPOINT_DIR}")
    log(f"Batch size        : {BATCH_SIZE}  |  Epochs : {EPOCHS}")

    pairs = collect_pairs()
    train_p, val_p, test_p = split_pairs(pairs)
    train_loader, val_loader, test_loader = make_loaders(train_p, val_p, test_p)

    log("📥 Initialisation du backbone ResNet18 (poids ImageNet préentraînés)...")
    model = PlankEyeV5().to(DEVICE)
    log(f"✅ Modèle V5 initialisé sur {DEVICE}")

    resume_epoch = get_checkpoint_start_epoch(LAST_CHECKPOINT_PATH) if ARGS.resume else 1
    set_backbone_trainable(model, resume_epoch)

    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler(enabled=AMP_ENABLED) if DEVICE_TYPE == "cuda" else None
    ema = ModelEMA(model)

    if ARGS.resume:
        start_epoch, histories, best_quality = load_ckpt(model, optimizer, scaler, ema, LAST_CHECKPOINT_PATH)
    else:
        start_epoch, histories, best_quality = 1, {"train_loss": [], "val_loss": [], "quality": []}, float("-inf")

    set_backbone_trainable(model, start_epoch)

    if start_epoch > 1:
        log(f"🔄 Reprise de l'entraînement à partir de l'epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch in (UNFREEZE_LAST_EPOCH, UNFREEZE_ALL_EPOCH):
            set_backbone_trainable(model, epoch)
            optimizer = build_optimizer(model)
            log(f"🔧 Optimizer reconstruit pour la phase de l'epoch {epoch}")

        t0 = time.time()
        train_stats = run_epoch(model, train_loader, optimizer=optimizer, scaler=scaler,
                                 epoch=epoch, total_epochs=EPOCHS, phase="train", ema=ema)
        val_stats = run_epoch(model, val_loader, optimizer=None, scaler=scaler,
                               epoch=epoch, total_epochs=EPOCHS, phase="val", ema=ema)
        dt = time.time() - t0

        histories["train_loss"].append(train_stats.get("loss", 0.0))
        histories["val_loss"].append(val_stats.get("loss", 0.0))
        q = -val_stats.get("loss", float("inf"))
        histories["quality"].append(q)

        log(
            f"📊 Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_stats.get('loss', 0):.4f} | "
            f"Val loss: {val_stats.get('loss', 0):.4f} | "
            f"Hmap: {val_stats.get('heatmap', 0):.4f} | "
            f"Corner: {val_stats.get('corner', 0):.4f} | "
            f"Geom: {val_stats.get('geometry', 0):.4f} | "
            f"{dt:.1f}s"
        )

        if q > best_quality:
            best_quality = q
            save_ckpt(BEST_MODEL_PATH, model, optimizer, scaler, ema, epoch, histories, best_quality)
            log(f"⭐ Nouveau meilleur modèle sauvegardé à l'epoch {epoch} (Qualité: {q:.4f}) → {BEST_MODEL_PATH}")

        save_ckpt(LAST_CHECKPOINT_PATH, model, optimizer, scaler, ema, epoch, histories, best_quality)

    log("✅ Entraînement terminé.")


if __name__ == "__main__":
    main()
