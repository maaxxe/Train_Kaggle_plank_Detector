

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm.auto import tqdm

PROJECT = Path("/kaggle/working/PlankEyev2_multipieces")
SCRIPT_DIR = PROJECT
DATA_DIR = SCRIPT_DIR / "data_kaggle_2"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.chdir(PROJECT)
if str(PROJECT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT))

from model_v5 import (
    IMG_SIZE,
    NUM_CLASSES,
    PlankEyeV5,
    combined_loss_v5,
)

try:
    from model_v5 import compute_detection_metrics
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

BATCH_SIZE = 12
VAL_BATCH_MULT = 2
WORLD_SIZE_ENV = max(1, int(os.environ.get("WORLD_SIZE", "1")))
GRAD_ACCUM = 1 if WORLD_SIZE_ENV > 1 else 2
EPOCHS = 180

LR_HEAD = 3.0e-4
LR_BACKBONE = 2.0e-5
LR_MIN_HEAD = 2.0e-6
LR_MIN_BACKBONE = 2.0e-7
HEAD_WARMUP_EPOCHS = 3
BACKBONE_WARMUP_EPOCHS = 3

UNFREEZE_LAST_EPOCH = 8
UNFREEZE_LAST_N_BLOCKS = 6
UNFREEZE_ALL_EPOCH = 20

WEIGHT_DECAY = 2.0e-4
GRAD_CLIP_NORM = 2.0
EMA_DECAY = 0.9998
EARLY_STOP_PATIENCE = 32
EARLY_STOP_MIN_DELTA = 1e-4

CLASS_BALANCE_POWER = 0.35
CLASS_WEIGHT_MIN = 0.70
CLASS_WEIGHT_MAX = 1.50

ROT_MAX_DEG = 30.0
ROT_MAX_OUT_OF_BOUNDS = 0.015
RANDOM_ERASE_P = 0.15

BEST_MODEL_PATH = SCRIPT_DIR / "best_checkpoint_v5.pt"
LAST_CHECKPOINT_PATH = SCRIPT_DIR / "last_checkpoint_v5.pt"

PERSIST_EVERY = 5
KAGGLE_DATASET_ID = "max778/chekpoints-backbone18"
PERSIST_DIR = Path("/kaggle/working/chekpoints-backbone18_persistent")

PLOT_PATH = SCRIPT_DIR / "training_curves_v5.png"
VIZ_DIR = SCRIPT_DIR / "viz_epochs_v5"
SPLIT_MANIFEST_PATH = SCRIPT_DIR / "split_v5.json"
VIZ_EVERY = 2

RUN_ID = "v5_run"
PRETRAINED_BACKBONE = True

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = max(1, int(os.environ.get("WORLD_SIZE", "1")))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_MAIN = RANK == 0

if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
    DEVICE_TYPE = "cuda"
else:
    DEVICE = torch.device("cpu")
    DEVICE_TYPE = "cpu"

AMP_ENABLED = DEVICE_TYPE == "cuda"


def rank0_print(*args, **kwargs) -> None:
    if IS_MAIN:
        print(*args, **kwargs)


def setup_distributed() -> None:
    if DEVICE_TYPE != "cuda":
        raise RuntimeError("Aucun GPU CUDA détecté. Active un runtime GPU Kaggle.")
    torch.cuda.set_device(LOCAL_RANK)
    if IS_DISTRIBUTED and not dist.is_initialized():
        store_path = "/kaggle/working/PlankEyev2_multipieces/ddp_filestore"
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{store_path}",
            rank=RANK,
            world_size=WORLD_SIZE,
            device_id=DEVICE,
        )
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if IS_DISTRIBUTED and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def wrap_for_training(model: nn.Module) -> nn.Module:
    if not IS_DISTRIBUTED:
        return model
    return DDP(
        model,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def broadcast_stop(stop: bool) -> bool:
    if not IS_DISTRIBUTED:
        return bool(stop)
    flag = torch.tensor([1 if stop else 0], device=DEVICE, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def set_seed(seed: int, rank: int = 0) -> None:
    process_seed = int(seed) + int(rank) * 1000
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


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
        if img_path.name in CORRUPTED_IMAGES:   # ✅ exclusion immédiate
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
    if IS_MAIN:
        print(f"✅ {len(pairs)} paires chargées ({len(CORRUPTED_IMAGES)} image(s) exclue(s) : {CORRUPTED_IMAGES})")
    return pairs
    
def _class_signature(label_path: Path) -> Tuple[int, ...]:
    objects = read_label(label_path)
    return (len(objects),)

def _write_split_manifest(train_p, val_p, test_p) -> None:
    # ✅ Seul rank0 écrit le fichier
    if int(os.environ.get("RANK", "0")) != 0:
        return

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

    # ── calcul du split (inchangé) ──────────────────────────────────────────
    groups = defaultdict(list)
    for pair in pairs:
        groups[_class_signature(pair[1])].append(pair)

    rng = random.Random(SEED)
    train_p, val_p, test_p = [], [], []

    for signature in sorted(groups, key=str):
        group = sorted(groups[signature])
        rng.shuffle(group)
        n = len(group)
        n_val  = int(round(n * VAL_RATIO))
        n_test = int(round(n * TEST_RATIO))
        if n >= 10:
            n_val  = max(1, n_val)
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

    # ✅ rank0 écrit, puis tout le monde attend avant de continuer
    _write_split_manifest(train_p, val_p, test_p)
    if dist.is_initialized():
        dist.barrier()

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
    def __init__(self, hflip: float=0.5, vflip: float=0.0, rot_deg: float=ROT_MAX_DEG, max_out_of_bounds: float=ROT_MAX_OUT_OF_BOUNDS):
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
                transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.035)], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.15, 1.0))], p=0.12)
            ])
        else:
            self.photo_tf = None
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        try:
            with Image.open(img_path) as im:
                image = im.convert('RGB')
        except Exception:
            # Image corrompue non détectée au scan → on prend la suivante
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
    workers = max(1, min(4, cpu // max(WORLD_SIZE, 1)))
    pin = DEVICE_TYPE == "cuda"
    common = {
        "num_workers": workers,
        "pin_memory": pin,
        "persistent_workers": False,                          # ✅ False (était workers > 0)
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker if workers > 0 else None,
        "prefetch_factor": 2 if workers > 0 else None,       # ✅ ajouté
    }
    train_ds = PlankDataset(train_p, augment=True)
    val_ds = PlankDataset(val_p, augment=False)
    test_ds = PlankDataset(test_p, augment=False)
    train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=SEED) if IS_DISTRIBUTED else None
    generator = torch.Generator()
    generator.manual_seed(SEED + RANK)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=train_sampler is None, sampler=train_sampler, generator=generator if train_sampler is None else None, **common)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * VAL_BATCH_MULT, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * VAL_BATCH_MULT, shuffle=False, **common)
    
    if IS_MAIN:
        print(f"📦 DataLoaders prêts : {len(train_p)} images d'entraînement, {len(val_p)} en validation.")
    return train_loader, val_loader, test_loader, train_sampler

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float=EMA_DECAY):
        base = unwrap_model(model)
        self.ema = copy.deepcopy(base).eval()
        self.decay = float(decay)
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        model_state = unwrap_model(model).state_dict()
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

    Le choix suit directement UNFREEZE_LAST_EPOCH et
    UNFREEZE_ALL_EPOCH definis plus haut.
    """
    base = unwrap_model(model)
    backbone = base.backbone

    # Tout geler d'abord.
    for param in backbone.parameters():
        param.requires_grad_(False)

    # Derniers blocs du ResNet18.
    if epoch >= UNFREEZE_LAST_EPOCH:
        for param in backbone.layer3.parameters():
            param.requires_grad_(True)
        for param in backbone.layer4.parameters():
            param.requires_grad_(True)

    # Backbone complet.
    if epoch >= UNFREEZE_ALL_EPOCH:
        for param in backbone.parameters():
            param.requires_grad_(True)

    trainable = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in backbone.parameters())

    if epoch < UNFREEZE_LAST_EPOCH:
        phase = "GELÉ"
    elif epoch < UNFREEZE_ALL_EPOCH:
        phase = "PARTIEL (layer3+layer4)"
    else:
        phase = "COMPLET"

    rank0_print(
        f"🔓 Backbone epoch {epoch}: {phase} | "
        f"{trainable:,}/{total:,} paramètres entraînables"
    )


def build_optimizer(model):
    base = unwrap_model(model)

    backbone_params = []
    head_params = []

    for name, param in base.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []

    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": LR_HEAD,
            "group_name": "head",
        })

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": LR_BACKBONE,
            "group_name": "backbone",
        })

    return torch.optim.AdamW(
        param_groups,
        weight_decay=WEIGHT_DECAY,
    )


def run_epoch(model, loader, optimizer=None, scaler=None, epoch=None, total_epochs=None, phase="train", ema=None):
    training = optimizer is not None
    active_model = model if training else (ema.ema if ema is not None else unwrap_model(model))

    if training:
        model.train()

        # model.train() remet tous les modules en mode train.
        # Pour le backbone gele, on remet les couches concernees
        # en eval afin que les BatchNorm ne modifient pas leurs
        # statistiques pendant la phase de gel.
        if epoch is not None:
            base = unwrap_model(model)
            backbone = base.backbone

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

    bar = tqdm(loader, desc=f"{phase.upper()} {epoch}/{total_epochs}" if epoch else phase.upper(), disable=(training and not IS_MAIN))

    for step, (images, batch_objects) in enumerate(bar):
        images = images.to(DEVICE, non_blocking=True)
        bs = images.shape[0]

        autocast_ctx = torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16 if DEVICE_TYPE == "cuda" else torch.bfloat16, enabled=AMP_ENABLED)

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
    if not IS_MAIN:
        return
    base = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "model": base.state_dict(),
        "optim": optimizer.state_dict(),
        "ema": ema.ema.state_dict() if ema else None,
        "histories": histories,
        "best_quality": best_quality,
        "config": {"run_id": RUN_ID, "pretrained_backbone": PRETRAINED_BACKBONE}
    }
    torch.save(payload, path)


def persist_checkpoints_to_kaggle(epoch: int) -> bool:
    if not IS_MAIN:
        return False
    try:
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BEST_MODEL_PATH, PERSIST_DIR / BEST_MODEL_PATH.name)
        shutil.copy2(LAST_CHECKPOINT_PATH, PERSIST_DIR / LAST_CHECKPOINT_PATH.name)
        
        metadata = {"title": "chekpoints-backbone18", "id": KAGGLE_DATASET_ID, "licenses": [{"name": "other"}], "isPrivate": True}
        with open(PERSIST_DIR / "dataset-metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        kaggle_exe = shutil.which("kaggle")
        if kaggle_exe:
            subprocess.run([kaggle_exe, "datasets", "version", "-p", str(PERSIST_DIR), "-m", f"Auto checkpoint epoch {epoch}", "--delete-old-versions"], check=False)
            print(f"[PERSIST] Checkpoint sauvegardé sur le dataset Kaggle (epoch {epoch}).")
    except Exception as exc:
        print(f"[PERSIST] Erreur : {exc}")
    return True


def get_checkpoint_start_epoch(path: Path) -> int:
    """Retourne l'epoch de reprise sans charger l'etat de l'optimizer."""
    if not path.exists():
        return 1

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", 0)) + 1


def load_ckpt(model, optimizer, scaler, ema, path: Path):
    empty_histories = {"train_loss": [], "val_loss": [], "quality": []}
    if not path.exists():
        return 1, empty_histories, float("-inf")

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"])

    # L'optimizer a deja ete construit avec le bon etat
    # freeze/unfreeze correspondant a l'epoch de reprise.
    if checkpoint.get("optim"):
        try:
            optimizer.load_state_dict(checkpoint["optim"])
        except (ValueError, RuntimeError) as exc:
            rank0_print(
                f"⚠️ Etat optimizer incompatible avec la nouvelle phase: {exc}"
            )
            rank0_print("   → Reprise des poids OK, optimizer réinitialisé pour cette phase.")

    if ema and checkpoint.get("ema"):
        ema.ema.load_state_dict(checkpoint["ema"])

    return (
        int(checkpoint.get("epoch", 0)) + 1,
        checkpoint.get("histories", empty_histories),
        float(checkpoint.get("best_quality", float("-inf"))),
    )


def main():
    setup_distributed()
    try:
        set_seed(SEED, rank=RANK)
        
        if IS_MAIN:
            print("=" * 60)
            print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT PLANKEYE V5")
            print("=" * 60)

        pairs = collect_pairs()
        train_p, val_p, test_p = split_pairs(pairs)
        barrier()

        train_loader, val_loader, test_loader, train_sampler = make_loaders(train_p, val_p, test_p)

        if IS_MAIN:
            print("📥 Téléchargement / Initialisation du backbone ResNet18 (Poids ImageNet préentraînés)...")
        
        raw_model = PlankEyeV5().to(DEVICE)
        
        if IS_MAIN:
            print("✅ Modèle ResNet18 et têtes V5 initialisés avec succès sur le device :", DEVICE)

        # Déterminer la phase AVANT de construire l'optimizer.
        # Cela permet une reprise propre même si le dernier checkpoint
        # a été sauvegardé pendant une phase partielle ou complète.
        resume_epoch = get_checkpoint_start_epoch(LAST_CHECKPOINT_PATH)
        set_backbone_trainable(raw_model, resume_epoch)

        optimizer = build_optimizer(raw_model)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
        ema = ModelEMA(raw_model)

        start_epoch, histories, best_quality = load_ckpt(
            raw_model, optimizer, scaler, ema, LAST_CHECKPOINT_PATH
        )

        # Sécurité : l'état final doit correspondre exactement à l'epoch reprise.
        set_backbone_trainable(raw_model, start_epoch)
        
        train_model = wrap_for_training(raw_model)

        if start_epoch > 1:
            print(f"🔄 Reprise de l'entraînement à partir de l'epoch {start_epoch}")

        for epoch in range(start_epoch, EPOCHS + 1):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            # Reconfiguration uniquement aux transitions de phase.
            # L'optimizer est reconstruit afin de ne contenir que les
            # paramètres actuellement entraînables.
            if epoch in (UNFREEZE_LAST_EPOCH, UNFREEZE_ALL_EPOCH):
                set_backbone_trainable(raw_model, epoch)
                optimizer = build_optimizer(raw_model)
                rank0_print(f"🔧 Optimizer reconstruit pour la phase de l'epoch {epoch}")

            train_stats = run_epoch(train_model, train_loader, optimizer=optimizer, scaler=scaler, epoch=epoch, total_epochs=EPOCHS, phase="train", ema=ema)
            barrier()
    
            if IS_MAIN:
                val_stats = run_epoch(raw_model, val_loader, optimizer=None, scaler=scaler, epoch=epoch, total_epochs=EPOCHS, phase="val", ema=ema)
                
                histories["train_loss"].append(train_stats.get("loss", 0.0))
                histories["val_loss"].append(val_stats.get("loss", 0.0))
                q = -val_stats.get("loss", float("inf"))
                histories["quality"].append(q)

                # ✅ Résumé de l'epoch
                print(
                    f"📊 Epoch {epoch}/{EPOCHS} | "
                    f"Train loss: {train_stats.get('loss', 0):.4f} | "
                    f"Val loss: {val_stats.get('loss', 0):.4f} | "
                    f"Hmap: {val_stats.get('heatmap', 0):.4f} | "
                    f"Corner: {val_stats.get('corner', 0):.4f} | "
                    f"Geom: {val_stats.get('geometry', 0):.4f}"
                )
    
                if q > best_quality:
                    best_quality = q
                    save_ckpt(BEST_MODEL_PATH, raw_model, optimizer, scaler, ema, epoch, histories, best_quality)
                    print(f"⭐ Nouveau meilleur modèle sauvegardé à l'epoch {epoch} (Qualité: {q:.4f})")
    
                save_ckpt(LAST_CHECKPOINT_PATH, raw_model, optimizer, scaler, ema, epoch, histories, best_quality)
    
                if epoch % PERSIST_EVERY == 0:
                    persist_checkpoints_to_kaggle(epoch)
    
            barrier()  # ✅ rank1 attend que rank0 finisse validation + sauvegarde
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
    
