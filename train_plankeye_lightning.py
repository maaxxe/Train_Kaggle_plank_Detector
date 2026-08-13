

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
import math
import os
import random
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

STUDIO_ROOT = Path("/teamspace/studios/this_studio")
PROJECT = STUDIO_ROOT / "Train_Kaggle_plank_Detector"
SCRIPT_DIR = PROJECT
DATASET_CANDIDATES = [
    PROJECT / "dataset" / "dataset_fusionne" / "dataset_fusionne",
    PROJECT / "dataset" / "dataset_fusionne",
    PROJECT / "dataset",
]
DATA_DIR = next(
    (
        p for p in DATASET_CANDIDATES
        if (p / "images").is_dir() and (p / "labels").is_dir()
    ),
    PROJECT / "dataset" / "dataset_fusionne" / "dataset_fusionne",
)
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset Lightning introuvable ou incomplet : {DATA_DIR}"
    )

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.chdir(PROJECT)
if str(PROJECT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT))

from model.model import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    build_model,
    combined_loss_v2,
    compute_detection_metrics,
)

if NUM_CLASSES != 1:
    raise RuntimeError(f"Le modèle doit être en 1 classe, NUM_CLASSES={NUM_CLASSES}")

SEED = 48

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Batch par GPU.
# Lightning 1 GPU : 8 images × accumulation 2 = batch effectif 16.
# Si 2 GPU sont disponibles : 8 × 2 GPU × accumulation 1 = batch global 16.
BATCH_SIZE = 8
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

MODEL_WEIGHTS_DIR = PROJECT / "model_poids"
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_WEIGHTS_DIR / "best_plankeye_v4_1class_512.pt"
LAST_CHECKPOINT_PATH = MODEL_WEIGHTS_DIR / "last_plankeye_v4_1class_512.pt"

# Sauvegarde persistante Kaggle :
# tous les 5 epochs, best.pt + last.pt remplacent la version précédente
# du dataset max778/checkpoints.
PERSIST_EVERY = 5
KAGGLE_DATASET_ID = "max778/checkpoints"
PERSIST_DIR = STUDIO_ROOT / "checkpoints_persistent"

PLOT_PATH = SCRIPT_DIR / "training_curves_v4_1class_512.png"
VIZ_DIR = SCRIPT_DIR / "viz_epochs_v4_1class_512"
SPLIT_MANIFEST_PATH = SCRIPT_DIR / "split_plankeye_v4_1class.json"
VIZ_EVERY = 2

WARMSTART_V2_CANDIDATES = [
    SCRIPT_DIR / "best_plankeye_v2_384.pt",
    SCRIPT_DIR / "best_plankeye_v2.pt",
]

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
        raise RuntimeError(
            "Aucun GPU CUDA détecté. Active une machine GPU dans Lightning AI."
        )

    torch.cuda.set_device(LOCAL_RANK)

    if IS_DISTRIBUTED and not dist.is_initialized():

        store_path = str(PROJECT / "ddp_filestore")

        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{store_path}",
            rank=RANK,
            world_size=WORLD_SIZE,
            device_id=DEVICE,
        )

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if IS_DISTRIBUTED and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def stage_name_for_epoch(epoch: int) -> str:
    if epoch < UNFREEZE_LAST_EPOCH:
        return "frozen"
    if epoch < UNFREEZE_ALL_EPOCH:
        return f"last_{UNFREEZE_LAST_N_BLOCKS}"
    return "all"


def wrap_for_training(model: nn.Module) -> nn.Module:
    if not IS_DISTRIBUTED:
        return model
    return DDP(
        model,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        broadcast_buffers=False,
        find_unused_parameters=True,
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

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
    """Lit les quadrilatères et fusionne toutes les anciennes classes en classe 0."""
    objects = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            try:
                _old_cls = int(float(parts[0]))
                coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
            except ValueError:
                continue

            if not np.isfinite(coords).all():
                continue

            cls = 0
            pts = coords.reshape(4, 2)

            if (pts < -0.02).any() or (pts > 1.02).any():
                continue

            pts = np.clip(pts, 0.0, 1.0)
            pts = reorder_corners(pts)

            if polygon_area_np(pts) < 1e-06:
                continue

            objects.append({'cls': cls, 'corners': pts})

    return objects

def collect_pairs():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise RuntimeError(f'Dataset absent : {IMAGES_DIR} / {LABELS_DIR}')
    pairs = []
    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = LABELS_DIR / f'{img_path.stem}.txt'
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs

def _class_signature(label_path: Path) -> Tuple[int, ...]:
    objects = read_label(label_path)
    return (len(objects),)

def _write_split_manifest(train_p, val_p, test_p) -> None:
    payload = {'version': 1, 'seed': SEED, 'dataset_dir': str(DATA_DIR), 'train': [p[0].name for p in train_p], 'val': [p[0].name for p in val_p], 'test': [p[0].name for p in test_p]}
    tmp = SPLIT_MANIFEST_PATH.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SPLIT_MANIFEST_PATH)

def _try_load_split_manifest(pairs):
    if not SPLIT_MANIFEST_PATH.exists():
        return None
    try:
        payload = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'Manifest split illisible, recréation : {exc}')
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
    print(f'Split persistant chargé : train={len(train_p)} val={len(val_p)} test={len(test_p)} | nouvelles images ajoutées au train={len(new_pairs)}')
    return (train_p, val_p, test_p)

def split_pairs(pairs):
    existing = _try_load_split_manifest(pairs)
    if existing is not None:
        return existing
    groups: Dict[Tuple[int, ...], List[Tuple[Path, Path]]] = defaultdict(list)
    for pair in pairs:
        groups[_class_signature(pair[1])].append(pair)
    rng = random.Random(SEED)
    train_p, val_p, test_p = ([], [], [])
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
    print(f'Split stratifié créé : train={len(train_p)} val={len(val_p)} test={len(test_p)} (~{TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})')
    return (train_p, val_p, test_p)

def split_stats(name: str, pairs) -> Counter:
    counts = Counter()
    empty = 0
    for _img, label in pairs:
        objs = read_label(label)
        if not objs:
            empty += 1
        for obj in objs:
            counts[int(obj['cls'])] += 1
    pretty = {CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c): counts[c] for c in range(NUM_CLASSES)}
    print(f'{name:>5} : images={len(pairs)} objets={sum(counts.values())} vides={empty} | {pretty}')
    return counts

def compute_class_weights(train_pairs) -> torch.Tensor:
    counts = split_stats('train', train_pairs)
    vals = np.asarray([max(counts[c], 1) for c in range(NUM_CLASSES)], dtype=np.float64)
    mean = vals.mean()
    weights = (mean / vals) ** CLASS_BALANCE_POWER
    weights = np.clip(weights, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)
    weights = weights / weights.mean()
    tensor = torch.tensor(weights, dtype=torch.float32)
    print('Poids classes :', {CLASS_NAMES[i]: round(float(w), 3) for i, w in enumerate(weights)})
    return tensor

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
        if random.random() < self.vflip:
            out_img = out_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            for obj in out_objs:
                obj['corners'][:, 1] = 1.0 - obj['corners'][:, 1]
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

def safe_random_erase(tensor: torch.Tensor, objects, p: float=RANDOM_ERASE_P, scale=(0.008, 0.05), ratio=(0.4, 2.5), max_overlap: float=0.1, max_tries: int=12, valid_box=None):
    if random.random() > p or not objects:
        return tensor
    _, H, W = tensor.shape
    if valid_box is None:
        vx0, vy0, vx1, vy1 = (0, 0, W, H)
    else:
        vx0, vy0, vx1, vy1 = valid_box
        vx0, vy0 = (max(0, vx0), max(0, vy0))
        vx1, vy1 = (min(W, vx1), min(H, vy1))
    valid_w, valid_h = (vx1 - vx0, vy1 - vy0)
    if valid_w <= 1 or valid_h <= 1:
        return tensor
    valid_area = valid_w * valid_h
    boxes = []
    for obj in objects:
        pts = np.asarray(obj['corners'])
        x0, y0 = pts.min(axis=0) * [W, H]
        x1, y1 = pts.max(axis=0) * [W, H]
        boxes.append((float(x0), float(y0), float(x1), float(y1)))
    for _ in range(max_tries):
        target_area = random.uniform(*scale) * valid_area
        aspect = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        erase_h = int(round(math.sqrt(target_area * aspect)))
        erase_w = int(round(math.sqrt(target_area / aspect)))
        if erase_h <= 0 or erase_w <= 0 or erase_h >= valid_h or (erase_w >= valid_w):
            continue
        top = random.randint(vy0, vy1 - erase_h)
        left = random.randint(vx0, vx1 - erase_w)
        worst = 0.0
        for x0, y0, x1, y1 in boxes:
            ix0, iy0 = (max(left, x0), max(top, y0))
            ix1, iy1 = (min(left + erase_w, x1), min(top + erase_h, y1))
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            box_area = max((x1 - x0) * (y1 - y0), 1e-06)
            worst = max(worst, inter / box_area)
        if worst <= max_overlap:
            tensor[:, top:top + erase_h, left:left + erase_w] = 0.0
            return tensor
    return tensor

class PlankDataset(Dataset):

    def __init__(self, pairs, augment=False):
        self.pairs = list(pairs)
        self.augment = augment
        self.geo_aug = GeometricAug() if augment else None
        if augment:
            self.photo_tf = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.035)], p=0.8), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.15, 1.0))], p=0.12), transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.12), transforms.RandomGrayscale(p=0.025)])
        else:
            self.photo_tf = None
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        with Image.open(img_path) as im:
            image = im.convert('RGB')
        objects = read_label(label_path)
        if self.geo_aug is not None:
            image, objects = self.geo_aug(image, objects)
        if self.photo_tf is not None:
            image = self.photo_tf(image)
        image, objects, valid_box = letterbox_resize(image, IMG_SIZE, objects, return_valid_box=True)
        tensor = self.to_tensor(image)
        if self.augment:
            tensor = safe_random_erase(tensor, objects, valid_box=valid_box)
        return (tensor, objects)

def make_loaders(train_p, val_p, test_p):
    cpu = os.cpu_count() or 4
    workers = max(1, min(4, cpu // max(WORLD_SIZE, 1)))
    pin = DEVICE_TYPE == "cuda"

    common = {
        "num_workers": workers,
        "pin_memory": pin,
        "persistent_workers": workers > 0,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker if workers > 0 else None,
    }
    if workers > 0:
        common["prefetch_factor"] = 3

    train_ds = PlankDataset(train_p, augment=True)
    val_ds = PlankDataset(val_p, augment=False)
    test_ds = PlankDataset(test_p, augment=False)

    train_sampler = None
    if IS_DISTRIBUTED:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=WORLD_SIZE,
            rank=RANK,
            shuffle=True,
            seed=SEED,
            drop_last=False,
        )

    generator = torch.Generator()
    generator.manual_seed(SEED + RANK)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator if train_sampler is None else None,
        drop_last=False,
        **common,
    )

    # Validation/test complets uniquement utilisés par rank 0.
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * VAL_BATCH_MULT,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * VAL_BATCH_MULT,
        shuffle=False,
        drop_last=False,
        **common,
    )

    rank0_print(
        f"DataLoader : workers/rank={workers} | world_size={WORLD_SIZE} | "
        f"pin_memory={pin} | prefetch={common.get('prefetch_factor', 0)}"
    )
    return train_loader, val_loader, test_loader, train_sampler

class ModelEMA:

    def __init__(self, model: nn.Module, decay: float=EMA_DECAY, tau: float=2000.0):
        base = unwrap_model(model)
        self.ema = copy.deepcopy(base).eval()
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        model_state = unwrap_model(model).state_dict()
        for key, value in self.ema.state_dict().items():
            src = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(d).add_(src, alpha=1.0 - d)
            else:
                value.copy_(src)

    def predict(self, x, **kwargs):
        return self.ema.predict(x, **kwargs)

def set_bn_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

def configure_backbone_stage(model, epoch: int, announce: bool=False) -> str:
    base = unwrap_model(model)
    if epoch < UNFREEZE_LAST_EPOCH:
        base.freeze_backbone()
        stage = "frozen"
    elif epoch < UNFREEZE_ALL_EPOCH:
        base.unfreeze_backbone(last_n=UNFREEZE_LAST_N_BLOCKS)
        stage = f"last_{UNFREEZE_LAST_N_BLOCKS}"
    else:
        base.unfreeze_backbone(last_n=None)
        stage = "all"

    if announce and IS_MAIN:
        trainable_bb = sum(p.numel() for p in base.backbone.parameters() if p.requires_grad)
        total_bb = sum(p.numel() for p in base.backbone.parameters())
        print(
            f"Backbone stage={stage} | "
            f"paramètres entraînables={trainable_bb:,}/{total_bb:,}"
        )
    return stage

def _cosine(start: float, end: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

def scheduled_lrs(epoch: int) -> Tuple[float, float]:
    if epoch <= HEAD_WARMUP_EPOCHS:
        head_lr = LR_HEAD * epoch / max(HEAD_WARMUP_EPOCHS, 1)
    else:
        progress = (epoch - HEAD_WARMUP_EPOCHS - 1) / max(EPOCHS - HEAD_WARMUP_EPOCHS - 1, 1)
        head_lr = _cosine(LR_HEAD, LR_MIN_HEAD, progress)
    if epoch < UNFREEZE_LAST_EPOCH:
        bb_lr = 0.0
    elif epoch < UNFREEZE_LAST_EPOCH + BACKBONE_WARMUP_EPOCHS:
        k = epoch - UNFREEZE_LAST_EPOCH + 1
        bb_lr = LR_BACKBONE * k / BACKBONE_WARMUP_EPOCHS
    else:
        start_ep = UNFREEZE_LAST_EPOCH + BACKBONE_WARMUP_EPOCHS
        progress = (epoch - start_ep) / max(EPOCHS - start_ep, 1)
        bb_lr = _cosine(LR_BACKBONE, LR_MIN_BACKBONE, progress)
    return (head_lr, bb_lr)

def apply_learning_rates(optimizer, head_lr: float, backbone_lr: float) -> None:
    for group in optimizer.param_groups:
        name = group.get('group_name', 'head')
        mult = float(group.get('lr_mult', 1.0))
        group['lr'] = head_lr if name == 'head' else backbone_lr * mult

def build_optimizer(model):
    base = unwrap_model(model)
    groups = base.param_groups(lr_head=LR_HEAD, lr_backbone=LR_BACKBONE)
    kwargs = dict(
        params=groups,
        lr=LR_HEAD,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=WEIGHT_DECAY,
    )
    if DEVICE_TYPE == "cuda":
        try:
            return torch.optim.AdamW(**kwargs, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(**kwargs)

def run_epoch(
    model,
    loader,
    class_weights: torch.Tensor,
    optimizer=None,
    scaler=None,
    epoch=None,
    total_epochs=None,
    phase="train",
    ema: ModelEMA | None=None,
):
    training = optimizer is not None
    active_model = model if training else (ema.ema if ema is not None else unwrap_model(model))

    if training:
        model.train()
        set_bn_eval(unwrap_model(model).backbone)
    else:
        active_model.eval()

    tracked_keys = (
        "loss",
        "hmap",
        "corner",
        "corner_px",
        "offset",
        "area",
        "edge_len",
        "edge_dir",
        "consistency",
    )

    totals = defaultdict(float)
    total_images = 0
    total_collisions = 0
    total_candidates = 0

    desc = f"{phase.upper()} {epoch}/{total_epochs}" if epoch is not None else phase.upper()
    bar = tqdm(
        loader,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=(training and not IS_MAIN),
    )

    total_steps = len(loader)
    remainder = total_steps % GRAD_ACCUM

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, batch_objects) in enumerate(bar):
        images = images.to(DEVICE, non_blocking=True)
        bs = images.shape[0]

        autocast_ctx = torch.amp.autocast(
            device_type=DEVICE_TYPE,
            dtype=torch.float16 if DEVICE_TYPE == "cuda" else torch.bfloat16,
            enabled=AMP_ENABLED,
        )

        if training:
            final_partial = remainder != 0 and step >= total_steps - remainder
            group_size = remainder if final_partial else GRAD_ACCUM
            should_step = (step + 1) % GRAD_ACCUM == 0 or step + 1 == total_steps

            sync_ctx = nullcontext()
            if IS_DISTRIBUTED and isinstance(model, DDP) and not should_step:
                sync_ctx = model.no_sync()

            with sync_ctx:
                with autocast_ctx:
                    outputs = active_model(images)
                    loss, breakdown = combined_loss_v2(
                        *outputs,
                        batch_objects,
                        class_weights=class_weights,
                    )

                if not torch.isfinite(loss):
                    rank0_print(
                        f"Loss NaN/Inf : epoch={epoch} step={step}, batch ignoré"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaled_loss = loss / max(group_size, 1)

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in unwrap_model(model).parameters() if p.requires_grad],
                    GRAD_CLIP_NORM,
                )

                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

        else:
            with torch.no_grad(), autocast_ctx:
                outputs = active_model(images)
                loss, breakdown = combined_loss_v2(
                    *outputs,
                    batch_objects,
                    class_weights=class_weights,
                )

        totals["loss"] += float(loss.detach()) * bs
        for key in tracked_keys[1:]:
            totals[key] += float(breakdown.get(key, 0.0)) * bs

        total_collisions += int(breakdown.get("n_collisions", 0))
        total_candidates += int(breakdown.get("n_candidates", 0))
        total_images += bs

        if IS_MAIN:
            denom = max(total_images, 1)
            coll = 100.0 * total_collisions / max(total_candidates, 1)
            bar.set_postfix(
                loss=f"{totals['loss'] / denom:.4f}",
                corner_px=f"{totals['corner_px'] / denom:.2f}",
                hmap=f"{totals['hmap'] / denom:.3f}",
                coll=f"{coll:.2f}%",
            )

    # TRAIN : agrégation des statistiques des deux GPU.
    if training and IS_DISTRIBUTED:
        values = [totals[k] for k in tracked_keys]
        values += [float(total_images), float(total_collisions), float(total_candidates)]
        vec = torch.tensor(values, device=DEVICE, dtype=torch.float64)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        values = vec.cpu().tolist()

        for i, key in enumerate(tracked_keys):
            totals[key] = values[i]
        total_images = int(values[len(tracked_keys)])
        total_collisions = int(values[len(tracked_keys) + 1])
        total_candidates = int(values[len(tracked_keys) + 2])

    denom = max(total_images, 1)
    stats = {key: value / denom for key, value in totals.items()}
    stats["collision_rate"] = 100.0 * total_collisions / max(total_candidates, 1)
    return stats

@torch.no_grad()
def evaluate_metrics(predictor, loader, conf_thresh=0.03, topk=100):
    predictor_model = predictor.ema if hasattr(predictor, 'ema') else predictor
    was_training = predictor_model.training
    predictor_model.eval()
    all_dets, all_gts = ([], [])
    for images, batch_objects in tqdm(loader, desc='METRICS', unit='batch', dynamic_ncols=True, leave=False):
        images = images.to(DEVICE, non_blocking=True)
        dets = predictor.predict(images, conf_thresh=conf_thresh, topk=topk)
        all_dets.extend(dets)
        all_gts.extend(batch_objects)
    predictor_model.train(was_training)
    return compute_detection_metrics(all_dets, all_gts, num_classes=NUM_CLASSES)

def quality_score(metrics: Dict[str, float]) -> float:
    corner_quality = metrics['pck4'] * metrics['match_recall']
    return 0.3 * metrics['map50'] + 0.5 * metrics['map75'] + 0.2 * corner_quality

def warmstart_from_compatible_checkpoint(model, candidates) -> Path | None:
    if LAST_CHECKPOINT_PATH.exists():
        return None
    source_path = next((Path(p) for p in candidates if Path(p).exists()), None)
    if source_path is None:
        return None
    try:
        checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
        source_state = checkpoint.get('ema') or checkpoint.get('model') or checkpoint
        current = model.state_dict()
        compatible = {key: value for key, value in source_state.items() if key in current and tuple(value.shape) == tuple(current[key].shape)}
        if not compatible:
            print(f'Warm-start v2 ignoré : aucun tenseur compatible dans {source_path.name}')
            return None
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        loaded_params = sum((current[k].numel() for k in compatible))
        total_params = sum((v.numel() for v in current.values()))
        print(f'Warm-start depuis {source_path.name} : {len(compatible)} tenseurs, {loaded_params:,}/{total_params:,} valeurs chargées ({100.0 * loaded_params / max(total_params, 1):.1f}%)')
        print(f'  Nouvelles/incompatibles conservées à leur init : {len(missing)} tenseurs')
        if unexpected:
            print(f'  Tenseurs source inattendus ignorés : {len(unexpected)}')
        return source_path
    except Exception as exc:
        print(f'Warm-start v2 impossible ({source_path.name}) : {exc}')
        return None

def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_current"] = torch.cuda.get_rng_state(DEVICE)
    return state


def _restore_rng_state(state) -> None:
    # En cas de changement de nombre de GPU, chaque rank reçoit son propre seed reproductible.
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda_current" in state:
            torch.cuda.set_rng_state(state["cuda_current"], device=DEVICE)
    except Exception as exc:
        rank0_print(f"État RNG non restauré : {exc}")

def save_ckpt(
    path: Path,
    model,
    optimizer,
    scaler,
    ema,
    epoch: int,
    histories: Dict[str, list],
    best_quality: float,
    best_epoch: int | None,
    best_metrics: Dict | None,
    last_improvement_epoch: int,
):
    if not IS_MAIN:
        return

    base = unwrap_model(model)
    payload = {
        "version": "plankeye_v4_1class_512_ddp",
        "epoch": epoch,
        "model": base.state_dict(),
        "optim": optimizer.state_dict(),
        "ema": ema.ema.state_dict() if ema is not None else None,
        "ema_updates": ema.updates if ema is not None else 0,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "histories": histories,
        "best_quality": best_quality,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_map50": (best_metrics or {}).get("map50", float("-inf")),
        "best_val": (
            min(histories.get("val_loss", [float("inf")]))
            if histories.get("val_loss")
            else float("inf")
        ),
        "last_improvement_epoch": last_improvement_epoch,
        "rng_state": _rng_state(),
        "config": {
            "img_size": IMG_SIZE,
            "batch_size_per_gpu": BATCH_SIZE,
            "world_size": WORLD_SIZE,
            "global_batch_size": BATCH_SIZE * WORLD_SIZE * GRAD_ACCUM,
            "grad_accum": GRAD_ACCUM,
            "lr_head": LR_HEAD,
            "lr_backbone": LR_BACKBONE,
        },
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def persist_checkpoints_to_kaggle(
    epoch: int,
    best_epoch=None,
    best_quality=None,
) -> bool:
    """
    Crée une nouvelle version du dataset Kaggle max778/checkpoints
    avec les best/last courants puis supprime les anciennes versions.

    Seul rank 0 exécute l'upload. Une erreur d'API/authentification
    est affichée mais ne fait jamais planter l'entraînement.
    """
    if not IS_MAIN:
        return False

    missing = [
        p.name
        for p in (BEST_MODEL_PATH, LAST_CHECKPOINT_PATH)
        if not p.exists()
    ]
    if missing:
        print(
            "[PERSIST] Upload ignoré, fichier(s) absent(s) : "
            + ", ".join(missing)
        )
        return False

    try:
        # Dossier propre pour ne pousser que les deux .pt.
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            BEST_MODEL_PATH,
            PERSIST_DIR / BEST_MODEL_PATH.name,
        )
        shutil.copy2(
            LAST_CHECKPOINT_PATH,
            PERSIST_DIR / LAST_CHECKPOINT_PATH.name,
        )

        # Pour une nouvelle version d'un dataset existant,
        # l'identifiant du dataset suffit dans les métadonnées.
        metadata = {
            "title": "checkpoints",
            "id": KAGGLE_DATASET_ID,
            "licenses": [{"name": "other"}],
        }
        with open(
            PERSIST_DIR / "dataset-metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

        message = f"PlankEye epoch {epoch}"
        if best_epoch is not None:
            message += f" | best epoch {best_epoch}"
        if best_quality is not None and math.isfinite(float(best_quality)):
            message += f" | quality {float(best_quality):.4f}"

        kaggle_exe = shutil.which("kaggle")
        if kaggle_exe is None:
            print(
                "[PERSIST] CLI Kaggle introuvable : "
                "checkpoint local conservé, entraînement poursuivi."
            )
            return False

        print()
        print("=" * 72)
        print(f"[PERSIST] Envoi des checkpoints après epoch {epoch}")
        print(f"[PERSIST] Dataset : {KAGGLE_DATASET_ID}")
        print("=" * 72)

        cmd = [
            kaggle_exe,
            "datasets",
            "version",
            "-p",
            str(PERSIST_DIR),
            "-m",
            message,
            "--delete-old-versions",
        ]

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout.strip():
            print(result.stdout.strip())

        if result.returncode != 0:
            print(
                f"[PERSIST] ÉCHEC upload (code {result.returncode}). "
                "L'entraînement continue."
            )
            if result.stderr.strip():
                print(result.stderr.strip())
            return False

        print(
            "[PERSIST] OK : best.pt + last.pt sont persistants "
            "et l'ancienne version a été remplacée."
        )
        return True

    except Exception as exc:
        print(
            f"[PERSIST] Erreur ignorée : {type(exc).__name__}: {exc}"
        )
        return False


def load_ckpt(model, optimizer, scaler, ema, path: Path):
    empty_histories = {
        "train_loss": [],
        "val_loss": [],
        "train_corner_px": [],
        "val_corner_px": [],
        "map50": [],
        "map75": [],
        "pck4": [],
        "match_recall": [],
        "quality": [],
        "lr_head": [],
        "lr_bb": [],
    }

    if not path.exists():
        return (1, empty_histories, float("-inf"), None, None, 0)

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    base = unwrap_model(model)

    try:
        base.load_state_dict(checkpoint["model"], strict=True)
    except Exception as exc:
        rank0_print(
            f"Checkpoint incompatible avec le nouveau modèle, démarrage neuf : {exc}"
        )
        return (1, empty_histories, float("-inf"), None, None, 0)

    try:
        optimizer.load_state_dict(checkpoint["optim"])
    except Exception as exc:
        rank0_print(f"Optimiseur non restauré : {exc}")

    if ema is not None and checkpoint.get("ema") is not None:
        try:
            ema.ema.load_state_dict(checkpoint["ema"], strict=True)
            ema.updates = int(checkpoint.get("ema_updates", 0))
        except Exception as exc:
            rank0_print(f"EMA non restaurée : {exc}")

    if scaler is not None and checkpoint.get("scaler"):
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as exc:
            rank0_print(f"GradScaler non restauré : {exc}")

    histories = empty_histories
    histories.update(checkpoint.get("histories", {}))

    # Changement de topologie possible (1 ou plusieurs GPU) :
    # on garde poids/optimiseur/EMA/scaler et on reprend à l'epoch suivant.
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_quality = float(checkpoint.get("best_quality", float("-inf")))
    best_epoch = checkpoint.get("best_epoch")
    best_metrics = checkpoint.get("best_metrics")
    last_improvement = int(
        checkpoint.get(
            "last_improvement_epoch",
            best_epoch or checkpoint.get("epoch", 0),
        )
    )

    rank0_print(
        f"Reprise checkpoint : epoch {start_epoch} | "
        f"best={best_epoch} quality={best_quality:.4f}"
    )
    return (
        start_epoch,
        histories,
        best_quality,
        best_epoch,
        best_metrics,
        last_improvement,
    )

def plot_curves(histories: Dict[str, list], best_epoch=None):
    if not histories['train_loss']:
        return
    epochs = np.arange(1, len(histories['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PlankEye v3 — entraînement', fontsize=14, fontweight='bold')
    ax = axes[0, 0]
    ax.plot(epochs, histories['train_loss'], label='train')
    ax.plot(epochs, histories['val_loss'], label='val')
    ax.set_title('Loss totale')
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(epochs, histories['train_corner_px'], label='train')
    ax.plot(epochs, histories['val_corner_px'], label='val')
    ax.set_title('Erreur coins — px entrée')
    ax.set_ylabel('pixels')
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(epochs, histories['map50'], label='mAP@0.5')
    ax.plot(epochs, histories['map75'], label='mAP@0.75')
    ax.plot(epochs, histories['pck4'], label='PCK@4px')
    ax.plot(epochs, histories['match_recall'], label='match recall', alpha=0.7)
    ax.set_title('Détection et précision')
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(epochs, histories['quality'], label='quality')
    ax2 = ax.twinx()
    ax2.plot(epochs, histories['lr_head'], ls='--', label='lr head')
    ax2.plot(epochs, histories['lr_bb'], ls=':', label='lr bb')
    ax.set_title('Score de sélection / learning rate')
    ax.set_ylabel('quality')
    ax2.set_ylabel('LR')
    ax.grid(alpha=0.3)
    if best_epoch is not None:
        for axis in axes.flat:
            axis.axvline(best_epoch, ls='--', alpha=0.5)
    for mark in (UNFREEZE_LAST_EPOCH, UNFREEZE_ALL_EPOCH):
        if mark <= len(epochs):
            for axis in axes.flat:
                axis.axvline(mark, ls=':', alpha=0.35)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)

def visualize_predictions(model, val_pairs, epoch: int, n: int=2, conf_thresh: float=0.3):
    if not val_pairs:
        return
    VIZ_DIR.mkdir(exist_ok=True)
    was_training = model.training
    model.eval()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    rng = random.Random(SEED + epoch)
    samples = rng.sample(val_pairs, min(n, len(val_pairs)))
    fig, axes = plt.subplots(1, len(samples), figsize=(7 * len(samples), 6))
    if len(samples) == 1:
        axes = [axes]
    for ax, (img_path, label_path) in zip(axes, samples):
        with Image.open(img_path) as im:
            raw = im.convert('RGB')
        gt = read_label(label_path)
        image, gt = letterbox_resize(raw, IMG_SIZE, gt)
        tensor = tf(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dets = model.predict(tensor, conf_thresh=conf_thresh, topk=50)[0]
        ax.imshow(np.asarray(image))
        for obj in gt:
            poly = np.asarray(obj['corners']) * IMG_SIZE
            ax.add_patch(plt.Polygon(poly, closed=True, fill=False, lw=1.5, ls='--'))
        for det in dets:
            poly = np.asarray(det['corners']) * IMG_SIZE
            ax.add_patch(plt.Polygon(poly, closed=True, fill=False, lw=1.8))
            center = poly.mean(axis=0)
            ax.text(center[0], center[1], f"c{det['cls']} {det['score']:.2f}", fontsize=7, ha='center', va='center', bbox={'boxstyle': 'round,pad=0.1', 'fc': 'white', 'alpha': 0.65})
        ax.set_title(f'{img_path.name} | GT={len(gt)} pred={len(dets)}')
        ax.axis('off')
    fig.suptitle(f'PlankEye v3 — epoch {epoch}')
    plt.tight_layout()
    out = VIZ_DIR / f'epoch_{epoch:03d}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    model.train(was_training)

def main():
    setup_distributed()

    try:
        set_seed(SEED, rank=RANK)

        if IS_MAIN:
            print("=" * 72)
            print("PlankEye — entraînement Lightning AI")
            print("=" * 72)
            print(f"PyTorch : {torch.__version__}")
            print(f"CUDA : {torch.version.cuda}")
            print(f"GPU disponibles : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(
                    f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
                    f"{props.total_memory / 1024**3:.2f} Go"
                )
            print(
                f"world_size={WORLD_SIZE} | batch/GPU={BATCH_SIZE} | "
                f"accum={GRAD_ACCUM} | "
                f"batch global={BATCH_SIZE * WORLD_SIZE * GRAD_ACCUM}"
            )
            print(f"Dataset : {DATA_DIR}")
            print(f"Best local : {BEST_MODEL_PATH}")
            print(f"Last local : {LAST_CHECKPOINT_PATH}")

        pairs = collect_pairs()
        if not pairs:
            raise RuntimeError(
                f"Aucune paire image/label dans {IMAGES_DIR} / {LABELS_DIR}"
            )

        # Un seul processus crée éventuellement le split persistant.
        if IS_MAIN:
            train_p, val_p, test_p = split_pairs(pairs)
        barrier()
        if not IS_MAIN:
            train_p, val_p, test_p = split_pairs(pairs)

        if IS_MAIN:
            print(f"Total images : {len(pairs)}")
            class_weights = compute_class_weights(train_p).to(DEVICE)
            split_stats("val", val_p)
            split_stats("test", test_p)
        else:
            class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=DEVICE)

        if IS_DISTRIBUTED:
            dist.broadcast(class_weights, src=0)

        train_loader, val_loader, test_loader, train_sampler = make_loaders(
            train_p,
            val_p,
            test_p,
        )

        raw_model = build_model(pretrained=False)
        warmstart_from_compatible_checkpoint(raw_model, WARMSTART_V2_CANDIDATES)
        raw_model = raw_model.to(DEVICE)

        optimizer = build_optimizer(raw_model)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
        ema = ModelEMA(raw_model)

        (
            start_epoch,
            histories,
            best_quality,
            best_epoch,
            best_metrics,
            last_improvement_epoch,
        ) = load_ckpt(
            raw_model,
            optimizer,
            scaler,
            ema,
            LAST_CHECKPOINT_PATH,
        )

        # Après le chargement, chaque rank reçoit un RNG distinct.
        set_seed(SEED + start_epoch, rank=RANK)

        start_time = time.time()
        previous_stage = None
        train_model = raw_model
        stop_training = False

        for epoch in range(start_epoch, EPOCHS + 1):
            wanted_stage = stage_name_for_epoch(epoch)

            # DDP enregistre les paramètres entraînables lors de sa construction.
            # On reconstruit donc le wrapper seulement aux changements de stage.
            if wanted_stage != previous_stage:
                barrier()
                if isinstance(train_model, DDP):
                    del train_model

                stage = configure_backbone_stage(
                    raw_model,
                    epoch,
                    announce=True,
                )
                train_model = wrap_for_training(raw_model)
                previous_stage = stage
                barrier()
            else:
                stage = previous_stage

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            head_lr, bb_lr = scheduled_lrs(epoch)
            apply_learning_rates(
                optimizer,
                head_lr=head_lr,
                backbone_lr=bb_lr,
            )

            train_stats = run_epoch(
                train_model,
                train_loader,
                class_weights=class_weights,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                total_epochs=EPOCHS,
                phase="train",
                ema=ema,
            )

            # Toutes les copies ont fini l'update avant la validation rank 0.
            barrier()

            if IS_MAIN:
                val_stats = run_epoch(
                    raw_model,
                    val_loader,
                    class_weights=class_weights,
                    optimizer=None,
                    scaler=scaler,
                    epoch=epoch,
                    total_epochs=EPOCHS,
                    phase="val",
                    ema=ema,
                )

                metrics = evaluate_metrics(
                    ema,
                    val_loader,
                    conf_thresh=0.03,
                    topk=100,
                )

                q = quality_score(metrics)

                histories["train_loss"].append(train_stats.get("loss", 0.0))
                histories["val_loss"].append(val_stats.get("loss", 0.0))
                histories["train_corner_px"].append(
                    train_stats.get("corner_px", 0.0)
                )
                histories["val_corner_px"].append(
                    val_stats.get("corner_px", 0.0)
                )
                histories["map50"].append(metrics["map50"])
                histories["map75"].append(metrics["map75"])
                histories["pck4"].append(metrics["pck4"])
                histories["match_recall"].append(metrics["match_recall"])
                histories["quality"].append(q)
                histories["lr_head"].append(head_lr)
                histories["lr_bb"].append(bb_lr)

                elapsed = time.time() - start_time
                done = epoch - start_epoch + 1
                eta = (
                    elapsed / max(done, 1) * max(EPOCHS - epoch, 0)
                )

                print(
                    f"[{epoch:03d}/{EPOCHS}] stage={stage} | "
                    f"lr_head={head_lr:.2e} lr_bb={bb_lr:.2e} | "
                    f"train={train_stats['loss']:.4f} "
                    f"corner={train_stats['corner_px']:.2f}px | "
                    f"val={val_stats['loss']:.4f} "
                    f"corner={val_stats['corner_px']:.2f}px | "
                    f"mAP50={metrics['map50']:.4f} "
                    f"mAP75={metrics['map75']:.4f} | "
                    f"PCK4={metrics['pck4']:.3f} "
                    f"match={metrics['match_recall']:.3f} | "
                    f"quality={q:.4f} | ETA={eta / 60:.1f} min"
                )

                print(f"  AP50/classes={metrics['ap50_per_class']}")
                print(
                    "  coins matchés : "
                    f"MAE={metrics['corner_mae_px']:.2f}px "
                    f"RMSE={metrics['corner_rmse_px']:.2f}px "
                    f"PCK2={metrics['pck2']:.3f} "
                    f"PCK4={metrics['pck4']:.3f} "
                    f"PCK8={metrics['pck8']:.3f}"
                )

                improved = q > best_quality + EARLY_STOP_MIN_DELTA

                if improved:
                    best_quality = q
                    best_epoch = epoch
                    best_metrics = dict(metrics)
                    best_metrics["quality"] = q
                    last_improvement_epoch = epoch

                    save_ckpt(
                        BEST_MODEL_PATH,
                        raw_model,
                        optimizer,
                        scaler,
                        ema,
                        epoch,
                        histories,
                        best_quality,
                        best_epoch,
                        best_metrics,
                        last_improvement_epoch,
                    )

                    print(
                        f"  ✓ Nouveau meilleur : quality={q:.4f} | "
                        f"mAP75={metrics['map75']:.4f} | "
                        f"corner={metrics['corner_mae_px']:.2f}px"
                    )

                save_ckpt(
                    LAST_CHECKPOINT_PATH,
                    raw_model,
                    optimizer,
                    scaler,
                    ema,
                    epoch,
                    histories,
                    best_quality,
                    best_epoch,
                    best_metrics,
                    last_improvement_epoch,
                )

                plot_curves(histories, best_epoch=best_epoch)

                if epoch % VIZ_EVERY == 0 or improved:
                    try:
                        visualize_predictions(
                            ema.ema,
                            val_p,
                            epoch,
                            n=2,
                            conf_thresh=0.15 if epoch <= 5 else 0.3,
                        )
                    except Exception as exc:
                        print(f"Visualisation ignorée : {exc}")

                early_stop_allowed = epoch >= UNFREEZE_ALL_EPOCH + 5
                stop_training = (
                    early_stop_allowed
                    and epoch - last_improvement_epoch >= EARLY_STOP_PATIENCE
                )

                if stop_training:
                    print(
                        f"Early stopping epoch {epoch} | "
                        f"best={best_epoch} quality={best_quality:.4f} | "
                        f"patience={EARLY_STOP_PATIENCE}"
                    )

                # Sauvegarde persistante toutes les 5 epochs.
                # En cas d'early stopping, on sauvegarde aussi immédiatement
                # même si l'epoch n'est pas un multiple de 5.
                if epoch % PERSIST_EVERY == 0 or stop_training:
                    persist_checkpoints_to_kaggle(
                        epoch=epoch,
                        best_epoch=best_epoch,
                        best_quality=best_quality,
                    )

            stop_training = broadcast_stop(stop_training)
            barrier()

            if stop_training:
                break

        barrier()

        if IS_MAIN:
            elapsed = time.time() - start_time
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            print(f"\nEntraînement terminé en {h:02d}h{m:02d}m{s:02d}s")

            best_path = (
                BEST_MODEL_PATH
                if BEST_MODEL_PATH.exists()
                else LAST_CHECKPOINT_PATH
            )

            if not best_path.exists():
                raise RuntimeError("Aucun checkpoint final disponible")

            checkpoint = torch.load(
                best_path,
                map_location=DEVICE,
                weights_only=False,
            )

            raw_model.load_state_dict(checkpoint["model"], strict=True)

            if checkpoint.get("ema") is not None:
                ema.ema.load_state_dict(checkpoint["ema"], strict=True)

            test_stats = run_epoch(
                raw_model,
                test_loader,
                class_weights=class_weights,
                optimizer=None,
                scaler=scaler,
                phase="test",
                ema=ema,
            )

            test_metrics = evaluate_metrics(
                ema,
                test_loader,
                conf_thresh=0.03,
                topk=100,
            )

            print(
                f"\nTEST | loss={test_stats['loss']:.4f} | "
                f"corner-loss-monitor={test_stats['corner_px']:.2f}px | "
                f"mAP50={test_metrics['map50']:.4f} "
                f"mAP75={test_metrics['map75']:.4f} | "
                f"corner MAE={test_metrics['corner_mae_px']:.2f}px | "
                f"PCK4={test_metrics['pck4']:.3f}"
            )
            print(f"AP50/classes={test_metrics['ap50_per_class']}")
            print(f"AP75/classes={test_metrics['ap75_per_class']}")
            print(f"Meilleur checkpoint : {best_path}")

        barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
