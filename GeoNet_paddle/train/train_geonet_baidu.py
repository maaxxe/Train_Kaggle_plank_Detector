# -*- coding: utf-8 -*-
"""
GeoNet PaddlePaddle - entrainement 1x GPU pour Baidu AI Studio.

Cible :
    PaddlePaddle 3.x
    Tesla V100 16 Go
    1 GPU

Par defaut, les hyperparametres reprennent le notebook GeoNet actuel :
    IMG_SIZE = 640
    EPOCHS = 120
    batch = 4
    accumulation = 4 sur 1 GPU
    -> batch effectif = 16, comme 2 GPU x 4 x accumulation 2.

Les checkpoints Paddle sont sauvegardes dans :
    last_multiforme.pdckpt
    best_multiforme.pdckpt
    history.json

Reprise :
    --resume-paddle /chemin/last_multiforme.pdckpt

Initialisation depuis un checkpoint PyTorch :
    1) exporter .pt -> .npz localement avec tools/export_torch_checkpoint_npz.py
    2) --init-torch-npz /chemin/checkpoint.npz

Attention : dans le cas NPZ, les poids et l'epoch/best_iou sont repris,
mais PAS l'etat AdamW PyTorch. L'optimizer repart donc neuf.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import paddle
from paddle.io import DataLoader, Dataset
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from model.multiforme_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    load_torch_npz_weights,
    multiforme_loss,
    segmentation_metrics,
)


# ============================================================
# Reproductibilite
# ============================================================

def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


# ============================================================
# Dataset
# ============================================================

def draw_gaussian_max(
    heatmap: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
) -> None:

    h, w = heatmap.shape

    radius = max(
        1,
        int(
            math.ceil(
                3.0 * sigma
            )
        ),
    )

    x0 = max(
        0,
        int(
            math.floor(cx)
        ) - radius,
    )

    x1 = min(
        w,
        int(
            math.ceil(cx)
        ) + radius + 1,
    )

    y0 = max(
        0,
        int(
            math.floor(cy)
        ) - radius,
    )

    y1 = min(
        h,
        int(
            math.ceil(cy)
        ) + radius + 1,
    )

    if (
        x0 >= x1
        or y0 >= y1
    ):
        return

    xs = np.arange(
        x0,
        x1,
        dtype=np.float32,
    )

    ys = np.arange(
        y0,
        y1,
        dtype=np.float32,
    )

    yy, xx = np.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    gaussian = np.exp(
        -(
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )
        / (
            2.0 * sigma ** 2
        )
    )

    region = heatmap[
        y0:y1,
        x0:x1,
    ]

    np.maximum(
        region,
        gaussian,
        out=region,
    )

    ix = int(
        round(cx)
    )

    iy = int(
        round(cy)
    )

    if (
        0 <= ix < w
        and 0 <= iy < h
    ):
        heatmap[
            iy,
            ix,
        ] = 1.0


class MultiFormeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        img_size: int = 512,
        indices: List[int] | None = None,
    ):
        super().__init__()

        self.root = Path(root)
        self.image_dir = (
            self.root
            / "images"
        )
        self.label_dir = (
            self.root
            / "labels"
        )

        self.img_size = int(
            img_size
        )

        self.center_size = (
            self.img_size // 2
        )

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier images introuvable : {self.image_dir}"
            )

        if not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier labels introuvable : {self.label_dir}"
            )

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        all_images = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower()
            in extensions
        )

        if not all_images:
            raise RuntimeError(
                f"Aucune image dans {self.image_dir}"
            )

        if indices is None:
            self.images = all_images
        else:
            self.images = [
                all_images[i]
                for i in indices
            ]

        self.mean = np.asarray(
            IMAGENET_MEAN,
            dtype=np.float32,
        ).reshape(
            1,
            1,
            3,
        )

        self.std = np.asarray(
            IMAGENET_STD,
            dtype=np.float32,
        ).reshape(
            1,
            1,
            3,
        )

    def __len__(
        self,
    ):
        return len(
            self.images
        )

    def _read_objects(
        self,
        label_path: Path,
    ):
        objects = []

        if not label_path.exists():
            return objects

        text = label_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            return objects

        for line_no, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            vals = (
                line.strip().split()
            )

            if not vals:
                continue

            if len(vals) < 7:
                raise ValueError(
                    f"Label invalide {label_path}:{line_no}: "
                    f"{len(vals)} valeurs."
                )

            cls = int(
                float(
                    vals[0]
                )
            )

            cx = float(
                vals[1]
            )

            cy = float(
                vals[2]
            )

            coords = [
                float(v)
                for v in vals[3:]
            ]

            if len(coords) % 2 != 0:
                raise ValueError(
                    "Nombre impair de coordonnees "
                    f"dans {label_path}:{line_no}"
                )

            points = np.asarray(
                coords,
                dtype=np.float32,
            ).reshape(
                -1,
                2,
            )

            center = np.asarray(
                [
                    cx,
                    cy,
                ],
                dtype=np.float32,
            )

            if points.shape[0] < 3:
                continue

            if (
                not np.isfinite(
                    points
                ).all()
                or not np.isfinite(
                    center
                ).all()
            ):
                raise ValueError(
                    f"NaN/Inf dans {label_path}:{line_no}"
                )

            objects.append(
                {
                    "cls": cls,
                    "center": np.clip(
                        center,
                        0.0,
                        1.0,
                    ),
                    "points": np.clip(
                        points,
                        0.0,
                        1.0,
                    ),
                }
            )

        return objects

    def __getitem__(
        self,
        index: int,
    ):
        image_path = (
            self.images[index]
        )

        label_path = (
            self.label_dir
            / f"{image_path.stem}.txt"
        )

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Impossible de lire {image_path}"
            )

        image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (
                self.img_size,
                self.img_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        objects = self._read_objects(
            label_path
        )

        mask = np.zeros(
            (
                self.img_size,
                self.img_size,
            ),
            dtype=np.uint8,
        )

        center_heatmap = np.zeros(
            (
                self.center_size,
                self.center_size,
            ),
            dtype=np.float32,
        )

        for obj in objects:
            pts_n = obj[
                "points"
            ]

            pts_px = pts_n.copy()

            pts_px[:, 0] *= (
                self.img_size - 1
            )

            pts_px[:, 1] *= (
                self.img_size - 1
            )

            poly = np.round(
                pts_px
            ).astype(
                np.int32
            )

            cv2.fillPoly(
                mask,
                [poly],
                255,
            )

            cx_n, cy_n = obj[
                "center"
            ]

            cx = float(
                cx_n
            ) * (
                self.center_size - 1
            )

            cy = float(
                cy_n
            ) * (
                self.center_size - 1
            )

            min_xy = pts_n.min(
                axis=0
            )

            max_xy = pts_n.max(
                axis=0
            )

            box_w = float(
                max_xy[0]
                - min_xy[0]
            ) * self.center_size

            box_h = float(
                max_xy[1]
                - min_xy[1]
            ) * self.center_size

            sigma = np.clip(
                min(
                    box_w,
                    box_h,
                ) / 6.0,
                1.5,
                8.0,
            )

            draw_gaussian_max(
                center_heatmap,
                cx,
                cy,
                float(sigma),
            )

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        image = (
            image
            - self.mean
        ) / self.std

        image = np.transpose(
            image,
            (
                2,
                0,
                1,
            ),
        ).copy()

        mask = (
            mask.astype(
                np.float32
            )
            / 255.0
        )[
            None,
            ...
        ]

        center_heatmap = (
            center_heatmap[
                None,
                ...
            ]
        )

        # Tuple simple = collate Paddle robuste.
        return (
            image,
            mask,
            center_heatmap,
        )


def count_images(
    root: Path,
) -> int:
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return len(
        [
            p
            for p in (
                root
                / "images"
            ).iterdir()
            if p.suffix.lower()
            in extensions
        ]
    )


def make_split(
    n: int,
    val_ratio: float,
    seed: int,
) -> Tuple[
    List[int],
    List[int],
]:

    if n < 2:
        raise RuntimeError(
            "Il faut au moins 2 images pour train/val."
        )

    indices = list(
        range(n)
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        indices
    )

    n_val = max(
        1,
        int(
            round(
                n * val_ratio
            )
        ),
    )

    n_val = min(
        n_val,
        n - 1,
    )

    return (
        indices[n_val:],
        indices[:n_val],
    )


# ============================================================
# Epoch
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    amp: bool,
    accumulation_steps: int,
    grad_clip: float,
    train: bool,
):
    if train:
        model.train()
        optimizer.clear_grad()
    else:
        model.eval()

    sums = {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "boundary": 0.0,
        "center": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "n_samples": 0.0,
    }

    iterator = tqdm(
        loader,
        leave=False,
        desc=(
            "TRAIN"
            if train
            else "VAL"
        ),
    )

    n_batches = len(
        loader
    )

    context = (
        paddle.enable_grad
        if train
        else paddle.no_grad
    )

    with context():
        for step, batch in enumerate(
            iterator
        ):
            images, masks, centers = (
                batch
            )

            # DataLoader fournit deja des Tensor,
            # mais to_tensor reste inutile ici.
            images = images.astype(
                "float32"
            )

            masks = masks.astype(
                "float32"
            )

            centers = centers.astype(
                "float32"
            )

            bs = int(
                images.shape[0]
            )

            should_step = (
                (
                    step + 1
                ) % accumulation_steps
                == 0
                or (
                    step + 1
                ) == n_batches
            )

            with paddle.amp.auto_cast(
                enable=amp,
                dtype="float16",
            ):
                outputs = model(
                    images
                )

            # Loss volontairement en FP32 :
            # meme protection que dans la version PyTorch.
            loss_outputs = {
                "mask_logits": outputs[
                    "mask_logits"
                ].astype(
                    "float32"
                ),
                "center_logits": outputs[
                    "center_logits"
                ].astype(
                    "float32"
                ),
            }

            raw_loss, parts = multiforme_loss(
                loss_outputs,
                masks,
                centers,
            )

            if not bool(
                paddle.isfinite(
                    raw_loss
                ).item()
            ):
                raise FloatingPointError(
                    "Loss non finie detectee : "
                    f"{float(raw_loss.item())}"
                )

            if train:
                loss_for_backward = (
                    raw_loss
                    / accumulation_steps
                )

                if amp:
                    scaled_loss = scaler.scale(
                        loss_for_backward
                    )
                    scaled_loss.backward()
                else:
                    loss_for_backward.backward()

                if should_step:
                    if amp:
                        # Important : on unscale avant le clipping,
                        # sinon le seuil de gradient serait appliqué
                        # aux gradients multipliés par le loss scale.
                        scaler.unscale_(
                            optimizer
                        )

                        paddle.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=grad_clip,
                        )

                        scaler.step(
                            optimizer
                        )
                        scaler.update()
                    else:
                        paddle.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=grad_clip,
                        )
                        optimizer.step()

                    optimizer.clear_grad()

            metrics = segmentation_metrics(
                outputs[
                    "mask_logits"
                ],
                masks,
            )

            sums[
                "loss"
            ] += (
                float(
                    raw_loss.item()
                )
                * bs
            )

            for key in [
                "bce",
                "dice_loss",
                "boundary",
                "center",
            ]:
                sums[
                    key
                ] += (
                    parts[
                        key
                    ]
                    * bs
                )

            sums[
                "iou"
            ] += (
                metrics[
                    "iou"
                ]
                * bs
            )

            sums[
                "dice"
            ] += (
                metrics[
                    "dice"
                ]
                * bs
            )

            sums[
                "n_samples"
            ] += bs

            n = max(
                1.0,
                sums[
                    "n_samples"
                ],
            )

            iterator.set_postfix(
                loss=(
                    f"{sums['loss'] / n:.4f}"
                ),
                iou=(
                    f"{sums['iou'] / n:.3f}"
                ),
                dice=(
                    f"{sums['dice'] / n:.3f}"
                ),
            )

    n = max(
        1.0,
        sums.pop(
            "n_samples"
        ),
    )

    return {
        key: value / n
        for key, value
        in sums.items()
    }


# ============================================================
# Checkpoints Paddle
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_iou: float,
    history: List[Dict],
    args,
) -> None:

    checkpoint = {
        "epoch": int(
            epoch
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_iou": float(
            best_iou
        ),
        "history": history,
        "config": vars(
            args
        ),
        "framework": "paddle",
    }

    tmp_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    paddle.save(
        checkpoint,
        str(tmp_path),
    )

    os.replace(
        tmp_path,
        path,
    )


def load_paddle_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
):
    checkpoint = paddle.load(
        str(path)
    )

    model.set_state_dict(
        checkpoint[
            "model"
        ]
    )

    if (
        "optimizer"
        in checkpoint
    ):
        optimizer.set_state_dict(
            checkpoint[
                "optimizer"
            ]
        )

    if (
        "scheduler"
        in checkpoint
    ):
        scheduler.set_state_dict(
            checkpoint[
                "scheduler"
            ]
        )

    epoch = int(
        checkpoint.get(
            "epoch",
            0,
        )
    )

    best_iou = float(
        checkpoint.get(
            "best_iou",
            -1.0,
        )
    )

    history = list(
        checkpoint.get(
            "history",
            [],
        )
    )

    return (
        epoch + 1,
        best_iou,
        history,
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="/home/aistudio/work/GeoNet_paddle_checkpoints",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch sur l'unique V100.",
    )

    parser.add_argument(
        "--accumulation",
        type=int,
        default=4,
        help=(
            "4 donne un batch effectif 16 avec batch=4, "
            "equivalent au notebook 2xT4: 4*2*2."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=5e-7,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--resume-paddle",
        type=str,
        default="",
    )

    parser.add_argument(
        "--init-torch-npz",
        type=str,
        default="",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if (
        args.resume_paddle
        and args.init_torch_npz
    ):
        raise ValueError(
            "Utilise soit --resume-paddle, soit --init-torch-npz, pas les deux."
        )

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError(
            "Cette installation Paddle n'est pas compilee avec CUDA."
        )

    if paddle.device.cuda.device_count() < 1:
        raise RuntimeError(
            "Aucun GPU detecte par Paddle."
        )

    paddle.set_device(
        "gpu:0"
    )

    seed_everything(
        args.seed
    )

    data_dir = Path(
        args.data
    ).resolve()

    output_dir = Path(
        args.output
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_total = count_images(
        data_dir
    )

    train_idx, val_idx = make_split(
        n_total,
        args.val_ratio,
        args.seed,
    )

    train_set = MultiFormeDataset(
        data_dir,
        img_size=args.img_size,
        indices=train_idx,
    )

    val_set = MultiFormeDataset(
        data_dir,
        img_size=args.img_size,
        indices=val_idx,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        return_list=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        return_list=True,
    )

    model = build_model(
        pretrained=False
    )

    # --------------------------------------------------------
    # Param groups :
    # base LR = lr_head
    # backbone utilise un multiplicateur fixe.
    # --------------------------------------------------------

    head_params = [
        p
        for name, p
        in model.named_parameters()
        if not name.startswith(
            "encoder."
        )
    ]

    backbone_params = list(
        model.encoder.parameters()
    )

    if args.lr_head <= 0:
        raise ValueError(
            "--lr-head doit etre > 0"
        )

    backbone_lr_scale = (
        args.lr_backbone
        / args.lr_head
    )

    min_lr_ratio = 0.03

    scheduler = (
        paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=args.lr_head,
            T_max=max(
                1,
                args.epochs,
            ),
            eta_min=(
                args.lr_head
                * min_lr_ratio
            ),
        )
    )

    optimizer = paddle.optimizer.AdamW(
        learning_rate=scheduler,
        parameters=[
            {
                "params": head_params,
                "learning_rate": 1.0,
            },
            {
                "params": backbone_params,
                "learning_rate": backbone_lr_scale,
            },
        ],
        weight_decay=args.weight_decay,
    )

    amp = not args.no_amp

    scaler = paddle.amp.GradScaler(
        enable=amp,
        init_loss_scaling=1024.0,
        use_dynamic_loss_scaling=True,
    )

    start_epoch = 1
    best_iou = -1.0
    history = []

    # --------------------------------------------------------
    # Reprise Paddle complete
    # --------------------------------------------------------

    if args.resume_paddle:
        resume_path = Path(
            args.resume_paddle
        ).resolve()

        if not resume_path.is_file():
            raise FileNotFoundError(
                resume_path
            )

        (
            start_epoch,
            best_iou,
            history,
        ) = load_paddle_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
        )

        print(
            f"Reprise Paddle : {resume_path}"
        )

        print(
            f"Prochaine epoch: {start_epoch}"
        )

        print(
            f"Best IoU      : {best_iou:.6f}"
        )

    # --------------------------------------------------------
    # Initialisation depuis poids PyTorch exportes
    # --------------------------------------------------------

    elif args.init_torch_npz:
        npz_path = Path(
            args.init_torch_npz
        ).resolve()

        meta_path = npz_path.with_suffix(
            ".meta.json"
        )

        init_meta = {}

        if meta_path.is_file():
            init_meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

        # Un export d'encodeur ImageNet est volontairement partiel.
        # Un checkpoint GeoNet complet, lui, doit matcher strictement.
        init_kind = init_meta.get(
            "kind",
            "geonet_checkpoint",
        )

        strict_init = (
            init_kind
            != "imagenet_backbone"
        )

        result = load_torch_npz_weights(
            model,
            npz_path,
            strict=strict_init,
        )

        meta = result.get(
            "metadata",
            init_meta,
        )

        if meta.get(
            "epoch"
        ) is not None:
            start_epoch = (
                int(
                    meta[
                        "epoch"
                    ]
                )
                + 1
            )

        if meta.get(
            "best_iou"
        ) is not None:
            best_iou = float(
                meta[
                    "best_iou"
                ]
            )

        if isinstance(
            meta.get(
                "history"
            ),
            list,
        ):
            history = meta[
                "history"
            ]

        # Si on importe un checkpoint PyTorch déjà avancé,
        # placer le scheduler Paddle sur l'epoch absolue correspondante.
        # Pour un simple backbone ImageNet, on démarre à l'epoch 1.
        if (
            init_kind
            == "geonet_checkpoint"
            and start_epoch > 1
        ):
            scheduler.step(
                start_epoch - 1
            )

        print(
            f"Poids PyTorch importes : {result['loaded']} tensors"
        )

        if result[
            "missing"
        ]:
            print(
                "Poids Paddle non initialises depuis le NPZ :",
                len(
                    result[
                        "missing"
                    ]
                ),
                "(normal pour un export ImageNet-backbone uniquement)",
            )

        print(
            "ATTENTION : optimizer PyTorch non converti ; "
            "AdamW Paddle repart neuf."
        )

        print(
            f"Prochaine epoch : {start_epoch}"
        )

        print(
            f"Best IoU       : {best_iou:.6f}"
        )

    n_params = sum(
        int(
            np.prod(
                p.shape
            )
        )
        for p
        in model.parameters()
    )

    print(
        "=" * 78
    )

    print(
        "GEONET - PADDLE / BAIDU AI STUDIO"
    )

    print(
        "=" * 78
    )

    print(
        "Paddle              :",
        paddle.__version__,
    )

    print(
        "Device              :",
        paddle.device.get_device(),
    )

    print(
        "GPU count           :",
        paddle.device.cuda.device_count(),
    )

    print(
        "Images              :",
        n_total,
    )

    print(
        "Train / Val         :",
        len(train_set),
        "/",
        len(val_set),
    )

    print(
        "Image size          :",
        args.img_size,
    )

    print(
        "Batch GPU           :",
        args.batch,
    )

    print(
        "Accumulation        :",
        args.accumulation,
    )

    print(
        "Batch effectif      :",
        args.batch
        * args.accumulation,
    )

    print(
        "AMP FP16            :",
        amp,
    )

    print(
        "LR head             :",
        args.lr_head,
    )

    print(
        "LR backbone         :",
        args.lr_backbone,
    )

    print(
        "Parametres          :",
        f"{n_params:,}",
    )

    print(
        "Output              :",
        output_dir,
    )

    print(
        "=" * 78
    )

    epochs_without_improvement = 0
    training_start = time.time()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        # Freeze selon numero d'epoch absolu, comme le training d'origine.
        encoder_frozen = (
            epoch
            <= args.freeze_backbone_epochs
        )

        if encoder_frozen:
            model.freeze_encoder()
        else:
            model.unfreeze_encoder()

        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            accumulation_steps=args.accumulation,
            grad_clip=args.grad_clip,
            train=True,
        )

        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            accumulation_steps=1,
            grad_clip=args.grad_clip,
            train=False,
        )

        current_head_lr = float(
            scheduler.get_lr()
        )

        current_backbone_lr = (
            0.0
            if encoder_frozen
            else current_head_lr
            * backbone_lr_scale
        )

        improved = (
            val_stats[
                "iou"
            ]
            > best_iou
        )

        if improved:
            best_iou = val_stats[
                "iou"
            ]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        elapsed_min = (
            time.time()
            - training_start
        ) / 60.0

        record = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "lr_head": current_head_lr,
            "lr_backbone": current_backbone_lr,
        }

        history.append(
            record
        )

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train={train_stats['loss']:.4f} "
            f"IoU={train_stats['iou']:.4f} "
            f"| val={val_stats['loss']:.4f} "
            f"IoU={val_stats['iou']:.4f} "
            f"Dice={val_stats['dice']:.4f} "
            f"Center={val_stats['center']:.4f} "
            f"| LR={current_head_lr:.2e}/"
            f"{current_backbone_lr:.2e} "
            f"| {elapsed_min:.1f} min"
        )

        last_path = (
            output_dir
            / "last_multiforme.pdckpt"
        )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_iou,
            history,
            args,
        )

        print(
            f"  -> LAST sauvegarde : {last_path}"
        )

        if improved:
            best_path = (
                output_dir
                / "best_multiforme.pdckpt"
            )

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_iou,
                history,
                args,
            )

            print(
                "  -> BEST : "
                f"IoU={best_iou:.4f}"
            )

        (
            output_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Scheduler apres l'epoch : nouveau LR pour l'epoch suivante.
        scheduler.step()

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "Early stopping : "
                f"{args.patience} epochs "
                "sans amelioration de l'IoU."
            )
            break

    print(
        "=" * 78
    )

    print(
        "Termine. Best IoU =",
        f"{best_iou:.6f}",
    )

    print(
        "LAST :",
        output_dir
        / "last_multiforme.pdckpt",
    )

    print(
        "BEST :",
        output_dir
        / "best_multiforme.pdckpt",
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
