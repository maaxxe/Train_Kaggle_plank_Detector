# -*- coding: utf-8 -*-
"""
GeoNet / MultiFormeNet - version native PaddlePaddle 3.x.

Port de l'architecture PyTorch d'origine :
    image
      -> EfficientNetV2-S (features 0..6)
      -> decodeur U-Net multi-echelle
      -> masque binaire pleine resolution
      -> heatmap de centres a stride 2

Le post-traitement masque -> instances -> N points reste identique
et repose sur NumPy/OpenCV.

IMPORTANT
---------
- Cette version n'importe PAS torch.
- L'encodeur EfficientNetV2-S est implemente ici pour conserver la meme
  architecture que torchvision EfficientNetV2-S sur les niveaux utilises
  par GeoNet.
- Les poids PyTorch .pt ne sont pas lisibles directement par Paddle.
  Utiliser tools/export_torch_checkpoint_npz.py sur une machine PyTorch,
  puis load_torch_npz_weights() sur Baidu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Utilitaires EfficientNetV2-S
# ============================================================

def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class ConvBNAct(nn.Sequential):
    """Equivalent de torchvision.ops.Conv2dNormActivation pour ce backbone."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2D(
                in_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=False,
            ),
            # PyTorch BatchNorm momentum=0.1 correspond a Paddle momentum=0.9.
            nn.BatchNorm2D(
                out_c,
                momentum=0.9,
                epsilon=1e-5,
            ),
        ]

        if activation:
            layers.append(nn.Silu())

        super().__init__(*layers)


class SqueezeExcitation(nn.Layer):
    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2D(squeeze_channels, input_channels, 1)
        self.activation = nn.Silu()
        self.scale_activation = nn.Sigmoid()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * x


class StochasticDepth(nn.Layer):
    """Stochastic depth mode='row', equivalent a torchvision."""

    def __init__(self, p: float):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p doit etre dans [0, 1]")
        self.p = float(p)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if not self.training or self.p == 0.0:
            return x

        survival_rate = 1.0 - self.p
        if survival_rate <= 0.0:
            return paddle.zeros_like(x)

        shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
        noise = paddle.rand(shape, dtype=x.dtype)
        noise = paddle.floor(noise + survival_rate)
        return x * noise / survival_rate


@dataclass
class MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int


@dataclass
class FusedMBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int


class MBConv(nn.Layer):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
    ):
        super().__init__()

        if cnf.stride not in (1, 2):
            raise ValueError("stride EfficientNet invalide")

        self.use_res_connect = (
            cnf.stride == 1
            and cnf.input_channels == cnf.out_channels
        )

        expanded_channels = _make_divisible(
            cnf.input_channels * cnf.expand_ratio,
            8,
        )

        layers = []

        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvBNAct(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                )
            )

        layers.append(
            ConvBNAct(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
            )
        )

        squeeze_channels = max(1, cnf.input_channels // 4)

        layers.append(
            SqueezeExcitation(
                expanded_channels,
                squeeze_channels,
            )
        )

        layers.append(
            ConvBNAct(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                activation=False,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(
            stochastic_depth_prob
        )
        self.out_channels = cnf.out_channels

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        y = self.block(x)

        if self.use_res_connect:
            y = self.stochastic_depth(y)
            y = y + x

        return y


class FusedMBConv(nn.Layer):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
    ):
        super().__init__()

        if cnf.stride not in (1, 2):
            raise ValueError("stride EfficientNet invalide")

        self.use_res_connect = (
            cnf.stride == 1
            and cnf.input_channels == cnf.out_channels
        )

        expanded_channels = _make_divisible(
            cnf.input_channels * cnf.expand_ratio,
            8,
        )

        layers = []

        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvBNAct(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                )
            )
            layers.append(
                ConvBNAct(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    activation=False,
                )
            )
        else:
            layers.append(
                ConvBNAct(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(
            stochastic_depth_prob
        )
        self.out_channels = cnf.out_channels

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        y = self.block(x)

        if self.use_res_connect:
            y = self.stochastic_depth(y)
            y = y + x

        return y


def _build_stage(
    cnf,
    stage_block_start: int,
    total_stage_blocks: int,
    stochastic_depth_prob: float = 0.2,
) -> nn.Sequential:
    blocks = []

    for layer_id in range(cnf.num_layers):
        if layer_id == 0:
            block_cnf = type(cnf)(
                cnf.expand_ratio,
                cnf.kernel,
                cnf.stride,
                cnf.input_channels,
                cnf.out_channels,
                cnf.num_layers,
            )
        else:
            block_cnf = type(cnf)(
                cnf.expand_ratio,
                cnf.kernel,
                1,
                cnf.out_channels,
                cnf.out_channels,
                cnf.num_layers,
            )

        block_id = stage_block_start + layer_id
        sd_prob = (
            stochastic_depth_prob
            * float(block_id)
            / float(total_stage_blocks)
        )

        if isinstance(block_cnf, FusedMBConvConfig):
            blocks.append(FusedMBConv(block_cnf, sd_prob))
        else:
            blocks.append(MBConv(block_cnf, sd_prob))

    return nn.Sequential(*blocks)


class EfficientNetV2SFeatures(nn.Layer):
    """
    Tronc EfficientNetV2-S compatible avec les features[0]..features[6]
    de torchvision. La feature[7] (conv 1280) n'est pas utilisee par GeoNet.
    """

    def __init__(self):
        super().__init__()

        configs = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]

        total_blocks = sum(c.num_layers for c in configs)

        features = [
            ConvBNAct(
                3,
                24,
                kernel_size=3,
                stride=2,
            )
        ]

        stage_start = 0
        for cnf in configs:
            features.append(
                _build_stage(
                    cnf,
                    stage_block_start=stage_start,
                    total_stage_blocks=total_blocks,
                    stochastic_depth_prob=0.2,
                )
            )
            stage_start += cnf.num_layers

        self.features = nn.Sequential(*features)

        self._init_weights()

    def _init_weights(self) -> None:
        conv_init = nn.initializer.KaimingNormal()
        one = nn.initializer.Constant(1.0)
        zero = nn.initializer.Constant(0.0)

        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                conv_init(layer.weight)
                if layer.bias is not None:
                    zero(layer.bias)
            elif isinstance(layer, nn.BatchNorm2D):
                if layer.weight is not None:
                    one(layer.weight)
                if layer.bias is not None:
                    zero(layer.bias)


# ============================================================
# Blocs GeoNet
# ============================================================

def _group_norm(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(
        num_groups=groups,
        num_channels=channels,
    )


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
    ):
        padding = kernel_size // 2

        super().__init__(
            nn.Conv2D(
                in_c,
                out_c,
                kernel_size,
                padding=padding,
                bias_attr=False,
            ),
            _group_norm(out_c),
            nn.Silu(),
        )


class ResidualConvBlock(nn.Layer):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        self.conv1 = ConvGNAct(in_c, out_c, 3)

        self.conv2 = nn.Sequential(
            nn.Conv2D(
                out_c,
                out_c,
                3,
                padding=1,
                bias_attr=False,
            ),
            _group_norm(out_c),
        )

        self.skip = (
            nn.Identity()
            if in_c == out_c
            else nn.Conv2D(
                in_c,
                out_c,
                1,
                bias_attr=False,
            )
        )

        self.act = nn.Silu()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(y + self.skip(x))


class UpBlock(nn.Layer):
    def __init__(
        self,
        in_c: int,
        skip_c: int,
        out_c: int,
    ):
        super().__init__()

        self.reduce = ConvGNAct(in_c, out_c, 1)
        self.fuse = ResidualConvBlock(
            out_c + skip_c,
            out_c,
        )

    def forward(
        self,
        x: paddle.Tensor,
        skip: paddle.Tensor,
    ) -> paddle.Tensor:

        x = F.interpolate(
            x,
            size=list(skip.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )

        x = self.reduce(x)
        x = paddle.concat([x, skip], axis=1)

        return self.fuse(x)


class EfficientNetV2SEncoder(nn.Layer):
    """
    Niveaux conserves exactement comme dans GeoNet PyTorch :
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

    def __init__(self):
        super().__init__()
        self.features = EfficientNetV2SFeatures().features

    def forward(
        self,
        x: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:

        x = self.features[0](x)
        x = self.features[1](x)
        s2 = x

        x = self.features[2](x)
        s4 = x

        x = self.features[3](x)
        s8 = x

        x = self.features[4](x)
        x = self.features[5](x)
        s16 = x

        x = self.features[6](x)
        s32 = x

        return {
            "s2": s2,
            "s4": s4,
            "s8": s8,
            "s16": s16,
            "s32": s32,
        }


class MultiFormeNet(nn.Layer):
    """
    Sorties :
        mask_logits   : (B, 1, H, W)
        center_logits : (B, 1, H/2, W/2)
    """

    def __init__(self):
        super().__init__()

        self.encoder = EfficientNetV2SEncoder()

        self.bottleneck = ResidualConvBlock(
            256,
            256,
        )

        self.up16 = UpBlock(
            256,
            160,
            192,
        )

        self.up8 = UpBlock(
            192,
            64,
            128,
        )

        self.up4 = UpBlock(
            128,
            48,
            96,
        )

        self.up2 = UpBlock(
            96,
            24,
            64,
        )

        self.center_head = nn.Sequential(
            ResidualConvBlock(64, 64),
            nn.Conv2D(64, 1, 1),
        )

        # Identique au biais initial PyTorch.
        nn.initializer.Constant(-2.19)(
            self.center_head[-1].bias
        )

        self.mask_refine = nn.Sequential(
            ConvGNAct(64 + 3, 48, 3),
            ResidualConvBlock(48, 32),
            nn.Conv2D(32, 1, 1),
        )

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.stop_gradient = True

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.stop_gradient = False

    def forward(
        self,
        x: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:

        input_size = list(x.shape[-2:])
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

        mask_logits = self.mask_refine(
            paddle.concat(
                [y_full, x],
                axis=1,
            )
        )

        return {
            "mask_logits": mask_logits,
            "center_logits": center_logits,
        }


# ============================================================
# Losses
# ============================================================

def dice_loss(
    logits: paddle.Tensor,
    target: paddle.Tensor,
    eps: float = 1e-6,
) -> paddle.Tensor:

    prob = F.sigmoid(logits)
    axes = [1, 2, 3]

    inter = paddle.sum(
        prob * target,
        axis=axes,
    )

    denom = (
        paddle.sum(prob, axis=axes)
        + paddle.sum(target, axis=axes)
    )

    dice = (
        2.0 * inter + eps
    ) / (
        denom + eps
    )

    return paddle.mean(
        1.0 - dice
    )


def _soft_boundary(
    x: paddle.Tensor,
    kernel_size: int = 3,
) -> paddle.Tensor:

    pad = kernel_size // 2

    max_v = F.max_pool2d(
        x,
        kernel_size,
        stride=1,
        padding=pad,
    )

    min_v = -F.max_pool2d(
        -x,
        kernel_size,
        stride=1,
        padding=pad,
    )

    return paddle.clip(
        max_v - min_v,
        min=0.0,
        max=1.0,
    )


def boundary_dice_loss(
    logits: paddle.Tensor,
    target: paddle.Tensor,
    eps: float = 1e-6,
) -> paddle.Tensor:

    pred_boundary = _soft_boundary(
        F.sigmoid(logits)
    )

    gt_boundary = _soft_boundary(
        target
    )

    axes = [1, 2, 3]

    inter = paddle.sum(
        pred_boundary * gt_boundary,
        axis=axes,
    )

    denom = (
        paddle.sum(
            pred_boundary,
            axis=axes,
        )
        + paddle.sum(
            gt_boundary,
            axis=axes,
        )
    )

    score = (
        2.0 * inter + eps
    ) / (
        denom + eps
    )

    return paddle.mean(
        1.0 - score
    )


def center_focal_loss(
    logits: paddle.Tensor,
    target: paddle.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> paddle.Tensor:

    pred = paddle.clip(
        F.sigmoid(logits),
        min=1e-6,
        max=1.0 - 1e-6,
    )

    pos_mask = (
        target >= 0.999
    ).astype("float32")

    neg_mask = (
        1.0 - pos_mask
    )

    neg_weights = paddle.pow(
        1.0 - target,
        beta,
    )

    pos_loss = (
        -paddle.log(pred)
        * paddle.pow(
            1.0 - pred,
            alpha,
        )
        * pos_mask
    )

    neg_loss = (
        -paddle.log(
            1.0 - pred
        )
        * paddle.pow(
            pred,
            alpha,
        )
        * neg_weights
        * neg_mask
    )

    n_pos = paddle.sum(
        pos_mask
    )

    # Dynamic graph : item() est volontaire ici pour reproduire
    # exactement la branche du code PyTorch.
    if float(n_pos.item()) > 0.0:
        return (
            paddle.sum(pos_loss)
            + paddle.sum(neg_loss)
        ) / n_pos

    return paddle.sum(
        neg_loss
    ) / max(
        1,
        int(logits.shape[0]),
    )


def multiforme_loss(
    outputs: Dict[str, paddle.Tensor],
    mask_target: paddle.Tensor,
    center_target: paddle.Tensor,
    w_bce: float = 1.0,
    w_dice: float = 1.0,
    w_boundary: float = 0.35,
    w_center: float = 0.50,
) -> Tuple[paddle.Tensor, Dict[str, float]]:

    mask_logits = outputs[
        "mask_logits"
    ]

    center_logits = outputs[
        "center_logits"
    ]

    bce = F.binary_cross_entropy_with_logits(
        mask_logits,
        mask_target,
        reduction="mean",
    )

    dsc = dice_loss(
        mask_logits,
        mask_target,
    )

    boundary = boundary_dice_loss(
        mask_logits,
        mask_target,
    )

    center = center_focal_loss(
        center_logits,
        center_target,
    )

    total = (
        w_bce * bce
        + w_dice * dsc
        + w_boundary * boundary
        + w_center * center
    )

    return total, {
        "bce": float(bce.item()),
        "dice_loss": float(dsc.item()),
        "boundary": float(boundary.item()),
        "center": float(center.item()),
        "total": float(total.item()),
    }


# ============================================================
# Metriques segmentation
# ============================================================

@paddle.no_grad()
def segmentation_metrics(
    mask_logits: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:

    pred = (
        F.sigmoid(mask_logits)
        >= threshold
    ).astype("float32")

    axes = [1, 2, 3]

    inter = paddle.sum(
        pred * target,
        axis=axes,
    )

    union = paddle.sum(
        (
            (
                pred + target
            ) > 0
        ).astype("float32"),
        axis=axes,
    )

    pred_sum = paddle.sum(
        pred,
        axis=axes,
    )

    gt_sum = paddle.sum(
        target,
        axis=axes,
    )

    iou = (
        inter + eps
    ) / (
        union + eps
    )

    dice = (
        2.0 * inter + eps
    ) / (
        pred_sum
        + gt_sum
        + eps
    )

    return {
        "iou": float(
            paddle.mean(iou).item()
        ),
        "dice": float(
            paddle.mean(dice).item()
        ),
    }


# ============================================================
# Import de poids PyTorch exportes en NPZ
# ============================================================

def _torch_candidates_for_paddle_key(
    paddle_key: str,
) -> List[str]:

    candidates = [
        paddle_key,
    ]

    if "._mean" in paddle_key:
        candidates.append(
            paddle_key.replace(
                "._mean",
                ".running_mean",
            )
        )

    if "._variance" in paddle_key:
        candidates.append(
            paddle_key.replace(
                "._variance",
                ".running_var",
            )
        )

    # Selon la version de Paddle, certains noms peuvent etre exposes
    # sans underscore. On garde des variantes defensives.
    if ".running_mean" in paddle_key:
        candidates.append(
            paddle_key.replace(
                ".running_mean",
                "._mean",
            )
        )

    if ".running_var" in paddle_key:
        candidates.append(
            paddle_key.replace(
                ".running_var",
                "._variance",
            )
        )

    return candidates


def load_torch_npz_weights(
    model: nn.Layer,
    npz_path: str | Path,
    strict: bool = True,
) -> Dict[str, object]:
    """
    Charge dans Paddle des poids exportes depuis un state_dict PyTorch.

    Le fichier NPZ est produit par :
        tools/export_torch_checkpoint_npz.py

    Les compteurs BatchNorm num_batches_tracked sont ignores.
    """

    npz_path = Path(npz_path)

    if not npz_path.is_file():
        raise FileNotFoundError(
            npz_path
        )

    archive = np.load(
        npz_path,
        allow_pickle=False,
    )

    destination = model.state_dict()

    loaded = {}
    missing = []
    shape_errors = []

    for dst_key, dst_tensor in destination.items():
        src_key = None

        for candidate in _torch_candidates_for_paddle_key(
            dst_key
        ):
            if candidate in archive.files:
                src_key = candidate
                break

        if src_key is None:
            missing.append(dst_key)
            continue

        arr = np.asarray(
            archive[src_key]
        )

        expected_shape = tuple(
            int(v)
            for v in dst_tensor.shape
        )

        if tuple(arr.shape) != expected_shape:
            # Utile si un Linear est ajoute plus tard :
            # PyTorch [out, in], Paddle [in, out].
            if (
                arr.ndim == 2
                and tuple(arr.T.shape) == expected_shape
            ):
                arr = arr.T
            else:
                shape_errors.append(
                    (
                        dst_key,
                        tuple(arr.shape),
                        expected_shape,
                    )
                )
                continue

        loaded[dst_key] = paddle.to_tensor(
            arr,
            dtype=dst_tensor.dtype,
        )

    if shape_errors:
        details = "\n".join(
            f"  {k}: torch={a} paddle={b}"
            for k, a, b in shape_errors[:20]
        )

        raise RuntimeError(
            "Dimensions incompatibles pendant l'import PyTorch -> Paddle:\n"
            + details
        )

    if strict and missing:
        raise RuntimeError(
            "Poids absents pendant l'import PyTorch -> Paddle "
            f"({len(missing)}):\n  "
            + "\n  ".join(
                missing[:30]
            )
        )

    # Conserve les parametres Paddle non trouves si strict=False.
    merged = dict(destination)
    merged.update(loaded)
    model.set_state_dict(merged)

    meta_path = npz_path.with_suffix(
        ".meta.json"
    )

    metadata = {}

    if meta_path.is_file():
        import json

        metadata = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

    return {
        "loaded": len(loaded),
        "missing": missing,
        "metadata": metadata,
    }


# ============================================================
# Post-traitement : mask -> instances -> N points
# ============================================================

def resample_contour(
    contour: np.ndarray,
    n_points: int = 32,
) -> Optional[np.ndarray]:

    contour = contour.reshape(
        -1,
        2,
    ).astype(
        np.float32
    )

    if len(contour) < 2:
        return None

    closed = np.vstack(
        [
            contour,
            contour[0],
        ]
    )

    segments = np.diff(
        closed,
        axis=0,
    )

    lengths = np.sqrt(
        (
            segments ** 2
        ).sum(
            axis=1
        )
    )

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(lengths),
        ]
    )

    total = float(
        cumulative[-1]
    )

    if total < 1.0:
        return None

    targets = np.linspace(
        0.0,
        total,
        n_points,
        endpoint=False,
    )

    output = []

    for t in targets:
        idx = (
            np.searchsorted(
                cumulative,
                t,
                side="right",
            )
            - 1
        )

        idx = min(
            max(
                0,
                idx,
            ),
            len(segments) - 1,
        )

        seg_len = lengths[
            idx
        ]

        if seg_len < 1e-6:
            p = closed[idx]
        else:
            alpha = (
                t
                - cumulative[idx]
            ) / seg_len

            p = (
                closed[idx]
                + alpha * segments[idx]
            )

        output.append(p)

    points = np.asarray(
        output,
        dtype=np.float32,
    )

    x = points[:, 0]
    y = points[:, 1]

    area = 0.5 * np.sum(
        x * np.roll(y, -1)
        - y * np.roll(x, -1)
    )

    if area < 0:
        points = points[::-1]

    idx = np.lexsort(
        (
            points[:, 0],
            points[:, 1],
        )
    )[0]

    points = np.roll(
        points,
        -idx,
        axis=0,
    )

    return points


def inside_center(
    instance_mask: np.ndarray,
) -> Optional[Tuple[float, float]]:

    binary = (
        instance_mask > 0
    ).astype(
        np.uint8
    )

    if cv2.countNonZero(
        binary
    ) == 0:
        return None

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5,
    )

    _, max_value, _, max_loc = cv2.minMaxLoc(
        distance
    )

    if max_value <= 0:
        return None

    return (
        float(max_loc[0]),
        float(max_loc[1]),
    )


def _local_center_peaks(
    center_prob: np.ndarray,
    foreground: np.ndarray,
    threshold: float,
    min_distance: int,
) -> np.ndarray:

    k = max(
        3,
        2 * int(min_distance) + 1,
    )

    kernel = np.ones(
        (k, k),
        np.uint8,
    )

    dilated = cv2.dilate(
        center_prob.astype(
            np.float32
        ),
        kernel,
    )

    peaks = (
        (center_prob >= threshold)
        & (
            center_prob
            >= dilated - 1e-7
        )
        & (foreground > 0)
    ).astype(
        np.uint8
    )

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            peaks,
            8,
        )
    )

    clean = np.zeros_like(
        peaks
    )

    for i in range(
        1,
        n,
    ):
        cx, cy = centroids[i]

        x = int(
            round(cx)
        )

        y = int(
            round(cy)
        )

        if (
            0 <= x < clean.shape[1]
            and 0 <= y < clean.shape[0]
        ):
            clean[y, x] = 1

    return clean


def split_instances_with_centers(
    binary_mask: np.ndarray,
    center_prob: Optional[np.ndarray] = None,
    center_threshold: float = 0.35,
    center_min_distance: int = 6,
    min_area: int = 80,
) -> List[np.ndarray]:

    binary = (
        binary_mask > 0
    ).astype(
        np.uint8
    )

    n_cc, cc = cv2.connectedComponents(
        binary,
        8,
    )

    instances: List[
        np.ndarray
    ] = []

    if center_prob is None:
        for cid in range(
            1,
            n_cc,
        ):
            inst = (
                cc == cid
            ).astype(
                np.uint8
            )

            if int(
                inst.sum()
            ) >= min_area:
                instances.append(
                    inst
                )

        return instances

    peaks = _local_center_peaks(
        center_prob,
        binary,
        threshold=center_threshold,
        min_distance=center_min_distance,
    )

    for cid in range(
        1,
        n_cc,
    ):
        component = (
            cc == cid
        ).astype(
            np.uint8
        )

        area = int(
            component.sum()
        )

        if area < min_area:
            continue

        peak_inside = (
            peaks * component
        ).astype(
            np.uint8
        )

        n_peak, peak_labels = (
            cv2.connectedComponents(
                peak_inside,
                8,
            )
        )

        n_centers = (
            n_peak - 1
        )

        if n_centers <= 1:
            instances.append(
                component
            )
            continue

        dist = cv2.distanceTransform(
            component,
            cv2.DIST_L2,
            5,
        )

        if dist.max() <= 0:
            instances.append(
                component
            )
            continue

        surface = (
            255.0
            * (
                1.0
                - dist
                / (
                    dist.max()
                    + 1e-6
                )
            )
        )

        surface = surface.astype(
            np.uint8
        )

        surface_bgr = cv2.cvtColor(
            surface,
            cv2.COLOR_GRAY2BGR,
        )

        markers = np.ones(
            component.shape,
            dtype=np.int32,
        )

        markers[
            component > 0
        ] = 0

        peak_coords = np.argwhere(
            peak_inside > 0
        )

        for marker_id, (
            py,
            px,
        ) in enumerate(
            peak_coords,
            start=2,
        ):
            cv2.circle(
                markers,
                (
                    int(px),
                    int(py),
                ),
                2,
                int(marker_id),
                -1,
            )

        cv2.watershed(
            surface_bgr,
            markers,
        )

        produced = 0

        for marker_id in range(
            2,
            2 + len(peak_coords),
        ):
            inst = (
                (
                    markers
                    == marker_id
                )
                & (
                    component > 0
                )
            ).astype(
                np.uint8
            )

            if int(
                inst.sum()
            ) >= min_area:
                instances.append(
                    inst
                )

                produced += 1

        if produced == 0:
            instances.append(
                component
            )

    return instances


def extract_instances(
    mask_prob: np.ndarray,
    center_prob: Optional[np.ndarray] = None,
    mask_threshold: float = 0.50,
    center_threshold: float = 0.35,
    n_points: int = 32,
    min_area: int = 80,
) -> List[Dict[str, np.ndarray]]:

    if mask_prob.ndim != 2:
        raise ValueError(
            "mask_prob doit etre 2D"
        )

    binary = (
        mask_prob >= mask_threshold
    ).astype(
        np.uint8
    )

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    if (
        center_prob is not None
        and center_prob.shape
        != binary.shape
    ):
        center_prob = cv2.resize(
            center_prob.astype(
                np.float32
            ),
            (
                binary.shape[1],
                binary.shape[0],
            ),
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
            (
                inst * 255
            ).astype(
                np.uint8
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if cv2.contourArea(
            contour
        ) < min_area:
            continue

        points = resample_contour(
            contour,
            n_points=n_points,
        )

        center = inside_center(
            inst
        )

        if (
            points is None
            or center is None
        ):
            continue

        results.append(
            {
                "center": np.asarray(
                    center,
                    dtype=np.float32,
                ),
                "points": points,
                "mask": inst,
                "area": float(
                    cv2.contourArea(
                        contour
                    )
                ),
            }
        )

    results.sort(
        key=lambda obj: obj[
            "area"
        ],
        reverse=True,
    )

    return results


# ============================================================
# Construction
# ============================================================

def build_model(
    pretrained: bool = False,
) -> MultiFormeNet:
    """
    pretrained est conserve pour compatibilite d'appel.

    En Paddle natif, les poids ImageNet torchvision ne peuvent pas etre
    telecharges directement. Pour initialiser depuis les poids PyTorch,
    exporter un checkpoint/poids vers NPZ puis appeler
    load_torch_npz_weights().
    """
    if pretrained:
        print(
            "ATTENTION: pretrained=True ne telecharge pas de poids dans "
            "la version Paddle. Utiliser --init-torch-npz."
        )

    return MultiFormeNet()


if __name__ == "__main__":
    paddle.set_device(
        "gpu:0"
        if paddle.is_compiled_with_cuda()
        else "cpu"
    )

    model = build_model(
        pretrained=False
    )

    model.eval()

    x = paddle.randn(
        [1, 3, 256, 256],
        dtype="float32",
    )

    with paddle.no_grad():
        out = model(x)

    print(
        "mask_logits  :",
        tuple(
            out[
                "mask_logits"
            ].shape
        ),
    )

    print(
        "center_logits:",
        tuple(
            out[
                "center_logits"
            ].shape
        ),
    )

    print(
        "parameters   :",
        f"{sum(int(np.prod(p.shape)) for p in model.parameters()):,}",
    )
