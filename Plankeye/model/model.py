"""

PlankEye v3 — détection multi-instance de quadrilatères avec priorité à la

précision des coins.

Principales évolutions par rapport à PlankEye v2 :

- MobileNetV3-Large préentraîné + BiFPN multi-échelle ;

- raffinement haute résolution au stride 2 avec skip s2 du backbone ;

- régressions coins/centre spécifiques à chaque classe ;

- coins régressés directement relativement à la cellule du centre (la précision

  des coins ne dépend plus de l'erreur de l'offset du centre) ;

- loss des coins en pixels, permutation-invariante sur les 8 ordres valides ;

- contraintes géométriques (aire, longueurs/directions des arêtes, cohérence

  centre/coins) ;

- IoU polygonale exacte pour quadrilatères convexes, utilisée en NMS et métriques ;

- métriques mAP@0.5, mAP@0.75 et précision directe des coins.

Format label :

    <classe> x1 y1 x2 y2 x3 y3 x4 y4

avec coordonnées normalisées dans [0, 1].



Letterbox obligatoire : utiliser la même transformation à l'entraînement et à l'inférence.

"""

from __future__ import annotations

import math

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large



# -----------------------------------------------------------------------------

# Configuration modèle*

# -----------------------------------------------------------------------------

print("[PlankEye v4] configuration : MobileNetV3-Large + BiFPN + refiner stride 2 ")
print("============================  NEW_VERSION   ============================= ")

print("============================ Depart 1 classe ============================= ")

NUM_CLASSES = 1

N_CORNERS = 4

CLASS_NAMES = ["planche"]

# 512 conserve sensiblement plus de détail que 384 tout en restant raisonnable*

# avec MobileNetV3 + AMP et un batch de 2 sur la majorité des GPU dédiés.*

IMG_SIZE = 512

PRED_STRIDE = 2

HEATMAP_SIZE = IMG_SIZE // PRED_STRIDE

MIN_OVERLAP = 0.70

MIN_SIGMA = 1.25

MAX_SIGMA = 14.0

HEATMAP_LABEL_SMOOTHING = 0.0

DEFAULT_CONF_THRESH = 0.30

DEFAULT_TOPK = 50

DEFAULT_NMS_IOU_THRESH = 0.42

DEFAULT_CROSS_CLASS_NMS_IOU = 0.82

HEAD_DROPOUT = 0.05

MIN_QUAD_AREA = 1e-5
MIN_EDGE_LENGTH = 1e-4
CORNER_MIN_DISTANCE = 1e-5

# Niveaux MobileNetV3-Large vérifiés par probe dynamique.*

_MNV3_LEVELS = {"s2": 1, "s4": 3, "s8": 6, "s16": 12}

# 4 décalages cycliques × 2 orientations. La loss choisit automatiquement*

# l'association la moins coûteuse : pas de discontinuité liée au "premier" coin.*

_CORNER_PERMUTATIONS = torch.tensor(

    [

        [0, 1, 2, 3],

        [1, 2, 3, 0],

        [2, 3, 0, 1],

        [3, 0, 1, 2],

        [0, 3, 2, 1],

        [3, 2, 1, 0],

        [2, 1, 0, 3],

        [1, 0, 3, 2],

    ],

    dtype=torch.long,

)



# -----------------------------------------------------------------------------

# Backbone*

# -----------------------------------------------------------------------------

class MobileNetV3Backbone(nn.Module):

    """MobileNetV3-Large utilisé uniquement comme extracteur de features."""

    def __init__(self, pretrained: bool = True):

        super().__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None

        base = mobilenet_v3_large(weights=weights)

        self.features = base.features

        self._out_channels = self._probe()

    def _probe(self) -> Dict[str, int]:

        was_training = self.features.training

        self.features.eval()

        x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

        channels: Dict[str, int] = {}

        with torch.no_grad():

            for i, block in enumerate(self.features):

                x = block(x)

                for name, idx in _MNV3_LEVELS.items():

                    if i == idx:

                        channels[name] = int(x.shape[1])

        self.features.train(was_training)

        missing = set(_MNV3_LEVELS) - set(channels)

        if missing:

            raise RuntimeError(f"Niveaux MobileNetV3 introuvables : {sorted(missing)}")

        return channels

    @property

    def out_channels(self) -> Dict[str, int]:

        return dict(self._out_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        feats: Dict[str, torch.Tensor] = {}

        for i, block in enumerate(self.features):

            x = block(x)

            for name, idx in _MNV3_LEVELS.items():

                if i == idx:

                    feats[name] = x

        return feats



# -----------------------------------------------------------------------------

# Blocs réseau*

# -----------------------------------------------------------------------------

def _norm(channels: int, groups: int = 16) -> nn.GroupNorm:

    g = min(groups, channels)

    while channels % g != 0:

        g -= 1

    return nn.GroupNorm(g, channels)



class ConvGNAct(nn.Sequential):

    def __init__(

        self,

        in_c: int,

        out_c: int,

        kernel_size: int = 3,

        stride: int = 1,

        dilation: int = 1,

    ):

        padding = (kernel_size // 2) * dilation

        super().__init__(

            nn.Conv2d(

                in_c,

                out_c,

                kernel_size,

                stride=stride,

                padding=padding,

                dilation=dilation,

                bias=False,

            ),

            _norm(out_c),

            nn.SiLU(inplace=True),

        )



class DepthwiseSeparable(nn.Sequential):

    """Bloc léger mais plus doux pour la régression que ReLU6."""

    def __init__(self, in_c: int, out_c: int, dilation: int = 1):

        super().__init__(

            nn.Conv2d(

                in_c,

                in_c,

                3,

                stride=1,

                padding=dilation,

                dilation=dilation,

                groups=in_c,

                bias=False,

            ),

            nn.Conv2d(in_c, out_c, 1, bias=False),

            _norm(out_c),

            nn.SiLU(inplace=True),

        )



class ResidualDepthwiseBlock(nn.Module):

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0):

        super().__init__()

        self.block = DepthwiseSeparable(channels, channels, dilation=dilation)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.dropout(self.block(x))



# -----------------------------------------------------------------------------

# BiFPN stride 4*

# -----------------------------------------------------------------------------

class BiFPNBlock(nn.Module):

    def __init__(self, feat_c: int):

        super().__init__()

        self.eps = 1e-4

        self.w_td4 = nn.Parameter(torch.ones(2))

        self.w_td3 = nn.Parameter(torch.ones(2))

        self.w_bu4 = nn.Parameter(torch.ones(3))

        self.w_bu5 = nn.Parameter(torch.ones(2))

        self.td_conv4 = DepthwiseSeparable(feat_c, feat_c)

        self.td_conv3 = DepthwiseSeparable(feat_c, feat_c)

        self.bu_conv4 = DepthwiseSeparable(feat_c, feat_c)

        self.bu_conv5 = DepthwiseSeparable(feat_c, feat_c)

    def _fuse(self, feats: Sequence[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:

        w = F.softplus(weights)

        w = w / (w.sum() + self.eps)

        out = feats[0] * w[0]

        for i in range(1, len(feats)):

            out = out + feats[i] * w[i]

        return out

    def forward(

        self,

        p3: torch.Tensor,

        p4: torch.Tensor,

        p5: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p5_td = p5

        p4_td = self.td_conv4(

            self._fuse(

                [p4, F.interpolate(p5_td, size=p4.shape[-2:], mode="bilinear", align_corners=False)],

                self.w_td4,

            )

        )

        p3_td = self.td_conv3(

            self._fuse(

                [p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="bilinear", align_corners=False)],

                self.w_td3,

            )

        )

        p3_out = p3_td

        p4_out = self.bu_conv4(

            self._fuse(

                [p4, p4_td, F.max_pool2d(p3_out, kernel_size=2, stride=2)],

                self.w_bu4,

            )

        )

        p5_out = self.bu_conv5(

            self._fuse(

                [p5, F.max_pool2d(p4_out, kernel_size=2, stride=2)],

                self.w_bu5,

            )

        )

        return p3_out, p4_out, p5_out



class BiFPNNeck(nn.Module):

    def __init__(self, in_channels: Dict[str, int], feat_c: int = 192, n_blocks: int = 3):

        super().__init__()

        self.lat = nn.ModuleDict(

            {

                name: nn.Sequential(

                    nn.Conv2d(in_channels[name], feat_c, 1, bias=False),

                    _norm(feat_c),

                    nn.SiLU(inplace=True),

                )

                for name in ("s4", "s8", "s16")

            }

        )

        self.blocks = nn.ModuleList([BiFPNBlock(feat_c) for _ in range(n_blocks)])

        self.out_weights = nn.Parameter(torch.ones(3))

        self.eps = 1e-4

        self.feat_c = feat_c

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:

        p3 = self.lat["s4"](feats["s4"])

        p4 = self.lat["s8"](feats["s8"])

        p5 = self.lat["s16"](feats["s16"])

        for block in self.blocks:

            p3, p4, p5 = block(p3, p4, p5)

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)

        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)

        w = F.softplus(self.out_weights)

        w = w / (w.sum() + self.eps)

        return w[0] * p3 + w[1] * p4_up + w[2] * p5_up



# -----------------------------------------------------------------------------

# Raffinement haute résolution stride 2*

# -----------------------------------------------------------------------------

class HighResolutionRefiner(nn.Module):

    """Fusionne le contexte BiFPN stride 4 avec les détails précoces stride 2."""

    def __init__(self, s2_channels: int, bifpn_channels: int, out_c: int = 96):

        super().__init__()

        self.s2_proj = nn.Sequential(

            nn.Conv2d(s2_channels, out_c, 1, bias=False),

            _norm(out_c),

            nn.SiLU(inplace=True),

        )

        self.context_proj = nn.Sequential(

            nn.Conv2d(bifpn_channels, out_c, 1, bias=False),

            _norm(out_c),

            nn.SiLU(inplace=True),

        )

        self.fuse_weights = nn.Parameter(torch.ones(2))

        self.eps = 1e-4

        self.refine = nn.Sequential(

            ResidualDepthwiseBlock(out_c, dilation=1),

            ResidualDepthwiseBlock(out_c, dilation=2),

            ConvGNAct(out_c, out_c, kernel_size=3),

        )

        self.out_c = out_c

    def forward(self, s2: torch.Tensor, context_s4: torch.Tensor) -> torch.Tensor:

        fine = self.s2_proj(s2)

        context = self.context_proj(context_s4)

        context = F.interpolate(context, size=fine.shape[-2:], mode="bilinear", align_corners=False)

        w = F.softplus(self.fuse_weights)

        w = w / (w.sum() + self.eps)

        x = w[0] * fine + w[1] * context

        return self.refine(x)



# -----------------------------------------------------------------------------

# Têtes de prédiction*

# -----------------------------------------------------------------------------

class PredictionTower(nn.Module):

    def __init__(self, channels: int, depth: int = 2, dropout: float = HEAD_DROPOUT):

        super().__init__()

        blocks: List[nn.Module] = []

        for i in range(depth):

            blocks.append(ResidualDepthwiseBlock(channels, dilation=1 + (i % 2)))

        blocks.append(nn.Dropout2d(dropout))

        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)



class HeatmapHead(nn.Module):

    def __init__(self, in_c: int, num_classes: int):

        super().__init__()

        self.tower = PredictionTower(in_c, depth=2, dropout=HEAD_DROPOUT)

        self.out = nn.Conv2d(in_c, num_classes, 1)

        nn.init.normal_(self.out.weight, std=1e-3)

        nn.init.constant_(self.out.bias, -4.595)  # p initiale ≈ 1 % (heatmap 256×256 très sparse)*

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.out(self.tower(x))



class CornerHead(nn.Module):

    """8 valeurs par classe : 4 coins × (dx, dy), relatifs à la cellule."""

    def __init__(self, in_c: int, num_classes: int):

        super().__init__()

        self.num_classes = num_classes

        self.tower = PredictionTower(in_c, depth=3, dropout=HEAD_DROPOUT)

        self.out = nn.Conv2d(in_c, num_classes * 2 * N_CORNERS, 1)

        nn.init.normal_(self.out.weight, std=1e-3)

        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.out(self.tower(x))



class CenterOffsetHead(nn.Module):

    """2 valeurs par classe, converties ensuite en [-0.5, 0.5] par tanh."""

    def __init__(self, in_c: int, num_classes: int):

        super().__init__()

        self.num_classes = num_classes

        self.tower = PredictionTower(in_c, depth=2, dropout=HEAD_DROPOUT)

        self.out = nn.Conv2d(in_c, num_classes * 2, 1)

        nn.init.normal_(self.out.weight, std=1e-3)

        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.out(self.tower(x))



# -----------------------------------------------------------------------------

# Modèle principal*

# -----------------------------------------------------------------------------

class PlankEyeV2(nn.Module):

    """

    Nom de classe conservé pour compatibilité d'import avec le projet existant.

    L'architecture interne correspond à PlankEye v3.

    """

    def __init__(

        self,

        num_classes: int = NUM_CLASSES,

        pretrained: bool = True,

        feat_c: int = 192,

        refine_c: int = 96,

        n_bifpn: int = 3,

    ):

        super().__init__()

        self.num_classes = num_classes

        self.backbone = MobileNetV3Backbone(pretrained=pretrained)

        self.neck = BiFPNNeck(self.backbone.out_channels, feat_c=feat_c, n_blocks=n_bifpn)

        self.refiner = HighResolutionRefiner(

            s2_channels=self.backbone.out_channels["s2"],

            bifpn_channels=feat_c,

            out_c=refine_c,

        )

        self.heatmap_head = HeatmapHead(refine_c, num_classes=num_classes)

        self.corner_head = CornerHead(refine_c, num_classes=num_classes)

        self.offset_head = CenterOffsetHead(refine_c, num_classes=num_classes)

        self.freeze_backbone()

        total = sum(p.numel() for p in self.parameters())

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"[PlankEye v3] backbone={self.backbone.out_channels}")

        print(

            f"[PlankEye v3] paramètres={total:,} | entraînables init={trainable:,} | "

            f"classes={num_classes} | stride={PRED_STRIDE} | input={IMG_SIZE}"

        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        feats = self.backbone(x)

        context = self.neck(feats)

        fused = self.refiner(feats["s2"], context)

        expected_h = x.shape[-2] // PRED_STRIDE

        expected_w = x.shape[-1] // PRED_STRIDE

        if fused.shape[-2:] != (expected_h, expected_w):

            raise RuntimeError(

                f"Feature map {tuple(fused.shape[-2:])} != attendu {(expected_h, expected_w)}"

            )

        return (

            self.heatmap_head(fused),

            self.corner_head(fused),

            self.offset_head(fused),

        )

    def freeze_backbone(self) -> None:

        for p in self.backbone.parameters():

            p.requires_grad_(False)

    def unfreeze_backbone(self, last_n: Optional[int] = None) -> None:

        blocks = list(self.backbone.features.children())

        if last_n is None:

            for p in self.backbone.parameters():

                p.requires_grad_(True)

            return

        self.freeze_backbone()

        last_n = max(1, min(int(last_n), len(blocks)))

        for block in blocks[-last_n:]:

            for p in block.parameters():

                p.requires_grad_(True)

    def param_groups(self, lr_head: float, lr_backbone: float):

        """Groupes avec layer-wise LR decay sur MobileNetV3."""

        blocks = list(self.backbone.features.children())

        n = len(blocks)

        cut1 = max(1, n // 3)

        cut2 = max(cut1 + 1, 2 * n // 3)

        def params_of(seq: Iterable[nn.Module]):

            out: List[nn.Parameter] = []

            for module in seq:

                out.extend(list(module.parameters()))

            return out

        head_params = (

            list(self.neck.parameters())

            + list(self.refiner.parameters())

            + list(self.heatmap_head.parameters())

            + list(self.corner_head.parameters())

            + list(self.offset_head.parameters())

        )

        return [

            {

                "params": params_of(blocks[:cut1]),

                "lr": lr_backbone * 0.25,

                "group_name": "bb_early",

                "lr_mult": 0.25,

            },

            {

                "params": params_of(blocks[cut1:cut2]),

                "lr": lr_backbone * 0.50,

                "group_name": "bb_mid",

                "lr_mult": 0.50,

            },

            {

                "params": params_of(blocks[cut2:]),

                "lr": lr_backbone,

                "group_name": "bb_late",

                "lr_mult": 1.0,

            },

            {

                "params": head_params,

                "lr": lr_head,

                "group_name": "head",

                "lr_mult": 1.0,

            },

        ]

    @torch.no_grad()

    def predict(

        self,

        x: torch.Tensor,

        conf_thresh: float = DEFAULT_CONF_THRESH,

        topk: int = DEFAULT_TOPK,

        nms_iou_thresh: float = DEFAULT_NMS_IOU_THRESH,

        cross_class_nms_iou: float = DEFAULT_CROSS_CLASS_NMS_IOU,

        original_sizes: Optional[Sequence[Tuple[int, int]]] = None,

        output_pixels: bool = False,

    ):

        was_training = self.training

        self.eval()

        outputs = self.forward(x)

        detections = decode_detections(

            *outputs,

            conf_thresh=conf_thresh,

            topk=topk,

            nms_iou_thresh=nms_iou_thresh,

            cross_class_nms_iou=cross_class_nms_iou,

            original_sizes=original_sizes,

            output_pixels=output_pixels,

        )

        self.train(was_training)

        return detections



def build_model(

    num_classes: int = NUM_CLASSES,

    pretrained: bool = True,

    feat_c: int = 192,

    refine_c: int = 96,

    n_bifpn: int = 3,

    **_: object,

) -> PlankEyeV2:

    return PlankEyeV2(

        num_classes=num_classes,

        pretrained=pretrained,

        feat_c=feat_c,

        refine_c=refine_c,

        n_bifpn=n_bifpn,

    )



# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Letterbox / coordonnées
# -----------------------------------------------------------------------------

def compute_letterbox(
    original_width: int,
    original_height: int,
    input_size: int = IMG_SIZE,
) -> Tuple[float, int, int, int, int]:
    """Retourne scale, new_w, new_h, pad_x, pad_y."""
    ow, oh = int(original_width), int(original_height)
    if ow <= 0 or oh <= 0:
        raise ValueError(f"Dimensions invalides: {ow}x{oh}")
    scale = min(float(input_size) / ow, float(input_size) / oh)
    new_w = max(1, int(round(ow * scale)))
    new_h = max(1, int(round(oh * scale)))
    pad_x = (int(input_size) - new_w) // 2
    pad_y = (int(input_size) - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def letterbox_corners_to_model(
    corners: np.ndarray,
    original_width: int,
    original_height: int,
    input_size: int = IMG_SIZE,
) -> np.ndarray:
    """Coins normalisés image originale -> coordonnées normalisées modèle."""
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    scale, _nw, _nh, pad_x, pad_y = compute_letterbox(
        original_width, original_height, input_size
    )
    px = pts.copy()
    px[:, 0] *= max(int(original_width) - 1, 1)
    px[:, 1] *= max(int(original_height) - 1, 1)
    px[:, 0] = px[:, 0] * scale + pad_x
    px[:, 1] = px[:, 1] * scale + pad_y
    return np.clip(px / max(int(input_size) - 1, 1), 0.0, 1.0).astype(np.float32)


def model_corners_to_original_pixels(
    corners: np.ndarray,
    original_width: int,
    original_height: int,
    input_size: int = IMG_SIZE,
) -> np.ndarray:
    """Coins normalisés modèle -> pixels image originale."""
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    scale, _nw, _nh, pad_x, pad_y = compute_letterbox(
        original_width, original_height, input_size
    )
    px = pts.copy() * max(int(input_size) - 1, 1)
    px[:, 0] = (px[:, 0] - pad_x) / max(scale, 1e-12)
    px[:, 1] = (px[:, 1] - pad_y) / max(scale, 1e-12)
    px[:, 0] = np.clip(px[:, 0], 0.0, max(int(original_width) - 1, 0))
    px[:, 1] = np.clip(px[:, 1], 0.0, max(int(original_height) - 1, 0))
    return px.astype(np.float32)


# Targets CenterNet*

# -----------------------------------------------------------------------------

def gaussian_radius(height: float, width: float, min_overlap: float = MIN_OVERLAP) -> float:

    height = max(float(height), 1e-6)

    width = max(float(width), 1e-6)

    a1 = 1.0

    b1 = height + width

    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)

    r1 = (b1 + math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / 2

    a2 = 4.0

    b2 = 2 * (height + width)

    c2 = (1 - min_overlap) * width * height

    r2 = (b2 + math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / 2

    a3 = 4 * min_overlap

    b3 = -2 * min_overlap * (height + width)

    c3 = (min_overlap - 1) * width * height

    r3 = (b3 + math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / (2 * a3)

    return max(0.0, min(r1, r2, r3))



def draw_gaussian_(

    heatmap: torch.Tensor,

    cx: float,

    cy: float,

    sigma: float,

    force_peak_xy: Optional[Tuple[int, int]] = None,

    peak_value: float = 1.0,

) -> None:

    H, W = heatmap.shape

    sigma = float(np.clip(sigma, MIN_SIGMA, MAX_SIGMA))

    radius = max(1, int(math.ceil(3.0 * sigma)))

    x0 = max(0, int(math.floor(cx)) - radius)

    x1 = min(W, int(math.ceil(cx)) + radius + 1)

    y0 = max(0, int(math.floor(cy)) - radius)

    y1 = min(H, int(math.ceil(cy)) + radius + 1)

    if x1 <= x0 or y1 <= y0:

        return

    gy = torch.arange(y0, y1, device=heatmap.device, dtype=torch.float32)

    gx = torch.arange(x0, x1, device=heatmap.device, dtype=torch.float32)

    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    gauss = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma * sigma))

    gauss = gauss * peak_value

    region = heatmap[y0:y1, x0:x1]

    torch.maximum(region, gauss, out=region)

    if force_peak_xy is not None:

        ix, iy = force_peak_xy

        heatmap[iy, ix] = peak_value



def build_targets(

    batch_objects,

    hmap_h: int,

    hmap_w: int,

    num_classes: int = NUM_CLASSES,

    device: torch.device | str = "cpu",

    label_smoothing: float = HEATMAP_LABEL_SMOOTHING,

    verbose: bool = False,

):

    """

    Les régressions étant spécifiques à la classe, deux classes différentes

    peuvent partager la même cellule sans gradient contradictoire.

    """

    peak_value = 1.0 - label_smoothing

    B = len(batch_objects)

    heatmaps = torch.zeros(B, num_classes, hmap_h, hmap_w, device=device)

    candidates = []

    scale_x = max(hmap_w - 1, 1)

    scale_y = max(hmap_h - 1, 1)

    for b, objects in enumerate(batch_objects):

        for obj in objects:

            cls = int(obj["cls"])

            if cls < 0 or cls >= num_classes:

                continue

            pts = obj["corners"]

            if not torch.is_tensor(pts):

                pts = torch.as_tensor(pts, dtype=torch.float32)

            pts = pts.to(device=device, dtype=torch.float32).reshape(4, 2)

            if not torch.isfinite(pts).all():

                continue

            pts = pts.clamp(0.0, 1.0)

            center = pts.mean(dim=0)

            cx = float(center[0]) * scale_x

            cy = float(center[1]) * scale_y

            # Cellule la plus proche -> offset symétrique dans [-0.5, 0.5].*

            ix = int(round(cx))

            iy = int(round(cy))

            ix = min(max(ix, 0), hmap_w - 1)

            iy = min(max(iy, 0), hmap_h - 1)

            box_w = float(pts[:, 0].max() - pts[:, 0].min()) * scale_x

            box_h = float(pts[:, 1].max() - pts[:, 1].min()) * scale_y

            sigma = np.clip(gaussian_radius(box_h, box_w) / 3.0, MIN_SIGMA, MAX_SIGMA)

            draw_gaussian_(

                heatmaps[b, cls],

                cx,

                cy,

                float(sigma),

                force_peak_xy=(ix, iy),

                peak_value=peak_value,

            )

            # Ancrage coin = cellule entière, pas centre prédit. Une erreur de*

            # center-offset ne translate donc plus les quatre sommets.*

            cell_center = torch.tensor(

                [ix / scale_x, iy / scale_y],

                dtype=torch.float32,

                device=device,

            )

            corner_off = (pts - cell_center.unsqueeze(0)).reshape(-1)

            center_off = torch.tensor([cx - ix, cy - iy], device=device)

            # Collision seulement si même image + même classe + même cellule.*

            # On conserve l'objet dont le vrai centre est le plus proche.*

            dist2 = (cx - ix) ** 2 + (cy - iy) ** 2

            candidates.append((b, cls, iy, ix, dist2, corner_off, center_off))

    best_by_cell = {}

    collisions = 0

    for cand in candidates:

        key = cand[:4]

        previous = best_by_cell.get(key)

        if previous is None:

            best_by_cell[key] = cand

            continue

        collisions += 1

        if cand[4] < previous[4]:

            best_by_cell[key] = cand

        if verbose:

            print(f"Collision target : batch={key[0]} cls={key[1]} cell=({key[3]},{key[2]})")

    idx_b: List[int] = []

    idx_cls: List[int] = []

    idx_y: List[int] = []

    idx_x: List[int] = []

    corner_targets: List[torch.Tensor] = []

    offset_targets: List[torch.Tensor] = []

    for b, cls, iy, ix, _dist2, corner_off, center_off in best_by_cell.values():

        idx_b.append(b)

        idx_cls.append(cls)

        idx_y.append(iy)

        idx_x.append(ix)

        corner_targets.append(corner_off)

        offset_targets.append(center_off)

    if not idx_b:

        empty_long = torch.zeros(0, dtype=torch.long, device=device)

        return (

            heatmaps,

            empty_long,

            empty_long.clone(),

            empty_long.clone(),

            empty_long.clone(),

            torch.zeros(0, 8, device=device),

            torch.zeros(0, 2, device=device),

            {"n_collisions": collisions, "n_candidates": len(candidates)},

        )

    return (

        heatmaps,

        torch.tensor(idx_b, dtype=torch.long, device=device),

        torch.tensor(idx_cls, dtype=torch.long, device=device),

        torch.tensor(idx_y, dtype=torch.long, device=device),

        torch.tensor(idx_x, dtype=torch.long, device=device),

        torch.stack(corner_targets, dim=0),

        torch.stack(offset_targets, dim=0),

        {"n_collisions": collisions, "n_candidates": len(candidates)},

    )



# -----------------------------------------------------------------------------

# Losses*

# -----------------------------------------------------------------------------

def focal_heatmap_loss(

    pred_logits: torch.Tensor,

    target_heatmaps: torch.Tensor,

    alpha: float = 2.0,

    beta: float = 4.0,

    class_weights: Optional[torch.Tensor] = None,

) -> torch.Tensor:

    pred = torch.sigmoid(pred_logits).clamp(1e-5, 1.0 - 1e-5)

    peak = 1.0 - HEATMAP_LABEL_SMOOTHING

    pos_mask = (target_heatmaps >= peak - 1e-5).to(pred.dtype)

    neg_mask = 1.0 - pos_mask

    neg_weights = (1.0 - target_heatmaps).pow(beta)

    pos_loss = -torch.log(pred) * (1.0 - pred).pow(alpha) * pos_mask

    neg_loss = -torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg_mask

    if class_weights is not None:

        cw = class_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)

        pos_loss = pos_loss * cw

        neg_loss = neg_loss * cw

        normalizer = (pos_mask * cw).sum().clamp(min=1.0)

    else:

        normalizer = pos_mask.sum().clamp(min=1.0)

    return (pos_loss.sum() + neg_loss.sum()) / normalizer



def gather_class_at_positions(

    pred_map: torch.Tensor,

    idx_b: torch.Tensor,

    idx_cls: torch.Tensor,

    idx_y: torch.Tensor,

    idx_x: torch.Tensor,

    values_per_class: int,

) -> torch.Tensor:

    if idx_b.numel() == 0:

        return pred_map.new_zeros((0, values_per_class))

    B, channels, H, W = pred_map.shape

    if channels % values_per_class != 0:

        raise ValueError(f"{channels=} incompatible avec {values_per_class=}")

    C = channels // values_per_class

    view = pred_map.view(B, C, values_per_class, H, W)

    return view[idx_b, idx_cls, :, idx_y, idx_x]



def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:

    if values.numel() == 0:

        return values.sum() * 0.0

    if weights is None:

        return values.mean()

    weights = weights.to(device=values.device, dtype=values.dtype)

    return (values * weights).sum() / weights.sum().clamp(min=1e-6)



def permutation_invariant_corner_loss(

    corner_pred_flat: torch.Tensor,

    corner_gt_flat: torch.Tensor,

    instance_weights: Optional[torch.Tensor] = None,

    beta_px: float = 1.0,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """

    Loss Huber avec transition à 1 pixel, mais calculée en coordonnées

    normalisées pour garder des gradients bornés. Les 8 permutations cycliques

    valides du quadrilatère sont testées et la meilleure est retenue.

    """

    if corner_pred_flat.numel() == 0:

        zero = corner_pred_flat.sum() * 0.0

        return zero, corner_gt_flat.reshape(-1, 4, 2), zero

    pred = corner_pred_flat.reshape(-1, 4, 2)

    gt = corner_gt_flat.reshape(-1, 4, 2)

    perms = _CORNER_PERMUTATIONS.to(gt.device)

    variants = gt[:, perms, :]  # (N, 8, 4, 2)*

    beta_norm = beta_px / float(IMG_SIZE)

    elem = F.smooth_l1_loss(

        pred[:, None, :, :].expand_as(variants),

        variants,

        beta=beta_norm,

        reduction="none",

    )

    per_perm = elem.mean(dim=(-1, -2))

    best_perm = per_perm.argmin(dim=1)

    batch_idx = torch.arange(pred.shape[0], device=pred.device)

    aligned_gt = variants[batch_idx, best_perm]

    per_instance = per_perm[batch_idx, best_perm]

    loss = _weighted_mean(per_instance, instance_weights)

    corner_err_px = torch.linalg.vector_norm(pred - aligned_gt, dim=-1) * IMG_SIZE

    corner_err_px = _weighted_mean(corner_err_px.mean(dim=1), instance_weights)

    return loss, aligned_gt, corner_err_px



def polygon_area_log_loss(

    pred: torch.Tensor,

    gt: torch.Tensor,

    weights: Optional[torch.Tensor] = None,

) -> torch.Tensor:

    """Loss d'aire stable (nom historique conservé pour compatibilité)."""

    if pred.numel() == 0:

        return pred.sum() * 0.0

    def area(points: torch.Tensor) -> torch.Tensor:

        x = points[..., 0]

        y = points[..., 1]

        return 0.5 * torch.abs(

            (x * torch.roll(y, -1, dims=-1) - torch.roll(x, -1, dims=-1) * y).sum(dim=-1)

        )

    a_pred = area(pred)

    a_gt = area(gt)

    per_instance = F.smooth_l1_loss(a_pred, a_gt, beta=0.01, reduction="none")

    return _weighted_mean(per_instance, weights)



def edge_geometry_loss(

    pred: torch.Tensor,

    gt: torch.Tensor,

    weights: Optional[torch.Tensor] = None,

) -> Tuple[torch.Tensor, torch.Tensor]:

    if pred.numel() == 0:

        z = pred.sum() * 0.0

        return z, z

    e_pred = torch.roll(pred, -1, dims=1) - pred

    e_gt = torch.roll(gt, -1, dims=1) - gt

    len_pred = torch.linalg.vector_norm(e_pred, dim=-1)

    len_gt = torch.linalg.vector_norm(e_gt, dim=-1)

    length_per_instance = F.smooth_l1_loss(

        len_pred, len_gt, beta=0.01, reduction="none"

    ).mean(dim=1)

    # La direction n'est activée que lorsque l'arête prédite a déjà une longueur*

    # suffisante. Cela évite le gradient 1/||edge|| très violent au démarrage.*

    valid = (len_pred.detach() > 0.02) & (len_gt.detach() > 0.02)

    safe_pred = len_pred.clamp(min=0.02)

    safe_gt = len_gt.clamp(min=0.02)

    unit_pred = e_pred / safe_pred.unsqueeze(-1)

    unit_gt = e_gt / safe_gt.unsqueeze(-1)

    cos = (unit_pred * unit_gt).sum(dim=-1).clamp(-1.0, 1.0)

    direction = (1.0 - cos) * valid.to(cos.dtype)

    valid_count = valid.sum(dim=1).clamp(min=1)

    direction_per_instance = direction.sum(dim=1) / valid_count

    return (

        _weighted_mean(length_per_instance, weights),

        _weighted_mean(direction_per_instance, weights),

    )



def combined_loss_v2(

    heatmap_logits: torch.Tensor,

    corner_pred: torch.Tensor,

    offset_pred: torch.Tensor,

    batch_objects,

    class_weights: Optional[torch.Tensor] = None,

    w_hmap: float = 1.0,

    w_corner: float = 6.0,

    w_offset: float = 0.45,

    w_area: float = 0.20,

    w_edge_length: float = 0.25,

    w_edge_direction: float = 0.04,

    w_consistency: float = 0.20,

):

    """Loss combinée orientée précision des coins."""

    B, C, H, W = heatmap_logits.shape

    device = heatmap_logits.device

    (

        target_hmaps,

        idx_b,

        idx_cls,

        idx_y,

        idx_x,

        corner_gt,

        offset_gt,

        collision_stats,

    ) = build_targets(batch_objects, H, W, num_classes=C, device=device)

    if class_weights is not None:

        class_weights = class_weights.to(device=device, dtype=heatmap_logits.dtype)

    l_hmap = focal_heatmap_loss(

        heatmap_logits,

        target_hmaps,

        class_weights=class_weights,

    )

    corner_at_pos = gather_class_at_positions(

        corner_pred, idx_b, idx_cls, idx_y, idx_x, values_per_class=8

    )

    offset_logits_at_pos = gather_class_at_positions(

        offset_pred, idx_b, idx_cls, idx_y, idx_x, values_per_class=2

    )

    # Offset borné et symétrique autour de 0.*

    offset_at_pos = 0.5 * torch.tanh(offset_logits_at_pos)

    if idx_b.numel() > 0:

        inst_weights = class_weights[idx_cls] if class_weights is not None else None

        l_corner, aligned_gt, corner_px = permutation_invariant_corner_loss(

            corner_at_pos,

            corner_gt,

            instance_weights=inst_weights,

        )

        offset_elem = F.smooth_l1_loss(

            offset_at_pos,

            offset_gt,

            beta=0.08,

            reduction="none",

        ).mean(dim=1)

        l_offset = _weighted_mean(offset_elem, inst_weights)

        pred_pts = corner_at_pos.reshape(-1, 4, 2)

        gt_pts = aligned_gt

        l_area = polygon_area_log_loss(pred_pts, gt_pts, inst_weights)

        l_edge_len, l_edge_dir = edge_geometry_loss(pred_pts, gt_pts, inst_weights)

        # La moyenne des offsets de coins doit retrouver le vrai centre relatif*

        # à la cellule. On compare au center-offset prédit pour lier les têtes.*

        mean_corner = pred_pts.mean(dim=1)

        cell_center_norm = torch.stack(
            [
                idx_x.to(pred_pts.dtype) / max(W - 1, 1),
                idx_y.to(pred_pts.dtype) / max(H - 1, 1),
            ],
            dim=1,
        )
        offset_norm = torch.stack(
            [
                offset_at_pos[:, 0] / max(W - 1, 1),
                offset_at_pos[:, 1] / max(H - 1, 1),
            ],
            dim=1,
        )
        predicted_center_norm = cell_center_norm + offset_norm
        consistency_delta = mean_corner - predicted_center_norm

        consistency_per = F.smooth_l1_loss(

            consistency_delta,

            torch.zeros_like(consistency_delta),

            beta=1.0 / IMG_SIZE,

            reduction="none",

        ).mean(dim=1)

        l_consistency = 2.0 * _weighted_mean(consistency_per, inst_weights)

    else:

        zero = heatmap_logits.sum() * 0.0

        l_corner = l_offset = l_area = l_edge_len = l_edge_dir = l_consistency = zero

        corner_px = zero

    total = (

        w_hmap * l_hmap

        + w_corner * l_corner

        + w_offset * l_offset

        + w_area * l_area

        + w_edge_length * l_edge_len

        + w_edge_direction * l_edge_dir

        + w_consistency * l_consistency

    )

    return total, {

        "hmap": float(l_hmap.detach()),

        "corner": float(l_corner.detach()),

        "corner_px": float(corner_px.detach()),

        "offset": float(l_offset.detach()),

        "area": float(l_area.detach()),

        "edge_len": float(l_edge_len.detach()),

        "edge_dir": float(l_edge_dir.detach()),

        "consistency": float(l_consistency.detach()),

        "n_obj": int(idx_b.numel()),

        "n_collisions": int(collision_stats["n_collisions"]),

        "n_candidates": int(collision_stats["n_candidates"]),

    }



# -----------------------------------------------------------------------------

# Géométrie polygonale exacte*

# -----------------------------------------------------------------------------

def _cross2(a: np.ndarray, b: np.ndarray) -> float:

    return float(a[0] * b[1] - a[1] * b[0])



def _convex_hull(points: np.ndarray) -> np.ndarray:

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    pts = np.unique(np.round(pts, decimals=12), axis=0)

    if len(pts) <= 1:

        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):

        return _cross2(a - o, b - o)

    lower: List[np.ndarray] = []

    for p in pts:

        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:

            lower.pop()

        lower.append(p)

    upper: List[np.ndarray] = []

    for p in pts[::-1]:

        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:

            upper.pop()

        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    return hull



def _signed_polygon_area(poly: np.ndarray) -> float:

    if len(poly) < 3:

        return 0.0

    x = poly[:, 0]

    y = poly[:, 1]

    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))



def _polygon_area_np(poly: np.ndarray) -> float:

    return abs(_signed_polygon_area(poly))



def _line_intersection_with_clip(

    s: np.ndarray,

    e: np.ndarray,

    a: np.ndarray,

    b: np.ndarray,

) -> np.ndarray:

    edge = b - a

    seg = e - s

    denom = _cross2(edge, seg)

    if abs(denom) < 1e-12:

        return e.copy()

    t = -_cross2(edge, s - a) / denom

    return s + t * seg



def _convex_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:

    if len(subject) < 3 or len(clip) < 3:

        return np.empty((0, 2), dtype=np.float64)

    subject = np.asarray(subject, dtype=np.float64)

    clip = np.asarray(clip, dtype=np.float64)

    if _signed_polygon_area(clip) < 0:

        clip = clip[::-1]

    output = subject.copy()

    for i in range(len(clip)):

        if len(output) == 0:

            break

        a = clip[i]

        b = clip[(i + 1) % len(clip)]

        input_list = output

        output_list: List[np.ndarray] = []

        s = input_list[-1]

        for e in input_list:

            e_inside = _cross2(b - a, e - a) >= -1e-12

            s_inside = _cross2(b - a, s - a) >= -1e-12

            if e_inside:

                if not s_inside:

                    output_list.append(_line_intersection_with_clip(s, e, a, b))

                output_list.append(e)

            elif s_inside:

                output_list.append(_line_intersection_with_clip(s, e, a, b))

            s = e

        output = (

            np.asarray(output_list, dtype=np.float64)

            if output_list

            else np.empty((0, 2), dtype=np.float64)

        )

    return output



def _order_polygon_np(points: np.ndarray) -> np.ndarray:
    """Ordre circulaire déterministe."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError(f"Un quadrilatère doit avoir 4 points, reçu {len(pts)}")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def canonicalize_corners(points: np.ndarray) -> np.ndarray:
    """Contrat stable : TL, TR, BR, BL."""
    pts = _order_polygon_np(points).astype(np.float64)
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    if _signed_polygon_area(pts) < 0:
        pts = pts[[0, 3, 2, 1]]
    return pts.astype(np.float32)


def _is_valid_quadrilateral_np(
    points: np.ndarray,
    min_area: float = MIN_QUAD_AREA,
    min_edge: float = MIN_EDGE_LENGTH,
) -> bool:
    """4 sommets distincts, arêtes non nulles et convexité stricte."""
    try:
        pts = canonicalize_corners(points).astype(np.float64)
    except (ValueError, TypeError):
        return False
    if not np.isfinite(pts).all():
        return False

    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < CORNER_MIN_DISTANCE:
                return False

    edges = np.roll(pts, -1, axis=0) - pts
    if np.any(np.linalg.norm(edges, axis=1) < min_edge):
        return False

    crosses = np.array(
        [_cross2(edges[i], edges[(i + 1) % 4]) for i in range(4)],
        dtype=np.float64,
    )
    if np.any(np.abs(crosses) < min_area):
        return False
    if np.any(crosses > 0) and np.any(crosses < 0):
        return False

    return _polygon_area_np(pts) >= min_area


def _polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray, grid_size: int | None = None) -> float:
    """IoU exacte, uniquement pour des quadrilatères convexes valides."""
    del grid_size
    if not _is_valid_quadrilateral_np(poly_a) or not _is_valid_quadrilateral_np(poly_b):
        return 0.0

    a = canonicalize_corners(poly_a).astype(np.float64)
    b = canonicalize_corners(poly_b).astype(np.float64)
    area_a = _polygon_area_np(a)
    area_b = _polygon_area_np(b)
    if area_a <= 1e-12 or area_b <= 1e-12:
        return 0.0

    inter_poly = _convex_clip(a, b)
    inter = _polygon_area_np(inter_poly) if len(inter_poly) >= 3 else 0.0
    union = area_a + area_b - inter
    if union <= 1e-12:
        return 0.0
    return float(np.clip(inter / union, 0.0, 1.0))


def _points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:

    """Conservé pour compatibilité avec d'anciens imports/tests."""

    points = np.asarray(points)

    poly = np.asarray(poly)

    n = poly.shape[0]

    inside = np.zeros(points.shape[0], dtype=bool)

    x, y = points[:, 0], points[:, 1]

    j = n - 1

    for i in range(n):

        xi, yi = poly[i]

        xj, yj = poly[j]

        hit = ((yi > y) != (yj > y)) & (

            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi

        )

        inside ^= hit

        j = i

    return inside



# -----------------------------------------------------------------------------

# Decode / NMS*

# -----------------------------------------------------------------------------

def _nms_heatmap(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:

    pad = (kernel - 1) // 2

    pooled = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=pad)

    return heatmap * (pooled == heatmap).to(heatmap.dtype)



def _nms_polygons(

    dets,

    iou_thresh: float = DEFAULT_NMS_IOU_THRESH,

    cross_class_iou: float = DEFAULT_CROSS_CLASS_NMS_IOU,

):

    if len(dets) <= 1:

        return dets

    ordered = sorted(dets, key=lambda d: d["score"], reverse=True)

    kept = []

    for det in ordered:

        suppress = False

        for other in kept:

            iou = _polygon_iou(det["corners"], other["corners"])

            if det["cls"] == other["cls"]:

                if iou > iou_thresh:

                    suppress = True

                    break

            elif iou > cross_class_iou:

                # Deux classes différentes ne sont supprimées que si les*

                # polygones sont pratiquement identiques.*

                suppress = True

                break

        if not suppress:

            kept.append(det)

    return kept



def decode_detections(
    heatmap_logits: torch.Tensor,
    corner_pred: torch.Tensor,
    offset_pred: torch.Tensor,
    conf_thresh: float = DEFAULT_CONF_THRESH,
    topk: int = DEFAULT_TOPK,
    nms_iou_thresh: float = DEFAULT_NMS_IOU_THRESH,
    cross_class_nms_iou: float = DEFAULT_CROSS_CLASS_NMS_IOU,
    original_sizes: Optional[Sequence[Tuple[int, int]]] = None,
    output_pixels: bool = False,
):
    """Décode les pics et inverse le letterbox si demandé.

    original_sizes[b] = (width, height). Si output_pixels=True, ``corners``
    et ``center`` sont exprimés dans le repère pixel de l'image originale.
    """
    B, C, H, W = heatmap_logits.shape
    if original_sizes is not None and len(original_sizes) != B:
        raise ValueError(
            f"original_sizes doit contenir {B} dimensions, reçu {len(original_sizes)}"
        )

    heat = _nms_heatmap(torch.sigmoid(heatmap_logits))
    corner_view = corner_pred.view(B, C, 8, H, W)
    offset_view = offset_pred.view(B, C, 2, H, W)
    scale_x = max(W - 1, 1)
    scale_y = max(H - 1, 1)
    results = []

    for b in range(B):
        dets = []
        for c in range(C):
            scores_flat = heat[b, c].reshape(-1)
            k = min(int(topk), scores_flat.numel())
            if k <= 0:
                continue

            top_scores, top_idx = torch.topk(scores_flat, k)
            valid = top_scores >= conf_thresh
            if not bool(valid.any()):
                continue

            scores = top_scores[valid]
            flat_idx = top_idx[valid]
            iy = torch.div(flat_idx, W, rounding_mode="floor")
            ix = flat_idx.remainder(W)

            offset_logits = offset_view[b, c, :, iy, ix].transpose(0, 1)
            center_off = 0.5 * torch.tanh(offset_logits)
            centers_norm = torch.stack(
                [
                    (ix.to(center_off.dtype) + center_off[:, 0]) / scale_x,
                    (iy.to(center_off.dtype) + center_off[:, 1]) / scale_y,
                ],
                dim=1,
            ).clamp(0.0, 1.0)

            corner_vals = (
                corner_view[b, c, :, iy, ix].transpose(0, 1).reshape(-1, 4, 2)
            )
            cell_centers = torch.stack(
                [
                    ix.to(corner_vals.dtype) / scale_x,
                    iy.to(corner_vals.dtype) / scale_y,
                ],
                dim=1,
            )
            corners_norm = (
                corner_vals + cell_centers[:, None, :]
            ).clamp(0.0, 1.0)

            scores_np = scores.detach().cpu().numpy()
            centers_np = centers_norm.detach().cpu().numpy()
            corners_np = corners_norm.detach().cpu().numpy()

            for score, center, poly in zip(scores_np, centers_np, corners_np):
                if not _is_valid_quadrilateral_np(poly):
                    continue

                poly_norm = canonicalize_corners(poly)

                if original_sizes is not None:
                    ow, oh = map(int, original_sizes[b])
                    poly_pixels = model_corners_to_original_pixels(
                        poly_norm, ow, oh, input_size=IMG_SIZE
                    )
                    scale, _nw, _nh, pad_x, pad_y = compute_letterbox(
                        ow, oh, IMG_SIZE
                    )
                    center_model_px = center * max(IMG_SIZE - 1, 1)
                    center_pixels = np.array(
                        [
                            np.clip(
                                (center_model_px[0] - pad_x) / max(scale, 1e-12),
                                0, max(ow - 1, 0)
                            ),
                            np.clip(
                                (center_model_px[1] - pad_y) / max(scale, 1e-12),
                                0, max(oh - 1, 0)
                            ),
                        ],
                        dtype=np.float32,
                    )
                else:
                    poly_pixels = None
                    center_pixels = None

                det = {
                    "cls": c,
                    "score": float(score),
                    "center": (
                        center_pixels
                        if output_pixels and center_pixels is not None
                        else np.asarray(center, dtype=np.float32)
                    ),
                    "corners": (
                        poly_pixels
                        if output_pixels and poly_pixels is not None
                        else poly_norm
                    ),
                }
                if output_pixels and poly_pixels is not None:
                    det["corners_norm"] = poly_norm
                dets.append(det)

        results.append(
            _nms_polygons(
                dets,
                iou_thresh=nms_iou_thresh,
                cross_class_iou=cross_class_nms_iou,
            )
        )
    return results


def detections_to_jsonable(detections, decimals: int = 2) -> List[Dict[str, object]]:
    """Convertit une liste de détections en contrat JSON minimal.

    Retour :
    [
        {"confidence": 0.94, "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
    ]
    """
    out = []
    for det in detections:
        corners = np.asarray(det["corners"], dtype=np.float32).reshape(4, 2)
        out.append(
            {
                "confidence": round(float(det["score"]), decimals),
                "corners": [
                    [round(float(x), decimals), round(float(y), decimals)]
                    for x, y in corners
                ],
            }
        )
    return out


def _compute_ap_at_iou(all_dets, all_gts, num_classes: int, iou_thresh: float):

    ap_per_class = {}

    for c in range(num_classes):

        gts_per_img = {

            img_idx: [np.asarray(g["corners"], dtype=np.float32) for g in gts if int(g["cls"]) == c]

            for img_idx, gts in enumerate(all_gts)

        }

        gt_count = sum(len(v) for v in gts_per_img.values())

        if gt_count == 0:

            continue

        entries = []

        for img_idx, dets in enumerate(all_dets):

            for d in dets:

                if int(d["cls"]) == c:

                    entries.append((float(d["score"]), img_idx, np.asarray(d["corners"])))

        entries.sort(key=lambda x: x[0], reverse=True)

        matched = {i: [False] * len(v) for i, v in gts_per_img.items()}

        tp = np.zeros(len(entries), dtype=np.float64)

        fp = np.zeros(len(entries), dtype=np.float64)

        for i, (_score, img_idx, corners) in enumerate(entries):

            candidates = gts_per_img.get(img_idx, [])

            best_iou = 0.0

            best_j = -1

            for j, gt_corners in enumerate(candidates):

                if matched[img_idx][j]:

                    continue

                iou = _polygon_iou(corners, gt_corners)

                if iou > best_iou:

                    best_iou = iou

                    best_j = j

            if best_j >= 0 and best_iou >= iou_thresh:

                tp[i] = 1.0

                matched[img_idx][best_j] = True

            else:

                fp[i] = 1.0

        if len(entries) == 0:

            ap_per_class[c] = 0.0

            continue

        tp_cum = np.cumsum(tp)

        fp_cum = np.cumsum(fp)

        recall = tp_cum / max(gt_count, 1)

        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        mrec = np.concatenate(([0.0], recall, [1.0]))

        mpre = np.concatenate(([0.0], precision, [0.0]))

        for j in range(len(mpre) - 2, -1, -1):

            mpre[j] = max(mpre[j], mpre[j + 1])

        idx = np.where(mrec[1:] != mrec[:-1])[0]

        ap_per_class[c] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    mean_ap = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0

    return mean_ap, ap_per_class



def compute_ap50(all_dets, all_gts, num_classes: int = NUM_CLASSES, iou_thresh: float = 0.5):

    """API historique conservée."""

    return _compute_ap_at_iou(all_dets, all_gts, num_classes, iou_thresh)



def _best_corner_errors_px(
    pred: np.ndarray,
    gt: np.ndarray,
    image_size: int | Tuple[int, int],
) -> np.ndarray:
    """Erreur euclidienne des coins en pixels réels."""
    pred = canonicalize_corners(pred).astype(np.float32)
    gt = canonicalize_corners(gt).astype(np.float32)

    if isinstance(image_size, (tuple, list, np.ndarray)):
        width, height = float(image_size[0]), float(image_size[1])
    else:
        width = height = float(image_size)

    scale = np.array(
        [max(width - 1.0, 1.0), max(height - 1.0, 1.0)],
        dtype=np.float32,
    )

    variants = [np.roll(gt, -shift, axis=0) for shift in range(4)]
    gt_rev = gt[::-1]
    variants += [np.roll(gt_rev, -shift, axis=0) for shift in range(4)]

    errors = [np.linalg.norm((pred - v) * scale, axis=1) for v in variants]
    mean_errors = [float(e.mean()) for e in errors]
    return errors[int(np.argmin(mean_errors))]


def compute_corner_metrics(
    all_dets,
    all_gts,
    image_sizes: Optional[Sequence[Tuple[int, int]]] = None,
    image_size: int = IMG_SIZE,
    match_iou_thresh: float = 0.30,
):
    """Mesure les quatre coins en pixels de l'image originale."""
    if image_sizes is not None and len(image_sizes) != len(all_gts):
        raise ValueError(
            f"image_sizes doit contenir {len(all_gts)} dimensions, reçu {len(image_sizes)}"
        )

    all_corner_errors: List[float] = []
    matched_objects = 0
    total_gt = 0

    for img_idx, (dets, gts) in enumerate(zip(all_dets, all_gts)):
        total_gt += len(gts)
        used = [False] * len(dets)
        current_size = image_sizes[img_idx] if image_sizes is not None else image_size

        for gt in gts:
            gt_cls = int(gt["cls"])
            gt_poly = np.asarray(gt["corners"], dtype=np.float32)
            best_iou = 0.0
            best_j = -1

            for j, det in enumerate(dets):
                if used[j] or int(det["cls"]) != gt_cls:
                    continue
                iou = _polygon_iou(np.asarray(det["corners"]), gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j < 0 or best_iou < match_iou_thresh:
                continue

            used[best_j] = True
            matched_objects += 1
            errs = _best_corner_errors_px(
                np.asarray(dets[best_j]["corners"]),
                gt_poly,
                current_size,
            )
            all_corner_errors.extend(errs.tolist())

    if not all_corner_errors:
        return {
            "corner_mae_px": float("inf"),
            "corner_rmse_px": float("inf"),
            "pck2": 0.0,
            "pck4": 0.0,
            "pck8": 0.0,
            "matched_objects": 0,
            "match_recall": 0.0,
        }

    arr = np.asarray(all_corner_errors, dtype=np.float64)
    return {
        "corner_mae_px": float(arr.mean()),
        "corner_rmse_px": float(np.sqrt(np.mean(arr ** 2))),
        "pck2": float(np.mean(arr <= 2.0)),
        "pck4": float(np.mean(arr <= 4.0)),
        "pck8": float(np.mean(arr <= 8.0)),
        "matched_objects": int(matched_objects),
        "match_recall": float(matched_objects / max(total_gt, 1)),
    }


def compute_detection_metrics(
    all_dets,
    all_gts,
    num_classes: int = NUM_CLASSES,
    image_sizes: Optional[Sequence[Tuple[int, int]]] = None,
):
    map50, ap50 = _compute_ap_at_iou(all_dets, all_gts, num_classes, 0.50)
    map75, ap75 = _compute_ap_at_iou(all_dets, all_gts, num_classes, 0.75)
    corner = compute_corner_metrics(
        all_dets,
        all_gts,
        image_sizes=image_sizes,
        image_size=IMG_SIZE,
    )

    return {

        "map50": map50,

        "map75": map75,

        "ap50_per_class": ap50,

        "ap75_per_class": ap75,

        **corner,

    }



# -----------------------------------------------------------------------------

# Smoke test*

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    model = build_model(pretrained=False, feat_c=128, refine_c=64, n_bifpn=1)

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    heatmap, corners, offset = model(dummy)

    print("heatmap", tuple(heatmap.shape))

    print("corners", tuple(corners.shape))

    print("offset", tuple(offset.shape))

    fake_batch = [

        [

            {

                "cls": 0,

                "corners": torch.tensor(

                    [[0.30, 0.40], [0.36, 0.38], [0.37, 0.44], [0.31, 0.46]],

                    dtype=torch.float32,

                ),

            }

        ]

    ]

    loss, breakdown = combined_loss_v2(heatmap, corners, offset, fake_batch)

    print("loss", float(loss), breakdown)

    dets = decode_detections(heatmap, corners, offset, conf_thresh=0.0, topk=2)

    print("dets", len(dets[0]))