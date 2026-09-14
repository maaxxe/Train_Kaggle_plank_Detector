from __future__ import annotations

"""PlankEye V5.2 model.

Detection scheme:
- ResNet18 backbone + lightweight FPN.
- Center heatmap at stride 2.
- Absolute normalized quadrilateral corners (4 x 2 values in [0, 1]).
- Sub-cell center offset in [-0.5, 0.5].

V5.2 intentionally changes the corner target semantics from the older V5 model.
Old V5 checkpoints must NOT be reused for training/inference with this file.
"""

from typing import Dict, List
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

MODEL_VERSION = "v5.2-absolute-corners" 
IMG_SIZE = 512
PRED_STRIDE = 2
HEATMAP_SIZE = IMG_SIZE // PRED_STRIDE
NUM_CLASSES = 1
NUM_CORNERS = 4

# Cyclic permutations + reversed cyclic permutations.  A quadrilateral can use
# any starting corner and either winding direction without changing its shape.
CORNER_PERMUTATIONS = torch.tensor(
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


class CoordConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_c + 2, out_c, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return self.conv(torch.cat([x, grid], dim=1))


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s2 = self.stem(x)
        s4 = self.layer1(F.max_pool2d(s2, 2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)
        return {"s2": s2, "s4": s4, "s8": s8, "s16": s16, "s32": s32}


class PlankEyeV5(nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        c = 128

        self.lat_s4 = nn.Conv2d(64, c, kernel_size=1)
        self.lat_s8 = nn.Conv2d(128, c, kernel_size=1)
        self.lat_s16 = nn.Conv2d(256, c, kernel_size=1)
        self.lat_s32 = nn.Conv2d(512, c, kernel_size=1)

        self.refiner = nn.Sequential(
            nn.Conv2d(c + 64, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        self.hmap_head = nn.Sequential(
            CoordConv2d(c, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, NUM_CLASSES, kernel_size=1),
        )
        self.corn_head = nn.Sequential(
            CoordConv2d(c, 128, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, NUM_CLASSES * 8, kernel_size=1),
        )
        self.offs_head = nn.Sequential(
            CoordConv2d(c, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, NUM_CLASSES * 2, kernel_size=1),
        )

        # CenterNet-style low initial foreground probability (~1%).
        nn.init.constant_(self.hmap_head[-1].bias, -4.59)

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)

        p4 = self.lat_s4(f["s4"])
        p8 = F.interpolate(self.lat_s8(f["s8"]), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p16 = F.interpolate(self.lat_s16(f["s16"]), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p32 = F.interpolate(self.lat_s32(f["s32"]), size=p4.shape[-2:], mode="bilinear", align_corners=False)

        p = p4 + p8 + p16 + p32
        p = F.interpolate(p, size=f["s2"].shape[-2:], mode="bilinear", align_corners=False)
        fused = self.refiner(torch.cat([p, f["s2"]], dim=1))

        hmap_logits = self.hmap_head(fused)
        # IMPORTANT: raw logits here. Sigmoid is applied only where needed.
        corner_logits = self.corn_head(fused)
        # A center offset is relative to the center of the selected heatmap cell.
        center_offsets = 0.5 * torch.tanh(self.offs_head(fused))
        return hmap_logits, corner_logits, center_offsets


# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

def draw_gaussian(heatmap: torch.Tensor, cx_i: int, cy_i: int, sigma: float = 2.0) -> None:
    h, w = heatmap.shape
    radius = int(3.0 * sigma)
    x0 = max(0, cx_i - radius)
    x1 = min(w, cx_i + radius + 1)
    y0 = max(0, cy_i - radius)
    y1 = min(h, cy_i + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy = torch.arange(y0, y1, device=heatmap.device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(x0, x1, device=heatmap.device, dtype=torch.float32).view(1, -1)
    g = torch.exp(-((xx - cx_i) ** 2 + (yy - cy_i) ** 2) / (2.0 * sigma * sigma))
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], g)


def build_targets_gpu(batch_objects, device):
    bsz = len(batch_objects)
    h = HEATMAP_SIZE
    w = HEATMAP_SIZE

    hmap = torch.zeros(bsz, NUM_CLASSES, h, w, device=device)
    corner_target = torch.zeros(bsz, NUM_CLASSES, 4, 2, h, w, device=device)
    offset_target = torch.zeros(bsz, NUM_CLASSES, 2, h, w, device=device)
    positive_mask = torch.zeros(bsz, NUM_CLASSES, h, w, device=device)
    best_dist = torch.full((bsz, NUM_CLASSES, h, w), float("inf"), device=device)

    for b, objects in enumerate(batch_objects):
        for obj in objects:
            cls = int(obj["cls"])
            if cls < 0 or cls >= NUM_CLASSES:
                continue

            corners = torch.as_tensor(obj["corners"], dtype=torch.float32, device=device).reshape(4, 2)
            corners = corners.clamp(0.0, 1.0)
            center = corners.mean(dim=0)

            # Keep centers inside the valid heatmap range even for coordinates 1.0.
            cx = torch.clamp(center[0] * w, 0.0, w - 1e-4)
            cy = torch.clamp(center[1] * h, 0.0, h - 1e-4)
            cx_i = int(torch.floor(cx).item())
            cy_i = int(torch.floor(cy).item())

            draw_gaussian(hmap[b, cls], cx_i, cy_i, sigma=2.0)

            # In the extremely rare case where two centers land in the same cell,
            # keep the object whose true center is closest to that cell center.
            dist = (cx - (cx_i + 0.5)) ** 2 + (cy - (cy_i + 0.5)) ** 2
            if dist >= best_dist[b, cls, cy_i, cx_i]:
                continue
            best_dist[b, cls, cy_i, cx_i] = dist

            positive_mask[b, cls, cy_i, cx_i] = 1.0
            offset_target[b, cls, 0, cy_i, cx_i] = cx - (cx_i + 0.5)
            offset_target[b, cls, 1, cy_i, cx_i] = cy - (cy_i + 0.5)

            # V5.2: absolute normalized corners. No clipping radius and no
            # dependence on the object's center position.
            corner_target[b, cls, :, :, cy_i, cx_i] = corners

    return hmap, corner_target, offset_target, positive_mask


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def focal_heatmap_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    pos_mask = target.eq(1.0)
    neg_mask = target.lt(1.0)
    neg_weights = (1.0 - target).pow(4)

    pos_loss = torch.log(pred) * (1.0 - pred).pow(2) * pos_mask
    neg_loss = torch.log(1.0 - pred) * pred.pow(2) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    if num_pos > 0:
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return -neg_loss.sum() / neg_mask.sum().clamp(min=1.0)


def _positive_corner_tensors(pred_corner_logits, target_corners, positive_mask):
    b, c, _, _, h, w = target_corners.shape
    pred = torch.sigmoid(pred_corner_logits).view(b, c, 4, 2, h, w)

    # CRITICAL: move spatial dimensions before (corner, xy) BEFORE boolean
    # indexing.  Indexing a [B,C,4,2,H,W] tensor directly with an expanded
    # mask mixes different objects whenever one image contains several
    # positives.  [B,C,H,W,4,2][positive_mask] correctly returns [N,4,2].
    pred_at_cells = pred.permute(0, 1, 4, 5, 2, 3).contiguous()
    target_at_cells = target_corners.permute(0, 1, 4, 5, 2, 3).contiguous()
    mask = positive_mask.bool()
    if mask.sum() == 0:
        return pred, None, None
    pred_pos = pred_at_cells[mask]
    target_pos = target_at_cells[mask]
    return pred, pred_pos, target_pos


def permutation_corner_loss(pred_corner_logits, target_corners, positive_mask):
    pred, pred_pos, target_pos = _positive_corner_tensors(pred_corner_logits, target_corners, positive_mask)
    if pred_pos is None:
        return pred.sum() * 0.0

    perms = CORNER_PERMUTATIONS.to(pred_pos.device)
    losses = []
    for perm in perms:
        target_perm = target_pos[:, perm, :]
        # Small beta: nearly L1 for meaningful pixel errors, smoother near zero.
        loss = F.smooth_l1_loss(pred_pos, target_perm, reduction="none", beta=0.02).mean(dim=(1, 2))
        losses.append(loss)
    return torch.stack(losses, dim=1).min(dim=1).values.mean()


def offset_loss(pred_offsets, target_offsets, positive_mask):
    pred = pred_offsets.view_as(target_offsets)
    mask = positive_mask.unsqueeze(2)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    loss = F.smooth_l1_loss(pred, target_offsets, reduction="none", beta=0.1) * mask
    return loss.sum() / (mask.sum() * 2.0 + 1e-6)


def polygon_area_torch(points: torch.Tensor) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    return 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1, dims=-1) - y * torch.roll(x, -1, dims=-1), dim=-1))


def geometry_loss(pred_corner_logits, target_corners, positive_mask):
    pred, pred_pos, target_pos = _positive_corner_tensors(pred_corner_logits, target_corners, positive_mask)
    if pred_pos is None:
        return pred.sum() * 0.0

    perms = CORNER_PERMUTATIONS.to(pred_pos.device)
    corner_losses = []
    permuted_targets = []
    for perm in perms:
        target_perm = target_pos[:, perm, :]
        permuted_targets.append(target_perm)
        corner_losses.append(
            F.smooth_l1_loss(pred_pos, target_perm, reduction="none", beta=0.02).mean(dim=(1, 2))
        )
    corner_losses = torch.stack(corner_losses, dim=1)
    best_idx = corner_losses.argmin(dim=1)
    all_targets = torch.stack(permuted_targets, dim=1)
    row = torch.arange(pred_pos.size(0), device=pred_pos.device)
    best_targets = all_targets[row, best_idx]

    pred_area = polygon_area_torch(pred_pos)
    target_area = polygon_area_torch(best_targets)
    area_loss = F.smooth_l1_loss(pred_area, target_area, beta=0.02)

    pred_edges = torch.roll(pred_pos, shifts=-1, dims=1) - pred_pos
    target_edges = torch.roll(best_targets, shifts=-1, dims=1) - best_targets
    pred_dirs = F.normalize(pred_edges, dim=-1, eps=1e-6)
    target_dirs = F.normalize(target_edges, dim=-1, eps=1e-6)
    direction_loss = (1.0 - (pred_dirs * target_dirs).sum(dim=-1)).mean()

    # Penalize self-crossing / non-convex corner ordering by requiring all
    # consecutive cross products to share the dominant winding sign.
    cross_values = []
    for i in range(4):
        a = pred_edges[:, i]
        b = pred_edges[:, (i + 1) % 4]
        cross_values.append(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    cross_values = torch.stack(cross_values, dim=1)
    orientation = torch.sign(cross_values.sum(dim=1, keepdim=True).detach())
    orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
    convexity_loss = F.relu(1e-4 - orientation * cross_values).mean()

    return 1.0 * area_loss + 0.5 * direction_loss + 0.25 * convexity_loss


def combined_loss_v5(preds, batch_objects, device):
    hmap_p, corner_p, offsets_p = preds
    hmap_t, corner_t, offset_t, positive_mask = build_targets_gpu(batch_objects, device)

    loss_hmap = focal_heatmap_loss(hmap_p, hmap_t)
    loss_corner = permutation_corner_loss(corner_p, corner_t, positive_mask)
    loss_offset = offset_loss(offsets_p, offset_t, positive_mask)
    loss_geometry = geometry_loss(corner_p, corner_t, positive_mask)

    total = 1.0 * loss_hmap + 8.0 * loss_corner + 1.0 * loss_offset + 2.0 * loss_geometry
    logs = {
        "total": float(total.detach().item()),
        "heatmap": float(loss_hmap.detach().item()),
        "corner": float(loss_corner.detach().item()),
        "offset": float(loss_offset.detach().item()),
        "geometry": float(loss_geometry.detach().item()),
    }
    return total, logs


# -----------------------------------------------------------------------------
# Polygon helpers + real quadrilateral NMS
# -----------------------------------------------------------------------------

def _order_quad_np(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(ang)]


def _polygon_area_np(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    subject = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    if len(subject) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=np.float64)

    signed = 0.5 * np.sum(clip[:, 0] * np.roll(clip[:, 1], -1) - clip[:, 1] * np.roll(clip[:, 0], -1))
    clip_pts = clip if signed >= 0 else clip[::-1]

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def intersect(p1, p2, a, b):
        d1 = p2 - p1
        d2 = b - a
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return p2
        t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / den
        return p1 + t * d1

    output = [p.copy() for p in subject]
    for i in range(len(clip_pts)):
        if not output:
            break
        a = clip_pts[i]
        b = clip_pts[(i + 1) % len(clip_pts)]
        input_list = output
        output = []
        prev = input_list[-1]
        prev_in = inside(prev, a, b)
        for cur in input_list:
            cur_in = inside(cur, a, b)
            if cur_in:
                if not prev_in:
                    output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(intersect(prev, cur, a, b))
            prev = cur
            prev_in = cur_in
    return np.asarray(output, dtype=np.float64) if output else np.empty((0, 2), dtype=np.float64)


def _is_convex_quad_np(pts: np.ndarray, eps: float = 1e-10) -> bool:
    pts = _order_quad_np(pts)
    cross = []
    for i in range(4):
        e1 = pts[(i + 1) % 4] - pts[i]
        e2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        cross.append(e1[0] * e2[1] - e1[1] * e2[0])
    pos = all(v > eps for v in cross)
    neg = all(v < -eps for v in cross)
    return pos or neg


def quad_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    a = _order_quad_np(a)
    b = _order_quad_np(b)
    if not _is_convex_quad_np(a) or not _is_convex_quad_np(b):
        return 0.0
    area_a = _polygon_area_np(a)
    area_b = _polygon_area_np(b)
    if area_a <= 1e-12 or area_b <= 1e-12:
        return 0.0
    inter_poly = _clip_polygon(a, b)
    inter = _polygon_area_np(inter_poly)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-12 else 0.0


def _polygon_nms(detections: List[dict], iou_thresh: float) -> List[dict]:
    if iou_thresh <= 0 or len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: List[dict] = []
    for det in ordered:
        quad = np.asarray(det["corners"], dtype=np.float64)
        if all(quad_iou_np(quad, np.asarray(k["corners"], dtype=np.float64)) < iou_thresh for k in kept):
            kept.append(det)
    return kept


@torch.no_grad()
def decode_predictions(
    hmap: torch.Tensor,
    corner_logits: torch.Tensor,
    offsets: torch.Tensor,
    score_thresh: float = 0.3,
    topk: int = 50,
    nms_iou: float = 0.5,
):
    bsz, classes, h, w = hmap.shape
    prob = hmap.sigmoid()
    keep_map = prob.eq(F.max_pool2d(prob, kernel_size=3, stride=1, padding=1))
    prob = prob * keep_map
    corner_prob = torch.sigmoid(corner_logits)

    results = []
    for b in range(bsz):
        detections = []
        for c in range(classes):
            flat = prob[b, c].flatten()
            k = min(topk, flat.numel())
            scores, idx = torch.topk(flat, k)
            keep = scores >= score_thresh
            scores = scores[keep]
            idx = idx[keep]

            for score, flat_idx in zip(scores.tolist(), idx.tolist()):
                cy_i = flat_idx // w
                cx_i = flat_idx % w
                off_x = float(offsets[b, c * 2, cy_i, cx_i].item())
                off_y = float(offsets[b, c * 2 + 1, cy_i, cx_i].item())

                quad = corner_prob[b, c * 8:(c + 1) * 8, cy_i, cx_i].view(4, 2)
                centroid = quad.mean(dim=0)
                angles = torch.atan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])
                quad = quad[torch.argsort(angles)].clamp(0.0, 1.0)

                detections.append(
                    {
                        "score": float(score),
                        "class": c,
                        "corners": quad.cpu().numpy(),
                        "center": (
                            float((cx_i + 0.5 + off_x) / w),
                            float((cy_i + 0.5 + off_y) / h),
                        ),
                    }
                )

        detections = _polygon_nms(detections, iou_thresh=nms_iou)
        results.append(detections)
    return results
