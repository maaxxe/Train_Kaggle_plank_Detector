"""
PlankEye V5 — model.py
-----------------------
Backbone : ResNet18 pretrained
Neck     : FPN léger (s4 + s8 + s16 + s32)
Heads    : Heatmap + 4 coins (offset local, tanh) + Offset centre
Loss     : Heatmap focal + permutation-invariant corners
           + offset + géométrie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


# =============================================================================
# PARAMÈTRES PARTAGÉS
# =============================================================================

IMG_SIZE = 512
PRED_STRIDE = 2
HEATMAP_SIZE = IMG_SIZE // PRED_STRIDE

NUM_CLASSES = 1
NUM_CORNERS = 4

CORNER_CLIP_RADIUS = HEATMAP_SIZE / 2.0  # = 128.0, utilisé au codage ET au décodage


# =============================================================================
# PERMUTATIONS DES COINS
# =============================================================================

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
    dtype=torch.long
)


# =============================================================================
# COORD CONV
# =============================================================================

class CoordConv2d(nn.Module):

    def __init__(self, in_c, out_c, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_c + 2, out_c, **kwargs)

    def forward(self, x):
        B, _, H, W = x.shape

        yy = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        xx = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)

        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([gx, gy], dim=0)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

        x = torch.cat([x, grid], dim=1)
        return self.conv(x)


# =============================================================================
# BACKBONE RESNET18
# =============================================================================

class ResNet18Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.stem(x)
        s4 = self.layer1(F.max_pool2d(x, 2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)

        return {"s2": x, "s4": s4, "s8": s8, "s16": s16, "s32": s32}


# =============================================================================
# MODÈLE
# =============================================================================

class PlankEyeV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18Backbone()

        C = 128

        self.lat_s4 = nn.Conv2d(64, C, kernel_size=1)
        self.lat_s8 = nn.Conv2d(128, C, kernel_size=1)
        self.lat_s16 = nn.Conv2d(256, C, kernel_size=1)
        self.lat_s32 = nn.Conv2d(512, C, kernel_size=1)

        self.refiner = nn.Sequential(
            nn.Conv2d(C + 64, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True),

            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True)
        )

        self.hmap_head = nn.Sequential(
            CoordConv2d(C, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, NUM_CLASSES, kernel_size=1)
        )

        self.corn_head = nn.Sequential(
            CoordConv2d(C, 128, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, NUM_CLASSES * 8, kernel_size=1)
        )

        self.offs_head = nn.Sequential(
            CoordConv2d(C, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, NUM_CLASSES * 2, kernel_size=1)
        )

        nn.init.constant_(self.hmap_head[-1].bias, -4.59)

    def forward(self, x):
        f = self.backbone(x)

        p4 = self.lat_s4(f["s4"])
        p8 = self.lat_s8(f["s8"])
        p16 = self.lat_s16(f["s16"])
        p32 = self.lat_s32(f["s32"])

        p8 = F.interpolate(p8, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p16 = F.interpolate(p16, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p32 = F.interpolate(p32, size=p4.shape[-2:], mode="bilinear", align_corners=False)

        p = p4 + p8 + p16 + p32
        p = F.interpolate(p, size=f["s2"].shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([p, f["s2"]], dim=1)
        fused = self.refiner(fused)

        hmap = self.hmap_head(fused)
        corners = torch.tanh(self.corn_head(fused))
        offsets = self.offs_head(fused)

        return hmap, corners, offsets


# =============================================================================
# GAUSSIENNE & CIBLES
# =============================================================================

def draw_gaussian(heatmap, cx, cy, sigma=2.0):
    H, W = heatmap.shape
    radius = int(3 * sigma)

    x0 = max(0, int(cx) - radius)
    x1 = min(W, int(cx) + radius + 1)
    y0 = max(0, int(cy) - radius)
    y1 = min(H, int(cy) + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return

    yy = torch.arange(y0, y1, device=heatmap.device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(x0, x1, device=heatmap.device, dtype=torch.float32).view(1, -1)

    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], g)


def build_targets_gpu(batch_objects, device, corner_clip=0.999):
    B = len(batch_objects)
    H = HEATMAP_SIZE
    W = HEATMAP_SIZE

    hmap = torch.zeros(B, NUM_CLASSES, H, W, device=device)
    corner_target = torch.zeros(B, NUM_CLASSES, 4, 2, H, W, device=device)
    offset_target = torch.zeros(B, NUM_CLASSES, 2, H, W, device=device)
    positive_mask = torch.zeros(B, NUM_CLASSES, H, W, device=device)

    best_dist = torch.full((B, NUM_CLASSES, H, W), float("inf"), device=device)

    for b, objects in enumerate(batch_objects):
        for obj in objects:
            cls = obj["cls"]

            if cls < 0 or cls >= NUM_CLASSES:
                continue

            corners = torch.as_tensor(obj["corners"], dtype=torch.float32, device=device)

            center = corners.mean(dim=0)
            cx = center[0] * W
            cy = center[1] * H

            cx_i = int(torch.clamp(cx, 0, W - 1).item())
            cy_i = int(torch.clamp(cy, 0, H - 1).item())

            draw_gaussian(hmap[b, cls], float(cx_i), float(cy_i), sigma=2.0)

            dist = (cx - cx_i) ** 2 + (cy - cy_i) ** 2

            if dist >= best_dist[b, cls, cy_i, cx_i]:
                continue

            best_dist[b, cls, cy_i, cx_i] = dist
            positive_mask[b, cls, cy_i, cx_i] = 1.0

            offset_target[b, cls, 0, cy_i, cx_i] = cx - cx_i
            offset_target[b, cls, 1, cy_i, cx_i] = cy - cy_i

            corners_px = corners.clone()
            corners_px[:, 0] = corners_px[:, 0] * W
            corners_px[:, 1] = corners_px[:, 1] * H

            corner_offset_x = (corners_px[:, 0] - (cx_i + 0.5)) / CORNER_CLIP_RADIUS
            corner_offset_y = (corners_px[:, 1] - (cy_i + 0.5)) / CORNER_CLIP_RADIUS

            corner_offset_x = corner_offset_x.clamp(-corner_clip, corner_clip)
            corner_offset_y = corner_offset_y.clamp(-corner_clip, corner_clip)

            corner_target[b, cls, :, 0, cy_i, cx_i] = corner_offset_x
            corner_target[b, cls, :, 1, cy_i, cx_i] = corner_offset_y

    return hmap, corner_target, offset_target, positive_mask


# =============================================================================
# LOSSES
# =============================================================================

def focal_heatmap_loss(pred, target):
    pred = pred.sigmoid()

    pos_mask = target.eq(1.0)
    neg_mask = target.lt(1.0)
    neg_weights = (1.0 - target).pow(4)

    pos_loss = torch.log(pred.clamp(min=1e-6)) * (1.0 - pred).pow(2) * pos_mask
    neg_loss = torch.log((1.0 - pred).clamp(min=1e-6)) * pred.pow(2) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    if num_pos > 0:
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    num_neg = neg_mask.sum().clamp(min=1.0)
    return -neg_loss.sum() / num_neg


def permutation_corner_loss(pred_corners, target_corners, positive_mask):
    B, C, _, H, W = target_corners.shape
    pred = pred_corners.view(B, C, 4, 2, H, W)
    mask = positive_mask.bool()

    if mask.sum() == 0:
        return pred.sum() * 0.0

    pred_pos = pred[mask].view(-1, 4, 2)
    target_pos = target_corners[mask].view(-1, 4, 2)
    perms = CORNER_PERMUTATIONS.to(pred.device)

    losses = []
    for perm in perms:
        target_perm = target_pos[:, perm, :]
        loss = F.smooth_l1_loss(pred_pos, target_perm, reduction="none").mean(dim=(1, 2))
        losses.append(loss)

    losses = torch.stack(losses, dim=1)
    best_loss = losses.min(dim=1).values
    return best_loss.mean()


def offset_loss(pred_offsets, target_offsets, positive_mask):
    # CORRECTIF: Alignement des shapes de pred_offsets et target_offsets
    pred = pred_offsets.view(target_offsets.shape) 
    mask = positive_mask.unsqueeze(2)

    if mask.sum() == 0:
        return pred.sum() * 0.0

    loss = F.smooth_l1_loss(pred, target_offsets, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() * 2.0 + 1e-6)


def polygon_area_torch(points):
    x = points[..., 0]
    y = points[..., 1]
    return 0.5 * torch.abs(
        torch.sum(x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1), dim=-1)
    )


def geometry_loss(pred_corners, target_corners, positive_mask):
    B, C, _, H, W = target_corners.shape
    pred = pred_corners.view(B, C, 4, 2, H, W)
    mask = positive_mask.bool()

    if mask.sum() == 0:
        return pred.sum() * 0.0

    pred_pos = pred[mask].view(-1, 4, 2)
    target_pos = target_corners[mask].view(-1, 4, 2)
    perms = CORNER_PERMUTATIONS.to(pred.device)

    all_corner_losses = []
    for perm in perms:
        target_perm = target_pos[:, perm, :]
        loss = F.smooth_l1_loss(pred_pos, target_perm, reduction="none").mean(dim=(1, 2))
        all_corner_losses.append(loss)

    all_corner_losses = torch.stack(all_corner_losses, dim=1)
    best_idx = all_corner_losses.argmin(dim=1)
    row_idx = torch.arange(target_pos.size(0), device=target_pos.device).unsqueeze(1)
    best_targets = target_pos[row_idx, perms[best_idx]]

    # Aire
    pred_area = polygon_area_torch(pred_pos)
    target_area = polygon_area_torch(best_targets)
    area_loss = F.smooth_l1_loss(pred_area, target_area)

    # Direction des côtés
    pred_edges = torch.roll(pred_pos, shifts=-1, dims=1) - pred_pos
    target_edges = torch.roll(best_targets, shifts=-1, dims=1) - best_targets
    pred_edges = F.normalize(pred_edges, dim=-1, eps=1e-6)
    target_edges = F.normalize(target_edges, dim=-1, eps=1e-6)
    direction_loss = (1.0 - (pred_edges * target_edges).sum(dim=-1)).mean()

    # Convexité
    cross_values = []
    for i in range(4):
        a = pred_edges[:, i]
        b = pred_edges[:, (i + 1) % 4]
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        cross_values.append(cross)
    cross_values = torch.stack(cross_values, dim=1)
    mean_cross = cross_values.mean(dim=1, keepdim=True)
    convexity_loss = F.relu(-mean_cross * cross_values).mean()

    return 0.5 * area_loss + 0.5 * direction_loss + 0.25 * convexity_loss


def combined_loss_v5(preds, batch_objects, device):
    hmap_p, corn_p, offs_p = preds
    hmap_t, corner_t, offset_t, positive_mask = build_targets_gpu(batch_objects, device)

    loss_hmap = focal_heatmap_loss(hmap_p, hmap_t)
    loss_corner = permutation_corner_loss(corn_p, corner_t, positive_mask)
    loss_offset = offset_loss(offs_p, offset_t, positive_mask)
    loss_geometry = geometry_loss(corn_p, corner_t, positive_mask)

    total = 1.0 * loss_hmap + 5.0 * loss_corner + 1.0 * loss_offset + 1.0 * loss_geometry

    logs = {
        "total": float(total.detach().item()),
        "heatmap": float(loss_hmap.detach().item()),
        "corner": float(loss_corner.detach().item()),
        "offset": float(loss_offset.detach().item()),
        "geometry": float(loss_geometry.detach().item()),
    }

    return total, logs


# =============================================================================
# DÉCODAGE (INFÉRENCE) AVEC NMS
# =============================================================================

@torch.no_grad()
def decode_predictions(hmap, corners, offsets, score_thresh=0.3, topk=50):
    B, C, H, W = hmap.shape
    prob = hmap.sigmoid()

    # CORRECTIF: NMS CenterNet standard via max-pooling 3x3
    keep_map = (prob == F.max_pool2d(prob, kernel_size=3, stride=1, padding=1))
    prob = prob * keep_map

    results = []

    for b in range(B):
        detections = []

        # CORRECTIF: Support du multi-classes
        for c in range(C):
            p = prob[b, c]

            flat = p.flatten()
            k = min(topk, flat.numel())
            scores, idx = torch.topk(flat, k)

            keep = scores > score_thresh
            scores = scores[keep]
            idx = idx[keep]

            for score, flat_idx in zip(scores.tolist(), idx.tolist()):
                cy_i = flat_idx // W
                cx_i = flat_idx % W

                off_x = offsets[b, c*2, cy_i, cx_i].item()
                off_y = offsets[b, c*2 + 1, cy_i, cx_i].item()

                corner_offsets = corners[b, c*8:(c+1)*8, cy_i, cx_i].view(4, 2)

                corner_x = (cx_i + 0.5 + CORNER_CLIP_RADIUS * corner_offsets[:, 0]) / W
                corner_y = (cy_i + 0.5 + CORNER_CLIP_RADIUS * corner_offsets[:, 1]) / H

                quad = torch.stack([corner_x, corner_y], dim=1)

                # CORRECTIF: Tri angulaire déterministe
                centroid = quad.mean(dim=0)
                angles = torch.atan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])
                order = torch.argsort(angles)
                quad = quad[order]

                detections.append({
                    "score": float(score),
                    "class": c,
                    "corners": quad.cpu().numpy(),
                    "center": ((cx_i + off_x) / W, (cy_i + off_y) / H),
                })

        results.append(detections)

    return results