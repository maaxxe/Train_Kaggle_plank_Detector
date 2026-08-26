from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset import PairedPlankDataset
from models import PatchDiscriminator, UNetGenerator, init_weights
from utils import save_triplet_grid, seed_everything


def ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0

    return distributed, rank, local_rank, world_size


def cleanup(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def split_indices(n, val_ratio, seed):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return train_idx, val_idx


@torch.no_grad()
def validate(generator, loader, device, latent_channels, amp_enabled):
    generator.eval()

    l1_sum = torch.tensor(0.0, device=device)
    n_sum = torch.tensor(0.0, device=device)

    for batch in loader:
        real = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        # Zéro bruit pour rendre la validation comparable epoch après epoch.
        noise = torch.zeros(
            real.size(0),
            latent_channels,
            real.size(2),
            real.size(3),
            device=device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            fake = generator(mask, noise)
            loss = torch.mean(torch.abs(fake - real))

        l1_sum += loss * real.size(0)
        n_sum += real.size(0)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(l1_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)

    generator.train()
    return (l1_sum / n_sum.clamp_min(1)).item()


def save_checkpoint(
    path,
    epoch,
    generator,
    discriminator,
    opt_g,
    opt_d,
    scaler_g,
    scaler_d,
    best_val,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "generator": unwrap(generator).state_dict(),
            "discriminator": unwrap(discriminator).state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
            "scaler_g": scaler_g.state_dict(),
            "scaler_d": scaler_d.state_dict(),
            "best_val_l1": best_val,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4, help="batch PAR GPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.10)

    parser.add_argument("--latent-channels", type=int, default=3)
    parser.add_argument("--base", type=int, default=64)

    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-l1", type=float, default=50.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()

    distributed, rank, local_rank, world_size = ddp_setup()

    seed_everything(args.seed + rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible. Active un GPU Kaggle.")

    device = torch.device(f"cuda:{local_rank}")
    amp_enabled = True

    out = Path(args.out)
    ckpt_dir = out / "checkpoints"
    sample_dir = out / "samples"

    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

    full_train = PairedPlankDataset(args.data, augment=True)
    full_val = PairedPlankDataset(args.data, augment=False)

    train_idx, val_idx = split_indices(
        len(full_train),
        args.val_ratio,
        args.seed,
    )

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )

    val_sampler = (
        DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=False,
    )

    G = UNetGenerator(
        latent_channels=args.latent_channels,
        base=args.base,
    ).to(device)

    D = PatchDiscriminator(base=args.base).to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    if distributed:
        G = DDP(
            G,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        D = DDP(
            D,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    opt_g = torch.optim.Adam(
        G.parameters(),
        lr=args.lr_g,
        betas=(args.beta1, args.beta2),
    )

    opt_d = torch.optim.Adam(
        D.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, args.beta2),
    )

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {resume_path}")

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        unwrap(G).load_state_dict(checkpoint["generator"])
        unwrap(D).load_state_dict(checkpoint["discriminator"])

        opt_g.load_state_dict(checkpoint["optimizer_g"])
        opt_d.load_state_dict(checkpoint["optimizer_d"])

        if "scaler_g" in checkpoint:
            scaler_g.load_state_dict(checkpoint["scaler_g"])
        if "scaler_d" in checkpoint:
            scaler_d.load_state_dict(checkpoint["scaler_d"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_l1", float("inf")))

        if is_main(rank):
            print(f"Reprise depuis: {resume_path}")
            print(f"Prochaine epoch: {start_epoch}")
            print(f"Best val L1    : {best_val:.6f}")

    if is_main(rank):
        print("=" * 80)
        print("GAN_PLANK EYE v2 — KAGGLE / PYTORCH")
        print("=" * 80)
        print(f"GPU(s)          : {world_size}")
        print(f"Train / Val     : {len(train_ds)} / {len(val_ds)}")
        print(f"Batch / GPU     : {args.batch}")
        print(f"Batch global    : {args.batch * world_size}")
        print(f"Epochs          : {args.epochs}")
        print(f"Lambda L1       : {args.lambda_l1}")
        print(f"AMP             : {amp_enabled}")
        print(f"Sortie          : {out}")
        print("=" * 80)

    history = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            G.train()
            D.train()

            epoch_g = 0.0
            epoch_d = 0.0
            epoch_adv = 0.0
            epoch_l1 = 0.0
            n_batches = 0

            t0 = time.time()

            for step, batch in enumerate(train_loader, start=1):
                real = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                noise = torch.randn(
                    real.size(0),
                    args.latent_channels,
                    real.size(2),
                    real.size(3),
                    device=device,
                )

                # -------------------------------------------------
                # 1. DISCRIMINATEUR
                # -------------------------------------------------
                opt_d.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    with torch.no_grad():
                        fake_detached = G(mask, noise)

                    pred_real = D(mask, real)
                    pred_fake = D(mask, fake_detached)

                    loss_d_real = bce(
                        pred_real,
                        torch.ones_like(pred_real),
                    )
                    loss_d_fake = bce(
                        pred_fake,
                        torch.zeros_like(pred_fake),
                    )

                    loss_d = 0.5 * (loss_d_real + loss_d_fake)

                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

                # -------------------------------------------------
                # 2. GÉNÉRATEUR
                # -------------------------------------------------
                opt_g.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    fake = G(mask, noise)
                    pred_fake_for_g = D(mask, fake)

                    loss_adv = bce(
                        pred_fake_for_g,
                        torch.ones_like(pred_fake_for_g),
                    )

                    loss_recon = l1(fake, real)
                    loss_g = loss_adv + args.lambda_l1 * loss_recon

                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()

                epoch_g += loss_g.detach().item()
                epoch_d += loss_d.detach().item()
                epoch_adv += loss_adv.detach().item()
                epoch_l1 += loss_recon.detach().item()
                n_batches += 1

                if is_main(rank) and (
                    step == 1
                    or step % 20 == 0
                    or step == len(train_loader)
                ):
                    print(
                        f"\r[{epoch:03d}/{args.epochs}] "
                        f"[{step:04d}/{len(train_loader):04d}] "
                        f"G={loss_g.item():.4f} "
                        f"D={loss_d.item():.4f} "
                        f"ADV={loss_adv.item():.4f} "
                        f"L1={loss_recon.item():.4f}",
                        end="",
                        flush=True,
                    )

            val_l1 = validate(
                G,
                val_loader,
                device,
                args.latent_channels,
                amp_enabled,
            )

            # Moyenne train multi-GPU.
            stats = torch.tensor(
                [
                    epoch_g,
                    epoch_d,
                    epoch_adv,
                    epoch_l1,
                    float(n_batches),
                ],
                device=device,
            )

            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            denom = max(stats[4].item(), 1.0)
            mean_g = stats[0].item() / denom
            mean_d = stats[1].item() / denom
            mean_adv = stats[2].item() / denom
            mean_l1 = stats[3].item() / denom

            elapsed = time.time() - t0

            if is_main(rank):
                print()
                print(
                    f"    train: G={mean_g:.4f} | D={mean_d:.4f} "
                    f"| ADV={mean_adv:.4f} | L1={mean_l1:.4f} "
                    f"| val L1={val_l1:.4f} | {elapsed/60:.1f} min"
                )

                # Preview avec le premier batch de validation.
                preview_batch = next(iter(val_loader))
                preview_real = preview_batch["image"].to(device)
                preview_mask = preview_batch["mask"].to(device)

                preview_noise = torch.randn(
                    preview_real.size(0),
                    args.latent_channels,
                    preview_real.size(2),
                    preview_real.size(3),
                    device=device,
                )

                G.eval()
                with torch.no_grad(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    preview_fake = G(preview_mask, preview_noise)

                save_triplet_grid(
                    preview_mask.cpu(),
                    preview_fake.cpu(),
                    preview_real.cpu(),
                    sample_dir / f"epoch_{epoch:03d}.jpg",
                    max_items=4,
                )
                G.train()

                # BEST = plus petite erreur L1 validation.
                improved = val_l1 < best_val
                if improved:
                    best_val = val_l1

                # LAST contient toujours la meilleure valeur connue,
                # ce qui permet une reprise propre après interruption.
                save_checkpoint(
                    ckpt_dir / "last.pt",
                    epoch,
                    G,
                    D,
                    opt_g,
                    opt_d,
                    scaler_g,
                    scaler_d,
                    best_val,
                    args,
                )

                if improved:
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        epoch,
                        G,
                        D,
                        opt_g,
                        opt_d,
                        scaler_g,
                        scaler_d,
                        best_val,
                        args,
                    )

                    print(
                        f"    -> nouveau BEST: val L1 = {best_val:.6f}"
                    )

                history.append(
                    {
                        "epoch": epoch,
                        "train_g": mean_g,
                        "train_d": mean_d,
                        "train_adv": mean_adv,
                        "train_l1": mean_l1,
                        "val_l1": val_l1,
                        "seconds": elapsed,
                    }
                )

                (out / "history.json").write_text(
                    json.dumps(history, indent=2),
                    encoding="utf-8",
                )

    except KeyboardInterrupt:
        if is_main(rank):
            print(
                "\nInterruption manuelle détectée. "
                "best.pt et last.pt des epochs déjà terminées restent sauvegardés."
            )
        raise
    finally:
        cleanup(distributed)


if __name__ == "__main__":
    main()
