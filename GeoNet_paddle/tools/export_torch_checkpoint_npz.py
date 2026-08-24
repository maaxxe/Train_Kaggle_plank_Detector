# -*- coding: utf-8 -*-
"""
Exporte un checkpoint GeoNet PyTorch (.pt) en NPZ lisible par la version Paddle.

A EXECUTER SUR UNE MACHINE OU PYTORCH FONCTIONNE
(local, Kaggle, Lightning...), PAS sur Baidu AI Studio.

Exemple :
    python tools/export_torch_checkpoint_npz.py \
        --checkpoint ../GeoNet/model_poids/last_multiforme.pt \
        --output model_poids/last_multiforme_torch.npz

Le fichier .meta.json associe contient epoch, best_iou et history.
L'optimizer PyTorch n'est PAS converti.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint GeoNet .pt existant.",
    )

    parser.add_argument(
        "--imagenet-backbone",
        action="store_true",
        help=(
            "Au lieu d'un checkpoint GeoNet, exporte les poids "
            "ImageNet de torchvision EfficientNetV2-S pour l'encodeur."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    return parser.parse_args()


def clean_state_dict(
    state,
):
    output = {}

    for key, value in state.items():
        if key.startswith(
            "module."
        ):
            key = key[
                len(
                    "module."
                ):
            ]

        if key.endswith(
            ".num_batches_tracked"
        ):
            continue

        if not torch.is_tensor(
            value
        ):
            continue

        output[
            key
        ] = (
            value.detach()
            .cpu()
            .numpy()
        )

    return output


def main():
    args = parse_args()

    if bool(
        args.checkpoint
    ) == bool(
        args.imagenet_backbone
    ):
        raise ValueError(
            "Choisis exactement une source : --checkpoint OU --imagenet-backbone."
        )

    output_path = Path(
        args.output
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "source": "",
        "kind": "",
        "epoch": None,
        "best_iou": None,
        "history": [],
    }

    if args.checkpoint:
        checkpoint_path = Path(
            args.checkpoint
        ).resolve()

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state = checkpoint.get(
            "model",
            checkpoint.get(
                "model_state_dict",
                checkpoint,
            ),
        )

        arrays = clean_state_dict(
            state
        )

        metadata[
            "source"
        ] = str(
            checkpoint_path
        )
        metadata[
            "kind"
        ] = "geonet_checkpoint"

        metadata[
            "epoch"
        ] = int(
            checkpoint.get(
                "epoch",
                checkpoint.get(
                    "current_epoch",
                    0,
                ),
            )
        )

        best = checkpoint.get(
            "best_iou",
            checkpoint.get(
                "best_quality",
                None,
            ),
        )

        metadata[
            "best_iou"
        ] = (
            None
            if best is None
            else float(best)
        )

        history = checkpoint.get(
            "history",
            [],
        )

        if isinstance(
            history,
            list,
        ):
            metadata[
                "history"
            ] = history

    else:
        from torchvision.models import (
            EfficientNet_V2_S_Weights,
            efficientnet_v2_s,
        )

        model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        state = {}

        for key, value in (
            model.features.state_dict().items()
        ):
            # GeoNet n'utilise que features 0..6.
            first = key.split(
                ".",
                1,
            )[0]

            if (
                first.isdigit()
                and int(first) <= 6
            ):
                state[
                    "encoder.features."
                    + key
                ] = value

        arrays = clean_state_dict(
            state
        )

        metadata[
            "source"
        ] = (
            "torchvision EfficientNet_V2_S_Weights.DEFAULT"
        )
        metadata[
            "kind"
        ] = "imagenet_backbone"

    np.savez_compressed(
        output_path,
        **arrays,
    )

    meta_path = output_path.with_suffix(
        ".meta.json"
    )

    meta_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    size_mb = (
        output_path.stat().st_size
        / 1024**2
    )

    print(
        "=" * 72
    )

    print(
        "EXPORT PYTORCH -> NPZ"
    )

    print(
        "=" * 72
    )

    print(
        "Tensors :",
        len(arrays),
    )

    print(
        "NPZ     :",
        output_path,
    )

    print(
        "Meta    :",
        meta_path,
    )

    print(
        "Taille  :",
        f"{size_mb:.2f} MB",
    )

    print(
        "Epoch   :",
        metadata[
            "epoch"
        ],
    )

    print(
        "Best IoU:",
        metadata[
            "best_iou"
        ],
    )


if __name__ == "__main__":
    main()
