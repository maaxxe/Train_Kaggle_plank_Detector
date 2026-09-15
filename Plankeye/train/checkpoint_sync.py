from __future__ import annotations

"""
Synchronise les checkpoints PlankEye V7 vers un Dataset Kaggle existant.

Exemple :
    python checkpoint_sync.py \
        --run-dir /kaggle/working/PlankEyev2_multipieces/runs/plankeye_v7 \
        --dataset max778/chekpoints-backbone50 \
        --epoch 5

Le script crée un snapshot stable des fichiers avec des hardlinks quand possible.
Ainsi, train_v7.py peut continuer à remplacer last.pt pendant que Kaggle lit le
snapshot sans risque de mélanger deux versions du checkpoint.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
print("====================Checkpoint_Sync_v7=====================")

SMALL_FILES = (
    "dataset_split.json",
    "history.csv",
    "history.json",
    "loss_curve.png",
    "model_v7.py",
    "run_config.json",
    "source_git.json",
    "train_v7.py",
)


def human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def stable_link_or_copy(src: Path, dst: Path) -> None:
    """Create a stable snapshot cheaply, falling back to a real copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_kaggle_cli() -> str:
    executable = shutil.which("kaggle")
    if executable:
        return executable

    raise FileNotFoundError(
        "Commande 'kaggle' introuvable. "
        "Le Kaggle CLI doit être installé/configuré dans le notebook."
    )


def prepare_snapshot(
    run_dir: Path,
    stage_dir: Path,
    dataset: str,
    epoch: int | None,
) -> list[Path]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []

    # Essential resume checkpoints.
    for name in ("last.pt", "best.pt"):
        src = run_dir / name
        if src.exists():
            stable_link_or_copy(src, stage_dir / name)
            selected.append(stage_dir / name)

    # Keep only the checkpoint corresponding to this periodic upload.
    # Previous periodic checkpoints remain available in older Kaggle Dataset versions.
    if epoch is not None:
        periodic = run_dir / f"epoch_{epoch:03d}.pt"
        if periodic.exists():
            stable_link_or_copy(periodic, stage_dir / periodic.name)
            selected.append(stage_dir / periodic.name)

    for name in SMALL_FILES:
        src = run_dir / name
        if src.exists():
            stable_link_or_copy(src, stage_dir / name)
            selected.append(stage_dir / name)

    if not (stage_dir / "last.pt").exists():
        raise FileNotFoundError(
            f"Aucun last.pt trouvé dans {run_dir}. "
            "Il n'y a pas encore de checkpoint terminé à sauvegarder."
        )

    sync_info = {
        "dataset": dataset,
        "epoch": epoch,
        "source_run_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "size": human_size(path),
            }
            for path in selected
        },
    }

    (stage_dir / "sync_info.json").write_text(
        json.dumps(sync_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    selected.append(stage_dir / "sync_info.json")

    return selected


def fetch_dataset_metadata(
    kaggle_cli: str,
    dataset: str,
    stage_dir: Path,
) -> None:
    """Prepare metadata for versioning an existing Kaggle Dataset.

    Some Kaggle CLI versions successfully download dataset-metadata.json but
    omit the `id`/slug field required by `kaggle datasets version`.  Therefore
    we fetch the official metadata when possible, then *always* enforce the
    existing dataset slug locally before versioning.
    """
    metadata_path = stage_dir / "dataset-metadata.json"

    command = [
        kaggle_cli,
        "datasets",
        "metadata",
        dataset,
        "-p",
        str(stage_dir),
    ]

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    metadata = {}

    if result.returncode == 0 and metadata_path.exists():
        try:
            loaded = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if isinstance(loaded, dict):
                metadata = loaded
        except Exception:
            metadata = {}

    # CRITICAL: `kaggle datasets version` requires one of `id` or `id_no`.
    # Force the canonical owner/slug even if the installed CLI omitted it.
    metadata["id"] = dataset

    # `title` is not required for a version, but keeping a valid fallback makes
    # the metadata easier to inspect and remains compatible with older CLIs.
    if not metadata.get("title"):
        metadata["title"] = dataset.split("/", 1)[-1]

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Final local validation before trying a ~1 GB upload.
    check = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not check.get("id") and not check.get("id_no"):
        raise RuntimeError(
            "dataset-metadata.json invalide : aucun 'id' ou 'id_no'."
        )

    if check.get("id") != dataset and not check.get("id_no"):
        raise RuntimeError(
            "dataset-metadata.json pointe vers un autre dataset : "
            f"{check.get('id')!r} != {dataset!r}"
        )


def upload_version(
    *,
    run_dir: Path,
    dataset: str,
    epoch: int | None,
    message: str | None,
) -> None:
    kaggle_cli = find_kaggle_cli()

    label = f"epoch_{epoch:03d}" if epoch is not None else "manual_stop"
    stage_dir = run_dir / "_kaggle_upload" / label

    print()
    print("=" * 88)
    print("KAGGLE CHECKPOINT SYNC")
    print("=" * 88)
    print(f"Dataset : {dataset}")
    print(f"Run     : {run_dir}")
    print(f"Snapshot: {stage_dir}")
    print(f"Epoch   : {epoch if epoch is not None else 'dernier checkpoint terminé'}")

    files = prepare_snapshot(
        run_dir=run_dir,
        stage_dir=stage_dir,
        dataset=dataset,
        epoch=epoch,
    )

    fetch_dataset_metadata(
        kaggle_cli=kaggle_cli,
        dataset=dataset,
        stage_dir=stage_dir,
    )

    print("Fichiers snapshot :")
    for path in files:
        print(f"  - {path.name:<24} {human_size(path)}")

    if message is None:
        if epoch is None:
            message = "PlankEye V7 - sauvegarde après arrêt manuel"
        else:
            message = f"PlankEye V7 - checkpoint epoch {epoch}"

    command = [
        kaggle_cli,
        "datasets",
        "version",
        "-p",
        str(stage_dir),
        "-m",
        message,
        "-q",
        "-r",
        "skip",
    ]

    print()
    print("Upload Kaggle en cours...")
    start = time.time()

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    duration = time.time() - start

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"Upload Kaggle échoué avec le code {result.returncode}."
        )

    print(
        f"✅ Dataset Kaggle mis à jour en {duration / 60.0:.2f} min : "
        f"{dataset}"
    )
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload PlankEye V7 checkpoints to an existing Kaggle Dataset."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--message", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_version(
        run_dir=args.run_dir.resolve(),
        dataset=args.dataset.strip(),
        epoch=args.epoch,
        message=args.message,
    )


if __name__ == "__main__":
    main()
