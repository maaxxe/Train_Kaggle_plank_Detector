# GAN_PlankEye v2 — Kaggle / PyTorch

Version PyTorch/CUDA du GAN conditionnel multi-planches.

## Dataset source attendu

```text
dataset/
├── images/
│   ├── image0.jpg
│   └── ...
└── labels/
    ├── image0.txt
    └── ...
```

Format d'une ligne de label:

```text
0 x1 y1 x2 y2 x3 y3 x4 y4
```

Les coordonnées sont normalisées entre 0 et 1.

## Préparation

```bash
python prepare_dataset.py \
  --source /kaggle/input/.../dataset_fusionne \
  --out /kaggle/working/GAN_PlankEye_v2/data/paired \
  --size 512 \
  --overwrite
```

Sans `--source`, le script recherche automatiquement dans `/kaggle/input`.

## Entraînement 2 x T4

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data /kaggle/working/GAN_PlankEye_v2/data/paired \
  --out /kaggle/working/GAN_PlankEye_v2/runs/plankgan_multi_512 \
  --epochs 150 \
  --batch 4 \
  --workers 2 \
  --lambda-l1 50
```

## Reprise

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data /kaggle/working/GAN_PlankEye_v2/data/paired \
  --out /kaggle/working/GAN_PlankEye_v2/runs/plankgan_multi_512 \
  --epochs 150 \
  --batch 4 \
  --workers 2 \
  --lambda-l1 50 \
  --resume /kaggle/working/GAN_PlankEye_v2/runs/plankgan_multi_512/checkpoints/last.pt
```

## Génération

```bash
python generate.py \
  --checkpoint /kaggle/working/GAN_PlankEye_v2/runs/plankgan_multi_512/checkpoints/best.pt \
  --prepared /kaggle/working/GAN_PlankEye_v2/data/paired \
  --out /kaggle/working/GAN_PlankEye_v2/generated_v2_512 \
  --count 1000 \
  --size 512 \
  --min-planks 1 \
  --max-planks 7 \
  --min-gap 8 \
  --max-overlap 0
```
