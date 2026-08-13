# PlankEye — Entraînement Kaggle 2× T4

Entraînement du modèle **PlankEye** pour la détection de planches et l'estimation de leurs **4 coins**, avec :

- dépôt GitHub **privé** ;
- récupération automatique du modèle depuis GitHub ;
- entraînement **DistributedDataParallel (DDP)** sur **2× Tesla T4** ;
- reprise automatique depuis le dernier checkpoint ;
- sauvegarde persistante des checkpoints dans un dataset Kaggle privé ;
- sauvegarde automatique tous les 5 epochs ;
- sauvegarde manuelle possible à tout moment.

---

## Structure du dépôt

```text
Train_Kaggle_plank_Detector/
├── model/
│   └── model.py
├── train/
│   └── PlankEye_Training_2xT4_Kaggle_PRIVATE_GITHUB_FINAL.ipynb
└── README.md
```

### Fichiers principaux

- `model/model.py` : architecture du modèle PlankEye.
- `train/PlankEye_Training_2xT4_Kaggle_PRIVATE_GITHUB_FINAL.ipynb` : notebook Kaggle complet pour cloner le dépôt privé, préparer les données, récupérer les checkpoints, reprendre l'entraînement et lancer le DDP sur 2 GPU.

---

# Architecture du modèle

Le modèle utilise notamment :

- **MobileNetV3-Large** comme backbone ;
- **BiFPN** pour fusionner les features multi-échelles ;
- un **refiner haute résolution** ;
- une tête de **heatmap** ;
- une tête de prédiction des **4 coins** ;
- une tête d'**offset du centre** ;
- des losses géométriques adaptées aux quadrilatères ;
- **AMP / FP16** ;
- **EMA** ;
- **AdamW**.

Le modèle actuel travaille avec une seule classe :

```text
planche
```

---

# Dataset

Le dataset attendu contient :

```text
dataset_fusionne/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels/
    ├── image001.txt
    ├── image002.txt
    └── ...
```

Chaque ligne de label suit le format :

```text
classe x1 y1 x2 y2 x3 y3 x4 y4
```

Les coordonnées sont normalisées entre `0` et `1`.

Dataset Kaggle utilisé :

```text
max778/plankeye
```

---

# Checkpoints

Deux checkpoints sont utilisés :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
```

Dataset Kaggle privé utilisé :

```text
max778/checkpoints
```

## `best_...pt`

Contient le meilleur modèle obtenu selon le score `quality`.

```python
quality = (
    0.3 * mAP50
    + 0.5 * mAP75
    + 0.2 * (PCK4 * match_recall)
)
```

## `last_...pt`

Contient le dernier epoch terminé et permet de reprendre exactement l'entraînement.

Exemple :

```text
epoch sauvegardé : 26
prochain epoch    : 27
best_epoch        : 25
best_quality      : 0.8453918484740378
```

Le notebook sélectionne automatiquement :

- pour `last.pt` : le checkpoint avec l'epoch le plus avancé ;
- pour `best.pt` : le checkpoint avec la meilleure `best_quality`.

Cela évite qu'un ancien checkpoint Kaggle écrase un checkpoint local plus récent.

---

# GitHub privé

Le dépôt est prévu pour rester **privé**.

Kaggle récupère automatiquement le code avec un **GitHub Personal Access Token** enregistré dans les **Kaggle Secrets**.

## Secret Kaggle

Créer dans Kaggle un secret nommé exactement :

```text
GITHUB_TOKEN
```

Le token ne doit jamais être écrit directement dans le notebook, dans le README ou dans Git.

Le token GitHub doit idéalement être limité à :

```text
Repository access:
Only select repositories

Repository:
Train_Kaggle_plank_Detector

Permission:
Contents -> Read-only
```

---

# Configuration Kaggle

Avant de lancer le notebook :

## 1. GPU

```text
Accelerator : GPU T4 x2
```

Le notebook vérifie que 2 GPU sont disponibles avant de lancer le DDP.

## 2. Internet

```text
Internet : On
```

Internet est nécessaire pour :

- cloner le dépôt GitHub privé ;
- envoyer les nouvelles versions du dataset de checkpoints vers Kaggle.

## 3. Ajouter les Inputs

Ajouter dans **Input** :

```text
max778/plankeye
max778/checkpoints
```

Les chemins ressemblent ensuite à :

```text
/kaggle/input/datasets/max778/plankeye
/kaggle/input/datasets/max778/checkpoints
```

## 4. Activer le secret GitHub

Dans Kaggle :

```text
Add-ons
→ Secrets
→ GITHUB_TOKEN
→ ON
```

---

# Fonctionnement du notebook

Le notebook suit cet ordre :

```text
1. Vérification GPU / CUDA / Internet / Kaggle CLI
                    ↓
2. Clone du dépôt GitHub privé
                    ↓
3. Copie model/model.py vers model_improved.py
                    ↓
4. Préparation du dataset
                    ↓
5. Sélection de best.pt / last.pt
                    ↓
6. Vérification de l'epoch de reprise
                    ↓
7. Génération du script DDP
                    ↓
8. Lancement GPU 0 + GPU 1
                    ↓
9. Entraînement
                    ↓
10. Sauvegarde locale + persistante
```

Le dépôt GitHub est cloné dans :

```text
/kaggle/working/Train_Kaggle_plank_Detector
```

Le modèle :

```text
model/model.py
```

est automatiquement copié vers :

```text
/kaggle/working/PlankEyev2_multipieces/model_improved.py
```

Le script d'entraînement continue donc d'utiliser l'import historique `model_improved` sans modifier l'architecture.

---

# Paramètres principaux

Configuration actuelle :

```python
IMG_SIZE = 512

BATCH_SIZE = 8
WORLD_SIZE = 2
GRAD_ACCUM = 1

EPOCHS = 180
```

Batch global :

```text
8 images/GPU × 2 GPU × 1 accumulation
= 16 images
```

---

# Learning rate

```python
LR_HEAD = 3.0e-4
LR_BACKBONE = 2.0e-5

LR_MIN_HEAD = 2.0e-6
LR_MIN_BACKBONE = 2.0e-7
```

Warmup :

```python
HEAD_WARMUP_EPOCHS = 3
BACKBONE_WARMUP_EPOCHS = 3
```

Déblocage progressif du backbone :

```text
Epoch 0–7   : backbone gelé
Epoch 8     : déblocage des 6 derniers blocs
Epoch 20+   : backbone entièrement dégelé
```

---

# DDP — 2× Tesla T4

Le notebook lance deux processus :

```text
rank 0 -> GPU 0
rank 1 -> GPU 1
```

Initialisation distribuée :

```text
backend = nccl
init_method = FileStore
```

Le modèle est enveloppé avec :

```python
DistributedDataParallel(
    model,
    device_ids=[LOCAL_RANK],
    output_device=LOCAL_RANK,
    broadcast_buffers=False,
    find_unused_parameters=True,
    gradient_as_bucket_view=True,
)
```

`find_unused_parameters=True` est nécessaire avec l'architecture actuelle car certains paramètres ne participent pas à chaque passe forward.

Il ne faut pas lancer directement :

```bash
python train_plankeye_ddp.py
```

car cela démarrerait un seul processus et donc un seul GPU.

Il faut utiliser la cellule de lancement DDP prévue dans le notebook.

Au démarrage, vérifier :

```text
world_size=2
batch/GPU=8
accum=1
batch global=16
```

et non :

```text
world_size=1
```

---

# Validation

L'entraînement est distribué sur les deux GPU.

La validation est effectuée uniquement par le **rank 0**.

Pendant la validation, il est donc normal de voir le GPU 1 peu utilisé ou à 0 %.

---

# Sauvegarde automatique

Les checkpoints sont d'abord sauvegardés dans :

```text
/kaggle/working/PlankEyev2_multipieces/
```

avec :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
```

Une copie est préparée dans :

```text
/kaggle/working/checkpoints_persistent/
```

Puis une nouvelle version du dataset Kaggle :

```text
max778/checkpoints
```

est créée automatiquement.

Fréquence :

```python
PERSIST_EVERY = 5
```

Donc sauvegarde persistante aux epochs :

```text
5
10
15
20
25
30
...
```

et également lors d'un early stopping.

---

# Sauvegarde manuelle

Le notebook contient également une cellule permettant de pousser manuellement :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
```

vers :

```text
max778/checkpoints
```

Elle peut être utilisée avant de fermer une session Kaggle.

---

# Reprise de l'entraînement

Le notebook lit automatiquement :

```python
ckpt["epoch"]
ckpt["best_epoch"]
ckpt["best_quality"]
```

Exemple :

```text
LAST ACTUEL
========================================================================
epoch sauvegardé : 26
prochain epoch    : 27
best_epoch        : 25
best_quality      : 0.8453918484740378
```

Dans ce cas l'entraînement reprend à :

```text
epoch 27
```

---

# Workflow de développement

Sur le PC / WSL :

```bash
cd ~/Train_Kaggle_plank_Detector

git add .
git commit -m "Update PlankEye"
git push origin main
```

Puis, dans une nouvelle session Kaggle, la cellule GitHub clone automatiquement la dernière version du dépôt privé.

```text
PC / WSL
   ↓
modification model.py / notebook
   ↓
git push
   ↓
GitHub privé
   ↓
Kaggle + GITHUB_TOKEN
   ↓
git clone
   ↓
entraînement 2× T4
   ↓
best.pt / last.pt
   ↓
dataset privé max778/checkpoints
```

---

# Fichiers à ne pas mettre dans Git

Les checkpoints et datasets ne doivent pas être versionnés dans GitHub.

Exemple `.gitignore` :

```gitignore
*.pt
*.pth
*.ckpt

__pycache__/
*.pyc

.ipynb_checkpoints/

output/
checkpoints_persistent/

.DS_Store
```

GitHub contient le **code**.

Kaggle contient :

- les **données** ;
- les **checkpoints** ;
- les ressources temporaires d'entraînement.

---

# Notebook

Notebook actuel :

```text
train/PlankEye_Training_2xT4_Kaggle_PRIVATE_GITHUB_FINAL.ipynb
```

À lancer dans Kaggle avec :

```text
GPU T4 x2
Internet On
GITHUB_TOKEN activé
max778/plankeye ajouté
max778/checkpoints ajouté
```

Puis exécuter les cellules dans l'ordr

# PlankEye — Entraînement Kaggle 2× T4

Entraînement du modèle **PlankEye** pour la détection de planches et de leurs coins, avec reprise de checkpoint et entraînement multi-GPU sur **Kaggle 2× Tesla T4**.

## Structure du dépôt

```text
Train_Kaggle_plank_Detector/
├── model/
│   └── model.py
├── train/
│   └── PlankEye_Training_2xT4_Kaggle.ipynb
└── README.md
```

- `model/model.py` : architecture du modèle PlankEye.
- `train/PlankEye_Training_2xT4_Kaggle.ipynb` : notebook Kaggle complet pour préparer les données, reprendre un checkpoint et entraîner avec 2 GPU.

---

## Objectif

Le modèle détecte les planches présentes dans une image et estime leurs **4 coins**.

L'entraînement utilise notamment :

- **MobileNetV3-Large**
- **BiFPN**
- refiner haute résolution
- heatmaps
- offsets
- losses géométriques sur les coins
- AMP / FP16
- EMA
- AdamW
- entraînement **DistributedDataParallel (DDP)** sur 2 GPU

---

## Environnement Kaggle

Dans Kaggle :

1. Créer ou ouvrir un Notebook.
2. Ajouter le dataset contenant les images et labels.
3. Ajouter le dataset contenant les checkpoints.
4. Sélectionner :

```text
Accelerator : GPU T4 x2
Internet    : ON
```

5. Importer le notebook :

```text
train/PlankEye_Training_2xT4_Kaggle.ipynb
```

6. Exécuter les cellules dans l'ordre.

---

## Dataset

Le notebook attend un dataset de la forme :

```text
dataset_fusionne/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels/
    ├── image001.txt
    ├── image002.txt
    └── ...
```

Chaque ligne de label contient :

```text
classe x1 y1 x2 y2 x3 y3 x4 y4
```

Les coordonnées sont normalisées entre `0` et `1`.

Le modèle actuel travaille en **1 classe** :

```text
planche
```

---

## Paramètres principaux

Configuration actuelle :

```python
IMG_SIZE = 512
BATCH_SIZE = 8       # par GPU
WORLD_SIZE = 2
GRAD_ACCUM = 1
```

Le batch global vaut donc :

```text
8 × 2 GPU = 16 images
```

L'entraînement peut reprendre automatiquement depuis un checkpoint existant.

---

## Checkpoints

Deux fichiers sont utilisés :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
```

### `best_...pt`

Contient le meilleur modèle obtenu selon le score `quality`.

### `last_...pt`

Contient le dernier epoch terminé et permet de reprendre l'entraînement exactement là où il s'est arrêté.

Exemple :

```text
epoch sauvegardé : 24
prochain epoch    : 25
```

---

## Score `quality`

Le meilleur checkpoint est sélectionné avec :

```text
quality =
0.30 × mAP50
+ 0.50 × mAP75
+ 0.20 × (PCK4 × match_recall)
```

Le score privilégie donc :

- la qualité globale de détection ;
- la précision géométrique ;
- la précision des coins.

---

## Sauvegarde persistante Kaggle

Les checkpoints sont sauvegardés localement dans :

```text
/kaggle/working/PlankEyev2_multipieces/
```

Le notebook peut aussi envoyer automatiquement les checkpoints vers le dataset Kaggle :

```text
max778/checkpoints
```

La sauvegarde persistante est effectuée périodiquement, par exemple tous les **5 epochs**.

Les deux fichiers envoyés sont :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
```

Une sauvegarde manuelle peut également être déclenchée depuis le notebook.

---

## Reprise de l'entraînement

Le notebook recherche les checkpoints existants et sélectionne le plus récent.

Au lancement, vérifier que la sortie ressemble à :

```text
world_size=2 | batch/GPU=8 | accum=1 | batch global=16
Reprise checkpoint : epoch 25
```

Si la sortie affiche :

```text
world_size=1
```

alors le notebook n'utilise pas les deux GPU et il faut relancer la cellule DDP prévue dans le notebook.

---

## Exemple de sortie

```text
TRAIN 23/180: 100%|██████████| 257/257
VAL 23/180:   100%|██████████| 33/33

[023/180]
train=0.2559
val=0.2644
mAP50=0.9884
mAP75=0.9244
PCK4=0.360
match=0.995
quality=0.8304
```

---

## Métriques principales

| Métrique        | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `mAP50`        | qualité de détection avec IoU 0.50                      |
| `mAP75`        | qualité de détection avec IoU 0.75                      |
| `MAE`          | erreur moyenne sur les coins                              |
| `RMSE`         | erreur quadratique moyenne sur les coins                  |
| `PCK2`         | coins à moins de 2 px                                    |
| `PCK4`         | coins à moins de 4 px                                    |
| `PCK8`         | coins à moins de 8 px                                    |
| `match_recall` | proportion d'objets correctement associés                |
| `quality`      | score global utilisé pour choisir le meilleur checkpoint |

---

## Entraînement multi-GPU

Le notebook lance un processus par GPU avec PyTorch DDP :

```text
Rank 0 -> GPU 0
Rank 1 -> GPU 1
```

Configuration attendue :

```text
world_size=2
batch/GPU=8
accum=1
batch global=16
```

Le modèle utilise également :

```python
find_unused_parameters=True
```

pour gérer les paramètres qui ne participent pas à toutes les branches de loss.

---

## Dépendances principales

- Python
- PyTorch
- torchvision
- NumPy
- Pillow
- matplotlib
- tqdm
- Kaggle CLI

Kaggle fournit déjà la majorité de ces dépendances.

---

## Lancement

Le plus simple est d'utiliser directement :

```text
train/PlankEye_Training_2xT4_Kaggle.ipynb
```

dans Kaggle puis d'exécuter les cellules dans l'ordre.

Le notebook :

1. vérifie les GPU ;
2. prépare le dataset ;
3. sélectionne le checkpoint le plus récent ;
4. prépare le modèle ;
5. écrit le script d'entraînement ;
6. vérifie sa syntaxe ;
7. lance DDP sur les 2 T4 ;
8. sauvegarde `best.pt` et `last.pt` ;
9. permet une sauvegarde persistante/manuelle des checkpoints.

---

## Notes

Le fichier `model/model.py` doit correspondre au fichier importé par le notebook.
Si le notebook attend un nom différent, adapter le chemin ou le nom copié dans `/kaggle/working`.

Les checkpoints `.pt` ne sont pas inclus dans ce dépôt Git afin d'éviter de versionner des fichiers binaires volumineux.
