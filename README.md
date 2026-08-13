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

| Métrique | Description |
|---|---|
| `mAP50` | qualité de détection avec IoU 0.50 |
| `mAP75` | qualité de détection avec IoU 0.75 |
| `MAE` | erreur moyenne sur les coins |
| `RMSE` | erreur quadratique moyenne sur les coins |
| `PCK2` | coins à moins de 2 px |
| `PCK4` | coins à moins de 4 px |
| `PCK8` | coins à moins de 8 px |
| `match_recall` | proportion d'objets correctement associés |
| `quality` | score global utilisé pour choisir le meilleur checkpoint |

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
