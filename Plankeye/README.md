# Train_Kaggle_plank_Detector

Entraînement du modèle **PlankEye** pour la détection de planches et de leurs coins, avec deux environnements d'exécution clairement séparés :

- **Kaggle** : entraînement prévu principalement sur **2 × NVIDIA T4** avec DDP.
- **Lightning AI** : entraînement prévu principalement sur **1 × NVIDIA T4**, avec accumulation de gradients pour conserver un batch effectif équivalent.

Le dépôt contient le code du modèle, les notebooks d'entraînement, les checkpoints et les utilitaires nécessaires pour reprendre un entraînement existant.

---

# 0. Cloner GIT dans Kaggle

from kaggle_secrets import UserSecretsClient
import subprocess
import os

token = UserSecretsClient().get_secret("GITHUB_TOKEN")

os.chdir("/kaggle/working")

subprocess.run(["rm", "-rf", "Train_Kaggle_plank_Detector"])

url = f"https://maaxxe:{token}@github.com/maaxxe/Train_Kaggle_plank_Detector.git"

subprocess.run(
    ["git", "clone", url],
    check=True
)

os.chdir("/kaggle/working/Train_Kaggle_plank_Detector")

subprocess.run(["git", "status"])
subprocess.run(["git", "log", "--oneline", "-5"])



%cd /kaggle/working/Train_Kaggle_plank_Detector
!ls

# Atente:

/kaggle/working
dataset  model_improved.py  README.md  train_plankeye_lightning.py
model	 model_poids	    train      voir_epoch.py


Skip to
content

    Home

    Competitions

    Benchmarks

    Game Arena

    Data Hub

    More

​

Draft saved
Draft Session
No Accelerator
Session
3m
12 hours
Disk
347.8MiB
Max 57.6GiB
CPU
CPU
0.00%
RAM
607.7MiB
Max 30GiB
Notebook
Input
DATASETS

    checkpoints

Output (203.7MiB / 19.5GiB)

    /kaggle/working

Table of contents
No sections detected

Add markdown headers to add a section
Session options
Accelerator

Language
Persistence
Environment

You won't get new packages, but your code is less likely to break. What is a notebook environment?
Internet
Internet on
Tags
Dependency Manager

Some accelerators, such as TPUs, require identity verification. Verify identity
GitHub

Upload this ipynb to GitHub under your GitHub account . This can only be undone directly on GitHub.
​
File

Schedule a notebook to run

Schedule this notebook to run and save a new version on a future date. View all your scheduled notebooks.
Trigger

Something went wrong please try again later.




## 1. Structure du dépôt

```text
Train_Kaggle_plank_Detector/
├── README.md
├── dataset/
│   ├── PlankEye_Colab.ipynb
│   ├── model_improved.py
│   └── dataset_fusionne/
│       └── dataset_fusionne/
│           ├── images/
│           └── labels/
├── model/
│   └── model.py
├── model_poids/
│   ├── best_plankeye_v4_1class_512.pt
│   └── last_plankeye_v4_1class_512.pt
├── train/
│   ├── PlankEye_Training_2xT4_Kaggle.ipynb
│   └── PlankEye_Training_Lightning.ipynb
└── voir_epoch.py
```

Le dossier `dataset/` n'est pas nécessairement versionné entièrement sur GitHub : les images représentent plusieurs Go. Le dataset complet peut être récupéré directement depuis Kaggle.

---

## 2. Dataset

Dataset Kaggle utilisé :

```text
max778/plankeye
```

Taille observée :

```text
environ 6.2 Go une fois téléchargé/décompressé
```

Contenu attendu :

```text
dataset/
└── dataset_fusionne/
    └── dataset_fusionne/
        ├── images/
        └── labels/
```

État actuellement attendu :

```text
5137 images
5137 labels
```

Chaque image doit avoir un fichier de label associé.

### Vérification rapide

Depuis la racine du dépôt :

```bash
du -sh dataset

find dataset -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  | wc -l

find dataset -type f -iname "*.txt" | wc -l
```

Résultat attendu :

```text
5137
5137
```

---

# PARTIE A — KAGGLE

## 3. Entraînement sur Kaggle

Notebook principal :

```text
train/PlankEye_Training_2xT4_Kaggle.ipynb
```

Cette version est conçue pour l'environnement Kaggle et utilise les chemins de type :

```text
/kaggle/input/
/kaggle/working/
```

Elle est distincte de la version Lightning AI.

### Configuration GPU

Dans Kaggle :

1. Ouvrir le notebook.
2. Aller dans les paramètres de la session.
3. Activer l'accélérateur GPU.
4. Sélectionner **GPU T4 x2** lorsque cette configuration est disponible.
5. Vérifier dans le notebook :

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

Configuration attendue pour le notebook Kaggle :

```text
GPU 0 : T4
GPU 1 : T4
WORLD_SIZE = 2
BATCH_SIZE = 8 par GPU
batch global = 16
```

Le notebook Kaggle utilise DDP avec deux processus :

```text
rank 0 -> GPU 0
rank 1 -> GPU 1
```

---

## 4. Ajouter le dataset dans Kaggle

Le dataset principal est :

```text
max778/plankeye
```

Le dataset de checkpoints persistants est :

```text
max778/checkpoints
```

Dans un notebook Kaggle, ajouter les datasets dans la section **Input / Add Input**.

Le notebook peut ensuite rechercher les données sous :

```text
/kaggle/input/
```

Selon le montage créé par Kaggle, le chemin peut être court ou complètement qualifié. Le notebook doit donc éviter de dépendre d'un seul chemin codé en dur lorsqu'une recherche récursive est déjà prévue.

Exemple de structure :

```text
/kaggle/input/
└── ...
    └── plankeye/
        └── dataset_fusionne/
            └── dataset_fusionne/
                ├── images/
                └── labels/
```

---

## 5. Checkpoints sur Kaggle

Les deux fichiers principaux sont :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
```

Rôle :

- `best_...pt` : meilleur modèle obtenu selon la métrique de qualité.
- `last_...pt` : dernier état complet de l'entraînement, utilisé pour reprendre après un arrêt.

Dataset Kaggle utilisé pour leur persistance :

```text
max778/checkpoints
```

Le notebook peut sauvegarder automatiquement ces checkpoints vers le dataset Kaggle à intervalles réguliers.

Une sauvegarde manuelle peut aussi être déclenchée après l'arrêt de l'entraînement.

### Vérifier l'epoch d'un checkpoint

Depuis le dépôt :

```bash
python voir_epoch.py
```

Ou en Python :

```python
import torch

ckpt = torch.load(
    "model_poids/last_plankeye_v4_1class_512.pt",
    map_location="cpu",
    weights_only=False,
)

print("epoch :", ckpt.get("epoch"))
print("best_epoch :", ckpt.get("best_epoch"))
print("best_quality :", ckpt.get("best_quality"))
```

---

# PARTIE B — LIGHTNING AI

## 6. Principe

Lightning AI utilise un **Studio** Linux persistant.

Le projet utilisé dans ce dépôt est prévu sous :

```text
/teamspace/studios/this_studio/Train_Kaggle_plank_Detector
```

Le workflow recommandé est :

```text
CPU
  |
  | préparation du dépôt + téléchargement du dataset
  v
GPU T4
  |
  | entraînement
  v
checkpoints dans model_poids/
```

Il est préférable de préparer le projet sur CPU et de passer sur GPU seulement lorsque le code, le dataset et les checkpoints sont prêts.

---

## 7. Cloner le dépôt dans Lightning AI

Dans le terminal du Studio :

```bash
cd /teamspace/studios/this_studio

git clone https://github.com/maaxxe/Train_Kaggle_plank_Detector.git

cd Train_Kaggle_plank_Detector

ls
```

Structure minimale attendue :

```text
README.md
model/
model_poids/
train/
voir_epoch.py
```

---

## 8. Importer le dataset Kaggle dans Lightning AI

C'est la méthode recommandée pour éviter d'envoyer manuellement environ 6 Go depuis le PC.

Le dataset peut rester **privé** sur Kaggle. Il faut simplement authentifier le CLI Kaggle dans Lightning.

### 8.1 Installer le CLI Kaggle

```bash
pip install -U kaggle
```

Vérification :

```bash
kaggle --version
```

---

## 9. Créer un token Kaggle

Sur Kaggle :

```text
Settings
-> API Tokens
-> Generate New Token
```

Donner par exemple le nom :

```text
LightningAI-PlankEye
```

Kaggle fournit ensuite un token de type :

```text
KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Ne jamais :

- mettre ce token dans Git ;
- l'écrire dans le README ;
- le committer dans un notebook ;
- le publier dans une capture d'écran.

Si un token est exposé, le révoquer immédiatement puis en générer un nouveau.

---

## 10. Authentifier Lightning auprès de Kaggle

### Méthode 1 — variable d'environnement

Dans le terminal Lightning :

```bash
export KAGGLE_API_TOKEN="TON_TOKEN_KAGGLE"
```

Tester :

```bash
kaggle datasets list --mine
```

Pour ce projet, le résultat doit notamment contenir :

```text
max778/plankeye
max778/checkpoints
```

### Méthode 2 — fichier `access_token`

Pour éviter de refaire l'export à chaque nouveau shell :

```bash
mkdir -p ~/.kaggle

echo "TON_TOKEN_KAGGLE" > ~/.kaggle/access_token

chmod 600 ~/.kaggle/access_token
```

Puis tester :

```bash
kaggle datasets list --mine
```

---

## 11. Télécharger le dataset privé Kaggle dans Lightning

Depuis la racine du dépôt :

```bash
cd /teamspace/studios/this_studio/Train_Kaggle_plank_Detector
```

Créer le dossier :

```bash
mkdir -p dataset
```

Télécharger et décompresser directement le dataset :

```bash
kaggle datasets download \
  -d max778/plankeye \
  -p dataset \
  --unzip
```

Le transfert se fait directement de Kaggle vers Lightning.

Après téléchargement :

```bash
du -sh dataset
```

Puis :

```bash
find dataset -maxdepth 4 -type d | sort
```

Structure attendue :

```text
dataset
dataset/dataset_fusionne
dataset/dataset_fusionne/dataset_fusionne
dataset/dataset_fusionne/dataset_fusionne/images
dataset/dataset_fusionne/dataset_fusionne/labels
```

Vérifier le contenu :

```bash
find dataset -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  | wc -l

find dataset -type f -iname "*.txt" | wc -l
```

Résultat attendu :

```text
5137
5137
```

Le chemin utilisé par le notebook Lightning est donc :

```text
/teamspace/studios/this_studio/Train_Kaggle_plank_Detector/dataset/dataset_fusionne/dataset_fusionne
```

---

## 12. Télécharger les checkpoints Kaggle dans Lightning

Si les checkpoints ne sont pas déjà présents dans Git ou dans `model_poids/`, ils peuvent également être récupérés depuis :

```text
max778/checkpoints
```

Commande :

```bash
cd /teamspace/studios/this_studio/Train_Kaggle_plank_Detector

mkdir -p model_poids

kaggle datasets download \
  -d max778/checkpoints \
  -p model_poids \
  --unzip
```

Vérifier :

```bash
ls -lh model_poids
```

Les fichiers attendus sont :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
```

---

## 13. Passer le Studio Lightning sur GPU T4

Ne pas essayer d'installer `nvidia-smi` manuellement.

Si cette commande :

```bash
nvidia-smi
```

renvoie :

```text
command not found: nvidia-smi
```

le Studio est encore sur CPU.

Dans l'interface Lightning :

```text
Machine
-> Switch to GPU
-> T4
-> Request / Switch
```

Attendre le redémarrage complet du Studio.

Puis vérifier :

```bash
nvidia-smi
```

Et :

```bash
python -c "import torch; \
print('PyTorch:', torch.__version__); \
print('CUDA:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun')"
```

Résultat attendu :

```text
CUDA: True
GPU: Tesla T4
```

---

## 14. Notebook Lightning

Notebook :

```text
train/PlankEye_Training_Lightning.ipynb
```

Il ne faut pas utiliser les chemins `/kaggle/...` dans cette version.

Chemins principaux :

```text
Studio :
/teamspace/studios/this_studio

Projet :
/teamspace/studios/this_studio/Train_Kaggle_plank_Detector

Dataset :
/teamspace/studios/this_studio/Train_Kaggle_plank_Detector/dataset/dataset_fusionne/dataset_fusionne

Modèle :
/teamspace/studios/this_studio/Train_Kaggle_plank_Detector/model/model.py

Poids :
/teamspace/studios/this_studio/Train_Kaggle_plank_Detector/model_poids
```

---

## 15. Lancer l'entraînement sur Lightning

Exécuter les cellules du notebook dans l'ordre.

Avant le lancement, vérifier impérativement :

```text
CUDA disponible : True
Nombre de GPU : 1
```

Puis vérifier le dataset :

```text
images : 5137
labels : 5137
```

Puis vérifier le checkpoint `last` :

```text
last_plankeye_v4_1class_512.pt
epoch        : ...
best_epoch   : ...
best_quality : ...
```

Le notebook doit afficher l'epoch à partir duquel il va reprendre.

Ne pas lancer l'entraînement si le checkpoint chargé n'est pas celui attendu.

---

## 16. Batch Lightning vs Kaggle

### Kaggle — 2 × T4

Configuration prévue :

```text
BATCH_SIZE = 8 par GPU
WORLD_SIZE = 2
GRAD_ACCUM = 1

batch global = 8 × 2 = 16
```

### Lightning — 1 × T4

Configuration prévue :

```text
BATCH_SIZE = 8
WORLD_SIZE = 1
GRAD_ACCUM = 2

batch effectif = 8 × 2 = 16
```

L'accumulation de gradients permet donc de conserver un batch effectif de 16 avec un seul GPU.

---

## 17. Dépendances

Vérification rapide :

```bash
python -c "import torch, torchvision, numpy, matplotlib, PIL, tqdm; print('DEPENDANCES OK')"
```

Si nécessaire :

```bash
pip install -U \
  torch \
  torchvision \
  numpy \
  matplotlib \
  pillow \
  tqdm \
  kaggle
```

Éviter de réinstaller PyTorch inutilement sur une machine GPU si l'environnement Lightning possède déjà une version CUDA fonctionnelle.

Toujours vérifier ensuite :

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 18. Problème d'import du modèle

Le dépôt contient :

```text
model/
└── model.py
```

L'import suivant peut être incorrect :

```python
from model import NUM_CLASSES, IMG_SIZE, build_model
```

Utiliser :

```python
from model.model import NUM_CLASSES, IMG_SIZE, build_model
```

si le code importe directement le module présent dans `model/model.py`.

Pour rechercher les imports incorrects :

```bash
grep -Rni "from model import" .
```

---

## 19. Vérifier le dernier epoch

Commande :

```bash
python voir_epoch.py
```

Ou directement :

```python
import torch

path = "model_poids/last_plankeye_v4_1class_512.pt"

ckpt = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

print("epoch        :", ckpt.get("epoch"))
print("best_epoch   :", ckpt.get("best_epoch"))
print("best_quality :", ckpt.get("best_quality"))
```

---

## 20. Sauvegarde des checkpoints

Pendant l'entraînement, conserver au minimum :

```text
model_poids/
├── best_plankeye_v4_1class_512.pt
└── last_plankeye_v4_1class_512.pt
```

Le fichier `last` est indispensable pour reprendre un entraînement interrompu.

Le fichier `best` permet de conserver la meilleure version obtenue indépendamment du dernier epoch.

---

## 21. Envoyer les checkpoints vers le dataset Kaggle

Dataset :

```text
max778/checkpoints
```

Le workflow du notebook peut effectuer automatiquement cette opération.

Pour une sauvegarde manuelle, préparer un dossier contenant :

```text
best_plankeye_v4_1class_512.pt
last_plankeye_v4_1class_512.pt
dataset-metadata.json
```

Exemple de `dataset-metadata.json` :

```json
{
  "title": "checkpoints",
  "id": "max778/checkpoints",
  "licenses": [
    {
      "name": "other"
    }
  ]
}
```

Puis créer une nouvelle version :

```bash
kaggle datasets version \
  -p checkpoints_persistent \
  -m "PlankEye checkpoint update" \
  --delete-old-versions
```

Avant cela, vérifier :

```bash
kaggle datasets list --mine
```

---

## 22. Arrêter et reprendre un entraînement

La reprise doit toujours se faire à partir de :

```text
last_plankeye_v4_1class_512.pt
```

Workflow :

```text
entraînement
    |
    v
last checkpoint
    |
    +--> arrêt de la session
    |
    v
nouvelle session
    |
    v
chargement du last checkpoint
    |
    v
epoch suivant
```

Avant chaque reprise :

```bash
python voir_epoch.py
```

---

## 23. Commandes utiles

### Voir l'espace disque

```bash
df -h
du -sh dataset
du -sh model_poids
```

### Voir le GPU

```bash
nvidia-smi
```

Surveiller en continu :

```bash
watch -n 1 nvidia-smi
```

### Vérifier CUDA avec PyTorch

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### Voir les fichiers du projet

Sans `tree` :

```bash
find . -maxdepth 3 -type f | sort
```

Installer `tree` si souhaité :

```bash
sudo apt update
sudo apt install tree -y
tree -L 3
```

### Voir l'état Git

```bash
git status
```

### Récupérer les dernières modifications

```bash
git pull
```

---

## 24. Résumé Kaggle / Lightning AI

| Élément             | Kaggle                                  | Lightning AI                                  |
| --------------------- | --------------------------------------- | --------------------------------------------- |
| Notebook              | `PlankEye_Training_2xT4_Kaggle.ipynb` | `PlankEye_Training_Lightning.ipynb`         |
| Chemins               | `/kaggle/...`                         | `/teamspace/studios/this_studio/...`        |
| GPU prévu            | 2 × T4                                 | 1 × T4                                       |
| Batch/GPU             | 8                                       | 8                                             |
| Accumulation          | 1                                       | 2                                             |
| Batch effectif/global | 16                                      | 16                                            |
| Dataset               | Input Kaggle                            | téléchargement via Kaggle CLI               |
| Dataset ID            | `max778/plankeye`                     | `max778/plankeye`                           |
| Checkpoints           | `max778/checkpoints`                  | `model_poids/` + sauvegarde Kaggle possible |
| DDP                   | oui, 2 GPU                              | automatique si plusieurs GPU                  |
| Reprise               | `last_...pt`                          | `last_...pt`                                |

---

## 25. Installation Lightning rapide de zéro

Résumé complet pour repartir sur un nouveau Studio :

```bash
cd /teamspace/studios/this_studio

git clone https://github.com/maaxxe/Train_Kaggle_plank_Detector.git

cd Train_Kaggle_plank_Detector

pip install -U kaggle

export KAGGLE_API_TOKEN="TON_TOKEN_KAGGLE"

kaggle datasets list --mine

mkdir -p dataset

kaggle datasets download \
  -d max778/plankeye \
  -p dataset \
  --unzip

find dataset -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  | wc -l

find dataset -type f -iname "*.txt" | wc -l

python voir_epoch.py
```

Ensuite :

```text
Lightning UI
-> Machine
-> Switch to GPU
-> T4
```

Puis :

```bash
nvidia-smi

python -c "import torch; \
print(torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'aucun')"
```

Enfin ouvrir :

```text
train/PlankEye_Training_Lightning.ipynb
```

et exécuter les cellules dans l'ordre.

---

## 26. Sécurité

Ne jamais versionner :

```text
KAGGLE_API_TOKEN
KGAT_...
~/.kaggle/access_token
~/.kaggle/kaggle.json
```

Vérifier `.gitignore` si nécessaire.

Exemple :

```gitignore
.kaggle/
*.token
.env
.env.*
```

Les datasets et checkpoints lourds doivent de préférence être stockés dans Kaggle ou un stockage dédié plutôt que dans l'historique Git classique.

---

## 27. Workflow recommandé

```text
GitHub
  |
  | code
  v
Kaggle / Lightning
  |
  | téléchargement dataset max778/plankeye
  v
Préparation
  |
  | vérification images / labels / checkpoints
  v
GPU
  |
  | Kaggle : 2 × T4
  | Lightning : 1 × T4
  v
Entraînement
  |
  +--> last checkpoint
  |
  +--> best checkpoint
  |
  v
max778/checkpoints
```

Ce découpage permet de garder :

- le **code** dans GitHub ;
- le **dataset lourd** dans Kaggle ;
- les **checkpoints persistants** dans Kaggle ;
- l'environnement d'entraînement interchangeable entre Kaggle et Lightning AI.
