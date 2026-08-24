# GeoNet Paddle — Baidu AI Studio

Cette branche/dossier est le port PaddlePaddle de GeoNet.

## Architecture conservee

- EfficientNetV2-S, niveaux `s2/s4/s8/s16/s32`
- decodeur U-Net
- tete segmentation pleine resolution
- tete heatmap de centres a stride 2
- memes losses : BCE + Dice + boundary Dice + focal centre
- meme post-traitement OpenCV vers les points de contour

Les labels ne changent pas.

## Structure

```text
GeoNet_paddle/
├── model/
│   └── multiforme_model.py
├── train/
│   ├── train_geonet_baidu.py
│   └── Train_GeoNet_Baidu_V100.ipynb
├── tools/
│   └── export_torch_checkpoint_npz.py
└── model_poids/
```

## 1. Si tu veux reprendre les poids de ton GeoNet PyTorch

A faire LOCAL / Kaggle, la ou `torch` fonctionne :

```bash
cd Train_Kaggle_plank_Detector/GeoNet_paddle

python tools/export_torch_checkpoint_npz.py \
  --checkpoint ../GeoNet/model_poids/last_multiforme.pt \
  --output model_poids/last_multiforme_torch.npz
```

Cela produit :

```text
model_poids/
├── last_multiforme_torch.npz
└── last_multiforme_torch.meta.json
```

Commit/push ces fichiers uniquement si leur taille et ta politique GitHub le permettent.
Sinon, envoie-les directement a AI Studio.

IMPORTANT : l'optimizer PyTorch n'est pas convertible directement.
Les poids, epoch, best IoU et history sont repris ; AdamW repart neuf.

## 2. Baidu AI Studio

Verifie :

```python
import paddle

print(paddle.__version__)
print(paddle.is_compiled_with_cuda())
print(paddle.device.cuda.device_count())
print(paddle.device.get_device())
```

Puis lance :

```bash
python /home/aistudio/work/Train_Kaggle_plank_Detector/GeoNet_paddle/train/train_geonet_baidu.py \
  --data /home/aistudio/data/TON_DATASET \
  --output /home/aistudio/work/GeoNet_checkpoints \
  --img-size 640 \
  --epochs 120 \
  --batch 4 \
  --accumulation 4 \
  --workers 2 \
  --lr-head 1e-5 \
  --lr-backbone 5e-7 \
  --init-torch-npz /home/aistudio/work/Train_Kaggle_plank_Detector/GeoNet_paddle/model_poids/last_multiforme_torch.npz
```

Sur 1 V100 :
- batch physique = 4
- accumulation = 4
- batch effectif = 16

Cela correspond au batch effectif du notebook Kaggle :
`4 par GPU × 2 GPU × accumulation 2 = 16`.

## 3. Checkpoints Baidu

Le training cree :

```text
/home/aistudio/work/GeoNet_checkpoints/
├── last_multiforme.pdckpt
├── best_multiforme.pdckpt
└── history.json
```

Pour reprendre plus tard :

```bash
python .../train_geonet_baidu.py \
  --data /home/aistudio/data/TON_DATASET \
  --output /home/aistudio/work/GeoNet_checkpoints \
  --resume-paddle /home/aistudio/work/GeoNet_checkpoints/last_multiforme.pdckpt
```

## 4. Si tu ne veux PAS reprendre un checkpoint GeoNet

Tu peux exporter seulement l'encodeur ImageNet depuis une machine PyTorch :

```bash
python tools/export_torch_checkpoint_npz.py \
  --imagenet-backbone \
  --output model_poids/efficientnet_v2_s_imagenet.npz
```

Puis utilise ce NPZ comme `--init-torch-npz`.

Le script détecte automatiquement qu'il s'agit d'un export partiel :
l'encodeur EfficientNetV2-S est initialisé avec les poids ImageNet et
le décodeur / les têtes restent initialisés par Paddle.

Pour reprendre réellement ton apprentissage actuel, préfère l'export
du `last_multiforme.pt` complet décrit en section 1.
