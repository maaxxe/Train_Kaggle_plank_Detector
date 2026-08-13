from pathlib import Path
import sys
import torch

# Usage :
#   python voir_epoch.py
# ou :
#   python voir_epoch.py "C:\chemin\vers\last_plankeye_v4_1class_512.pt"

DEFAULT_CHECKPOINT = Path("model_poids") / "last_plankeye_v4_1class_512.pt"

checkpoint_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT

if not checkpoint_path.exists():
    print(f"ERREUR : checkpoint introuvable : {checkpoint_path.resolve()}")
    print()
    print("Place voir_epoch.py dans le même dossier que le .pt")
    print("ou lance :")
    print(r'python voir_epoch.py "C:\chemin\vers\last_plankeye_v4_1class_512.pt"')
    sys.exit(1)

try:
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
except Exception as exc:
    print(f"ERREUR lors de la lecture du checkpoint : {exc}")
    sys.exit(1)

epoch = ckpt.get("epoch")
best_epoch = ckpt.get("best_epoch")
best_quality = ckpt.get("best_quality")

print("=" * 60)
print("PLANK EYE - CHECKPOINT")
print("=" * 60)
print("Fichier       :", checkpoint_path.resolve())
print("Epoch actuel  :", epoch)

if epoch is not None:
    print("Prochain epoch:", int(epoch) + 1)

print("Best epoch    :", best_epoch)
print("Best quality  :", best_quality)
print("=" * 60)
