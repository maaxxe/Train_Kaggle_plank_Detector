from pathlib import Path

PROJECT_DIR = Path("/kaggle/working/GAN_PlankEye_v2")

# Entraînement
IMAGE_SIZE = 512
SEED = 42
LATENT_CHANNELS = 3
BASE_CHANNELS = 64
BATCH_PER_GPU = 4
NUM_WORKERS = 2
EPOCHS = 150
LR_G = 2e-4
LR_D = 2e-4
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 50.0
VAL_RATIO = 0.10

# Dossiers
PAIRED_DIR = PROJECT_DIR / "data" / "paired"
RUN_DIR = PROJECT_DIR / "runs" / "plankgan_multi_512"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
SAMPLE_DIR = RUN_DIR / "samples"
GENERATED_DIR = PROJECT_DIR / "generated"
