from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "colorfusionnet_best.pth"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1
NUM_WORKERS = 0
SEED = 42

PRETRAINED = True
FREEZE_ENCODER = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
