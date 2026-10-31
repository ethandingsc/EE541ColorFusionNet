import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torch.utils.data import DataLoader, Dataset

import config


class ImageFileDataset(Dataset):
    """Load unlabeled images recursively from a directory."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root):
        self.root = Path(root)
        self.image_paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in: {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB").copy()
        return image, 0


def lab_to_rgb(lightness, chrominance):
    """Convert L in [0, 100] and normalized ab in [-1, 1] to RGB in [0, 1]."""
    lab = np.concatenate([lightness, chrominance * 128.0], axis=-1)
    return np.clip(lab2rgb(lab.astype(np.float64)), 0.0, 1.0)


class ColorizationDataset(Dataset):
    def __init__(self, base_dataset, augment=False):
        self.base_dataset = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image = self.base_dataset[index][0].convert("RGB")

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image = image.resize(
            (config.IMG_SIZE, config.IMG_SIZE),
            Image.Resampling.BILINEAR,
        )
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)

        lightness = torch.from_numpy(lab[..., :1]).permute(2, 0, 1)
        chrominance = torch.from_numpy(lab[..., 1:] / 128.0).permute(2, 0, 1)
        return lightness, chrominance


def _make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE.type == "cuda",
    )


def get_train_val_loaders():
    """Load unlabeled images from data/train and data/val."""
    train_set = ColorizationDataset(
        ImageFileDataset(config.DATA_ROOT / "train"),
        augment=True,
    )
    val_set = ColorizationDataset(
        ImageFileDataset(config.DATA_ROOT / "val"),
        augment=False,
    )
    return _make_loader(train_set, shuffle=True), _make_loader(val_set)


def get_test_loader():
    """Load unlabeled images from data/test for final evaluation."""
    base = ImageFileDataset(config.DATA_ROOT / "test")
    return _make_loader(ColorizationDataset(base, augment=False))
