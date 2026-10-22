import random

import numpy as np
import torch
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

import config


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
    """Use STL10 train data and a fixed 90/10 train-validation split."""
    base = datasets.STL10(root=str(config.DATA_ROOT), split="train", download=True)
    generator = torch.Generator().manual_seed(config.SEED)
    indices = torch.randperm(len(base), generator=generator).tolist()
    val_size = max(1, int(len(indices) * config.VAL_RATIO))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(ColorizationDataset(base, augment=True), train_indices)
    val_set = Subset(ColorizationDataset(base, augment=False), val_indices)
    return _make_loader(train_set, shuffle=True), _make_loader(val_set)


def get_test_loader():
    """Use the official STL10 test split only for final evaluation."""
    base = datasets.STL10(root=str(config.DATA_ROOT), split="test", download=True)
    return _make_loader(ColorizationDataset(base, augment=False))
