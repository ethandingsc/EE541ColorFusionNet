import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import config
from ColorFusionNet_resnet18_baseline import Colorizer
from data import get_train_val_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    for lightness, target_ab in loader:
        lightness = lightness.to(config.DEVICE, non_blocking=True)
        target_ab = target_ab.to(config.DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predicted_ab = model(lightness)
            loss = criterion(predicted_ab, target_ab)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * lightness.size(0)

    return total_loss / len(loader.dataset)


def save_training_curve(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "training_curve.png", dpi=150)
    plt.close()


def train():
    set_seed(config.SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_train_val_loaders()

    model = Colorizer(
        pretrained=config.PRETRAINED,
        freeze_encoder=config.FREEZE_ENCODER,
    ).to(config.DEVICE)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"Device: {config.DEVICE}")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"train {train_loss:.5f} | val {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "img_size": config.IMG_SIZE,
                },
                config.CHECKPOINT_PATH,
            )

    with open(config.RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    save_training_curve(history)
    print(f"Best checkpoint: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
