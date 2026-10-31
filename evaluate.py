import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.color import deltaE_ciede2000
from skimage.metrics import structural_similarity

import config
from ColorFusionNet_resnet18_baseline import Colorizer
from data import get_test_loader, lab_to_rgb


def load_trained_model():
    if not config.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {config.CHECKPOINT_PATH}. Run train.py first."
        )

    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model = Colorizer(freeze_encoder=False).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_loader = get_test_loader()
    model = load_trained_model()
    criterion = nn.MSELoss(reduction="sum")

    squared_error = 0.0
    absolute_error = 0.0
    pixel_values = 0
    psnr_scores = []
    ssim_scores = []
    delta_e_scores = []
    examples = []

    with torch.no_grad():
        for lightness, target_ab in test_loader:
            lightness_device = lightness.to(config.DEVICE, non_blocking=True)
            target_device = target_ab.to(config.DEVICE, non_blocking=True)
            predicted_ab = model(lightness_device)

            squared_error += criterion(predicted_ab, target_device).item()
            absolute_error += torch.abs(predicted_ab - target_device).sum().item()
            pixel_values += target_ab.numel()

            lightness_np = lightness.numpy()
            target_np = target_ab.numpy()
            prediction_np = predicted_ab.cpu().numpy()

            for index in range(lightness.size(0)):
                l_channel = np.transpose(lightness_np[index], (1, 2, 0))
                target_ab_image = np.transpose(target_np[index], (1, 2, 0))
                prediction_ab_image = np.transpose(
                    prediction_np[index],
                    (1, 2, 0),
                )
                ground_truth = lab_to_rgb(
                    l_channel,
                    target_ab_image,
                )
                prediction = lab_to_rgb(
                    l_channel,
                    prediction_ab_image,
                )

                ground_truth_lab = np.concatenate(
                    [l_channel, target_ab_image * 128.0],
                    axis=-1,
                )
                prediction_lab = np.concatenate(
                    [l_channel, prediction_ab_image * 128.0],
                    axis=-1,
                )
                delta_e_scores.append(
                    float(
                        np.mean(
                            deltaE_ciede2000(ground_truth_lab, prediction_lab)
                        )
                    )
                )

                rgb_mse = float(np.mean((ground_truth - prediction) ** 2))
                psnr_scores.append(10.0 * np.log10(1.0 / max(rgb_mse, 1e-10)))
                ssim_scores.append(
                    structural_similarity(
                        ground_truth,
                        prediction,
                        channel_axis=2,
                        data_range=1.0,
                    )
                )

                if len(examples) < 8:
                    examples.append((l_channel[..., 0], ground_truth, prediction))

    metrics = {
        "test_ab_mse": squared_error / pixel_values,
        "test_ab_mae": absolute_error / pixel_values,
        "mean_psnr_db": float(np.mean(psnr_scores)),
        "std_psnr_db": float(np.std(psnr_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
        "std_ssim": float(np.std(ssim_scores)),
        "mean_delta_e_2000": float(np.mean(delta_e_scores)),
        "std_delta_e_2000": float(np.std(delta_e_scores)),
        "test_images": len(psnr_scores),
    }

    with open(config.RESULTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    figure, axes = plt.subplots(3, len(examples), figsize=(2.5 * len(examples), 7))
    for index, (gray, ground_truth, prediction) in enumerate(examples):
        axes[0, index].imshow(gray, cmap="gray", vmin=0, vmax=100)
        axes[1, index].imshow(ground_truth)
        axes[2, index].imshow(prediction)
        for row in range(3):
            axes[row, index].axis("off")
    axes[0, 0].set_ylabel("Input")
    axes[1, 0].set_ylabel("Ground truth")
    axes[2, 0].set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "colorization_results.png", dpi=150)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    evaluate()
