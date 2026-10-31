# ColorFusionNet

ColorFusionNet is an EE541 course project for automatic image colorization with PyTorch. The model receives the **L channel** of a CIE Lab image and predicts its normalized **a/b chrominance channels**. The predicted Lab image is then converted back to RGB for evaluation and visualization.

## Current Status

The repository currently provides a complete trainable baseline:

- ResNet-18 encoder with random initialization and no pretrained weights.
- U-Net-style decoder with multi-scale skip connections.
- Unlabeled image-folder data loading for custom datasets.
- RGB-to-Lab preprocessing and Lab-to-RGB reconstruction.
- Training, validation, checkpointing, evaluation, and result visualization.
- MSE, MAE, PSNR, SSIM, and CIEDE2000 evaluation metrics.

`DCSAF.py` contains an experimental fusion module prototype. It is not enabled in the current baseline, so the baseline remains a standard concatenation-based ResNet-U-Net model. The fusion module and possible gated variants will be evaluated separately through ablation experiments.

The current encoder uses the torchvision ResNet-18 architecture only as the baseline implementation. It is designed to be replaced by the project team's custom ResNet implementation while preserving the same multi-scale feature interface.

## Baseline Architecture

```text
RGB ground-truth image
        |
Convert RGB to CIE Lab
        |
L channel [B, 1, H, W]
        |
ResNet-18 encoder
        |
Multi-scale features: 64 / 64 / 128 / 256 / 512 channels
        |
U-Net decoder + concatenation-based skip fusion
        |
Predicted normalized ab channels [B, 2, H, W]
        |
Combine L + predicted ab and convert to RGB
```

For a `128 x 128` input, the encoder produces the following feature maps:

| Feature | Channels | Spatial size |
| --- | ---: | ---: |
| `stem` | 64 | `64 x 64` |
| `layer1` | 64 | `32 x 32` |
| `layer2` | 128 | `16 x 16` |
| `layer3` | 256 | `8 x 8` |
| `layer4` | 512 | `4 x 4` |

The decoder progressively upsamples the `layer4` bottleneck and concatenates it with the corresponding encoder features. The final `Tanh` layer predicts normalized Lab `a/b` values in approximately `[-1, 1]`.

## Dataset Structure

The project reads ordinary image files without requiring class labels. Organize the dataset as follows:

```text
ColorFusionNet/
|-- data/
|   |-- train/
|   |   `-- images or nested image folders
|   |-- val/
|   |   `-- images or nested image folders
|   `-- test/
|       `-- images or nested image folders
```

Supported formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`. Images are loaded recursively, converted to RGB, resized to the configured image size, and transformed into CIE Lab. Horizontal flipping is applied only to the training set.

The three splits must contain different images. Near-duplicate images or frames from the same sequence should not be placed across different splits.

## Project Structure

```text
ColorFusionNet/
|-- ColorFusionNet_resnet18_baseline.py  # Baseline model
|-- DCSAF.py                             # Experimental fusion module
|-- config.py                            # Paths and training settings
|-- data.py                              # Dataset loading and Lab conversion
|-- train.py                             # Training and validation
|-- evaluate.py                          # Metrics and visualization
|-- requirements.txt                     # Python dependencies
`-- results/                             # Generated checkpoints and results
```

## Installation

Activate the project environment and install the dependencies:

```powershell
conda activate ee541
python -m pip install -r requirements.txt
```

The code automatically uses CUDA when a CUDA-enabled PyTorch installation and compatible NVIDIA GPU are available. Otherwise, it falls back to the CPU.

## Training

After placing the dataset under `data/train`, `data/val`, and `data/test`, run:

```powershell
python train.py
```

The default configuration uses:

- Image size: `128 x 128`
- Batch size: `16`
- Epochs: `20`
- Optimizer: AdamW
- Loss: MSE on normalized Lab `a/b` channels
- Learning-rate scheduler: StepLR

The best validation checkpoint is saved as:

```text
results/colorfusionnet_best.pth
```

Training also produces:

```text
results/training_history.json
results/training_curve.png
```

## Evaluation

After training, run:

```powershell
python evaluate.py
```

The evaluation script reports:

| Metric | Interpretation | Direction |
| --- | --- | ---: |
| `ab MSE` | Squared chrominance prediction error | Lower is better |
| `ab MAE` | Absolute chrominance prediction error | Lower is better |
| PSNR | RGB reconstruction quality | Higher is better |
| SSIM | RGB structural similarity | Higher is better |
| CIEDE2000 (`Delta E 00`) | Perceptual color difference in Lab space | Lower is better |

PSNR, SSIM, and CIEDE2000 are reported with both their mean and standard deviation across test images.

Evaluation produces:

```text
results/evaluation_metrics.json
results/colorization_results.png
```

## Planned Experiments

The planned comparison is:

1. ResNet-U-Net baseline with standard skip concatenation.
2. Baseline with the team-developed ResNet encoder.
3. Enhanced encoder-decoder fusion module.
4. Gated fusion variant, if implemented.

All variants should use the same dataset splits, preprocessing, training configuration, checkpoint criterion, and evaluation samples to ensure a fair comparison.

Final quantitative results will be added after training and ablation experiments are completed.
