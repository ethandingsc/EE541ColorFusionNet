# ColorFusionNet

ColorFusionNet is a PyTorch course project for automatic grayscale image colorization. The model takes the **L channel** in the CIE Lab color space as input and predicts the corresponding **a/b color channels**.

## Project Goal

The project is expected to:

- Implement a ResNet-18-based encoder for multi-scale grayscale feature extraction.
- Build a U-Net-style decoder to recover spatial details through skip connections.
- Implement a dual enhancement module for bidirectional interaction between encoder and decoder features.
- Train the network to predict normalized Lab `a/b` channels.
- Evaluate the trained model with quantitative metrics and visual colorization results.

## Model Architecture

```text
Grayscale image
      |
CIE Lab L channel
      |
ResNet-18 encoder
      |
Multi-scale encoder features
      |
U-Net decoder + dual feature enhancement
      |
Predicted Lab a/b channels
      |
Reconstructed RGB image
```

The encoder extracts features at different spatial scales. The decoder progressively upsamples the high-level features and combines them with corresponding encoder features through skip connections.

The **Dynamic Channel-Spatial Cross Attention Fusion (DCSAF)** module performs two-way enhancement:

1. Decoder information enhances encoder features.
2. Encoder information enhances decoder features.

Channel attention, spatial attention, and dynamic feature weights are used to control the interaction between the two feature streams.

## Project Structure

```text
ColorFusionNet/
├── config.py
├── data.py
├── ColorFusionNet_resnet18_baseline.py
├── DCSAF.py
├── train.py
├── evaluate.py
├── requirements.txt
└── results/
```

| File | Purpose |
| --- | --- |
| `config.py` | Training settings, paths, and device selection |
| `data.py` | Image preprocessing and CIE Lab conversion |
| `ColorFusionNet_resnet18_baseline.py` | ResNet-18 encoder and U-Net decoder |
| `DCSAF.py` | Dual encoder-decoder feature enhancement module |
| `train.py` | Model training, validation, and checkpoint saving |
| `evaluate.py` | Quantitative evaluation and result visualization |
| `results/` | Saved checkpoints, metrics, curves, and output images |

## Installation

Install the required packages:

```powershell
pip install -r requirements.txt
```

The code automatically uses CUDA when an NVIDIA GPU and a CUDA-enabled PyTorch build are available. Otherwise, it runs on the CPU.

## Training

Run:

```powershell
python train.py
```

The training process predicts normalized Lab `a/b` channels from the input L channel and saves the best validation checkpoint in `results/`.

## Evaluation

After training, run:

```powershell
python evaluate.py
```

The evaluation stage is expected to:

- Load the best trained checkpoint.
- Calculate image-colorization metrics such as MSE, PSNR, and SSIM.
- Save comparisons of grayscale inputs, ground-truth images, and colorized predictions.

## Expected Outcome

The expected result is an end-to-end image-colorization pipeline that can learn semantic and spatial information from grayscale images and generate visually plausible colors. The dual enhancement module is expected to improve the interaction between encoder details and decoder semantics, helping the network recover more coherent object colors and image structures.

Final quantitative results will be added after model training and evaluation.
