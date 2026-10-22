"""ResNet-18 U-Net baseline for Lab image colorization.

Input:
    L:  [B, 1, H, W], CIE Lab lightness in [0, 100]
Output:
    ab: [B, 2, H, W], normalized chrominance in [-1, 1]

This file intentionally contains only the baseline model. Color-aware
downsampling, attention, and improved skip fusion should be added as separate
experimental variants so that their effects can be evaluated independently.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from DCSAF import DCSAF


class DecoderBlock(nn.Module):
    """Upsample a decoder feature and fuse it with one encoder skip feature."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        # DCSAF attention fusion
       # self.dcsaf = DCSAF(out_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder_feature: torch.Tensor, skip_feature: torch.Tensor):
        decoder_feature = self.upsample(decoder_feature)

        # This makes concatenation robust when H or W is not divisible by 32.
        if decoder_feature.shape[-2:] != skip_feature.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=skip_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused_feature = torch.cat([decoder_feature, skip_feature], dim=1)
        return self.fusion(fused_feature)

        # # DCSAF cross-feature enhancement
        # skip_feature, decoder_feature = self.dcsaf(skip_feature, decoder_feature)
        # # Original U-Net concatenation
        # fused_feature = torch.cat([decoder_feature, skip_feature], dim=1)

        # return self.fusion(fused_feature)


class ColorFusionNetBaseline(nn.Module):
    """ResNet-18 encoder with a U-Net-style convolutional decoder."""

    def __init__(self, pretrained: bool = True, freeze_encoder: bool = False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)

        # ResNet-18 feature sizes:
        # stem: 64, layer1: 64, layer2: 128, layer3: 256, layer4: 512
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.decoder0 = DecoderBlock(64, 64, 64)

        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh(),
        )

        # ImageNet normalization used by the pretrained ResNet-18 encoder.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        if freeze_encoder:
            self.set_encoder_trainable(False)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze all encoder parameters."""

        encoder_parts = (
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        for part in encoder_parts:
            for parameter in part.parameters():
                parameter.requires_grad = trainable

    def normalize_l_for_resnet(self, lightness: torch.Tensor) -> torch.Tensor:
        """Convert Lab L in [0, 100] to a 3-channel ImageNet-style input."""

        image = (lightness / 100.0).repeat(1, 3, 1, 1)
        return (image - self.imagenet_mean) / self.imagenet_std

    def forward(self, lightness: torch.Tensor) -> torch.Tensor:
        input_size = lightness.shape[-2:]
        x = self.normalize_l_for_resnet(lightness)

        skip0 = self.stem(x)                         # 64,  H/2
        skip1 = self.layer1(self.maxpool(skip0))     # 64,  H/4
        skip2 = self.layer2(skip1)                   # 128, H/8
        skip3 = self.layer3(skip2)                   # 256, H/16

        bottleneck = self.layer4(skip3)              # 512, H/32

        x = self.decoder3(bottleneck, skip3)          # 256, H/16
        x = self.decoder2(x, skip2)                   # 128, H/8
        x = self.decoder1(x, skip1)                   # 64,  H/4
        x = self.decoder0(x, skip0)                   # 64,  H/2
        ab = self.output_block(x)                     # 2,   H

        # Preserve the original spatial size for non-standard input dimensions.
        if ab.shape[-2:] != input_size:
            ab = F.interpolate(ab, size=input_size, mode="bilinear", align_corners=False)

        return ab


# Compatibility alias: existing training code can continue importing Colorizer.
Colorizer = ColorFusionNetBaseline


if __name__ == "__main__":
    model = ColorFusionNetBaseline(pretrained=False)
    sample_l = torch.rand(2, 1, 256, 256) * 100.0
    sample_ab = model(sample_l)
    print("Input shape: ", tuple(sample_l.shape))
    print("Output shape:", tuple(sample_ab.shape))
