import torch
import torch.nn as nn

# FAE-Det: Frequency-Aware Enhancement Network for Infrared-Visible Object Detection

class DCSAF(nn.Module):
    """
    Dynamic Channel-Spatial Cross Attention Fusion

    Adapted for Encoder-Decoder Feature Fusion
    in Grayscale Image Colorization.
    """
    def __init__(self, c, r=16):
        super().__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(c // r, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(c // r, 1), c, 1),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        # Dynamic Feature Weight
        self.dw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, 2, 1),
            nn.Softmax(dim=1)
        )
    def attention(self, x):
        # Channel attention
        c = self.ca(x)
        # Spatial attention
        s = self.sa(torch.cat([
            x.mean(1, keepdim=True),
            x.max(1, keepdim=True)[0]
        ], dim=1))
        return c * s

    def forward(self, enc, dec):
        # Cross attention
        enc_a = self.attention(enc)
        dec_a = self.attention(dec)
        # Adaptive feature weighting
        w = self.dw(enc + dec)
        w_enc = w[:, 0:1]
        w_dec = w[:, 1:2]
        # Cross-feature enhancement
        enc_out = enc + w_dec * enc * dec_a
        dec_out = dec + w_enc * dec * enc_a
        return enc_out, dec_out


"""
DCSAF: Dynamic Channel-Spatial Cross Attention Fusion

Adapted for ResNet-18 Encoder and U-Net Decoder feature fusion
in grayscale image colorization.

The module computes channel-spatial attention for encoder and decoder
features, generates adaptive fusion weights via Softmax, and performs
bidirectional cross-attention enhancement with residual connections.

It enhances interactions between low-level texture details and
high-level semantic features, potentially improving colorization quality.

Input: Encoder and Decoder features (same shape).
Output: Two enhanced feature maps for subsequent fusion.
"""

if __name__ == "__main__":

    enc = torch.randn(2, 64, 80, 80)
    dec = torch.randn(2, 64, 80, 80)
    model = DCSAF(64)
    enc_out, dec_out = model(enc, dec)
    print(enc_out.shape, dec_out.shape)


