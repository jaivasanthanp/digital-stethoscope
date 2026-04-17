"""
ResNet-10 for heart sound classification.

Input:  (batch, 1, 64, 64)  — grayscale log-mel spectrogram
Output: (batch, 4)          — softmax probabilities

Architecture:
  Conv1     : 7×7, stride 2 → (16, 32, 32)
  MaxPool   : 2×2, stride 2 → (16, 16, 16)
  ResBlock1 : 16→16, no downsample         (16, 16, 16)
  ResBlock2 : 16→32, stride 2 downsample   (32,  8,  8)
  ResBlock3 : 32→64, stride 2 downsample   (64,  4,  4)
  GlobalAvgPool → (64,)
  FC + Softmax  → (4,)

Each ResBlock = Conv→BN→ReLU→Conv→BN + skip, ReLU after add.
Skip uses 1×1 conv projection when channels change.

Optional SE (Squeeze-and-Excitation) blocks enabled by use_se=True.
SE: GAP → FC(C→C//4) → ReLU → FC(C//4→C) → Sigmoid → channel multiply.
~+3% accuracy, ~2K extra params — recommended.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ResBlock(nn.Module):
    """Basic residual block (two 3×3 convolutions)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        # Projection shortcut when spatial or channel dimensions change
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet10(nn.Module):
    """ResNet-10 for 4-class heart sound classification."""

    def __init__(self, n_classes: int = 4, use_se: bool = True):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.layer1 = ResBlock(16, 16, stride=1, use_se=use_se)
        self.layer2 = ResBlock(16, 32, stride=2, use_se=use_se)
        self.layer3 = ResBlock(32, 64, stride=2, use_se=use_se)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)    # → (B, 16, 16, 16)
        x = self.layer1(x)  # → (B, 16, 16, 16)
        x = self.layer2(x)  # → (B, 32,  8,  8)
        x = self.layer3(x)  # → (B, 64,  4,  4)
        x = self.gap(x)     # → (B, 64,  1,  1)
        x = x.flatten(1)    # → (B, 64)
        return self.fc(x)   # → (B, 4)  (logits; softmax applied in loss / at inference)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ResNet10(use_se=True)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print(f"Output shape : {y.shape}")
    print(f"Total params : {model.count_params():,}")
