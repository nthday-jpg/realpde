from __future__ import annotations

import torch
import torch.nn as nn

from realpde.models.base import PretrainModel
from realpde.models.unet.config import UNetConfig


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(PretrainModel):
    """Per-frame 2D UNet for velocity field prediction.

    Operates on each frame independently:

    1. Reshape ``(B, T, H, W, C)`` -> ``(B*T, H, W, C)``
    2. Run 2D UNet encoder-decoder with skip connections
    3. Reshape ``(B*T, out_c, H, W)`` -> ``(B, T, H, W, out_c)``
    """

    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        if config is None:
            config = UNetConfig()
        self.config = config

        in_ch = config.in_channels
        out_ch = config.out_channels
        base = config.channels
        n_layers = config.n_layers

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_ch
        for _ in range(n_layers):
            self.encoder.append(DoubleConv(ch, base))
            ch = base

        self.bottleneck = DoubleConv(base, base * 2)

        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        ch = base * 2
        for _ in range(n_layers):
            self.upconv.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.decoder.append(DoubleConv(ch // 2 + base, ch // 2))
            ch = ch // 2

        self.head = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        # (B, T, H, W, C) -> (B*T, C, H, W) for Conv2d (channels-first)
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)

        skips = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upconv, self.decoder, reversed(skips)):
            h = up(h)
            if h.shape != skip.shape:
                h = nn.functional.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = self.head(h)
        # (B*T, out_c, H, W) -> (B, T, H, W, out_c)
        return h.reshape(B, T, H, W, self.config.out_channels)
