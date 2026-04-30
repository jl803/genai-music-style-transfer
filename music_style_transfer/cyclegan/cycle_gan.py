"""Minimal CycleGAN-style 2D conv generator and PatchGAN discriminator for mel (1, n_mels, T)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """Maps mel [B,1,H,W] in [0,1] to mel same shape; uses Tanh then caller maps to [0,1]."""

    def __init__(self, n_mels: int = 128, time_len: int = 128) -> None:
        super().__init__()
        _ = (n_mels, time_len)  # fixed at train time; kept for API compatibility
        ch = 64
        self.down = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 2),
            nn.ReLU(True),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 4),
            nn.ReLU(True),
        )
        self.res = nn.Sequential(ResidualBlock(ch * 4), ResidualBlock(ch * 4), ResidualBlock(ch * 4))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 2),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(nn.Conv2d(ch // 2, 1, kernel_size=7, padding=3), nn.Tanh())

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        x = self.out(x)
        return (x + 1.0) * 0.5


class Discriminator(nn.Module):
    """PatchGAN: [B,1,H,W] -> logits [B,1,h',w']."""

    def __init__(self) -> None:
        super().__init__()

        def conv(in_c: int, out_c: int, stride: int, norm: bool) -> nn.Module:
            layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            conv(1, 64, stride=2, norm=False),
            conv(64, 128, stride=2, norm=True),
            conv(128, 256, stride=2, norm=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
