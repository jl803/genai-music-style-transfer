"""Conditional Glow-style normalizing flow for mel spectrogram crops.

This is a compact Glow implementation for tensors shaped [B, 1, n_mels, T].
It learns a class-conditional invertible mapping:

    mel crop <-> latent z

For style transfer, encode with the source genre label and decode the same
latent with the target genre label.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def squeeze2d(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Glow squeeze requires even H/W, got {height}x{width}.")
    x = x.view(batch, channels, height // 2, 2, width // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(batch, channels * 4, height // 2, width // 2)


def unsqueeze2d(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if channels % 4 != 0:
        raise ValueError(f"Glow unsqueeze requires channels divisible by 4, got {channels}.")
    x = x.view(batch, channels // 4, 2, 2, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(batch, channels // 4, height * 2, width * 2)


class ActNorm(nn.Module):
    """Activation normalization with data-dependent initialization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))

    def initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
            self.bias.copy_(-mean)
            self.log_scale.copy_(torch.log(1.0 / std))
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.initialized):
            self.initialize(x)

        _, _, height, width = x.shape
        logdet = height * width * self.log_scale.sum()
        if reverse:
            y = x * torch.exp(-self.log_scale) - self.bias
            return y, -logdet.expand(x.shape[0])

        y = (x + self.bias) * torch.exp(self.log_scale)
        return y, logdet.expand(x.shape[0])


class Invertible1x1Conv(nn.Module):
    """Glow's learned channel permutation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        self.weight = nn.Parameter(q)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        _, channels, height, width = x.shape
        sign, logabsdet = torch.linalg.slogdet(self.weight)
        if torch.any(sign == 0).item():
            raise RuntimeError("Invertible1x1Conv weight became singular.")

        if reverse:
            weight = torch.inverse(self.weight).view(channels, channels, 1, 1)
            y = F.conv2d(x, weight)
            return y, (-height * width * logabsdet).expand(x.shape[0])

        weight = self.weight.view(channels, channels, 1, 1)
        y = F.conv2d(x, weight)
        return y, (height * width * logabsdet).expand(x.shape[0])


class AffineCoupling(nn.Module):
    """Class-conditional affine coupling layer."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        cond_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.channels_a = channels // 2
        self.channels_b = channels - self.channels_a
        self.embedding = nn.Embedding(num_classes, cond_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.channels_a + cond_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.channels_b * 2, kernel_size=3, padding=1),
        )
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def condition(self, labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
        cond = self.embedding(labels).unsqueeze(-1).unsqueeze(-1)
        return cond.expand(-1, -1, height, width)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xa, xb = x[:, : self.channels_a], x[:, self.channels_a :]
        cond = self.condition(labels, xa.shape[2], xa.shape[3])
        shift, log_scale = self.net(torch.cat([xa, cond], dim=1)).chunk(2, dim=1)
        log_scale = 1.5 * torch.tanh(log_scale)

        if reverse:
            yb = (xb - shift) * torch.exp(-log_scale)
            logdet = -log_scale.flatten(1).sum(dim=1)
        else:
            yb = xb * torch.exp(log_scale) + shift
            logdet = log_scale.flatten(1).sum(dim=1)

        return torch.cat([xa, yb], dim=1), logdet


class FlowStep(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        cond_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = Invertible1x1Conv(channels)
        self.coupling = AffineCoupling(channels, hidden_channels, cond_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_logdet = x.new_zeros(x.shape[0])
        layers = (self.actnorm, self.invconv, self.coupling)
        if reverse:
            layers = tuple(reversed(layers))

        for layer in layers:
            if isinstance(layer, AffineCoupling):
                x, logdet = layer(x, labels, reverse=reverse)
            else:
                x, logdet = layer(x, reverse=reverse)
            total_logdet = total_logdet + logdet
        return x, total_logdet


class ConditionalGlow(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        time_len: int = 128,
        n_flows: int = 8,
        hidden_channels: int = 64,
        cond_channels: int = 16,
        num_classes: int = 2,
        logit_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if n_mels % 2 != 0 or time_len % 2 != 0:
            raise ValueError("n_mels and time_len must be even for Glow squeeze.")
        self.n_mels = n_mels
        self.time_len = time_len
        self.logit_eps = logit_eps
        flow_channels = 4
        self.flows = nn.ModuleList(
            [
                FlowStep(
                    channels=flow_channels,
                    hidden_channels=hidden_channels,
                    cond_channels=cond_channels,
                    num_classes=num_classes,
                )
                for _ in range(n_flows)
            ]
        )

    def logit_preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.clamp(self.logit_eps, 1.0 - self.logit_eps)
        y = torch.log(x) - torch.log1p(-x)
        logdet = (-torch.log(x) - torch.log1p(-x)).flatten(1).sum(dim=1)
        return y, logdet

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, total_logdet = self.logit_preprocess(x)
        x = squeeze2d(x)
        for flow in self.flows:
            x, logdet = flow(x, labels, reverse=False)
            total_logdet = total_logdet + logdet
        return x, total_logdet

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = z
        for flow in reversed(self.flows):
            x, _ = flow(x, labels, reverse=True)
        x = unsqueeze2d(x)
        return torch.sigmoid(x).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(x, labels)


def glow_nll_per_sample(
    model: ConditionalGlow,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    z, logdet = model(x, labels)
    log_prob = -0.5 * (z.pow(2) + LOG_2PI).flatten(1).sum(dim=1)
    n_dims = x[0].numel()
    return -(log_prob + logdet) / n_dims


def glow_nll(model: ConditionalGlow, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return glow_nll_per_sample(model, x, labels).mean()
