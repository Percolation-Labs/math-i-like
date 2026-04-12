"""Shared neural network building blocks.

RMSNorm, rotary position embeddings, and other reusable components
for the stigmergic attention architecture.
"""

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root mean square layer normalisation (Zhang & Sennrich 2019)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al. 2021)."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims — helper for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to queries or keys."""
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin
