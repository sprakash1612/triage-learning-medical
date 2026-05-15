"""
Input corruptions for Phase 3 robustness (torch tensors, normalized space).
After each corruption, values are clamped to [-1, 1] per project spec.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

SEVERITY_LEVELS = ("mild", "moderate", "severe")

GAUSSIAN_NOISE_STD: Dict[str, float] = {
    "mild": 0.05,
    "moderate": 0.15,
    "severe": 0.30,
}

BRIGHTNESS_SHIFT: Dict[str, float] = {
    "mild": 0.2,
    "moderate": 0.5,
    "severe": 0.8,
}

CONTRAST_FACTOR: Dict[str, float] = {
    "mild": 0.8,
    "moderate": 0.5,
    "severe": 0.2,
}

GAUSSIAN_BLUR: Dict[str, Tuple[int, float]] = {
    "mild": (3, 0.5),
    "moderate": (5, 1.0),
    "severe": (7, 2.0),
}

SALT_PEPPER_DENSITY: Dict[str, float] = {
    "mild": 0.02,
    "moderate": 0.08,
    "severe": 0.15,
}


def _clamp_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    k2d = g[:, None] * g[None, :]
    return k2d


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + torch.randn_like(x) * std


def brightness_shift(x: torch.Tensor, delta: float) -> torch.Tensor:
    return x + delta


def contrast_reduce(x: torch.Tensor, factor: float) -> torch.Tensor:
    # Per-image, per-channel mean over H, W
    mean = x.mean(dim=(2, 3), keepdim=True)
    return mean + factor * (x - mean)


def gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    c = x.shape[1]
    k2d = _gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    weight = k2d.expand(c, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    return F.conv2d(x, weight, padding=pad, groups=c)


def salt_pepper(x: torch.Tensor, density: float, generator: torch.Generator | None = None) -> torch.Tensor:
    out = x.clone()
    n = x.numel()
    k = int(n * density / 2)
    if k <= 0:
        return out
    flat = out.view(-1)
    idx = torch.randperm(n, device=x.device, generator=generator)[: 2 * k]
    flat[idx[:k]] = -1.0
    flat[idx[k : 2 * k]] = 1.0
    return out.view_as(x)


def apply_corruption(
    x: torch.Tensor,
    corruption_name: str,
    severity: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Apply named corruption at severity level. x is (N,C,H,W) normalized batch.
    """
    if severity not in SEVERITY_LEVELS:
        raise ValueError(severity)
    name = corruption_name.lower().replace(" ", "_").replace("-", "_")
    z = x
    if name in ("gaussian_noise", "noise"):
        z = gaussian_noise(x, GAUSSIAN_NOISE_STD[severity])
    elif name in ("brightness", "brightness_shift"):
        z = brightness_shift(x, BRIGHTNESS_SHIFT[severity])
    elif name in ("contrast", "contrast_reduction"):
        z = contrast_reduce(x, CONTRAST_FACTOR[severity])
    elif name in ("gaussian_blur", "blur"):
        ks, sig = GAUSSIAN_BLUR[severity]
        z = gaussian_blur(x, ks, sig)
    elif name in ("salt_pepper", "saltandpepper", "salt_and_pepper"):
        z = salt_pepper(x, SALT_PEPPER_DENSITY[severity], generator=generator)
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    return _clamp_norm(z)


CORRUPTION_DISPLAY_NAMES = [
    "gaussian_noise",
    "brightness",
    "contrast",
    "gaussian_blur",
    "salt_pepper",
]
