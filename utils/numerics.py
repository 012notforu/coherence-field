from __future__ import annotations

import numpy as np

Array = np.ndarray


def gaussian_kernel(size: int, sigma: float) -> Array:
    radius = size // 2
    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def fft2d(field: Array) -> Array:
    return np.fft.fftshift(np.fft.fft2(field))


def ema_update(prev: float, value: float, tau: float) -> float:
    decay = np.exp(-1.0 / max(tau, 1.0))
    return float(decay * prev + (1.0 - decay) * value)
