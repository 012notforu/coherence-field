from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_step(dx: float | tuple[float, float] | None) -> tuple[float, float]:
    if dx is None:
        return 1.0, 1.0
    if isinstance(dx, tuple):
        return float(dx[0]), float(dx[1])
    return float(dx), float(dx)


def grad(field: Array, dx: float | tuple[float, float] | None = None) -> tuple[Array, Array]:
    field = np.asarray(field)
    hx, hy = _as_step(dx)
    gx = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * hx)
    gy = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * hy)
    return gx, gy


def divergence(fx: Array, fy: Array, dx: float | tuple[float, float] | None = None) -> Array:
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    hx, hy = _as_step(dx)
    div_x = (np.roll(fx, -1, axis=0) - np.roll(fx, 1, axis=0)) / (2.0 * hx)
    div_y = (np.roll(fy, -1, axis=1) - np.roll(fy, 1, axis=1)) / (2.0 * hy)
    return div_x + div_y


def laplacian(field: Array, dx: float | tuple[float, float] | None = None) -> Array:
    field = np.asarray(field)
    hx, hy = _as_step(dx)
    lap_x = (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)) / (hx ** 2)
    lap_y = (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / (hy ** 2)
    return lap_x + lap_y


def bilaplacian(field: Array, dx: float | tuple[float, float] | None = None) -> Array:
    return laplacian(laplacian(field, dx=dx), dx=dx)


def hessian_action(field: Array, dx: float | tuple[float, float] | None = None) -> Array:
    hx, hy = _as_step(dx)
    gx, gy = grad(field, dx=dx)
    gxx = (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)) / (hx ** 2)
    gyy = (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / (hy ** 2)
    gxy = (
        np.roll(np.roll(field, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(field, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(field, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), 1, axis=1)
    ) / (4.0 * hx * hy)
    return gxx * gx ** 2 + 2.0 * gxy * gx * gy + gyy * gy ** 2


def norm_sq_grad(field: Array, dx: float | tuple[float, float] | None = None) -> Array:
    gx, gy = grad(field, dx=dx)
    return gx ** 2 + gy ** 2
