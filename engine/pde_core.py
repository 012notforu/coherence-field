from __future__ import annotations

import numpy as np

from .energy import coherence_p_to_m
from .ops import bilaplacian, grad, hessian_action, laplacian, norm_sq_grad, divergence
from .params import PhysicsParams

Array = np.ndarray


def coherence_force(theta: Array, params: PhysicsParams, dx: float | tuple[float, float] | None = None) -> Array:
    grad_sq = norm_sq_grad(theta, dx=dx)
    q = grad_sq + params.epsilon ** 2
    p = float(max(params.coherence_p, 1))
    m = coherence_p_to_m(p)
    f = params.alpha * p * q ** (-m)
    lap = laplacian(theta, dx=dx)
    h_term = hessian_action(theta, dx=dx)
    coherence_extras = 2.0 * params.alpha * p * m * q ** (-(m + 1.0)) * h_term
    return (params.gamma - f) * lap + coherence_extras


def accel_theta(theta: Array, params: PhysicsParams, dx: float | tuple[float, float] | None = None) -> Array:
    force = coherence_force(theta, params, dx=dx)
    force -= params.potential.derivative(theta)
    if params.curvature_penalty.enabled:
        force -= 2.0 * params.curvature_penalty.delta * bilaplacian(theta, dx=dx)
    return force / params.beta


def accel_CK(
    C: Array,
    K: Array,
    params: PhysicsParams,
    dx: float | tuple[float, float] | None = None,
) -> tuple[Array, Array]:
    cfg = params.cross_gradient
    if not cfg.enabled:
        zeros = np.zeros_like(C)
        return zeros, zeros
    gx_c, gy_c = grad(C, dx=dx)
    gx_k, gy_k = grad(K, dx=dx)
    eps = cfg.epsilon
    mag_c = np.sqrt(gx_c ** 2 + gy_c ** 2 + eps ** 2)
    mag_k = np.sqrt(gx_k ** 2 + gy_k ** 2 + eps ** 2)
    ratio_ck = mag_k / mag_c
    ratio_kc = mag_c / mag_k
    flux_cx = ratio_ck * gx_c
    flux_cy = ratio_ck * gy_c
    flux_kx = ratio_kc * gx_k
    flux_ky = ratio_kc * gy_k
    delta_c = cfg.gamma * divergence(flux_cx, flux_cy, dx=dx)
    delta_k = cfg.gamma * divergence(flux_kx, flux_ky, dx=dx)
    accel_c = -delta_c / cfg.beta
    accel_k = -delta_k / cfg.beta
    if cfg.parabolic_coeff > 0.0:
        accel_c -= cfg.parabolic_coeff * delta_c
        accel_k -= cfg.parabolic_coeff * delta_k
    return accel_c, accel_k
