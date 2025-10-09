from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .params import IntegrationParams

Array = np.ndarray


def leapfrog_step(
    state: Array,
    velocity: Array,
    accel_fn: Callable[[Array], Array],
    dt: float,
    *,
    accel_args: Optional[Sequence] = None,
    noise_cfg: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    max_step: Optional[float] = None,
) -> tuple[Array, Array, Array, dict]:
    """
    Symplectic leapfrog integrator:
      v_{t+1/2} = v_t + dt/2 * a(u_t)
      u_{t+1}   = u_t + dt * v_{t+1/2}  
      v_{t+1}   = v_{t+1/2} + dt/2 * a(u_{t+1})
    
    Conserves symplectic structure with small bounded energy drift expected.
    See tests/test_symplectic_energy.py for energy conservation validation.
    
    Args:
        state: Current position/field values
        velocity: Current velocity/momentum  
        accel_fn: Function computing acceleration from state
        dt: Time step size
        accel_args: Additional arguments for acceleration function
        noise_cfg: Optional noise configuration for stochastic dynamics
        rng: Random number generator for noise
        max_step: Optional step size clamping for stability
        
    Returns:
        (new_state, new_velocity, acceleration, info_dict)
    """
    accel_args = accel_args or ()
    acc0 = accel_fn(state, *accel_args)
    v_half = velocity + 0.5 * dt * acc0
    if noise_cfg and noise_cfg.get("enabled", False):
        rng = rng or np.random.default_rng(noise_cfg.get("seed", None))
        sigma = float(noise_cfg.get("sigma_scale", 0.0))
        if sigma > 0.0:
            v_half = v_half + rng.normal(scale=sigma * np.sqrt(dt), size=v_half.shape)
    delta = dt * v_half
    clamp_triggered = False
    if max_step is not None:
        clipped = np.clip(delta, -max_step, max_step)
        clamp_triggered = bool(np.any(np.abs(clipped) >= max_step - 1e-12))
        delta = clipped
    new_state = state + delta
    acc1 = accel_fn(new_state, *accel_args)
    new_velocity = v_half + 0.5 * dt * acc1
    info = {
        "rms_accel": float(np.sqrt(np.mean(acc0 ** 2))),
        "rms_accel_new": float(np.sqrt(np.mean(acc1 ** 2))),
        "rms_velocity_half": float(np.sqrt(np.mean(v_half ** 2))),
        "rms_step": float(np.sqrt(np.mean(delta ** 2))),
        "clamp_triggered": clamp_triggered,
    }
    return new_state, new_velocity, acc1, info


def suggest_cfl_dt(integration: IntegrationParams, wave_speed: float, dx: float) -> float:
    if wave_speed <= 0.0:
        return integration.dt
    cfl_dt = integration.cfl_limit * dx / max(wave_speed, 1e-8)
    return min(integration.dt, cfl_dt)
