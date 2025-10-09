"""Core physics engine for SCFD vNext."""

from .ops import grad, laplacian, bilaplacian, hessian_action, divergence, norm_sq_grad
from .pde_core import accel_CK, accel_theta, coherence_force
from .energy import (
    coherence_energy_density,
    cross_gradient_energy_density,
    kinetic_energy_density,
    potential_energy_density,
    curvature_energy_density,
    total_energy_density,
    metropolis_accept,
    coherence_p_to_m,
)
from .integrators import leapfrog_step, suggest_cfl_dt
from .params import (
    load_config,
    SimulationConfig,
    PhysicsParams,
    IntegrationParams,
    SchedulerParams,
)
from .scheduler import AsyncScheduler
from .symbolizer import extract_symbols
from .diagnostics import (
    compute_energy_report,
    radial_spectrum,
    energy_drift,
    predictability_horizon,
    edge_metrics,
    impulse_response,
)

__all__ = [
    "accel_theta",
    "accel_CK",
    "coherence_force",
    "grad",
    "laplacian",
    "bilaplacian",
    "hessian_action",
    "divergence",
    "norm_sq_grad",
    "coherence_energy_density",
    "cross_gradient_energy_density",
    "kinetic_energy_density",
    "potential_energy_density",
    "curvature_energy_density",
    "total_energy_density",
    "metropolis_accept",
    "coherence_p_to_m",
    "leapfrog_step",
    "suggest_cfl_dt",
    "load_config",
    "SimulationConfig",
    "PhysicsParams",
    "IntegrationParams",
    "SchedulerParams",
    "AsyncScheduler",
    "extract_symbols",
    "compute_energy_report",
    "radial_spectrum",
    "energy_drift",
    "predictability_horizon",
    "edge_metrics",
    "impulse_response",
]
