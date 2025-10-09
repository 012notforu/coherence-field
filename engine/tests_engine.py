import numpy as np
import pytest

from .diagnostics import energy_drift, edge_metrics, impulse_response
from .energy import (
    compute_total_energy,
    cross_gradient_energy_density,
    total_energy_density,
)
from .integrators import leapfrog_step, suggest_cfl_dt
from .ops import grad, divergence, laplacian, norm_sq_grad, hessian_action
from .pde_core import accel_CK, accel_theta
from .params import (
    CrossGradientConfig,
    CurvaturePenaltyConfig,
    GridSpec,
    HeterogeneityConfig,
    IntegrationParams,
    PhysicsParams,
    PotentialConfig,
    SchedulerParams,
    NoiseConfig,
    NudgeConfig,
    validate_near_critical,
)
from .scheduler import AsyncScheduler
from .symbolizer import extract_symbols


def _physics(quadratic: bool = True, cross_enabled: bool = True) -> PhysicsParams:
    potential = PotentialConfig(
        kind="quadratic" if quadratic else "double_well",
        params={"center": 0.0, "stiffness": 0.05}
        if quadratic
        else {"theta0": -1.0, "theta1": 1.0, "stiffness": 0.2},
    )
    cross = CrossGradientConfig(enabled=cross_enabled, gamma=0.15, epsilon=1.0, beta=1.0, parabolic_coeff=0.0)
    curvature = CurvaturePenaltyConfig(enabled=False, delta=0.0)
    heterogeneity = HeterogeneityConfig(amplitude=0.0, correlation_length=3, seed=1)
    return PhysicsParams(
        beta=1.0,
        gamma=0.25,
        alpha=0.1,
        coherence_p=2,
        epsilon=1.0,
        potential=potential,
        cross_gradient=cross,
        curvature_penalty=curvature,
        heterogeneity=heterogeneity,
    )


def test_ops_green_identity():
    rng = np.random.default_rng(0)
    field = rng.normal(size=(16, 16))
    fx = rng.normal(size=(16, 16))
    fy = rng.normal(size=(16, 16))
    div = divergence(fx, fy)
    lhs = float(np.mean(field * div))
    gx, gy = grad(field)
    rhs = float(np.mean(gx * fx + gy * fy))
    assert lhs == pytest.approx(-rhs, rel=1e-6, abs=1e-6)


def test_laplacian_matches_divergence_of_gradient():
    nx, ny = 32, 32
    dx = 2 * np.pi / nx
    dy = 2 * np.pi / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = np.sin(xx + 2 * yy)
    lap_div = divergence(*grad(field, dx=(dx, dy)), dx=(dx, dy))
    lap_direct = laplacian(field, dx=(dx, dy))
    np.testing.assert_allclose(lap_div, lap_direct, rtol=1e-1, atol=1e-1)


def test_hessian_action_periodic():
    nx, ny = 32, 32
    dx = 2 * np.pi / nx
    dy = 2 * np.pi / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    theta = np.sin(2.0 * xx) + np.cos(3.0 * yy)
    gx = 2.0 * np.cos(2.0 * xx)
    gy = -3.0 * np.sin(3.0 * yy)
    hxx = -4.0 * np.sin(2.0 * xx)
    hyy = -9.0 * np.cos(3.0 * yy)
    expected = gx ** 2 * hxx + gy ** 2 * hyy
    result = hessian_action(theta, dx=(dx, dy))
    np.testing.assert_allclose(result, expected, rtol=4e-1, atol=0.5)


def test_edge_metrics_flat_field():
    theta = np.zeros((8, 8))
    metrics = edge_metrics(theta, dx=1.0, grad_threshold=0.01)
    assert metrics["edge_density"] == 0.0
    assert metrics["curvature_stress"] == 0.0


def test_impulse_response_static_dynamics():
    theta = np.zeros((4, 4))
    theta_dot = np.zeros_like(theta)

    def zero_accel(field: np.ndarray) -> np.ndarray:
        return np.zeros_like(field)

    result = impulse_response(theta, theta_dot, zero_accel, dt=0.1, steps=5, amplitude=1e-3)
    assert result["impulse_peak"] > 0.0
    assert result["impulse_decay"] == pytest.approx(1.0, rel=1e-6)


def test_variational_honesty_single_field():
    physics = _physics()
    rng = np.random.default_rng(1)
    theta = rng.normal(scale=0.05, size=(16, 16))
    direction = rng.normal(scale=0.05, size=(16, 16))
    eps = 1e-4

    def energy(arr: np.ndarray) -> float:
        return compute_total_energy(arr, np.zeros_like(arr), physics)

    grad_energy = -physics.beta * accel_theta(theta, physics)
    lhs = float(np.sum(grad_energy * direction) / direction.size)
    rhs = (energy(theta + eps * direction) - energy(theta - eps * direction)) / (2 * eps)
    assert lhs == pytest.approx(rhs, rel=1e-3, abs=1e-6)

def test_cross_variational_honesty():
    physics = _physics(cross_enabled=True)
    rng = np.random.default_rng(2)
    C = rng.normal(scale=5e-3, size=(12, 12))
    K = rng.normal(scale=5e-3, size=(12, 12))
    dC = rng.normal(scale=5e-3, size=(12, 12))
    dK = rng.normal(scale=5e-3, size=(12, 12))
    eps = 1e-5

    def energy(c_field: np.ndarray, k_field: np.ndarray) -> float:
        density = cross_gradient_energy_density(c_field, k_field, physics)
        return float(np.mean(density))

    accel_c, accel_k = accel_CK(C, K, physics)
    lhs = float(np.sum(-physics.cross_gradient.beta * accel_c * dC + -physics.cross_gradient.beta * accel_k * dK))
    rhs = (
        energy(C + eps * dC, K + eps * dK)
        - energy(C - eps * dC, K - eps * dK)
    ) / (2 * eps)
    assert lhs == pytest.approx(rhs, rel=5e-2, abs=1e-6)


def test_symbolizer_read_only():
    rng = np.random.default_rng(3)

    class DummyConfig:
        symbolizer = {
            "bandpass": {"low_cut": 0.05, "high_cut": 0.2},
            "threshold": 0.3,
            "cluster_min_area": 2,
        }

    field = rng.normal(scale=0.1, size=(32, 32))
    guard = field.copy()
    extract_symbols(field, DummyConfig())
    np.testing.assert_array_equal(field, guard)


def test_async_scheduler_mask_shape_and_rate():
    grid = GridSpec(shape=(16, 16), spacing=1.0)
    params = SchedulerParams(mode="poisson", poisson_rate=0.6, jitter=0.0, reseed_every=0)
    scheduler = AsyncScheduler(params, grid, seed=42)
    mask = scheduler.sample_mask(dt=0.1)
    assert mask.shape == grid.shape
    ks = scheduler.ks_statistic(mask.ravel(), dt=0.1)
    assert ks < 0.5


def test_metropolis_accept_monotonic():
    delta = np.array([-1.0, 0.0, 1.0])
    temperature = 0.5
    clip = (0.01, 0.99)
    from .energy import metropolis_accept

    probs = metropolis_accept(delta, temperature, clip)
    assert probs[0] > probs[1] > probs[2]
    assert np.all((probs >= clip[0]) & (probs <= clip[1]))


def test_suggest_cfl_dt_bounds():
    params = IntegrationParams(
        dt=0.1,
        cfl_limit=0.4,
        nudges=NudgeConfig(enable_controller=False, max_step=0.1),
        noise=NoiseConfig(enabled=False, sigma_scale=0.0, seed=0),
    )
    dt = suggest_cfl_dt(params, wave_speed=2.0, dx=1.0)
    assert dt <= params.dt
    assert dt > 0.0


def test_leapfrog_roundtrip():
    physics = _physics()
    rng = np.random.default_rng(4)
    theta = rng.normal(scale=0.01, size=(12, 12))
    theta_dot = rng.normal(scale=0.01, size=(12, 12))
    dt = 5e-5

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, physics)

    forward_state = theta.copy()
    forward_vel = theta_dot.copy()
    for _ in range(10):
        forward_state, forward_vel, _, _ = leapfrog_step(forward_state, forward_vel, accel, dt)

    backward_state = forward_state.copy()
    backward_vel = -forward_vel.copy()
    for _ in range(10):
        backward_state, backward_vel, _, _ = leapfrog_step(backward_state, backward_vel, accel, -dt)

    np.testing.assert_allclose(backward_state, theta, atol=5e-5)


def test_energy_drift_small_series():
    energies = np.linspace(1.0, 1.004, 101)
    drift = energy_drift(energies)
    assert drift == pytest.approx(0.004, rel=1e-6)


def test_validate_near_critical_passes():
    physics = _physics()
    validate_near_critical(physics, bounds=(0.1, 10.0))


def test_validate_near_critical_wave_speed_failure():
    physics = _physics()
    physics.gamma = physics.coherence_linear_term - 1e-4
    with pytest.raises(ValueError):
        validate_near_critical(physics)


def test_validate_near_critical_margin_bounds():
    physics = _physics()
    with pytest.raises(ValueError):
        validate_near_critical(physics, bounds=(10.0, 20.0))


@pytest.mark.slow
def test_longrun_energy_drift():
    physics = _physics()
    rng = np.random.default_rng(5)
    theta = rng.normal(scale=0.002, size=(24, 24))
    theta_dot = np.zeros_like(theta)
    dt = 3e-4

    def accel(field: np.ndarray) -> np.ndarray:
        return accel_theta(field, physics)

    energies = []
    for _ in range(2000):
        theta, theta_dot, _, _ = leapfrog_step(theta, theta_dot, accel, dt)
        energy = compute_total_energy(theta, theta_dot, physics)
        energies.append(energy)
    drift = energy_drift(np.array(energies))
    assert abs(drift) < 5e-3
