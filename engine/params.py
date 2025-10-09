from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml


@dataclass
class RunParams:
    tag: str
    steps: int
    seed: int
    outdir: Optional[str]


@dataclass
class GridSpec:
    shape: tuple[int, int]
    spacing: float

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass
class PotentialConfig:
    kind: str
    params: Dict[str, float]

    def derivative(self, theta: np.ndarray) -> np.ndarray:
        if self.kind == "double_well":
            theta0 = self.params.get("theta0", 0.0)
            theta1 = self.params.get("theta1", 1.0)
            stiffness = self.params.get("stiffness", 0.2)
            term0 = theta - theta0
            term1 = theta - theta1
            return 2.0 * stiffness * term0 * term1 * (2.0 * theta - theta0 - theta1)
        if self.kind == "quadratic":
            center = self.params.get("center", 0.0)
            stiffness = self.params.get("stiffness", 1.0)
            return 2.0 * stiffness * (theta - center)
        raise ValueError(f"Unsupported potential kind: {self.kind}")

    def energy(self, theta: np.ndarray) -> np.ndarray:
        if self.kind == "double_well":
            theta0 = self.params.get("theta0", 0.0)
            theta1 = self.params.get("theta1", 1.0)
            stiffness = self.params.get("stiffness", 0.2)
            term0 = theta - theta0
            term1 = theta - theta1
            return stiffness * term0 ** 2 * term1 ** 2
        if self.kind == "quadratic":
            center = self.params.get("center", 0.0)
            stiffness = self.params.get("stiffness", 1.0)
            return stiffness * (theta - center) ** 2
        raise ValueError(f"Unsupported potential kind: {self.kind}")


@dataclass
class CurvaturePenaltyConfig:
    enabled: bool
    delta: float


@dataclass
class CrossGradientConfig:
    enabled: bool
    gamma: float
    epsilon: float
    beta: float = 1.0
    parabolic_coeff: float = 0.0


@dataclass
class HeterogeneityConfig:
    amplitude: float
    correlation_length: int
    seed: int

    def generate_field(self, grid: GridSpec) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(size=grid.shape)
        kernel = self._gaussian_kernel(grid)
        field = np.fft.ifft2(np.fft.fft2(noise) * np.fft.fft2(kernel, s=grid.shape)).real
        field = field / (np.std(field) + 1e-8)
        return self.amplitude * field

    def _gaussian_kernel(self, grid: GridSpec) -> np.ndarray:
        lx, ly = grid.shape
        x = np.arange(lx) - lx // 2
        y = np.arange(ly) - ly // 2
        xx, yy = np.meshgrid(x, y, indexing="ij")
        sigma = max(1.0, self.correlation_length)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel


@dataclass
class FreeEnergyGateTemperature:
    min: float
    max: float
    ema_tau: float


@dataclass
class MetropolisConfig:
    clip: tuple[float, float]
    logit_scale: float


@dataclass
class FreeEnergyGateConfig:
    enabled: bool
    temperature: FreeEnergyGateTemperature
    metropolis: MetropolisConfig


@dataclass
class PhysicsParams:
    beta: float
    gamma: float
    alpha: float
    coherence_p: int
    epsilon: float
    potential: PotentialConfig
    cross_gradient: CrossGradientConfig
    curvature_penalty: CurvaturePenaltyConfig
    heterogeneity: HeterogeneityConfig

    @property
    def coherence_exponent(self) -> float:
        return (self.coherence_p + 2.0) / 2.0

    @property
    def coherence_linear_term(self) -> float:
        p = float(max(self.coherence_p, 1))
        return self.alpha * p / (self.epsilon ** (self.coherence_p + 2))

    @property
    def wave_speed_sq(self) -> float:
        return (self.gamma - self.coherence_linear_term) / self.beta

    def coherence_margin(self) -> float:
        denom = max(self.coherence_linear_term, 1e-12)
        return self.gamma / denom


@dataclass
class NoiseConfig:
    enabled: bool
    sigma_scale: float
    seed: int


@dataclass
class NudgeConfig:
    enable_controller: bool
    max_step: float


@dataclass
class IntegrationParams:
    dt: float
    cfl_limit: float
    nudges: NudgeConfig
    noise: NoiseConfig


@dataclass
class SchedulerParams:
    mode: str
    poisson_rate: float
    jitter: float
    reseed_every: int


@dataclass
class LoggingParams:
    interval: int
    spectra_interval: int
    impulse_interval: int


@dataclass
class ObservationParams:
    trackers: Dict[str, Any]
    autoencoder: Dict[str, Any]
    controller: Dict[str, Any]


@dataclass
class EmBaselineParams:
    grid_shape: tuple[int, int]
    steps: int
    kernel_radius: int
    kernel_sigma: float
    activation: str
    halting: Dict[str, Any]
    optimizer: Dict[str, Any]


@dataclass
class SimulationConfig:
    run: RunParams
    grid: GridSpec
    physics: PhysicsParams
    integration: IntegrationParams
    scheduler: SchedulerParams
    free_energy_gate: FreeEnergyGateConfig
    symbolizer: Dict[str, Any]
    observation: ObservationParams
    logging: LoggingParams
    em_baseline: EmBaselineParams

    def startup_summary(self) -> str:
        c2 = self.physics.wave_speed_sq
        margin = self.physics.coherence_margin()
        dt = self.integration.dt
        dx = self.grid.spacing
        cfl = np.sqrt(max(c2, 0.0)) * dt / dx
        temp = self.free_energy_gate.temperature
        return (
            f"[SCFD] c^2={c2:.5e}, margin={margin:.3f}, CFL={cfl:.3f}\n"
            f"[Gate] T=[{temp.min:.3f},{temp.max:.3f}], ema_tau={temp.ema_tau:.1f}, "
            f"scheduler={self.scheduler.mode}"
        )


def load_config(path: str | Path) -> SimulationConfig:
    data = _read_yaml(path)
    run = RunParams(**data["run"])
    grid = GridSpec(shape=tuple(data["grid"]["shape"]), spacing=float(data["grid"]["spacing"]))
    potential = PotentialConfig(**data["physics"]["potential"])
    curvature = CurvaturePenaltyConfig(**data["physics"]["curvature_penalty"])
    cross = CrossGradientConfig(**data["physics"]["cross_gradient"])
    heterogeneity = HeterogeneityConfig(**data["physics"]["heterogeneity"])
    physics = PhysicsParams(
        beta=float(data["physics"]["beta"]),
        gamma=float(data["physics"]["gamma"]),
        alpha=float(data["physics"]["alpha"]),
        coherence_p=int(data["physics"]["coherence_p"]),
        epsilon=float(data["physics"]["epsilon"]),
        potential=potential,
        cross_gradient=cross,
        curvature_penalty=curvature,
        heterogeneity=heterogeneity,
    )
    noise = NoiseConfig(**data["integration"]["noise"])
    nudges = NudgeConfig(**data["integration"]["nudges"])
    integration = IntegrationParams(
        dt=float(data["integration"]["dt"]),
        cfl_limit=float(data["integration"]["cfl_limit"]),
        nudges=nudges,
        noise=noise,
    )
    scheduler = SchedulerParams(**data["scheduler"])
    temp = FreeEnergyGateTemperature(**data["free_energy_gate"]["temperature"])
    metropolis = MetropolisConfig(
        clip=tuple(data["free_energy_gate"]["metropolis"]["clip"]),
        logit_scale=float(data["free_energy_gate"]["metropolis"]["logit_scale"]),
    )
    gate = FreeEnergyGateConfig(
        enabled=bool(data["free_energy_gate"]["enabled"]),
        temperature=temp,
        metropolis=metropolis,
    )
    observation = ObservationParams(**data["observation"])
    logging_cfg = LoggingParams(**data["logging"])
    em_params = EmBaselineParams(
        grid_shape=tuple(data["em_baseline"]["grid_shape"]),
        steps=int(data["em_baseline"]["steps"]),
        kernel_radius=int(data["em_baseline"]["kernel_radius"]),
        kernel_sigma=float(data["em_baseline"]["kernel_sigma"]),
        activation=data["em_baseline"]["activation"],
        halting=data["em_baseline"]["halting"],
        optimizer=data["em_baseline"]["optimizer"],
    )
    config = SimulationConfig(
        run=run,
        grid=grid,
        physics=physics,
        integration=integration,
        scheduler=scheduler,
        free_energy_gate=gate,
        symbolizer=data["symbolizer"],
        observation=observation,
        logging=logging_cfg,
        em_baseline=em_params,
    )
    validate_near_critical(config.physics)
    return config


def validate_near_critical(physics: PhysicsParams, bounds: Tuple[float, float] = (0.1, 10.0)) -> None:
    if physics.wave_speed_sq <= 0.0:
        raise ValueError("Effective wave speed squared must be positive; adjust parameters")
    margin = physics.coherence_margin()
    if not (bounds[0] <= margin <= bounds[1]):
        raise ValueError(
            f"Coherence margin {margin:.3f} outside acceptable range {bounds}"
        )


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

