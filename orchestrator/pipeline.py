from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.heat_diffusion import HeatDiffusionSimulator
from benchmarks.heat_diffusion_arc import HeatDiffusionArcSimulator

Numeric = (int, float, np.floating, np.integer)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def _default_tuple(value: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    return tuple(value)


@dataclass(frozen=True)
class ProbeReport:
    physics: str
    grid_shape: Tuple[int, int]
    target_kind: str
    transform_cycle: Tuple[str, ...] = field(default_factory=tuple)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def transform_cycle_length(self) -> int:
        return len(self.transform_cycle)


@dataclass(frozen=True)
class VectorEntry:
    vector_id: str
    path: Path
    physics: str
    objective: str
    tags: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PlanStep:
    vector: VectorEntry
    mode: str = "sequence"  # future: blend/parallel
    duration: Optional[int] = None


@dataclass
class Plan:
    probe: ProbeReport
    steps: List[PlanStep]

    def summary(self) -> Dict[str, object]:
        return {
            "physics": self.probe.physics,
            "target_kind": self.probe.target_kind,
            "steps": [step.vector.vector_id for step in self.steps],
        }


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------


def probe_environment(
    simulator_factory: Callable[[], object],
    *,
    steps: int = 200,
    record_interval: int = 40,
) -> ProbeReport:
    """Run a short rollout to sense the environment."""

    simulator = simulator_factory()
    physics = _infer_physics(simulator)
    grid_shape = _infer_grid_shape(simulator)

    target_kind = _infer_target_kind(simulator)
    transform_cycle = _infer_transform_cycle(simulator)

    energy_values: List[float] = []
    mse_values: List[float] = []

    # Attempt to access target for error estimation
    target = getattr(simulator, "target", None)

    for step in range(steps):
        stats = simulator.step()
        energy_values.append(float(stats.get("energy", 0.0)))
        if target is not None and "mse" in stats:
            mse_values.append(float(stats["mse"]))
        if (step % record_interval) == 0 and hasattr(simulator, "_advance_transform"):
            # allow ARC simulators to expose transform changes early
            transform_cycle = _infer_transform_cycle(simulator)

    stats: Dict[str, float] = {}
    if energy_values:
        stats["energy_mean"] = float(np.mean(energy_values))
        stats["energy_std"] = float(np.std(energy_values))
        stats["energy_drift"] = float(energy_values[-1] - energy_values[0])
    if mse_values:
        stats["mse_mean"] = float(np.mean(mse_values))
        stats["mse_std"] = float(np.std(mse_values))
        stats["mse_final"] = float(mse_values[-1])

    return ProbeReport(
        physics=physics,
        grid_shape=grid_shape,
        target_kind=target_kind,
        transform_cycle=_default_tuple(transform_cycle),
        stats=stats,
    )


def _infer_physics(simulator: object) -> str:
    name = simulator.__class__.__name__.lower()
    if "heat" in name:
        return "heat"
    if "flow" in name:
        return "flow"
    if "wave" in name:
        return "wave"
    return "unknown"


def _infer_grid_shape(simulator: object) -> Tuple[int, int]:
    params = getattr(simulator, "params", None)
    shape = getattr(params, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    if hasattr(simulator, "temp"):
        arr = simulator.temp
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return (arr.shape[0], arr.shape[1])
    return (0, 0)


def _infer_transform_cycle(simulator: object) -> Tuple[str, ...]:
    cycle = getattr(simulator, "transform_cycle", None)
    if cycle is None:
        return tuple()
    if isinstance(cycle, (list, tuple)):
        return tuple(str(item) for item in cycle)
    return (str(cycle),)


def _infer_target_kind(simulator: object) -> str:
    if hasattr(simulator, "hidden_alpha"):
        return "param_id"
    if isinstance(simulator, HeatDiffusionArcSimulator):
        return "arc_cycle"

    target = getattr(simulator, "target", None)
    if isinstance(target, np.ndarray):
        return _classify_target_array(target)
    return "unknown"


def _classify_target_array(target: np.ndarray) -> str:
    h, w = target.shape
    center_val = float(target[h // 2, w // 2])
    yy, xx = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing="ij")
    dist = np.sqrt(xx ** 2 + yy ** 2)
    max_radius = np.sqrt(0.5 ** 2 + 0.5 ** 2)
    norm_dist = dist / max_radius

    bins = np.linspace(0.0, 1.0, 8)
    radial_means = []
    for i in range(len(bins) - 1):
        mask = (norm_dist >= bins[i]) & (norm_dist < bins[i + 1])
        if np.any(mask):
            radial_means.append(float(target[mask].mean()))
        else:
            radial_means.append(0.0)
    peak_idx = int(np.argmax(radial_means))
    peak_radius = (bins[peak_idx] + bins[peak_idx + 1]) * 0.5 if len(bins) > 1 else 0.0
    if center_val < 0.2 and radial_means[peak_idx] > 0.3 and 0.15 < peak_radius < 0.45:
        return "ring"

    max_pos = np.unravel_index(np.argmax(target), target.shape)
    min_pos = np.unravel_index(np.argmin(target), target.shape)
    max_norm = (max_pos[0] / max(1, h - 1), max_pos[1] / max(1, w - 1))
    min_norm = (min_pos[0] / max(1, h - 1), min_pos[1] / max(1, w - 1))

    if max_norm[0] < 0.35 and max_norm[1] < 0.35 and min_norm[0] > 0.65 and min_norm[1] > 0.65:
        return "hot_corner"
    if min_norm[0] < 0.35 and min_norm[1] < 0.35 and max_norm[0] > 0.65 and max_norm[1] > 0.65:
        return "cool_spot"

    flat = target.flatten()
    if flat.size >= 4:
        top_indices = np.argpartition(flat, -4)[-4:]
        coords = np.array([np.unravel_index(idx, target.shape) for idx in top_indices], dtype=np.float32)
        coords[:, 0] /= max(1, h - 1)
        coords[:, 1] /= max(1, w - 1)
        max_dist = 0.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                max_dist = max(max_dist, float(np.linalg.norm(coords[i] - coords[j])))
        if max_dist > 0.25 and center_val < 0.5:
            return "multi_blob"

    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    gradient = np.broadcast_to(x, (h, w))
    gradient_rev = np.broadcast_to(x[::-1], (h, w))
    diff_grad = np.linalg.norm(target - gradient) / (np.linalg.norm(gradient) + 1e-6)
    diff_grad_rev = np.linalg.norm(target - gradient_rev) / (np.linalg.norm(gradient_rev) + 1e-6)
    if diff_grad < 0.25 or diff_grad_rev < 0.25:
        return "gradient"

    return "unknown"


# ---------------------------------------------------------------------------
# Vector registry
# ---------------------------------------------------------------------------


def load_vector_registry(runs_root: Path | str = "runs") -> List[VectorEntry]:
    runs_path = Path(runs_root)
    entries: List[VectorEntry] = []
    if not runs_path.exists():
        return entries

    for best_vector in runs_path.glob("*/best_vector.json"):
        vector_id = best_vector.parent.name
        physics, objective = _infer_vector_metadata(vector_id)
        try:
            payload = json.loads(best_vector.read_text())
        except json.JSONDecodeError:
            payload = {}
        metrics = payload.get("metrics", {})
        metadata: Dict[str, object] = {"metrics": metrics}
        if "transform_cycle" in payload:
            cycle = tuple(str(x) for x in payload.get("transform_cycle", []))
            metadata["transform_cycle"] = cycle
            metadata["transform_cycle_length"] = len(cycle)
        if "transform_cycle_interval" in payload:
            metadata["transform_cycle_interval"] = int(payload["transform_cycle_interval"])
        if "base_target_kind" in payload:
            metadata["base_target_kind"] = payload["base_target_kind"]
        if "alpha" in payload:
            metadata["alpha"] = float(payload["alpha"])
        if "dt" in payload:
            metadata["dt"] = float(payload["dt"])
        if "noise" in payload:
            metadata["noise"] = float(payload["noise"])
        if "shape" in payload:
            shape = tuple(int(x) for x in payload.get("shape", []))
            if shape:
                metadata["shape"] = shape
        if "initial_centers" in payload:
            centers = tuple(tuple(float(coord) for coord in center) for center in payload.get("initial_centers", []))
            if centers:
                metadata["initial_centers"] = centers
                metadata["initial_center_count"] = len(centers)
        if "target_centers" in payload:
            centers = tuple(tuple(float(coord) for coord in center) for center in payload.get("target_centers", []))
            if centers:
                metadata["target_centers"] = centers
        if "blob_sigma" in payload:
            metadata["blob_sigma"] = float(payload["blob_sigma"])
        if "collision_radius" in payload:
            metadata["collision_radius"] = float(payload["collision_radius"])
        if "front_radius" in payload:
            metadata["front_radius"] = float(payload["front_radius"])
        if "front_width" in payload:
            metadata["front_width"] = float(payload["front_width"])
        if "alpha_low" in payload:
            metadata["alpha_low"] = float(payload["alpha_low"])
        if "alpha_high" in payload:
            metadata["alpha_high"] = float(payload["alpha_high"])
        if "split_axis" in payload:
            metadata["split_axis"] = str(payload["split_axis"])
        if "controller_config" in payload:
            metadata["controller_config"] = payload["controller_config"]
        if "training" in payload:
            metadata["training"] = payload["training"]
        if "physics_override" in payload:
            metadata["physics_override"] = payload["physics_override"]
        if "task" in payload:
            metadata["task"] = payload["task"]
        tags = tuple(sorted({physics, objective}))
        entries.append(
            VectorEntry(
                vector_id=vector_id,
                path=best_vector,
                physics=physics,
                objective=objective,
                tags=tags,
                metadata=metadata,
            )
        )
    return entries


def _infer_vector_metadata(vector_id: str) -> Tuple[str, str]:
    name = vector_id.lower()
    if name.startswith("heat_arc"):
        return "heat", "arc_cycle"
    if "heat_diffusion" in name and "hotcorner" in name:
        return "heat", "hot_corner"
    if "heat_diffusion" in name and "coolspot" in name:
        return "heat", "cool_spot"
    if "heat_diffusion" in name and "periodic" in name:
        return "heat", "periodic"
    if "heat_diffusion" in name and "routing" in name:
        return "heat", "routing"
    if "heat_diffusion" in name and "front" in name:
        return "heat", "front"
    if "heat_diffusion" in name and "param" in name:
        return "heat", "param_id"
    if "heat_diffusion" in name and "anisotropic" in name:
        return "heat", "anisotropic"
    if "heat_diffusion" in name:
        return "heat", "gradient"
    if name.startswith("flow"):
        return "flow", vector_id
    if name.startswith("wave"):
        return "wave", vector_id
    if "cartpole" in name:
        return "cartpole", "balance"
    return "unknown", vector_id


# ---------------------------------------------------------------------------
# Planning heuristics
# ---------------------------------------------------------------------------


def build_plan(probe: ProbeReport, registry: Sequence[VectorEntry]) -> Plan:
    candidates = [entry for entry in registry if entry.physics == probe.physics]
    if not candidates:
        candidates = list(registry)

    desired_objectives: List[str] = []
    if probe.transform_cycle_length > 1:
        desired_objectives.append("arc_cycle")
    if probe.target_kind == "hot_corner":
        desired_objectives.append("hot_corner")
    if probe.target_kind == "cool_spot":
        desired_objectives.append("cool_spot")
    if probe.target_kind == "periodic":
        desired_objectives.append("periodic")
    if probe.target_kind == "anisotropic":
        desired_objectives.append("anisotropic")
    desired_objectives.append("gradient")  # fallback for heat

    selected: Optional[VectorEntry] = None
    for objective in desired_objectives:
        matching = [entry for entry in candidates if entry.objective == objective]
        if not matching:
            continue
        if objective == "arc_cycle" and probe.transform_cycle_length > 0:
            probe_cycle = tuple(name.lower() for name in probe.transform_cycle)
            exact = [
                entry
                for entry in matching
                if tuple(str(name).lower() for name in entry.metadata.get("transform_cycle", ()))
                == probe_cycle
            ]
            if exact:
                selected = exact[0]
                break
            length_match = [
                entry
                for entry in matching
                if entry.metadata.get("transform_cycle_length") == probe.transform_cycle_length
            ]
            if length_match:
                selected = length_match[0]
                break
        selected = matching[0]
        break

    if selected is None and candidates:
        selected = candidates[0]

    if selected is None:
        raise RuntimeError("No vector entries available to build a plan")

    return Plan(probe=probe, steps=[PlanStep(vector=selected)])


def plan_for_environment(
    simulator_factory: Callable[[], object],
    *,
    registry: Optional[Sequence[VectorEntry]] = None,
    steps: int = 200,
    record_interval: int = 40,
) -> Plan:
    probe = probe_environment(simulator_factory, steps=steps, record_interval=record_interval)
    if registry is None:
        registry = load_vector_registry()
    return build_plan(probe, registry)


__all__ = [
    "ProbeReport",
    "VectorEntry",
    "PlanStep",
    "Plan",
    "probe_environment",
    "load_vector_registry",
    "build_plan",
    "plan_for_environment",
]
