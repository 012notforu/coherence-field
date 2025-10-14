#!/usr/bin/env python3
"""Smoke-test every registered vector on its native benchmark."""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.pipeline import load_vector_registry

def run_maze(vector_entry, steps: int) -> Dict[str, float]:
    from benchmarks.maze_solving import AdaptiveMazeSolver, MazeParams, generate_maze
    params = MazeParams(shape=(16, 16))
    maze = generate_maze(params.shape, wall_density=0.30, seed=hash(vector_entry.vector_id) % 10_000)
    solver = AdaptiveMazeSolver(params, maze, [vector_entry])
    result = solver.solve_adaptively(max_generations=3, steps_per_generation=steps)
    return {
        "solved": bool(result["solved"]),
        "total_steps": int(result["total_steps"]),
        "generations_used": int(result["generations_used"]),
    }

def run_cartpole(vector_entry, steps: int) -> Dict[str, float]:
    from benchmarks.cartpole import CartPoleEnv
    from benchmarks.cartpole_controller import CartPoleController
    env = CartPoleEnv()
    controller = CartPoleController.from_vector_path(vector_entry.path)
    obs, _ = env.reset(seed=hash(vector_entry.vector_id) % 10_000)
    total_reward = 0.0
    for _ in range(steps):
        action = controller.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return {"total_reward": float(total_reward)}

def run_heat(vector_entry, steps: int) -> Dict[str, float]:
    from benchmarks.heat_diffusion import HeatDiffusionController, HeatDiffusionSimulator, HeatDiffusionControlConfig
    cfg = HeatDiffusionControlConfig()
    controller = HeatDiffusionController.from_vector_path(cfg, vector_entry.path)
    sim = HeatDiffusionSimulator()
    metrics = {}
    for _ in range(steps):
        stats = sim.step(controller)
        metrics = stats
    return {k: float(v) for k, v in metrics.items()}

def run_flow(vector_entry, steps: int) -> Dict[str, float]:
    from benchmarks.flow_cylinder import FlowCylinderSimulator, FlowCylinderController, FlowCylinderParams, FlowCylinderControlConfig
    params = FlowCylinderParams()
    control_cfg = FlowCylinderControlConfig()
    controller = FlowCylinderController.from_vector_path(control_cfg, vector_entry.path)
    sim = FlowCylinderSimulator(params, controller)
    metrics = {}
    for _ in range(steps):
        metrics = sim.step()
    return {k: float(v) for k, v in metrics.items()}

def run_wave(vector_entry, steps: int) -> Dict[str, float]:
    from benchmarks.wave_control import WaveController, WaveSimulator, WaveParams, WaveControlConfig
    params = WaveParams()
    control_cfg = WaveControlConfig()
    controller = WaveController.from_vector_path(control_cfg, vector_entry.path)
    sim = WaveSimulator(params, controller)
    metrics = {}
    for _ in range(steps):
        metrics = sim.step()
    return {k: float(v) for k, v in metrics.items()}

SCENARIO_MAP: Tuple[Tuple[str, Callable], ...] = (
    ("maze", run_maze),
    ("grid", run_maze),
    ("cartpole", run_cartpole),
    ("heat", run_heat),
    ("flow", run_flow),
    ("wave", run_wave),
)

def pick_runner(vector_entry) -> Callable[[object, int], Dict[str, float]]:
    tags = {str(t).lower() for t in getattr(vector_entry, "tags", tuple())}
    physics = vector_entry.physics.lower()
    for keyword, runner in SCENARIO_MAP:
        if keyword in physics or keyword in tags:
            return runner
    objective = str(vector_entry.objective).lower()
    for keyword, runner in SCENARIO_MAP:
        if keyword in objective:
            return runner
    raise RuntimeError(f"No scenario mapping for vector '{vector_entry.vector_id}' (physics='{vector_entry.physics}', objective='{vector_entry.objective}')")

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test every registered vector.")
    parser.add_argument("--steps", type=int, default=120, help="Max steps per scenario (default: 120).")
    parser.add_argument("--out", type=str, default="logs/vector_smoke.jsonl", help="JSONL results file.")
    args = parser.parse_args()

    registry = load_vector_registry("runs")
    results = []
    for entry in registry:
        try:
            runner = pick_runner(entry)
        except Exception as exc:
            results.append({
                "vector_id": entry.vector_id,
                "status": "unsupported",
                "error": repr(exc),
            })
            continue

        try:
            metrics = runner(entry, args.steps)
            results.append({
                "vector_id": entry.vector_id,
                "status": "ok",
                "metrics": metrics,
                "physics": entry.physics,
                "objective": entry.objective,
            })
            print(f"[OK] {entry.vector_id:30s} → {metrics}")
        except Exception as exc:
            results.append({
                "vector_id": entry.vector_id,
                "status": "error",
                "error": repr(exc),
                "physics": entry.physics,
                "objective": entry.objective,
            })
            print(f"[FAIL] {entry.vector_id:30s} → {exc!r}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(results)} results to {out_path}")

if __name__ == "__main__":
    main()
