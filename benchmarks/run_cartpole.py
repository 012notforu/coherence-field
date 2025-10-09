"""Head-to-head cart-pole benchmark for EM and SCFD controllers."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .em_cartpole import EMCartPoleController, EMCartConfig
from .scfd_cartpole import SCFDCartPoleController, SCFDControllerConfig
from run.train_cma_scfd import _config_from_metadata, _config_from_vector


def _parse_weights(weights: Optional[str], expected: int) -> Optional[np.ndarray]:
    if not weights:
        return None
    values = [float(w) for w in weights.split(",") if w.strip()]
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != expected:
        raise ValueError(f"Expected {expected} SCFD weights, got {arr.size}")
    return arr


def _load_scfd_config(args: argparse.Namespace) -> Tuple[SCFDControllerConfig, Optional[Dict[str, object]]]:
    base_cfg = SCFDControllerConfig()
    payload: Optional[Dict[str, object]] = None
    cfg = base_cfg

    if args.vector:
        payload = json.loads(Path(args.vector).read_text())
        controller_meta = payload.get("controller_config")
        if controller_meta:
            cfg = _config_from_metadata(controller_meta)
        else:
            vector = np.asarray(payload["vector"], dtype=np.float32)
            cfg = _config_from_vector(base_cfg, vector)

    if args.scfd_cfg is not None:
        cfg.scfd_cfg_path = args.scfd_cfg
    if args.scfd_micro_steps is not None:
        cfg.micro_steps = int(args.scfd_micro_steps)
    if args.scfd_micro_steps_calm is not None:
        cfg.micro_steps_calm = int(args.scfd_micro_steps_calm)
    if cfg.micro_steps_calm is None:
        cfg.micro_steps_calm = max(4, cfg.micro_steps // 2)
    if args.scfd_encode_gain is not None:
        cfg.encode_gain = float(args.scfd_encode_gain)
    if args.scfd_encode_width is not None:
        cfg.encode_width = int(args.scfd_encode_width)
    if args.scfd_decay is not None:
        cfg.decay = float(args.scfd_decay)
    if args.scfd_smooth_lambda is not None:
        cfg.smooth_lambda = float(args.scfd_smooth_lambda)
    if args.scfd_deadzone_angle is not None:
        cfg.deadzone_angle = float(np.deg2rad(args.scfd_deadzone_angle))
    if args.scfd_deadzone_ang_vel is not None:
        cfg.deadzone_ang_vel = float(args.scfd_deadzone_ang_vel)
    if args.scfd_action_delta is not None:
        cfg.action_delta_clip = float(args.scfd_action_delta)
    weight_override = _parse_weights(args.scfd_weights, cfg.policy_weights.size)
    if weight_override is not None:
        cfg.policy_weights = weight_override.astype(np.float32)
    if args.scfd_gain_energy is not None:
        cfg.gain_energy = float(args.scfd_gain_energy)
    if args.scfd_gain_angle is not None:
        cfg.gain_angle = float(args.scfd_gain_angle)
    if args.scfd_gain_ang_vel is not None:
        cfg.gain_ang_vel = float(args.scfd_gain_ang_vel)

    return cfg, payload


def run_em(args: argparse.Namespace) -> Tuple[Dict[str, float], EMCartPoleController]:
    cfg = EMCartConfig(
        B=args.em_population,
        n_control_steps=args.steps,
        GA_interval=args.ga_interval,
    )
    controller = EMCartPoleController(cfg)
    start = time.perf_counter()
    metrics = controller.run(steps=args.steps)
    metrics["elapsed_sec"] = time.perf_counter() - start
    return metrics, controller


def run_scfd(args: argparse.Namespace) -> Tuple[Dict[str, float], SCFDCartPoleController]:
    cfg, payload = _load_scfd_config(args)
    rng = np.random.default_rng(args.scfd_seed) if args.scfd_seed is not None else None
    controller = SCFDCartPoleController(cfg, rng=rng)
    results = []
    start = time.perf_counter()
    for _ in range(args.episodes):
        results.append(controller.run_episode(steps=args.steps))
    elapsed = time.perf_counter() - start
    steps = np.array([r["steps"] for r in results], dtype=np.float32)
    metrics = {
        "mean_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "max_steps": float(steps.max()),
        "min_steps": float(steps.min()),
        "elapsed_sec": elapsed,
    }
    if payload is not None:
        metrics["vector_path"] = args.vector
    return metrics, controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cart-pole benchmark for EM and SCFD controllers")
    parser.add_argument("--controller", choices=["em", "scfd", "both"], default="both")
    parser.add_argument("--steps", type=int, default=20000, help="Number of control steps per evaluation")
    parser.add_argument("--episodes", type=int, default=5, help="SCFD episodes to average over")
    parser.add_argument("--em-population", type=int, default=256, dest="em_population")
    parser.add_argument("--ga-interval", type=int, default=10, dest="ga_interval")
    parser.add_argument("--vector", type=str, default=None, help="Path to CMA-trained SCFD vector JSON")
    parser.add_argument("--scfd-cfg", type=str, default=None, help="Override SCFD YAML config path")
    parser.add_argument("--scfd-micro-steps", type=int, default=None)
    parser.add_argument("--scfd-micro-steps-calm", type=int, metavar="MICRO", default=None)
    parser.add_argument("--scfd-encode-gain", type=float, default=None)
    parser.add_argument("--scfd-encode-width", type=int, default=None)
    parser.add_argument("--scfd-decay", type=float, default=None)
    parser.add_argument("--scfd-smooth-lambda", type=float, default=None)
    parser.add_argument("--scfd-deadzone-angle", type=float, default=None, help="Deadzone angle in degrees")
    parser.add_argument("--scfd-deadzone-ang-vel", type=float, default=None)
    parser.add_argument("--scfd-action-delta", type=float, default=None)
    parser.add_argument("--scfd-weights", type=str, default=None, help="Comma-separated weights for feature policy")
    parser.add_argument("--scfd-gain-energy", type=float, default=None)
    parser.add_argument("--scfd-gain-angle", type=float, default=None)
    parser.add_argument("--scfd-gain-ang-vel", type=float, default=None)
    parser.add_argument("--scfd-seed", type=int, default=None, help="Seed for SCFD controller rollouts")
    parser.add_argument("--viz", choices=["none", "em", "scfd", "both"], default="none")
    parser.add_argument("--deterministic", action="store_true", help="freeze non-essential RNG")
    parser.add_argument("--video-format", choices=["auto", "mp4", "gif"], default="auto", help="Format for cart-pole visualization output")
    parser.add_argument("--viz-steps", type=int, default=800)
    parser.add_argument("--outdir", default="cartpole_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set deterministic behavior if requested
    if args.deterministic:
        import random
        seed = getattr(args, 'scfd_seed', 42) or 42
        random.seed(seed)
        np.random.seed(seed)
    
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    em_controller = None
    if args.controller in ("em", "both"):
        em_metrics, em_controller = run_em(args)
        print("EM controller metrics:")
        for k, v in em_metrics.items():
            print(f"  {k}: {v}")

    scfd_controller = None
    if args.controller in ("scfd", "both"):
        scfd_metrics, scfd_controller = run_scfd(args)
        print("SCFD controller metrics:")
        for k, v in scfd_metrics.items():
            print(f"  {k}: {v}")

    if args.viz in ("em", "both") and em_controller is not None:
        viz_dir = out_dir / "em"
        result = em_controller.generate_visualization(horizon=args.viz_steps, out_dir=viz_dir)
        print("EM visualization outputs:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.viz in ("scfd", "both") and scfd_controller is not None:
        viz_dir = out_dir / "scfd"
        result = scfd_controller.generate_visualization(
            steps=args.viz_steps,
            out_dir=viz_dir,
            video_format=args.video_format,
        )
        print("SCFD visualization outputs:")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
