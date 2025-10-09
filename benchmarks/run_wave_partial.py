from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .wave_field_partial import (
    WavePartialParams,
    WavePartialControlConfig,
    WavePartialController,
    WavePartialSimulator,
    random_sensor_mask,
)
from .wave_field import synthetic_wave_target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wave-field benchmark with partial sensors and delay")
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--record-interval", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="wave_partial_outputs")
    parser.add_argument("--target", type=str, default="focus", choices=["focus", "defocus", "standing"])
    parser.add_argument("--sensor-fraction", type=float, default=0.2)
    parser.add_argument("--action-delay", type=int, default=3)
    parser.add_argument("--wave-speed", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    parser.add_argument("--metrics-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = WavePartialParams(
        wave_speed=args.wave_speed,
        dt=args.dt,
        init_seed=args.seed,
        sensor_fraction=args.sensor_fraction,
        action_delay=args.action_delay,
    )
    control_cfg = WavePartialControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = WavePartialController(control_cfg, params.shape, params.action_delay)
    rng = np.random.default_rng(args.seed)
    mask = random_sensor_mask(params.shape, params.sensor_fraction, rng)
    target = synthetic_wave_target(params.shape, kind=args.target if args.target != "standing" else "focus")
    simulator = WavePartialSimulator(params, controller, target, mask)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(out_dir, history)

    print("Wave partial metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")

    if args.metrics_json:
        payload = {
            "metrics": history["metrics"],
            "last": history["metrics"][-1],
            "sensor_fraction": args.sensor_fraction,
            "action_delay": args.action_delay,
        }
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
