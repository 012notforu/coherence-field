from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flow_cylinder import FlowCylinderControlConfig
from .flow_regime_sweep import FlowRegimeParams, FlowRegimeSweep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flow cylinder regime sweep benchmark")
    parser.add_argument("--steps", type=int, default=800, help="Steps per regime")
    parser.add_argument("--record-interval", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="flow_regime_outputs")
    parser.add_argument("--inflows", type=float, nargs="+", default=[0.6, 1.0, 1.4])
    parser.add_argument("--viscosity", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    parser.add_argument("--metrics-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = FlowRegimeParams(
        viscosity=args.viscosity,
        dt=args.dt,
        init_seed=args.seed,
        inflow_values=tuple(args.inflows),
        steps_per_regime=args.steps,
        record_interval=args.record_interval,
    )
    control_cfg = FlowCylinderControlConfig(scfd_cfg_path=args.scfd_cfg)
    sweep = FlowRegimeSweep(params, control_cfg)
    result = sweep.run(seeds=[args.seed])
    aggregate = sweep.aggregate_metrics()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "metrics.npy", np.array(result["metrics"], dtype=object))
    figs = sweep.save_visualizations(out_dir)

    print("Flow regime metrics:")
    for metrics in result["metrics"]:
        print(metrics)
    print("Aggregate:", aggregate)
    print("Artifacts saved to:")
    for key, value in figs.items():
        print(f"  {key}: {value}")

    if args.metrics_json:
        payload = {
            "per_regime": result["metrics"],
            "aggregate": aggregate,
        }
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
