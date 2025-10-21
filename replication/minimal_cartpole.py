#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 012loop / Looptronics
"""Minimal reproducibility harness for SCFD vs. EM cart-pole control."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from benchmarks.em_cartpole import EMCartConfig, EMCartPoleController
from benchmarks.scfd_cartpole import SCFDControllerConfig, SCFDCartPoleController


@dataclass(frozen=True)
class ControllerResult:
    controller: str
    seeds: Sequence[int]
    steps: List[float]
    extra: dict

    def to_summary(self) -> dict:
        mean_steps = statistics.fmean(self.steps)
        std_steps = statistics.pstdev(self.steps) if len(self.steps) > 1 else 0.0
        payload = {
            "controller": self.controller,
            "mean_steps": round(mean_steps, 2),
            "std_steps": round(std_steps, 2),
            "min_steps": round(float(min(self.steps)), 2),
            "max_steps": round(float(max(self.steps)), 2),
            "seeds": list(self.seeds),
        }
        payload.update(self.extra)
        return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_em(seed: int, horizon: int) -> dict:
    """Run the EM baseline with a trimmed-down population for speed."""
    cfg = EMCartConfig(
        B=32,
        n_control_steps=horizon,
        GA_interval=5,
    )
    controller = EMCartPoleController(cfg, rng=np.random.default_rng(seed))
    metrics = controller.run(steps=horizon)
    return {
        "best_steps": float(metrics["best_steps"]),
        "alive": int(metrics["alive"]),
        "mean_steps_alive": float(metrics["mean_steps_alive"]),
    }


def run_scfd(seed: int, horizon: int) -> dict:
    """Run SCFD cart-pole for a short episode budget."""
    cfg = SCFDControllerConfig(
        scfd_cfg_path=str(_repo_root() / "cfg" / "defaults.yaml"),
        micro_steps=40,
        micro_steps_calm=16,
    )
    controller = SCFDCartPoleController(cfg, rng=np.random.default_rng(seed))
    metrics = controller.run_episode(steps=horizon)
    return {
        "steps": float(metrics["steps"]),
        "rms_step": float(metrics.get("rms_step", 0.0)),
    }


def evaluate(seeds: Iterable[int], horizon: int) -> list[ControllerResult]:
    em_steps: list[float] = []
    em_alive: list[int] = []
    for seed in seeds:
        result = run_em(seed, horizon)
        em_steps.append(result["best_steps"])
        em_alive.append(result["alive"])

    scfd_steps: list[float] = []
    scfd_rms: list[float] = []
    for seed in seeds:
        result = run_scfd(seed, horizon)
        scfd_steps.append(result["steps"])
        scfd_rms.append(result["rms_step"])

    em_extra = {"mean_alive": round(statistics.fmean(em_alive), 2)}
    scfd_extra = {"mean_rms_step": round(statistics.fmean(scfd_rms), 4)}
    return [
        ControllerResult("EM baseline", tuple(seeds), em_steps, em_extra),
        ControllerResult("SCFD controller", tuple(seeds), scfd_steps, scfd_extra),
    ]


def render_table(results: Sequence[ControllerResult]) -> str:
    header = ["Controller", "Mean Steps", "Std", "Min", "Max", "Seeds", "Notes"]
    rows = []
    for res in results:
        summary = res.to_summary()
        note_parts = []
        if "mean_alive" in summary:
            note_parts.append(f"mean_alive={summary['mean_alive']}")
        if "mean_rms_step" in summary:
            note_parts.append(f"mean_rms_step={summary['mean_rms_step']}")
        rows.append(
            [
                summary["controller"],
                f"{summary['mean_steps']:.2f}",
                f"{summary['std_steps']:.2f}",
                f"{summary['min_steps']:.2f}",
                f"{summary['max_steps']:.2f}",
                ",".join(str(s) for s in summary["seeds"]),
                "; ".join(note_parts) if note_parts else "-",
            ]
        )

    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    lines = [
        " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
        "-+-".join("-" * widths[i] for i in range(len(header))),
    ]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(header))))
    return "\n".join(lines)


def write_expected(results: Sequence[ControllerResult], path: Path) -> None:
    payload = {res.controller: res.to_summary() for res in results}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal SCFD vs EM replication table.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to evaluate for both controllers.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=600,
        help="Step horizon for each controller evaluation.",
    )
    parser.add_argument(
        "--update-expected",
        action="store_true",
        help="Rewrite expected.json with the fresh metrics.",
    )
    parser.add_argument(
        "--expected-path",
        type=str,
        default=str(_repo_root() / "replication" / "expected.json"),
        help="Override path for expected metrics dump.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    results = evaluate(seeds, args.horizon)
    print(render_table(results))
    if args.update_expected:
        write_expected(results, Path(args.expected_path))


if __name__ == "__main__":
    main()
