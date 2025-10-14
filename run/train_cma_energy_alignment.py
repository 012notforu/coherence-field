#!/usr/bin/env python3
"""
Evolve MNIST ⇢ SCFD energy alignment parameters with CMA-ES.

This script optimizes the parameters (mass_scale, accel_scale, c_const) so that
the expected energy matches the SCFD energy computed from CNN activations.

Usage:

    python -m run.train_cma_energy_alignment --generations 12 --population 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cma
import numpy as np

from .mnist_energy_core import (
    EnergyParams,
    energy_reward,
    load_or_train_cnn,
    sample_activation_fields,
    get_mnist_dataloaders,
)


def evaluate_vector(
    vector: List[float],
    fields: List[np.ndarray],
) -> float:
    params = EnergyParams.from_vector(vector)
    diffs = []
    for field in fields:
        metrics = energy_reward(field, params)
        diffs.append(abs(metrics["diff"]))
    # CMA minimises objective -> mean absolute difference
    return float(np.mean(diffs))


def run_cma(
    model_path: Path,
    outdir: Path,
    generations: int,
    population: int,
    sigma: float,
    sample_size: int,
    train_limit: int,
) -> Dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    model = load_or_train_cnn(model_path, train_limit=train_limit)
    _, test_loader = get_mnist_dataloaders(batch_size=64, test_limit=sample_size)
    fields = sample_activation_fields(model, test_loader, count=sample_size)

    initial_vector = [1.0, 1.0, 0.12]
    options = {
        "popsize": population,
        "maxiter": generations,
        "verb_disp": 1,
    }
    es = cma.CMAEvolutionStrategy(initial_vector, sigma, options)

    history = []
    while not es.stop():
        solutions = es.ask()
        fitness = [evaluate_vector(list(sol), fields) for sol in solutions]
        es.tell(solutions, fitness)
        best_idx = int(np.argmin(fitness))
        history.append(
            {
                "iter": len(history),
                "best_vector": list(map(float, solutions[best_idx])),
                "best_fitness": float(fitness[best_idx]),
                "mean_fitness": float(np.mean(fitness)),
            }
        )
        es.disp()

    result = es.result
    best_vector = list(map(float, result.xbest))
    best_fitness = float(result.fbest)

    history_path = outdir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    metrics = {
        "best_vector": best_vector,
        "best_fitness": best_fitness,
        "iterations": len(history),
        "evaluations": int(result.evaluations),
        "sigma": sigma,
        "sample_size": sample_size,
    }
    return metrics


def save_vector(outdir: Path, metrics: Dict[str, float]) -> None:
    vector_path = outdir / "best_vector.json"
    payload = {
        "vector": metrics["best_vector"],
        "metrics": metrics,
        "physics_domain": "nn_interpretability",
        "objective": "mnist_energy_alignment",
        "activation_layer": "conv2",
        "tags": ["mnist", "energy_alignment", "interpretability"],
    }
    vector_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote best vector to {vector_path}")


def write_manifest(outdir: Path) -> None:
    manifest = {
        "manifest_version": "1.0",
        "vector_id": outdir.name,
        "description": "CMA-evolved MNIST ⇢ SCFD energy alignment parameters.",
        "element_count": 3,
        "element_roles": ["mass_scale", "accel_scale", "c_const"],
        "element_ranges": [[0.01, 10.0], [0.01, 10.0], [0.01, 1.0]],
        "element_units": ["scale", "scale", "scale"],
        "element_dtypes": ["float_linear", "float_linear", "float_linear"],
        "param_mappings": {"0": "mass_scale", "1": "accel_scale", "2": "c_const"},
        "domain_tags": ["nn_interpretability", "grid"],
        "selection_predicate": "task in ['nn_interpretability', 'grid_nav']",
        "priority": 1.2,
        "invariants": ["mass_scale>0", "accel_scale>0", "c_const>0"],
        "coverage_requirement": 0.0,
        "trust_region": {
            "mass_scale": [0.05, 8.0],
            "accel_scale": [0.05, 8.0],
            "c_const": [0.02, 0.5],
        },
        "cooldown_steps": 4,
        "success_history_window": 50,
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CMA-ES for MNIST energy alignment.")
    parser.add_argument("--weights", type=Path, default=Path("artifacts/mnist_cnn.pt"))
    parser.add_argument("--outdir", type=Path, default=Path("runs/mnist_energy_alignment"))
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_cma(
        model_path=args.weights,
        outdir=args.outdir,
        generations=args.generations,
        population=args.population,
        sigma=args.sigma,
        sample_size=args.sample_size,
        train_limit=args.train_limit,
    )
    print(json.dumps(metrics, indent=2))
    save_vector(args.outdir, metrics)
    write_manifest(args.outdir)


if __name__ == "__main__":
    main()
