#!/usr/bin/env python3
"""
MNIST ⇢ SCFD interpretability harness.

This script loads (or trains) a small CNN on MNIST, extracts spatial activations,
maps them to an SCFD field, and logs the resulting energy / reward metrics.

Usage examples:

    python -m run.mnist_energy_alignment --train --epochs 3
    python -m run.mnist_energy_alignment --vector runs/mnist_energy_alignment/best_vector.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from .mnist_energy_core import (
    EnergyParams,
    activation_map,
    load_energy_params,
    energy_reward,
    load_or_train_cnn,
    sample_activation_fields,
    get_mnist_dataloaders,
)





def analyze_samples(
    model_path: Path,
    vector: EnergyParams,
    limit: int,
    output: Path,
) -> None:
    model = load_or_train_cnn(model_path)
    model.eval()
    _, test_loader = get_mnist_dataloaders(batch_size=32, test_limit=limit)
    fields = sample_activation_fields(model, test_loader, count=limit)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for idx, field in enumerate(fields):
            metrics = energy_reward(field, vector)
            record = {
                "index": idx,
                **metrics,
            }
            fh.write(json.dumps(record) + "\n")
            if idx < 5:
                print(
                    f"[sample {idx}] energy={metrics['energy']:.4f} "
                    f"expected={metrics['expected_energy']:.4f} diff={metrics['diff']:.4f}"
                )
    print(f"Wrote {len(fields)} samples to {output}")


def single_example(
    model_path: Path,
    vector: EnergyParams,
    image_index: int,
) -> Dict[str, float]:
    model = load_or_train_cnn(model_path)
    model.eval()
    _, test_loader = get_mnist_dataloaders(batch_size=1, test_limit=image_index + 1)
    iterator = iter(test_loader)
    image = None
    label = None
    for _ in range(image_index + 1):
        image, label = next(iterator)
    with torch.no_grad():
        logits = model(image)
    pred = logits.argmax(dim=1).item()
    field = activation_map(model, image, layer_name="conv2")
    metrics = energy_reward(field, vector)
    metrics.update({"label": int(label.item()), "pred": pred})
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST ⇢ SCFD interpretability harness.")
    parser.add_argument("--weights", type=Path, default=Path("artifacts/mnist_cnn.pt"), help="Path to CNN weights.")
    parser.add_argument(
        "--vector",
        type=Path,
        default=None,
        help="Optional vector JSON to use for expected energy parameters.",
    )
    parser.add_argument("--train", action="store_true", help="Force re-training of the CNN before analysis.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs to train when --train is set or weights missing.")
    parser.add_argument(
        "--limit",
        type=int,
        default=128,
        help="Number of test samples to analyze (for --log-samples).",
    )
    parser.add_argument(
        "--log-samples",
        type=Path,
        default=Path("logs/mnist_energy_samples.jsonl"),
        help="Where to write JSONL sample metrics.",
    )
    parser.add_argument("--single-index", type=int, default=None, help="Run a single example and print metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train or not args.weights.exists():
        model = load_or_train_cnn(args.weights, epochs=args.epochs)
        print("Finished training CNN.")
    else:
        model = load_or_train_cnn(args.weights)
        print(f"Loaded CNN weights from {args.weights}")
    vector = load_energy_params(args.vector)

    if args.single_index is not None:
        metrics = single_example(args.weights, vector, args.single_index)
        print(json.dumps(metrics, indent=2))
        return

    analyze_samples(args.weights, vector, args.limit, args.log_samples)


if __name__ == "__main__":
    main()
