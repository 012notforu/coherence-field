#!/usr/bin/env python3
"""Run the MNIST energy observer over a dataset and log metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from observe.nn_interpretability import MNISTEnergyObserver
from run.mnist_energy_core import get_mnist_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST energy monitor")
    parser.add_argument("--weights", type=Path, default=Path("artifacts/mnist_cnn.pt"))
    parser.add_argument("--vector", type=Path, default=Path("runs/mnist_energy_alignment/best_vector.json"))
    parser.add_argument("--limit", type=int, default=None, help="Number of test samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diff-threshold", type=float, default=0.05, help="Flag samples whose |diff| exceeds this value")
    parser.add_argument("--out", type=Path, default=Path("logs/mnist_energy_monitor.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observer = MNISTEnergyObserver(weights_path=args.weights, vector_path=args.vector)
    _, test_loader = get_mnist_dataloaders(batch_size=args.batch_size, test_limit=args.limit)
    summary = observer.log_dataset(test_loader, out_path=args.out, limit=args.limit, diff_threshold=args.diff_threshold)
    print("MNIST energy monitor summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
