#!/usr/bin/env python3
"""Always-on MNIST -> SCFD energy observer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from run.mnist_energy_core import (
    EnergyParams,
    activation_map,
    energy_reward,
    get_mnist_dataloaders,
    load_energy_params,
    load_or_train_cnn,
)


@dataclass
class Observation:
    index: int
    label: Optional[int]
    prediction: int
    metrics: Dict[str, float]


class MNISTEnergyObserver:
    """Wraps the MNIST -> SCFD energy bridge as an always-on diagnostic."""

    def __init__(
        self,
        weights_path: Path = Path("artifacts/mnist_cnn.pt"),
        vector_path: Optional[Path] = Path("runs/mnist_energy_alignment/best_vector.json"),
        layer_name: str = "conv2",
    ) -> None:
        self.weights_path = weights_path
        self.vector_path = vector_path
        self.layer_name = layer_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_or_train_cnn(self.weights_path)
        self.model.to(self.device).eval()
        self.params = load_energy_params(self.vector_path)

    def _ensure_batch(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    def analyze_tensor(
        self,
        image: torch.Tensor,
        label: Optional[int] = None,
    ) -> Observation:
        image = self._ensure_batch(image)
        with torch.no_grad():
            logits = self.model(image.to(self.device))
        pred = int(logits.argmax(dim=1).item())
        field = activation_map(self.model, image.cpu(), layer_name=self.layer_name)
        metrics = energy_reward(field, self.params)
        return Observation(index=0, label=int(label) if label is not None else None, prediction=pred, metrics=metrics)

    def analyze_batch(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> List[Observation]:
        observations: List[Observation] = []
        images = self._ensure_batch(images)
        batch_size = images.size(0)
        labels_iter: Iterable[Optional[int]]
        if labels is None:
            labels_iter = [None] * batch_size
        else:
            labels_iter = [int(x) for x in labels]
        for idx, (img, lab) in enumerate(zip(images, labels_iter)):
            obs = self.analyze_tensor(img.unsqueeze(0), label=lab)
            obs.index = idx
            observations.append(obs)
        return observations

    def log_dataset(
        self,
        dataloader: DataLoader,
        out_path: Path,
        limit: Optional[int] = None,
        diff_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, float]] = []
        total = 0
        flagged = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for images, labels in dataloader:
                observations = self.analyze_batch(images, labels)
                for obs in observations:
                    record = {
                        "index": total,
                        "label": obs.label,
                        "prediction": obs.prediction,
                        **obs.metrics,
                    }
                    if diff_threshold is not None and abs(obs.metrics["diff"]) > diff_threshold:
                        record["flagged"] = True
                        flagged += 1
                    fh.write(json.dumps(record) + "
")
                    records.append(record)
                    total += 1
                    if limit is not None and total >= limit:
                        break
                if limit is not None and total >= limit:
                    break
        mean_reward = float(sum(r["reward"] for r in records) / max(len(records), 1))
        mean_diff = float(sum(abs(r["diff"]) for r in records) / max(len(records), 1))
        return {
            "samples": total,
            "flagged": flagged,
            "mean_reward": mean_reward,
            "mean_abs_diff": mean_diff,
        }

    def to_probe_stats(self, image: torch.Tensor, label: Optional[int] = None) -> Dict[str, float]:
        obs = self.analyze_tensor(image, label)
        stats = {
            "task": "nn_interpretability",
            "reward": obs.metrics["reward"],
            "energy": obs.metrics["energy"],
            "expected_energy": obs.metrics["expected_energy"],
            "energy_diff": obs.metrics["diff"],
            "mass_metric": obs.metrics["mass_metric"],
            "acceleration_metric": obs.metrics["acceleration_metric"],
            "prediction": obs.prediction,
        }
        if obs.label is not None:
            stats["label"] = obs.label
        return stats


def build_default_observer() -> MNISTEnergyObserver:
    return MNISTEnergyObserver()

