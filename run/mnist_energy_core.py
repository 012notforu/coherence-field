#!/usr/bin/env python3
"""
Core utilities for MNIST ⇢ SCFD interpretability bridge.

This module exposes three layers:

1.  A light-weight CNN feature extractor for MNIST images.
2.  Hooks that convert intermediate activations into SCFD-compatible fields.
3.  Energy/reward helpers that compare SCFD energy against an expected target
    derived from activation-driven mass and acceleration metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from engine import load_config, coherence_energy_density, norm_sq_grad, total_energy_density

try:
    import torchvision
    from torchvision import transforms
except Exception as exc:  # pragma: no cover - torchvision is optional at import
    raise RuntimeError(
        "torchvision is required for the MNIST interpretability bridge. "
        "Install it via `pip install torchvision`."
    ) from exc


# ---------------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------------


class SimpleMNISTCNN(nn.Module):
    """Minimal CNN that achieves >98% accuracy on MNIST with a few epochs."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_dataloaders(
    batch_size: int = 128,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=transform, download=True)

    if train_limit is not None:
        train_dataset = Subset(train_dataset, range(train_limit))
    if test_limit is not None:
        test_dataset = Subset(test_dataset, range(test_limit))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_mnist_model(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    lr: float = 1e-3,
) -> None:
    device = _device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        accuracy = correct / max(total, 1)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[Epoch {epoch + 1}] loss={avg_loss:.4f}  accuracy={accuracy * 100:.2f}%")


def load_or_train_cnn(
    weights_path: Path,
    epochs: int = 3,
    train_limit: Optional[int] = 20_000,
) -> SimpleMNISTCNN:
    model = SimpleMNISTCNN()
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return model

    print("Training MNIST CNN (weights not found)...")
    train_loader, test_loader = get_mnist_dataloaders(train_limit=train_limit, test_limit=10_000)
    train_mnist_model(model, epochs=epochs, train_loader=train_loader, test_loader=test_loader)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Saved trained weights to {weights_path}")
    return model


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def activation_map(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str = "conv2",
) -> np.ndarray:
    """
    Run a single image through the model and return the activation map from the
    specified convolutional layer. The map is averaged over channels.
    """
    activations: Dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        activations["map"] = out.detach().cpu()

    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(image.to(_device()))
    finally:
        handle.remove()

    act = activations["map"]  # shape: (1, C, H, W)
    heat = act.abs().mean(dim=1)[0]  # (H, W)
    heat = heat / (heat.max().item() + 1e-6)
    return heat.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# SCFD energy helpers
# ---------------------------------------------------------------------------


@dataclass
class EnergyParams:
    mass_scale: float
    accel_scale: float
    c_const: float

    @classmethod
    def from_vector(cls, vector: Iterable[float]) -> "EnergyParams":
        vals = list(vector)
        if len(vals) != 3:
            raise ValueError("Energy parameter vector must have length 3.")
        mass_scale = float(vals[0])
        accel_scale = float(vals[1])
        c_const = float(vals[2])
        return cls(mass_scale=mass_scale, accel_scale=accel_scale, c_const=c_const)


def load_energy_params(vector_path: Optional[Path]) -> EnergyParams:
    if vector_path is None:
        return EnergyParams(mass_scale=1.0, accel_scale=1.0, c_const=0.12)

    payload = json.loads(Path(vector_path).read_text())
    if isinstance(payload, dict) and "vector" in payload:
        vector = payload["vector"]
    elif isinstance(payload, list):
        vector = payload
    else:
        raise ValueError(f"Unrecognized vector format in {vector_path}")
    return EnergyParams.from_vector(vector)


def _physics_params():
    cfg = load_config(Path("cfg/defaults.yaml"))
    return cfg.physics, cfg.grid.spacing


def compute_scfd_metrics(field: np.ndarray) -> Dict[str, float]:
    physics, spacing = _physics_params()
    theta = np.asarray(field, dtype=np.float32)
    theta_dot = np.zeros_like(theta)

    energy_density = total_energy_density(theta, theta_dot, physics, dx=spacing)
    energy = float(np.mean(energy_density))

    coherence = coherence_energy_density(theta, physics, dx=spacing)
    mass_metric = float(np.mean(coherence))

    grad_sq = norm_sq_grad(theta, dx=spacing)
    acceleration_metric = float(np.mean(np.sqrt(np.maximum(grad_sq, 1e-12))))

    return {
        "energy": energy,
        "mass_metric": mass_metric,
        "acceleration_metric": acceleration_metric,
    }


def expected_energy(
    metrics: Dict[str, float],
    params: EnergyParams,
) -> float:
    expected_mass = params.mass_scale * metrics["mass_metric"] * (params.c_const ** 2)
    expected_accel = params.accel_scale * metrics["acceleration_metric"]
    return expected_mass + expected_accel


def energy_reward(
    field: np.ndarray,
    params: EnergyParams,
) -> Dict[str, float]:
    metrics = compute_scfd_metrics(field)
    expected = expected_energy(metrics, params)
    diff = metrics["energy"] - expected
    reward = -abs(diff)
    return {
        "energy": metrics["energy"],
        "mass_metric": metrics["mass_metric"],
        "acceleration_metric": metrics["acceleration_metric"],
        "expected_energy": expected,
        "diff": diff,
        "reward": reward,
    }


# ---------------------------------------------------------------------------
# Dataset helpers for CMA/analysis
# ---------------------------------------------------------------------------


def sample_activation_fields(
    model: nn.Module,
    dataloader: DataLoader,
    count: int,
    layer_name: str = "conv2",
) -> List[np.ndarray]:
    fields: List[np.ndarray] = []
    device = _device()
    model.to(device)
    model.eval()
    collected = 0
    with torch.no_grad():
        for images, _labels in dataloader:
            for idx in range(images.size(0)):
                image = images[idx : idx + 1].to(device)
                field = activation_map(model, image, layer_name=layer_name)
                fields.append(field)
                collected += 1
                if collected >= count:
                    return fields
    return fields
