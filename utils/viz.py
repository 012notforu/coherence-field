from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def _prepare_path(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_energy_breakdown(energy: dict, path: Path | str) -> None:
    labels = list(energy.keys())
    values = [energy[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel("Energy density")
    ax.set_title("Energy components")
    fig.tight_layout()
    fig.savefig(_prepare_path(path))
    plt.close(fig)


def plot_spectrum(freqs: Sequence[float], spectrum: Sequence[float], path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, spectrum, color="darkorange")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Radial spectrum")
    fig.tight_layout()
    fig.savefig(_prepare_path(path))
    plt.close(fig)


def plot_impulse_response(series: Sequence[float], path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(series, color="forestgreen")
    ax.set_xlabel("Step")
    ax.set_ylabel("Response magnitude")
    ax.set_title("Impulse response")
    fig.tight_layout()
    fig.savefig(_prepare_path(path))
    plt.close(fig)


def plot_horizon(horizon: Sequence[float], path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizon, color="slateblue")
    ax.set_xlabel("Window")
    ax.set_ylabel("Steps")
    ax.set_title("Predictability horizon")
    fig.tight_layout()
    fig.savefig(_prepare_path(path))
    plt.close(fig)
