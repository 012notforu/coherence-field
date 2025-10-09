from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.optim import Adam
except ImportError:  # pragma: no cover
    torch = None
    nn = object  # type: ignore
    Adam = object  # type: ignore


@dataclass
class AutoEncoderPredictor:
    input_dim: int
    latent_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    device: str = "cpu"

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for AutoEncoderPredictor")
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.predictor.to(self.device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.predictor.parameters())
        self.optim = Adam(params, lr=self.lr)
        self.loss_history: list[float] = []

    def _to_tensor(self, batch: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(batch, dtype=torch.float32, device=self.device)

    def reconstruct(self, batch: np.ndarray) -> tuple[np.ndarray, float]:
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            x = self._to_tensor(batch)
            latent = self.encoder(x)
            recon = self.decoder(latent)
            loss = torch.mean((x - recon) ** 2).item()
            return recon.cpu().numpy(), loss

    def step(self, batch: np.ndarray) -> float:
        self.encoder.train()
        self.decoder.train()
        x = self._to_tensor(batch)
        latent = self.encoder(x)
        recon = self.decoder(latent)
        loss = torch.mean((x - recon) ** 2)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        value = float(loss.item())
        self.loss_history.append(value)
        return value

    def regime_change_score(self, batch: np.ndarray) -> float:
        self.encoder.eval()
        with torch.no_grad():
            x = self._to_tensor(batch)
            latent = self.encoder(x)
            score = torch.mean(torch.abs(self.predictor(latent))).item()
        return float(score)

    def update_predictor(self, batch: np.ndarray, targets: np.ndarray) -> float:
        self.predictor.train()
        x = self._to_tensor(batch)
        y = self._to_tensor(targets).unsqueeze(-1)
        pred = self.predictor(x)
        loss = torch.mean((pred - y) ** 2)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())
