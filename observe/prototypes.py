from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class PrototypeBank:
    capacity: int = 64
    decay: float = 0.95
    prototypes: Dict[int, np.ndarray] = field(default_factory=dict)
    counts: Dict[int, int] = field(default_factory=dict)

    def update(self, symbol_id: int, patch: np.ndarray) -> None:
        patch = patch.astype(np.float32)
        if symbol_id not in self.prototypes and len(self.prototypes) >= self.capacity:
            # Evict the least frequent symbol
            victim = min(self.counts.items(), key=lambda kv: kv[1])[0]
            self.prototypes.pop(victim, None)
            self.counts.pop(victim, None)
        if symbol_id not in self.prototypes:
            self.prototypes[symbol_id] = patch.copy()
            self.counts[symbol_id] = 1
        else:
            self.prototypes[symbol_id] = (
                self.decay * self.prototypes[symbol_id] + (1.0 - self.decay) * patch
            )
            self.counts[symbol_id] += 1

    def score(self, symbol_id: int, patch: np.ndarray) -> float:
        if symbol_id not in self.prototypes:
            return float(np.linalg.norm(patch))
        proto = self.prototypes[symbol_id]
        return float(np.linalg.norm(proto - patch) / (np.linalg.norm(proto) + 1e-8))

    def ranked(self) -> Dict[int, float]:
        return {sid: float(val.mean()) for sid, val in self.prototypes.items()}
