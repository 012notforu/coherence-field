from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np

from engine.diagnostics import radial_spectrum, predictability_horizon


@dataclass
class SpectrumTracker:
    window: int = 256

    def __post_init__(self) -> None:
        self._history: Deque[np.ndarray] = deque(maxlen=self.window)

    def update(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centers, spectrum = radial_spectrum(field)
        self._history.append(spectrum)
        avg = np.mean(np.stack(self._history, axis=0), axis=0) if self._history else spectrum
        return centers, avg

    @property
    def width(self) -> float:
        if not self._history:
            return 0.0
        spectrum = np.mean(np.stack(self._history, axis=0), axis=0)
        weights = spectrum / (spectrum.sum() + 1e-8)
        indices = np.arange(len(spectrum))
        mean = (indices * weights).sum()
        variance = ((indices - mean) ** 2 * weights).sum()
        return float(np.sqrt(variance))


@dataclass
class SymbolTracker:
    history_window: int = 512

    def __post_init__(self) -> None:
        self._symbols: Deque[int] = deque(maxlen=self.history_window)

    def update(self, symbols: Iterable[int]) -> Dict[str, float]:
        for symbol in symbols:
            self._symbols.append(int(symbol))
        counts = Counter(self._symbols)
        total = sum(counts.values())
        if total == 0:
            return {"rate": 0.0, "entropy": 0.0, "perplexity": 0.0}
        probs = np.array([c / total for c in counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        perplexity = float(2 ** entropy)
        rate = len(symbols) / max(1, self.history_window)
        return {"rate": rate, "entropy": float(entropy), "perplexity": perplexity}

    def most_common(self, n: int = 5) -> List[Tuple[int, int]]:
        return Counter(self._symbols).most_common(n)


@dataclass
class HorizonTracker:
    threshold: float = 1e-2
    window: int = 64

    def __post_init__(self) -> None:
        self._hist_a: Deque[np.ndarray] = deque(maxlen=self.window)
        self._hist_b: Deque[np.ndarray] = deque(maxlen=self.window)

    def update(self, state_a: np.ndarray, state_b: np.ndarray) -> int:
        self._hist_a.append(state_a.copy())
        self._hist_b.append(state_b.copy())
        if len(self._hist_a) < 2:
            return self.window
        a = np.stack(self._hist_a, axis=0)
        b = np.stack(self._hist_b, axis=0)
        diff = np.linalg.norm(a - b, axis=(1, 2))
        return predictability_horizon(diff, np.zeros_like(diff), self.threshold)
