from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Tuple

import numpy as np


@dataclass
class NGramLanguageModel:
    order: int = 3
    smoothing: float = 1.0

    def __post_init__(self) -> None:
        self._context: Deque[int] = deque([0] * (self.order - 1), maxlen=self.order - 1)
        self._counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)

    def update(self, symbols: Iterable[int]) -> None:
        for symbol in symbols:
            context = tuple(self._context)
            self._counts[context][symbol] += 1
            self._context.append(symbol)

    def distribution(self, context: Iterable[int]) -> Dict[int, float]:
        context_tuple = tuple(context)[-(self.order - 1) :]
        counts = self._counts.get(context_tuple, Counter())
        total = sum(counts.values()) + self.smoothing * max(len(counts), 1)
        dist = {}
        for symbol, count in counts.items():
            dist[symbol] = (count + self.smoothing) / total
        return dist

    def perplexity(self, sequence: Iterable[int]) -> float:
        context = deque([0] * (self.order - 1), maxlen=self.order - 1)
        log_prob = 0.0
        n = 0
        for symbol in sequence:
            counts = self._counts.get(tuple(context), Counter())
            total = sum(counts.values()) + self.smoothing * max(len(counts), 1)
            prob = (counts.get(symbol, 0) + self.smoothing) / total
            log_prob += -np.log2(prob)
            context.append(symbol)
            n += 1
        if n == 0:
            return 0.0
        return float(2 ** (log_prob / n))

    def reset(self) -> None:
        self._context = deque([0] * (self.order - 1), maxlen=self.order - 1)
        self._counts.clear()
