from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label

from .params import SimulationConfig

Array = np.ndarray


@dataclass
class Symbol:
    label: int
    area: int
    centroid: tuple[float, float]
    mean_value: float


def _bandpass(field: Array, cfg: Dict[str, Any]) -> Array:
    shape_min = min(field.shape)
    low_sigma = max(1.0, cfg["low_cut"] * shape_min)
    high_sigma = max(low_sigma + 1.0, cfg["high_cut"] * shape_min)
    low = gaussian_filter(field, sigma=high_sigma)
    high = gaussian_filter(field, sigma=low_sigma)
    return high - low


def extract_symbols(field: Array, config: SimulationConfig) -> Dict[str, Any]:
    field_guard = np.array(field, copy=True)
    symbol_cfg = config.symbolizer
    bandpassed = _bandpass(field, symbol_cfg["bandpass"])
    thresh = symbol_cfg.get("threshold", 0.6)
    mask = bandpassed > thresh
    labeled, count = label(mask)
    symbols: List[Symbol] = []
    if count > 0:
        for idx in range(1, count + 1):
            region = labeled == idx
            area = int(region.sum())
            if area < symbol_cfg.get("cluster_min_area", 1):
                continue
            centroid = center_of_mass(region)
            mean_value = float(bandpassed[region].mean())
            symbols.append(
                Symbol(
                    label=idx,
                    area=area,
                    centroid=(float(centroid[0]), float(centroid[1])),
                    mean_value=mean_value,
                )
            )
    if not np.array_equal(field, field_guard):
        raise RuntimeError("Symbolizer must not mutate the physics state")
    return {
        "bandpassed": bandpassed,
        "mask": mask,
        "symbols": symbols,
        "count": len(symbols),
    }
