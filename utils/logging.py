from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Any, List

import numpy as np


Array = np.ndarray


@dataclass
class PhysicsContext:
    """Per-cell physics state snapshot."""
    coherence: float
    curvature: float
    grad_magnitude: float
    energy_density: float
    entropy: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class VectorInvocation:
    """Record of vector being called by a grid cell."""
    vector_id: str
    reason: str
    confidence: float
    selection_method: str  # "orchestrator", "pattern_match", "exploration"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutcomeMetrics:
    """Results of vector application."""
    delta_energy: float
    accepted: bool
    progress_made: float
    stuck_counter: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert numpy types to native Python types for JSON serialization
        for key, value in result.items():
            if hasattr(value, 'item'):  # numpy scalar
                result[key] = value.item()
            elif isinstance(value, np.bool_):
                result[key] = bool(value)
        return result


@dataclass
class CellLogEntry:
    """Complete log entry for a single cell at one timestep."""
    cell_pos: Tuple[int, int]
    timestep: int
    physics: PhysicsContext
    vector: VectorInvocation
    outcome: OutcomeMetrics
    neighbors: Optional[Dict[str, float]] = None  # Summary of neighbor states
    
    def to_dict(self) -> Dict[str, Any]:
        entry = {
            "cell_pos": self.cell_pos,
            "timestep": self.timestep,
            "physics": self.physics.to_dict(),
            "vector": self.vector.to_dict(),
            "outcome": self.outcome.to_dict(),
        }
        if self.neighbors:
            entry["neighbors"] = self.neighbors
        return entry


def create_run_directory(tag: str, outdir: Optional[str] = None, root: str = "runs") -> Path:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(outdir) if outdir else Path(root) / f"{timestamp}_{tag}"
    base.mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(exist_ok=True)
    (base / "pathway_analysis").mkdir(exist_ok=True)
    (base / "interaction_networks").mkdir(exist_ok=True)
    return base


@dataclass
class RunLogger:
    directory: Path

    def __post_init__(self) -> None:
        self.jsonl_path = self.directory / "logs.jsonl"
        self.csv_files: Dict[str, Path] = {}
        
        # Enhanced logging files
        self.pathway_path = self.directory / "pathway_log.jsonl"
        self.interactions_path = self.directory / "interactions.jsonl"
        self.vector_stats_path = self.directory / "vector_stats.jsonl"
        
        # In-memory tracking for aggregations
        self.vector_outcomes: Dict[str, List[float]] = {}
        self.pattern_frequencies: Dict[str, int] = {}

    def log_step(self, payload: Dict[str, float]) -> None:
        payload = dict(payload)
        payload.setdefault("timestamp", time.time())
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def log_csv(self, name: str, headers: Iterable[str], row: Iterable[float]) -> None:
        path = self.csv_files.setdefault(name, self.directory / f"{name}.csv")
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if is_new:
                writer.writerow(list(headers))
            writer.writerow(list(row))

    def dump_config(self, cfg_text: str) -> None:
        (self.directory / "cfg.yaml").write_text(cfg_text, encoding="utf-8")
    
    # Enhanced logging methods for multi-agent systems
    def log_cell_action(self, entry: CellLogEntry) -> None:
        """Log a complete cell action with full context."""
        # Write detailed entry
        with self.pathway_path.open("a", encoding="utf-8") as fh:
            entry_dict = entry.to_dict()
            entry_dict["timestamp"] = time.time()
            fh.write(json.dumps(entry_dict) + "\n")
        
        # Update statistics
        vector_id = entry.vector.vector_id
        if vector_id not in self.vector_outcomes:
            self.vector_outcomes[vector_id] = []
        self.vector_outcomes[vector_id].append(entry.outcome.delta_energy)
        
        # Track pattern frequencies
        reason = entry.vector.reason
        self.pattern_frequencies[reason] = self.pattern_frequencies.get(reason, 0) + 1
    
    def log_interaction(self, 
                       source_cell: Tuple[int, int], 
                       target_cell: Tuple[int, int],
                       interaction_type: str,
                       strength: float,
                       timestep: int) -> None:
        """Log cell-to-cell interactions through field dynamics."""
        interaction = {
            "source": source_cell,
            "target": target_cell,
            "type": interaction_type,
            "strength": strength,
            "timestep": timestep,
            "timestamp": time.time()
        }
        
        with self.interactions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(interaction) + "\n")
    
    def compute_physics_context(self, 
                               theta: Array, 
                               theta_dot: Array,
                               physics,  # PhysicsParams - avoiding import here
                               cell_pos: Tuple[int, int],
                               dx: Optional[float] = None) -> PhysicsContext:
        """Extract physics context for a specific cell."""
        from engine.ops import grad, laplacian
        from engine.energy import total_energy_density
        
        i, j = cell_pos
        
        # Local coherence (simplified - count similar neighbors)
        h, w = theta.shape
        neighborhood = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i + di) % h, (j + dj) % w
                neighborhood.append(theta[ni, nj])
        
        center_val = theta[i, j]
        coherence = float(np.mean([1.0 if abs(val - center_val) < 0.1 else 0.0 for val in neighborhood]))
        
        # Local curvature
        curvature = float(laplacian(theta, dx=dx)[i, j])
        
        # Gradient magnitude
        gx, gy = grad(theta, dx=dx)
        grad_magnitude = float(np.sqrt(gx[i, j]**2 + gy[i, j]**2))
        
        # Energy density at this cell
        energy_density = float(total_energy_density(theta, theta_dot, physics, dx=dx)[i, j])
        
        # Local entropy (diversity of neighbor values)
        unique_vals = len(set(np.round(neighborhood, 2)))
        entropy = float(np.log(max(unique_vals, 1)))
        
        return PhysicsContext(
            coherence=coherence,
            curvature=curvature,
            grad_magnitude=grad_magnitude,
            energy_density=energy_density,
            entropy=entropy
        )
    
    def compute_neighbor_summary(self, 
                                theta: Array,
                                cell_pos: Tuple[int, int]) -> Dict[str, float]:
        """Compute summary statistics of neighboring cells."""
        i, j = cell_pos
        h, w = theta.shape
        
        neighbor_vals = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = (i + di) % h, (j + dj) % w
                neighbor_vals.append(theta[ni, nj])
        
        return {
            "mean": float(np.mean(neighbor_vals)),
            "std": float(np.std(neighbor_vals)),
            "min": float(np.min(neighbor_vals)),
            "max": float(np.max(neighbor_vals))
        }
    
    def log_vector_stats_summary(self, timestep: int) -> None:
        """Periodically log aggregated vector performance statistics."""
        stats = {
            "timestep": timestep,
            "timestamp": time.time(),
            "vector_performance": {},
            "pattern_frequencies": dict(self.pattern_frequencies)
        }
        
        # Compute performance metrics per vector
        for vector_id, outcomes in self.vector_outcomes.items():
            if outcomes:
                stats["vector_performance"][vector_id] = {
                    "mean_delta_energy": float(np.mean(outcomes)),
                    "std_delta_energy": float(np.std(outcomes)),
                    "success_rate": float(np.mean([1.0 if x > 0 else 0.0 for x in outcomes])),
                    "invocation_count": len(outcomes)
                }
        
        with self.vector_stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(stats) + "\n")
        
        # Clear counters for next period
        self.vector_outcomes.clear()
        self.pattern_frequencies.clear()
