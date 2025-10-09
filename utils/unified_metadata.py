"""
Unified Metadata System

Comprehensive metadata management for SCFD multi-agent meta-learning system.
Tracks vectors, compositions, test runs, and performance analytics in one unified file.
"""
from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VectorMetrics:
    """Complete metrics for a single vector."""
    vector_id: str
    physics_domain: str
    total_uses: int = 0
    successful_uses: int = 0
    success_rate: float = 0.0
    mean_delta_energy: float = 0.0
    std_delta_energy: float = 0.0
    mean_acceptance_prob: float = 0.0
    total_energy_change: float = 0.0
    last_used: Optional[str] = None
    creation_date: Optional[str] = None
    source_type: str = "cma"  # "cma", "meta_learned", "manual"
    
    # Performance by context
    performance_by_task: Dict[str, Dict] = None
    performance_by_pattern: Dict[str, Dict] = None
    recent_performance: List[Dict] = None
    
    def __post_init__(self):
        if self.performance_by_task is None:
            self.performance_by_task = {}
        if self.performance_by_pattern is None:
            self.performance_by_pattern = {}
        if self.recent_performance is None:
            self.recent_performance = []


@dataclass
class CompositionMetrics:
    """Metrics for vector composition rules."""
    composition_id: str
    vector_ids: List[str]
    composition_type: str
    total_uses: int = 0
    successful_uses: int = 0
    success_rate: float = 0.0
    mean_energy_change: float = 0.0
    weight_adjustments: int = 0
    current_weights: List[float] = None
    performance_history: List[Dict] = None
    
    def __post_init__(self):
        if self.current_weights is None:
            self.current_weights = []
        if self.performance_history is None:
            self.performance_history = []


@dataclass
class TestRunSummary:
    """Summary of a complete test run."""
    run_id: str
    test_type: str
    timestamp: str
    success: bool
    duration_seconds: float
    key_metrics: Dict[str, Any]
    vector_usage: Dict[str, int]
    composition_usage: Dict[str, int]
    failure_patterns: Dict[str, int]
    notable_events: List[str]
    run_directory: str


@dataclass
class LoggedMove:
    """Individual logged move from a run with full context."""
    move_id: str
    run_id: str
    timestamp: str
    vector_id: str
    composition_id: Optional[str]
    agent_id: str
    agent_position: Tuple[int, int]
    reason: str  # Why this vector/composition was selected
    context: Dict[str, Any]  # Local features, physics state, etc.
    action_taken: Dict[str, Any]  # What the vector did
    outcome: Dict[str, Any]  # Results of the action
    accepted: bool
    delta_energy: float
    progress_made: float


class UnifiedMetadataManager:
    """Unified metadata management system."""
    
    def __init__(self, metadata_file: str = "unified_metadata.json"):
        self.metadata_file = Path(metadata_file)
        self.metadata = self._load_or_create_metadata()
        
    def _load_or_create_metadata(self) -> Dict:
        """Load existing metadata or create new structure."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                return self._create_empty_metadata()
        else:
            return self._create_empty_metadata()
    
    def _create_empty_metadata(self) -> Dict:
        """Create empty metadata structure."""
        return {
            "system_info": {
                "version": "1.0.0",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_vectors": 0,
                "total_compositions": 0,
                "total_test_runs": 0,
                "total_logged_moves": 0
            },
            "vectors": {},
            "compositions": {},
            "test_runs": {},
            "logged_moves": {},
            "performance_analytics": {
                "top_vectors_by_success": [],
                "top_compositions_by_success": [],
                "most_common_failure_patterns": {},
                "task_performance_summary": {}
            }
        }
    
    def save(self):
        """Save metadata to file."""
        self.metadata["system_info"]["last_updated"] = datetime.now().isoformat()
        
        # Convert any numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=convert_numpy)
    
    def register_vector(self, vector_id: str, physics_domain: str, source_type: str = "cma") -> VectorMetrics:
        """Register a new vector in the metadata."""
        if vector_id not in self.metadata["vectors"]:
            vector_metrics = VectorMetrics(
                vector_id=vector_id,
                physics_domain=physics_domain,
                source_type=source_type,
                creation_date=datetime.now().isoformat()
            )
            
            self.metadata["vectors"][vector_id] = asdict(vector_metrics)
            self.metadata["system_info"]["total_vectors"] += 1
            
            print(f"Registered new vector: {vector_id} ({physics_domain})")
        
        return VectorMetrics(**self.metadata["vectors"][vector_id])
    
    def register_composition(self, composition_id: str, vector_ids: List[str], 
                           composition_type: str, initial_weights: List[float]) -> CompositionMetrics:
        """Register a new composition rule."""
        if composition_id not in self.metadata["compositions"]:
            composition_metrics = CompositionMetrics(
                composition_id=composition_id,
                vector_ids=vector_ids,
                composition_type=composition_type,
                current_weights=initial_weights
            )
            
            self.metadata["compositions"][composition_id] = asdict(composition_metrics)
            self.metadata["system_info"]["total_compositions"] += 1
            
            print(f"Registered new composition: {composition_id} ({len(vector_ids)} vectors)")
        
        return CompositionMetrics(**self.metadata["compositions"][composition_id])
    
    def log_move(self, 
                 run_id: str,
                 vector_id: str,
                 agent_id: str,
                 agent_position: Tuple[int, int],
                 reason: str,
                 context: Dict[str, Any],
                 action_taken: Dict[str, Any],
                 outcome: Dict[str, Any],
                 composition_id: Optional[str] = None) -> str:
        """Log a single move with full context."""
        
        move_id = f"{run_id}_{agent_id}_{len(self.metadata['logged_moves'])}"
        
        logged_move = LoggedMove(
            move_id=move_id,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            vector_id=vector_id,
            composition_id=composition_id,
            agent_id=agent_id,
            agent_position=agent_position,
            reason=reason,
            context=context,
            action_taken=action_taken,
            outcome=outcome,
            accepted=outcome.get("accepted", False),
            delta_energy=outcome.get("delta_energy", 0.0),
            progress_made=outcome.get("progress_made", 0.0)
        )
        
        self.metadata["logged_moves"][move_id] = asdict(logged_move)
        self.metadata["system_info"]["total_logged_moves"] += 1
        
        # Update vector metrics
        self._update_vector_metrics(vector_id, logged_move)
        
        # Update composition metrics if applicable
        if composition_id:
            self._update_composition_metrics(composition_id, logged_move)
        
        return move_id
    
    def _update_vector_metrics(self, vector_id: str, logged_move: LoggedMove):
        """Update vector metrics based on logged move."""
        if vector_id not in self.metadata["vectors"]:
            self.register_vector(vector_id, "unknown")
        
        metrics = self.metadata["vectors"][vector_id]
        
        # Update usage counts
        metrics["total_uses"] += 1
        if logged_move.accepted:
            metrics["successful_uses"] += 1
        
        # Update success rate
        metrics["success_rate"] = metrics["successful_uses"] / metrics["total_uses"]
        
        # Update energy statistics
        old_total_energy = metrics["total_energy_change"]
        new_total_energy = old_total_energy + logged_move.delta_energy
        metrics["total_energy_change"] = new_total_energy
        
        # Update mean delta energy (running average)
        old_mean = metrics["mean_delta_energy"]
        metrics["mean_delta_energy"] = (old_mean * (metrics["total_uses"] - 1) + logged_move.delta_energy) / metrics["total_uses"]
        
        # Update last used
        metrics["last_used"] = logged_move.timestamp
        
        # Update performance by pattern
        pattern = logged_move.reason
        if pattern not in metrics["performance_by_pattern"]:
            metrics["performance_by_pattern"][pattern] = {
                "uses": 0, "successes": 0, "success_rate": 0.0, "avg_energy": 0.0
            }
        
        pattern_data = metrics["performance_by_pattern"][pattern]
        pattern_data["uses"] += 1
        if logged_move.accepted:
            pattern_data["successes"] += 1
        pattern_data["success_rate"] = pattern_data["successes"] / pattern_data["uses"]
        pattern_data["avg_energy"] = (pattern_data["avg_energy"] * (pattern_data["uses"] - 1) + logged_move.delta_energy) / pattern_data["uses"]
        
        # Add to recent performance (keep last 50)
        if len(metrics["recent_performance"]) >= 50:
            metrics["recent_performance"].pop(0)
        
        metrics["recent_performance"].append({
            "timestamp": logged_move.timestamp,
            "accepted": logged_move.accepted,
            "delta_energy": logged_move.delta_energy,
            "reason": logged_move.reason,
            "progress": logged_move.progress_made
        })
    
    def _update_composition_metrics(self, composition_id: str, logged_move: LoggedMove):
        """Update composition metrics based on logged move."""
        if composition_id not in self.metadata["compositions"]:
            return  # Should not happen, but guard against it
        
        metrics = self.metadata["compositions"][composition_id]
        
        # Update usage counts
        metrics["total_uses"] += 1
        if logged_move.accepted:
            metrics["successful_uses"] += 1
        
        # Update success rate
        metrics["success_rate"] = metrics["successful_uses"] / metrics["total_uses"]
        
        # Update energy statistics
        old_mean = metrics["mean_energy_change"]
        metrics["mean_energy_change"] = (old_mean * (metrics["total_uses"] - 1) + logged_move.delta_energy) / metrics["total_uses"]
        
        # Add to performance history (keep last 100)
        if len(metrics["performance_history"]) >= 100:
            metrics["performance_history"].pop(0)
        
        metrics["performance_history"].append({
            "timestamp": logged_move.timestamp,
            "accepted": logged_move.accepted,
            "delta_energy": logged_move.delta_energy,
            "reason": logged_move.reason,
            "agent_position": logged_move.agent_position
        })
    
    def import_run_data(self, run_directory: str, test_type: str) -> str:
        """Import data from a test run directory."""
        run_dir = Path(run_directory)
        
        if not run_dir.exists():
            raise ValueError(f"Run directory does not exist: {run_directory}")
        
        run_id = run_dir.name
        
        print(f"Importing run data: {run_id}")
        
        # Import logs
        imported_moves = 0
        if (run_dir / "pathway_log.jsonl").exists():
            imported_moves += self._import_pathway_logs(run_id, run_dir / "pathway_log.jsonl", test_type)
        
        # Import vector stats
        if (run_dir / "vector_stats.jsonl").exists():
            self._import_vector_stats(run_id, run_dir / "vector_stats.jsonl")
        
        # Create test run summary
        self._create_test_run_summary(run_id, run_directory, test_type, imported_moves)
        
        self.save()
        
        print(f"Imported {imported_moves} moves from run {run_id}")
        
        return run_id
    
    def _import_pathway_logs(self, run_id: str, log_file: Path, test_type: str) -> int:
        """Import pathway logs from a run."""
        imported_count = 0
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    entry = json.loads(line)
                    
                    # Extract move information
                    vector_info = entry.get("vector", {})
                    outcome_info = entry.get("outcome", {})
                    physics_info = entry.get("physics", {})
                    neighbors_info = entry.get("neighbors", {})
                    
                    # Log the move
                    self.log_move(
                        run_id=run_id,
                        vector_id=vector_info.get("vector_id", "unknown"),
                        agent_id=f"agent_{entry.get('cell_pos', [0, 0])}",
                        agent_position=tuple(entry.get("cell_pos", [0, 0])),
                        reason=vector_info.get("reason", "unknown"),
                        context={
                            "test_type": test_type,
                            "timestep": entry.get("timestep", 0),
                            "physics": physics_info,
                            "neighbors": neighbors_info
                        },
                        action_taken={
                            "vector_id": vector_info.get("vector_id"),
                            "selection_method": vector_info.get("selection_method", "unknown"),
                            "confidence": vector_info.get("confidence", 0.0)
                        },
                        outcome={
                            "accepted": outcome_info.get("accepted", False),
                            "delta_energy": outcome_info.get("delta_energy", 0.0),
                            "progress_made": outcome_info.get("progress_made", 0.0),
                            "stuck_counter": outcome_info.get("stuck_counter", 0)
                        }
                    )
                    
                    imported_count += 1
                    
        except Exception as e:
            print(f"Error importing pathway logs: {e}")
        
        return imported_count
    
    def _import_vector_stats(self, run_id: str, stats_file: Path):
        """Import vector statistics from a run."""
        try:
            with open(stats_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    stats = json.loads(line)
                    vector_performance = stats.get("vector_performance", {})
                    
                    for vector_id, perf_data in vector_performance.items():
                        # Register vector if not exists
                        if vector_id not in self.metadata["vectors"]:
                            physics_domain = "unknown"
                            # Try to infer physics domain from vector name
                            if "heat" in vector_id:
                                physics_domain = "heat"
                            elif "flow" in vector_id:
                                physics_domain = "flow"
                            elif "gray_scott" in vector_id:
                                physics_domain = "gray_scott"
                            elif "wave" in vector_id:
                                physics_domain = "wave"
                            elif "cartpole" in vector_id:
                                physics_domain = "cartpole"
                            elif "meta" in vector_id:
                                physics_domain = "meta"
                            
                            self.register_vector(vector_id, physics_domain)
                        
                        # Update performance by task
                        vector_metrics = self.metadata["vectors"][vector_id]
                        if "performance_by_task" not in vector_metrics:
                            vector_metrics["performance_by_task"] = {}
                        
                        task_name = f"{run_id}_summary"
                        vector_metrics["performance_by_task"][task_name] = {
                            "success_rate": perf_data.get("success_rate", 0.0),
                            "mean_delta_energy": perf_data.get("mean_delta_energy", 0.0),
                            "invocation_count": perf_data.get("invocation_count", 0)
                        }
                        
        except Exception as e:
            print(f"Error importing vector stats: {e}")
    
    def _create_test_run_summary(self, run_id: str, run_directory: str, test_type: str, moves_imported: int):
        """Create a test run summary."""
        
        # Calculate run metrics from logged moves
        run_moves = [move for move in self.metadata["logged_moves"].values() if move["run_id"] == run_id]
        
        if run_moves:
            total_moves = len(run_moves)
            successful_moves = sum(1 for move in run_moves if move["accepted"])
            success_rate = successful_moves / total_moves if total_moves > 0 else 0.0
            
            vector_usage = {}
            failure_patterns = {}
            
            for move in run_moves:
                vector_id = move["vector_id"]
                vector_usage[vector_id] = vector_usage.get(vector_id, 0) + 1
                
                if not move["accepted"]:
                    reason = move["reason"]
                    failure_patterns[reason] = failure_patterns.get(reason, 0) + 1
            
            # Determine if run was successful based on test type
            success = success_rate > 0.5  # Basic success criteria
            
            test_run_summary = TestRunSummary(
                run_id=run_id,
                test_type=test_type,
                timestamp=datetime.now().isoformat(),
                success=success,
                duration_seconds=0.0,  # Would need to calculate from logs
                key_metrics={
                    "total_moves": total_moves,
                    "successful_moves": successful_moves,
                    "success_rate": success_rate,
                    "moves_imported": moves_imported
                },
                vector_usage=vector_usage,
                composition_usage={},  # Would need composition data
                failure_patterns=failure_patterns,
                notable_events=[],
                run_directory=run_directory
            )
            
            self.metadata["test_runs"][run_id] = asdict(test_run_summary)
            self.metadata["system_info"]["total_test_runs"] += 1
    
    def get_vector_analytics(self) -> Dict[str, Any]:
        """Get comprehensive vector analytics."""
        vectors = self.metadata["vectors"]
        
        if not vectors:
            return {"message": "No vector data available"}
        
        # Top performers by success rate
        top_by_success = sorted(
            [(vid, data) for vid, data in vectors.items() if data["total_uses"] > 0],
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )[:10]
        
        # Most used vectors
        most_used = sorted(
            vectors.items(),
            key=lambda x: x[1]["total_uses"],
            reverse=True
        )[:10]
        
        # Physics domain analysis
        domain_stats = {}
        for vector_id, data in vectors.items():
            domain = data["physics_domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"count": 0, "total_uses": 0, "avg_success_rate": 0.0}
            
            domain_stats[domain]["count"] += 1
            domain_stats[domain]["total_uses"] += data["total_uses"]
        
        for domain, stats in domain_stats.items():
            domain_vectors = [v for v in vectors.values() if v["physics_domain"] == domain and v["total_uses"] > 0]
            if domain_vectors:
                stats["avg_success_rate"] = sum(v["success_rate"] for v in domain_vectors) / len(domain_vectors)
        
        return {
            "total_vectors": len(vectors),
            "active_vectors": len([v for v in vectors.values() if v["total_uses"] > 0]),
            "top_by_success_rate": [(vid, data["success_rate"], data["total_uses"]) for vid, data in top_by_success],
            "most_used_vectors": [(vid, data["total_uses"], data["success_rate"]) for vid, data in most_used],
            "physics_domain_stats": domain_stats
        }
    
    def get_composition_analytics(self) -> Dict[str, Any]:
        """Get comprehensive composition analytics."""
        compositions = self.metadata["compositions"]
        
        if not compositions:
            return {"message": "No composition data available"}
        
        # Active compositions
        active_compositions = [(cid, data) for cid, data in compositions.items() if data["total_uses"] > 0]
        
        # Top performers
        top_performers = sorted(
            active_compositions,
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )
        
        return {
            "total_compositions": len(compositions),
            "active_compositions": len(active_compositions),
            "top_performers": [(cid, data["success_rate"], data["total_uses"]) for cid, data in top_performers],
            "composition_types": list(set(data["composition_type"] for data in compositions.values()))
        }
    
    def export_vector_data(self, vector_id: str) -> Dict[str, Any]:
        """Export complete data for a specific vector."""
        if vector_id not in self.metadata["vectors"]:
            return {"error": f"Vector {vector_id} not found"}
        
        vector_data = self.metadata["vectors"][vector_id].copy()
        
        # Add related logged moves
        related_moves = [
            move for move in self.metadata["logged_moves"].values()
            if move["vector_id"] == vector_id
        ]
        
        vector_data["logged_moves"] = related_moves
        vector_data["export_timestamp"] = datetime.now().isoformat()
        
        return vector_data
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get complete system summary."""
        return {
            "system_info": self.metadata["system_info"],
            "vector_analytics": self.get_vector_analytics(),
            "composition_analytics": self.get_composition_analytics(),
            "recent_test_runs": list(self.metadata["test_runs"].keys())[-10:],
            "metadata_file_size": self.metadata_file.stat().st_size if self.metadata_file.exists() else 0
        }


__all__ = [
    "VectorMetrics", "CompositionMetrics", "TestRunSummary", "LoggedMove", "UnifiedMetadataManager"
]