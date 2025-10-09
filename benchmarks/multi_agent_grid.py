"""Multi-agent grid system with autonomous cell orchestrators."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from engine import load_config, accel_theta
from engine.integrators import leapfrog_step
from engine.energy import metropolis_accept, total_energy_density
from engine.ops import grad, laplacian
from orchestrator.pipeline import load_vector_registry, plan_for_environment
from utils.logging import (
    RunLogger, PhysicsContext, VectorInvocation, OutcomeMetrics, CellLogEntry
)

Array = np.ndarray


@dataclass
class VectorTag:
    """Metadata tags for vectors to enable pattern recognition."""
    primary_purpose: str  # "explore", "stabilize", "align", "smooth_front", "bridge_gap"
    physics_domain: str   # "heat", "flow", "wave", "cartpole", "gray_scott"
    activation_patterns: List[str]  # Conditions when this vector should be used
    confidence_history: List[float] = None  # Recent success rates
    cooldown_steps: int = 0  # Steps to wait before reusing
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = []


class VectorRegistry:
    """Enhanced vector registry with tagging and performance tracking."""
    
    def __init__(self, registry_path: str = "runs"):
        self.base_registry = load_vector_registry(registry_path)
        self.tags = self._initialize_tags()
        self.performance_history = {}
        
    def _initialize_tags(self) -> Dict[str, VectorTag]:
        """Initialize vector tags based on naming conventions."""
        tags = {}
        
        for vector_entry in self.base_registry:
            vector_id = vector_entry.vector_id
            physics = vector_entry.physics
            
            # Infer purpose from vector name and physics
            if "explore" in vector_id or "smoke" in vector_id:
                purpose = "explore"
                patterns = ["stuck_in_loop", "no_exploration"]
            elif "stable" in vector_id or "quick" in vector_id:
                purpose = "stabilize"
                patterns = ["local_oscillation", "high_gradient"]
            elif "front" in vector_id or "arc" in vector_id:
                purpose = "smooth_front"
                patterns = ["rough_interface", "high_curvature"]
            elif "bridge" in vector_id or "routing" in vector_id:
                purpose = "bridge_gap"
                patterns = ["disconnected_regions"]
            else:
                purpose = "align"
                patterns = ["low_coherence", "making_progress"]
            
            tags[vector_id] = VectorTag(
                primary_purpose=purpose,
                physics_domain=physics,
                activation_patterns=patterns
            )
        
        return tags
    
    def get_vectors_by_tag(self, purpose: str) -> List:
        """Get all vectors with specified purpose tag."""
        return [
            vec for vec in self.base_registry 
            if self.tags[vec.vector_id].primary_purpose == purpose
        ]
    
    def get_vectors_for_pattern(self, pattern: str) -> List:
        """Get vectors that should activate for this failure pattern."""
        return [
            vec for vec in self.base_registry
            if pattern in self.tags[vec.vector_id].activation_patterns
        ]
    
    def update_performance(self, vector_id: str, success: bool, energy_change: float):
        """Update performance tracking for vector."""
        if vector_id not in self.performance_history:
            self.performance_history[vector_id] = []
        
        self.performance_history[vector_id].append({
            "success": success,
            "energy_change": energy_change,
            "timestamp": time.time()
        })
        
        # Update confidence history (rolling average of last 10 uses)
        recent_successes = [
            1.0 if record["success"] else 0.0 
            for record in self.performance_history[vector_id][-10:]
        ]
        confidence = sum(recent_successes) / len(recent_successes) if recent_successes else 0.5
        self.tags[vector_id].confidence_history.append(confidence)


class GridCellAgent:
    """Autonomous grid cell agent with orchestrator workflow and field communication."""
    
    def __init__(self, 
                 cell_pos: Tuple[int, int], 
                 grid_shape: Tuple[int, int],
                 vector_registry: VectorRegistry,
                 physics_cfg,
                 agent_id: str = None):
        self.pos = cell_pos
        self.grid_shape = grid_shape
        self.registry = vector_registry
        self.physics_cfg = physics_cfg
        self.agent_id = agent_id or f"agent_{cell_pos[0]}_{cell_pos[1]}"
        
        # Activity state
        self.active = False
        self.activation_threshold = 0.01  # Minimum field change to activate (reduced from 0.1)
        self.timestep = 0
        
        # Decision state tracking
        self.last_vector_id = None
        self.stuck_counter = 0
        self.last_progress_step = 0
        self.recent_actions = []  # History for oscillation detection
        
        # Loop detection and recovery
        self.exploration_mode = False
        self.exploration_steps_remaining = 0
        
        # Field influence tracking
        self.last_field_value = 0.0
        self.field_gradient_history = []
        
    def should_activate(self, global_theta: Array) -> bool:
        """Determine if agent should activate based on local field activity."""
        i, j = self.pos
        current_value = global_theta[i, j]
        
        # Check for significant field changes
        field_change = abs(current_value - self.last_field_value)
        
        # Check local gradient magnitude
        gx, gy = grad(global_theta)
        local_gradient = float(np.sqrt(gx[i, j]**2 + gy[i, j]**2))
        
        # Activate if there's significant activity or we haven't acted recently
        should_activate = (
            field_change > self.activation_threshold or
            local_gradient > 0.02 or  # Reduced from 0.2 to 0.02
            (self.timestep - self.last_progress_step) > 20
        )
        
        self.last_field_value = current_value
        return should_activate
    
    def sense_local_environment(self, global_theta: Array) -> Dict[str, float]:
        """Extract local Moore neighborhood features."""
        i, j = self.pos
        h, w = self.grid_shape
        
        # Extract 3x3 neighborhood
        neighborhood = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = (i + di) % h, (j + dj) % w
                neighborhood.append(global_theta[ni, nj])
        
        center_val = neighborhood[4]  # Center of 3x3
        
        # Compute local features
        local_features = {
            "center_value": float(center_val),
            "neighbor_mean": float(np.mean(neighborhood[:4] + neighborhood[5:])),
            "neighbor_std": float(np.std(neighborhood)),
            "gradient_magnitude": float(np.std(neighborhood)),  # Proxy for local gradient
            "coherence": float(np.mean([
                1.0 if abs(val - center_val) < 0.1 else 0.0 
                for val in neighborhood
            ])),
            "entropy": float(-np.sum([
                p * np.log(p + 1e-8) for p in np.histogram(neighborhood, bins=5, density=True)[0] 
                if p > 0
            ]))
        }
        
        return local_features
    
    def detect_failure_pattern(self) -> str:
        """Analyze current state to detect failure patterns."""
        if self.exploration_mode:
            return "exploration_mode"
        
        if self.stuck_counter > 10:
            return "stuck_in_loop"
        
        if len(self.recent_actions) >= 6:
            # Check for oscillation in recent actions
            recent_positions = [action.get("outcome_pos", self.pos) for action in self.recent_actions[-6:]]
            unique_positions = len(set(recent_positions))
            if unique_positions < 3:
                return "local_oscillation"
        
        if self.timestep - self.last_progress_step > 15:
            return "no_recent_progress"
        
        # Check field gradient patterns
        if len(self.field_gradient_history) >= 5:
            recent_gradients = self.field_gradient_history[-5:]
            if all(g < 0.05 for g in recent_gradients):
                return "low_gradient_region"
            if any(g > 1.0 for g in recent_gradients):
                return "high_gradient"
        
        return "making_progress"
    
    def select_vector_by_pattern(self, pattern: str, local_features: Dict[str, float]):
        """Select vector based on detected pattern and local features."""
        # Get vectors that match the pattern
        candidate_vectors = self.registry.get_vectors_for_pattern(pattern)
        
        if not candidate_vectors:
            # Fallback to exploration vectors
            candidate_vectors = self.registry.get_vectors_by_tag("explore")
        
        if not candidate_vectors:
            # Final fallback to any vector
            candidate_vectors = self.registry.base_registry
        
        # Filter out recently used vectors (simple cooldown)
        if len(candidate_vectors) > 1 and self.last_vector_id:
            candidate_vectors = [
                vec for vec in candidate_vectors 
                if vec.vector_id != self.last_vector_id
            ]
        
        # Select based on confidence scores and local features
        best_vector = None
        best_score = -1.0
        
        for vector in candidate_vectors:
            # Base score from confidence history
            tag = self.registry.tags[vector.vector_id]
            confidence = tag.confidence_history[-1] if tag.confidence_history else 0.5
            
            # Adjust score based on local features
            score = confidence
            
            # Bonus for vectors that match current field characteristics
            if vector.physics == "heat" and local_features["gradient_magnitude"] > 0.3:
                score += 0.2
            elif vector.physics == "flow" and local_features["coherence"] < 0.5:
                score += 0.2
            elif vector.physics == "wave" and local_features["entropy"] > 1.0:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_vector = vector
        
        return best_vector or candidate_vectors[0], "pattern_match", best_score
    
    def apply_vector_with_physics(self, 
                                 vector_entry, 
                                 global_theta: Array, 
                                 global_theta_dot: Array,
                                 reason: str) -> Tuple[VectorInvocation, OutcomeMetrics, bool]:
        """Apply vector with full SCFD physics and metropolis acceptance."""
        # Load vector data
        with open(vector_entry.path) as f:
            data = json.load(f)
        vector = np.array(data["vector"], dtype=np.float32)
        
        i, j = self.pos
        
        # Record initial state
        initial_energy = float(total_energy_density(
            global_theta, global_theta_dot, self.physics_cfg
        )[i, j])
        
        # Apply vector influence (can be made more sophisticated based on vector type)
        field_backup = global_theta[i, j]
        
        if len(vector) >= 1:
            # Scale influence based on vector magnitude and current field state
            influence_strength = float(vector[0]) * 0.05  # Adjustable scale
            proposed_change = influence_strength
            
            # Test the change
            global_theta[i, j] += proposed_change
            
            # Compute new energy
            new_energy = float(total_energy_density(
                global_theta, global_theta_dot, self.physics_cfg
            )[i, j])
            
            delta_energy = new_energy - initial_energy
            
            # Metropolis acceptance (using free energy if in exploration mode)
            if self.exploration_mode:
                # Use higher temperature for exploration
                temperature = 1.0
                acceptance_prob = metropolis_accept(
                    np.array([delta_energy]), temperature, (0.01, 0.99)
                )[0]
            else:
                # Standard acceptance with lower temperature
                temperature = 0.4
                acceptance_prob = metropolis_accept(
                    np.array([delta_energy]), temperature, (0.01, 0.99)
                )[0]
            
            # Decision: accept or reject
            accepted = np.random.random() < acceptance_prob
            
            if not accepted:
                # Revert change
                global_theta[i, j] = field_backup
                delta_energy = 0.0
                progress_made = 0.0
            else:
                # Track progress
                if abs(delta_energy) > 0.01:
                    self.last_progress_step = self.timestep
                    self.stuck_counter = 0
                    progress_made = 1.0
                else:
                    self.stuck_counter += 1
                    progress_made = 0.0
        else:
            # Empty vector case
            accepted = False
            delta_energy = 0.0
            progress_made = 0.0
        
        # Create invocation record
        vector_invocation = VectorInvocation(
            vector_id=vector_entry.vector_id,
            reason=reason,
            confidence=0.8,  # Could be improved with actual confidence calculation
            selection_method="pattern_match"
        )
        
        # Create outcome record
        outcome = OutcomeMetrics(
            delta_energy=delta_energy,
            accepted=accepted,
            progress_made=progress_made,
            stuck_counter=self.stuck_counter
        )
        
        # Update performance tracking
        self.registry.update_performance(
            vector_entry.vector_id, 
            accepted and progress_made > 0, 
            delta_energy
        )
        
        # Track action history
        self.recent_actions.append({
            "vector_id": vector_entry.vector_id,
            "reason": reason,
            "accepted": accepted,
            "outcome_pos": self.pos
        })
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        self.last_vector_id = vector_entry.vector_id
        
        return vector_invocation, outcome, accepted
    
    def update_exploration_mode(self, pattern: str):
        """Manage exploration mode based on failure patterns."""
        if pattern in ["stuck_in_loop", "local_oscillation", "no_recent_progress"]:
            if not self.exploration_mode:
                self.exploration_mode = True
                self.exploration_steps_remaining = 5  # Explore for 5 steps
        
        if self.exploration_mode:
            self.exploration_steps_remaining -= 1
            if self.exploration_steps_remaining <= 0:
                self.exploration_mode = False
    
    def step(self, 
             global_theta: Array, 
             global_theta_dot: Array, 
             logger: RunLogger) -> bool:
        """Execute one timestep of autonomous agent behavior."""
        self.timestep += 1
        
        # Check if agent should activate
        if not self.should_activate(global_theta):
            self.active = False
            return False
        
        self.active = True
        
        # Sense local environment
        local_features = self.sense_local_environment(global_theta)
        
        # Compute physics context for logging
        physics_ctx = logger.compute_physics_context(
            global_theta, global_theta_dot, self.physics_cfg, self.pos
        )
        
        # Update field gradient history
        self.field_gradient_history.append(physics_ctx.grad_magnitude)
        if len(self.field_gradient_history) > 10:
            self.field_gradient_history.pop(0)
        
        # Detect current failure pattern
        pattern = self.detect_failure_pattern()
        
        # Update exploration mode
        self.update_exploration_mode(pattern)
        
        # Select vector based on pattern
        vector_entry, selection_method, confidence = self.select_vector_by_pattern(
            pattern, local_features
        )
        
        # Apply vector with full physics
        vector_invocation, outcome, accepted = self.apply_vector_with_physics(
            vector_entry, global_theta, global_theta_dot, pattern
        )
        
        # Compute neighbor summary for logging
        neighbor_summary = logger.compute_neighbor_summary(global_theta, self.pos)
        
        # Create and log complete entry
        log_entry = CellLogEntry(
            cell_pos=self.pos,
            timestep=self.timestep,
            physics=physics_ctx,
            vector=vector_invocation,
            outcome=outcome,
            neighbors=neighbor_summary
        )
        
        logger.log_cell_action(log_entry)
        
        # Log significant interactions with neighbors
        if accepted and abs(outcome.delta_energy) > 0.05:
            self._log_field_interactions(logger, outcome.delta_energy)
        
        return accepted
    
    def _log_field_interactions(self, logger: RunLogger, strength: float):
        """Log field-mediated interactions with neighboring cells."""
        i, j = self.pos
        h, w = self.grid_shape
        
        # Log interactions with immediate neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % h, (j + dj) % w
            logger.log_interaction(
                source_cell=self.pos,
                target_cell=(ni, nj),
                interaction_type="field_influence",
                strength=abs(strength),
                timestep=self.timestep
            )


class MultiAgentGridSystem:
    """Full multi-agent grid system with autonomous cell orchestrators."""
    
    def __init__(self, grid_shape: Tuple[int, int], config_path: str = "cfg/defaults.yaml"):
        self.grid_shape = grid_shape
        self.cfg = load_config(config_path)
        
        # Initialize global field state
        self.global_theta = np.random.random(grid_shape).astype(np.float32) * 0.1
        self.global_theta_dot = np.zeros_like(self.global_theta)
        
        # Initialize vector registry with tagging
        self.vector_registry = VectorRegistry("runs")
        
        # Create grid of agents (sparse activation - only create agents at interesting positions)
        self.agents = {}
        self._initialize_agents()
        
        # System state
        self.timestep = 0
        self.active_agents = set()
        
    def _initialize_agents(self):
        """Initialize agents at strategic grid positions."""
        h, w = self.grid_shape
        
        # Create agents at regular intervals for now (can be made smarter)
        agent_spacing = max(4, min(h, w) // 8)
        
        agent_id = 0
        for i in range(agent_spacing, h - agent_spacing, agent_spacing):
            for j in range(agent_spacing, w - agent_spacing, agent_spacing):
                pos = (i, j)
                agent = GridCellAgent(
                    pos, self.grid_shape, self.vector_registry, 
                    self.cfg.physics, f"agent_{agent_id}"
                )
                self.agents[pos] = agent
                agent_id += 1
        
        print(f"Initialized {len(self.agents)} agents on {h}x{w} grid")
    
    def step(self, logger: RunLogger) -> Dict[str, int]:
        """Execute one timestep of the multi-agent system."""
        self.timestep += 1
        
        # Update global physics first
        self.global_theta, self.global_theta_dot, _, _ = leapfrog_step(
            self.global_theta,
            self.global_theta_dot,
            lambda f: accel_theta(f, self.cfg.physics, dx=1.0),
            self.cfg.integration.dt
        )
        
        # Update agents
        stats = {
            "total_agents": len(self.agents),
            "active_agents": 0,
            "successful_actions": 0,
            "exploration_mode_agents": 0
        }
        
        self.active_agents.clear()
        
        for pos, agent in self.agents.items():
            accepted = agent.step(self.global_theta, self.global_theta_dot, logger)
            
            if agent.active:
                stats["active_agents"] += 1
                self.active_agents.add(pos)
                
                if accepted:
                    stats["successful_actions"] += 1
                
                if agent.exploration_mode:
                    stats["exploration_mode_agents"] += 1
        
        # Log global system state
        global_energy = float(np.mean(self.global_theta ** 2))
        logger.log_step({
            "timestep": self.timestep,
            "global_energy": global_energy,
            **stats
        })
        
        return stats
    
    def run_simulation(self, steps: int, log_dir: str = None) -> str:
        """Run multi-agent simulation for specified steps."""
        from utils.logging import create_run_directory
        
        # Create logging directory
        if log_dir:
            run_dir = Path(log_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = create_run_directory("multi_agent_grid")
        
        logger = RunLogger(run_dir)
        
        print(f"Running {steps} steps of multi-agent simulation...")
        print(f"Grid size: {self.grid_shape}")
        print(f"Agents: {len(self.agents)}")
        print(f"Vectors available: {len(self.vector_registry.base_registry)}")
        
        # Run simulation
        for step in range(steps):
            stats = self.step(logger)
            
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}: "
                      f"Active={stats['active_agents']}/{stats['total_agents']}, "
                      f"Success={stats['successful_actions']}, "
                      f"Explore={stats['exploration_mode_agents']}")
        
        # Generate final summary
        logger.log_vector_stats_summary(self.timestep)
        
        print(f"\nSimulation complete. Results saved to: {run_dir}")
        return str(run_dir)


__all__ = [
    "VectorTag", "VectorRegistry", "GridCellAgent", "MultiAgentGridSystem"
]