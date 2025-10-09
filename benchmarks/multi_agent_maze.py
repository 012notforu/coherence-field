"""Multi-agent maze solving using distributed grid cell orchestrators."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.multi_agent_grid import MultiAgentGridSystem, GridCellAgent
from benchmarks.maze_solving import generate_maze, MazeParams
from utils.logging import create_run_directory, RunLogger

Array = np.ndarray


@dataclass
class MazeAgentConfig:
    """Configuration for maze-solving agents."""
    goal_attraction_strength: float = 0.3
    path_marking_strength: float = 0.1
    wall_repulsion_strength: float = -0.5
    exploration_bonus: float = 0.2
    communication_range: int = 2  # How far agents can sense each other


class MazeGridCellAgent(GridCellAgent):
    """Specialized grid cell agent for maze solving."""
    
    def __init__(self, 
                 cell_pos: Tuple[int, int], 
                 grid_shape: Tuple[int, int],
                 vector_registry,
                 physics_cfg,
                 maze_layout: Array,
                 start_pos: Tuple[int, int],
                 goal_pos: Tuple[int, int],
                 config: MazeAgentConfig,
                 agent_id: str = None):
        super().__init__(cell_pos, grid_shape, vector_registry, physics_cfg, agent_id)
        
        self.maze_layout = maze_layout
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.config = config
        
        # Maze-specific state
        self.visited_positions = set()
        self.path_history = []
        self.distance_to_goal = self._manhattan_distance(cell_pos, goal_pos)
        self.last_distance_to_goal = self.distance_to_goal
        
        # Agent communication
        self.messages_received = []
        self.last_message_sent = None
        
        # Failure pattern tracking
        self.last_failure_pattern = None
        
        # Override activation threshold for maze solving
        self.activation_threshold = 0.05  # More sensitive for maze navigation
        
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not a wall)."""
        i, j = pos
        h, w = self.maze_layout.shape
        return (0 <= i < h and 0 <= j < w and self.maze_layout[i, j] == 0)
    
    def sense_maze_environment(self, global_theta: Array, agent_positions: Dict) -> Dict[str, float]:
        """Enhanced environment sensing for maze navigation."""
        # Get base features
        base_features = self.sense_local_environment(global_theta)
        
        # Add maze-specific features
        i, j = self.pos
        
        # Goal direction and distance
        goal_direction_x = self.goal_pos[0] - i
        goal_direction_y = self.goal_pos[1] - j
        current_distance = self._manhattan_distance(self.pos, self.goal_pos)
        
        # Check valid neighbors (not walls)
        valid_neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_pos = (i + di, j + dj)
            if self._is_valid_position(neighbor_pos):
                valid_neighbors.append(neighbor_pos)
        
        # Count nearby agents
        nearby_agents = 0
        for agent_pos in agent_positions.keys():
            if (agent_pos != self.pos and 
                self._manhattan_distance(self.pos, agent_pos) <= self.config.communication_range):
                nearby_agents += 1
        
        # Check if we're making progress toward goal
        progress_toward_goal = self.last_distance_to_goal - current_distance
        
        # Compute physics context for meta-vector selection
        from engine.ops import grad, laplacian
        from engine.energy import total_energy_density
        
        # Local coherence (count similar neighbors)
        h, w = global_theta.shape
        neighborhood = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i + di) % h, (j + dj) % w
                neighborhood.append(global_theta[ni, nj])
        
        center_val = global_theta[i, j]
        coherence = float(np.mean([1.0 if abs(val - center_val) < 0.1 else 0.0 for val in neighborhood]))
        
        # Local curvature
        curvature = float(laplacian(global_theta)[i, j])
        
        # Energy density at this cell
        try:
            energy_density = float(total_energy_density(global_theta, np.zeros_like(global_theta), self.physics_cfg)[i, j])
        except:
            energy_density = float(np.sum(global_theta[i, j]**2))
        
        maze_features = {
            **base_features,
            "goal_direction_x": float(goal_direction_x),
            "goal_direction_y": float(goal_direction_y),
            "distance_to_goal": float(current_distance),
            "progress_toward_goal": float(progress_toward_goal),
            "valid_neighbors": len(valid_neighbors),
            "nearby_agents": nearby_agents,
            "visited_this_position": 1.0 if self.pos in self.visited_positions else 0.0,
            "is_start": 1.0 if self.pos == self.start_pos else 0.0,
            "is_goal": 1.0 if self.pos == self.goal_pos else 0.0,
            # Physics context for meta-vector
            "coherence": coherence,
            "curvature": curvature,
            "energy_density": energy_density,
        }
        
        # Update tracking
        self.last_distance_to_goal = current_distance
        self.visited_positions.add(self.pos)
        
        return maze_features
    
    def detect_maze_failure_pattern(self, maze_features: Dict[str, float]) -> str:
        """Detect maze-specific failure patterns."""
        # Check if at goal
        if self.pos == self.goal_pos:
            return "goal_reached"
        
        # Check if stuck at walls
        if maze_features["valid_neighbors"] <= 1:
            return "dead_end"
        
        # Check if revisiting positions frequently
        if len(self.visited_positions) > 0:
            revisit_rate = len(self.path_history) / len(self.visited_positions)
            if revisit_rate > 3.0:
                return "excessive_backtracking"
        
        # Check if making progress toward goal
        if maze_features["progress_toward_goal"] < 0:
            return "moving_away_from_goal"
        
        # Check for local oscillation
        if len(self.path_history) >= 4:
            recent_positions = self.path_history[-4:]
            if len(set(recent_positions)) <= 2:
                return "local_oscillation"
        
        # Check general progress
        if self.timestep - self.last_progress_step > 10:
            return "no_recent_progress"
        
        # Check if other agents are nearby (might need coordination)
        if maze_features["nearby_agents"] > 0:
            return "agent_coordination_needed"
        
        return "making_progress"
    
    def select_maze_vector(self, pattern: str, maze_features: Dict[str, float]):
        """Select vector specialized for maze solving patterns."""
        # Define maze-specific vector preferences with multiple fallbacks
        maze_pattern_map = {
            "goal_reached": ["align", "stable"],  # Stabilize at goal
            "dead_end": ["explore", "unstable"],   # Need to backtrack/explore
            "excessive_backtracking": ["explore", "unstable"],  # Try different exploration strategy
            "moving_away_from_goal": ["align", "stable", "flow"],     # Realign toward goal
            "local_oscillation": ["explore", "unstable"],      # Break out of loop
            "no_recent_progress": ["explore", "unstable"],     # Try exploration
            "agent_coordination_needed": ["align", "wave"], # Coordinate with others
            "making_progress": ["align", "stable"]           # Continue current strategy
        }
        
        # Check for precision approach conditions (meta-learned vector)
        distance = maze_features["distance_to_goal"]
        if (distance <= 5 and 
            maze_features.get("coherence", 0.5) < 0.2 and
            maze_features.get("energy_density", 0.1) > 0.2 and
            maze_features.get("nearby_agents", 1) == 0):
            # Precision approach conditions met - prioritize meta-learned vector
            precision_vectors = [v for v in self.registry.base_registry 
                               if v.vector_id == "precision_approach_meta"]
            if precision_vectors:
                return precision_vectors[0], "meta_precision_approach", 0.95
        
        preferred_purposes = maze_pattern_map.get(pattern, ["explore", "align"])
        
        # Get vectors with preferred purposes
        candidate_vectors = []
        for purpose in preferred_purposes:
            vectors = self.registry.get_vectors_by_tag(purpose)
            candidate_vectors.extend(vectors)
        
        # Also include physics-based preferences by filtering vectors
        distance = maze_features["distance_to_goal"]
        if distance > 10:  # Far from goal - use exploration vectors from flow/heat
            for vector in self.registry.base_registry:
                if vector.physics in ["flow", "heat"]:
                    candidate_vectors.append(vector)
        elif distance <= 3:  # Close to goal - use precision vectors from cartpole
            for vector in self.registry.base_registry:
                if vector.physics == "cartpole":
                    candidate_vectors.append(vector)
            candidate_vectors.extend(self.registry.get_vectors_by_tag("stabilize"))
        
        # Remove duplicates by vector ID
        seen_ids = set()
        unique_vectors = []
        for vector in candidate_vectors:
            if vector.vector_id not in seen_ids:
                unique_vectors.append(vector)
                seen_ids.add(vector.vector_id)
        candidate_vectors = unique_vectors
        
        if not candidate_vectors:
            candidate_vectors = self.registry.base_registry
        
        # Score vectors based on maze context
        best_vector = None
        best_score = -1.0
        
        for vector in candidate_vectors:
            # Base confidence score
            tag = self.registry.tags[vector.vector_id]
            confidence = tag.confidence_history[-1] if tag.confidence_history else 0.5
            score = confidence
            
            # Strong maze-specific bonuses
            if pattern == "dead_end":
                if "explore" in vector.vector_id or vector.physics in ["flow", "heat"]:
                    score += 0.4
            elif pattern == "moving_away_from_goal":
                if vector.physics in ["flow", "heat"] or "align" in vector.vector_id:
                    score += 0.3
            elif pattern == "making_progress":
                if "stable" in vector.vector_id or vector.physics == "cartpole":
                    score += 0.2
            
            # Distance-based bonuses
            if distance <= 3 and "stable" in vector.vector_id:
                score += 0.3  # Precision near goal
            elif distance > 8 and vector.physics in ["flow", "heat"]:
                score += 0.2  # Exploration when far
            
            # Penalize vectors that recently failed in this pattern
            if (vector.vector_id == self.last_vector_id and 
                not maze_features.get("progress_toward_goal", 0) > 0 and
                pattern == self.last_failure_pattern):
                score -= 0.3
            
            if score > best_score:
                best_score = score
                best_vector = vector
        
        # Track failure pattern for next selection
        if maze_features.get("progress_toward_goal", 0) <= 0:
            self.last_failure_pattern = pattern
        
        return best_vector or candidate_vectors[0], "maze_pattern_match", best_score
    
    def compute_movement_utility(self, global_theta, global_theta_dot, maze_features):
        """Compute U_move = w_C*C + w_E*(-E) + w_F*(-F) + w_G*G"""
        # Movement weights (configurable)
        w_C = 1.0    # Maximize coherence
        w_E = 0.0    # Energy term (disabled for now)
        w_F = 0.0    # Free energy term (disabled for now) 
        w_G = 0.3    # Goal attraction
        
        # Get physics values at current position
        coherence = maze_features['coherence']
        energy_density = maze_features['energy_density']
        
        # Goal potential (negative distance = attraction)
        goal_distance = maze_features['distance_to_goal']
        goal_potential = -goal_distance  # Closer to goal = higher potential
        
        # Compute utility
        U_move = w_C * coherence + w_G * goal_potential
        
        return U_move, {'w_C': w_C, 'w_E': w_E, 'w_F': w_F, 'w_G': w_G}
    
    def _softmax(self, scores, temperature=1.0):
        """Convert scores to probabilities via softmax"""
        # Handle infinite values (invalid moves)
        finite_scores = [s if s != float('-inf') else -1e6 for s in scores]
        
        # Apply temperature scaling
        scaled_scores = [s / temperature for s in finite_scores]
        
        # Compute softmax
        max_score = max(scaled_scores)
        exp_scores = [np.exp(s - max_score) for s in scaled_scores]
        sum_exp = sum(exp_scores)
        
        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)  # Uniform if all invalid
        
        return [exp_s / sum_exp for exp_s in exp_scores]
    
    def compute_movement_direction(self, global_theta, maze_features):
        """Compute movement probabilities using gradient of U_move"""
        from engine.ops import grad
        
        # Compute coherence gradient
        try:
            coherence_grad_x, coherence_grad_y = grad(global_theta)
        except:
            # Fallback: simple finite difference
            i, j = self.pos
            h, w = global_theta.shape
            coherence_grad_x = global_theta[(i+1)%h, j] - global_theta[(i-1)%h, j]
            coherence_grad_y = global_theta[i, (j+1)%w] - global_theta[i, (j-1)%w]
        
        # Goal gradient (simple: points toward goal)
        i, j = self.pos
        goal_i, goal_j = self.goal_pos
        goal_grad_x = 1.0 if goal_i > i else -1.0 if goal_i < i else 0.0
        goal_grad_y = 1.0 if goal_j > j else -1.0 if goal_j < j else 0.0
        
        # Movement gradient (from disone.txt: ∇U_move = w_C*∇C + w_G*∇G)
        w_C, w_G = 1.0, 0.3
        if isinstance(coherence_grad_x, np.ndarray):
            movement_grad_x = w_C * coherence_grad_x[i, j] + w_G * goal_grad_x
            movement_grad_y = w_C * coherence_grad_y[i, j] + w_G * goal_grad_y
        else:
            movement_grad_x = w_C * coherence_grad_x + w_G * goal_grad_x
            movement_grad_y = w_C * coherence_grad_y + w_G * goal_grad_y
        
        movement_gradient = np.array([movement_grad_x, movement_grad_y])
        
        # Score each valid direction
        directions = {
            'up': np.array([-1, 0]),
            'down': np.array([1, 0]), 
            'left': np.array([0, -1]),
            'right': np.array([0, 1])
        }
        
        movement_scores = {}
        beta = 5.0  # Greediness parameter
        
        for direction_name, direction_vector in directions.items():
            next_pos = (i + direction_vector[0], j + direction_vector[1])
            
            if self._is_valid_position(next_pos):
                # Score = β * (gradient · direction)
                score = beta * np.dot(movement_gradient, direction_vector)
                movement_scores[direction_name] = score
            else:
                movement_scores[direction_name] = float('-inf')  # Invalid move
        
        # Convert to probabilities via softmax
        score_list = list(movement_scores.values())
        movement_probs_list = self._softmax(score_list)
        movement_probs = dict(zip(movement_scores.keys(), movement_probs_list))
        
        return movement_probs, movement_gradient
    
    def _apply_meta_vector(self, 
                          vector_entry, 
                          global_theta: Array, 
                          global_theta_dot: Array,
                          maze_features: Dict[str, float],
                          pattern: str):
        """Apply the meta-learned precision approach vector."""
        from vectors.precision_approach_meta import apply_precision_approach_meta
        from utils.logging import VectorInvocation, OutcomeMetrics
        
        # Create context from maze features
        context = {
            'goal_distance': maze_features['distance_to_goal'],
            'coherence': maze_features['coherence'],
            'curvature': maze_features['curvature'],
            'energy_density': maze_features['energy_density'],
            'valid_neighbors': maze_features['valid_neighbors'],
            'nearby_agents': maze_features['nearby_agents'],
            'progress_made': max(0.0, maze_features['progress_toward_goal'])
        }
        
        # Apply meta-learned vector
        try:
            new_theta, new_theta_dot = apply_precision_approach_meta(
                global_theta, global_theta_dot, self.physics_cfg, context
            )
            
            # Calculate field change at agent position
            i, j = self.pos
            delta_energy = float(new_theta[i, j] - global_theta[i, j])
            
            # Update global field
            global_theta[:] = new_theta
            global_theta_dot[:] = new_theta_dot
            
            # Create invocation record
            vector_invocation = VectorInvocation(
                vector_id="precision_approach_meta",
                reason=pattern,
                confidence=0.95,
                selection_method="meta_precision_approach"
            )
            
            # Create outcome record
            outcome = OutcomeMetrics(
                delta_energy=delta_energy,
                accepted=True,  # Meta-learned vector is always accepted
                progress_made=1.0 if abs(delta_energy) > 0.01 else 0.0,
                stuck_counter=0
            )
            
            return vector_invocation, outcome, True
            
        except Exception as e:
            print(f"Meta-vector application failed: {e}")
            # Fallback to no change
            vector_invocation = VectorInvocation(
                vector_id="precision_approach_meta",
                reason=pattern,
                confidence=0.0,
                selection_method="meta_precision_approach_failed"
            )
            
            outcome = OutcomeMetrics(
                delta_energy=0.0,
                accepted=False,
                progress_made=0.0,
                stuck_counter=1
            )
            
            return vector_invocation, outcome, False
    
    def apply_maze_vector(self, 
                         vector_entry, 
                         global_theta: Array, 
                         global_theta_dot: Array,
                         maze_features: Dict[str, float],
                         pattern: str):
        """Apply vector with maze-specific field modifications."""
        # Check if this is the meta-learned precision approach vector
        if vector_entry.vector_id == "precision_approach_meta":
            vector_invocation, outcome, accepted = self._apply_meta_vector(
                vector_entry, global_theta, global_theta_dot, maze_features, pattern
            )
        else:
            # Apply standard vector
            vector_invocation, outcome, accepted = self.apply_vector_with_physics(
                vector_entry, global_theta, global_theta_dot, pattern
            )
        
        # Always apply maze-specific field modifications for navigation
        i, j = self.pos
        goal_i, goal_j = self.goal_pos
        
        # Create strong gradient field pointing toward goal
        distance_to_goal = self._manhattan_distance(self.pos, self.goal_pos)
        
        # Apply goal attraction field - stronger when closer to goal
        attraction_strength = self.config.goal_attraction_strength * (1.0 + 1.0 / max(distance_to_goal, 1))
        
        # Check all 4 directions and strengthen field toward goal
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_pos = (i + di, j + dj)
            if self._is_valid_position(neighbor_pos):
                ni, nj = neighbor_pos
                
                # Calculate if this direction moves toward goal
                current_dist = self._manhattan_distance(self.pos, self.goal_pos)
                neighbor_dist = self._manhattan_distance(neighbor_pos, self.goal_pos)
                
                if neighbor_dist < current_dist:  # This direction is toward goal
                    global_theta[ni, nj] += attraction_strength
                elif neighbor_dist > current_dist:  # This direction is away from goal
                    global_theta[ni, nj] -= attraction_strength * 0.5
        
        if accepted:
            # Mark successful path for other agents
            global_theta[i, j] += self.config.path_marking_strength
            
            # If we made progress, strengthen the path we came from
            if maze_features["progress_toward_goal"] > 0 and len(self.path_history) > 0:
                prev_pos = self.path_history[-1]
                prev_i, prev_j = prev_pos
                if self._is_valid_position(prev_pos):
                    global_theta[prev_i, prev_j] += self.config.path_marking_strength * 0.5
        
        # Track path
        self.path_history.append(self.pos)
        if len(self.path_history) > 20:  # Keep recent history
            self.path_history.pop(0)
        
        return vector_invocation, outcome, accepted
    
    def step_maze_solving(self, 
                         global_theta: Array, 
                         global_theta_dot: Array,
                         agent_positions: Dict,
                         logger: RunLogger) -> bool:
        """Execute one maze-solving timestep."""
        self.timestep += 1
        
        # Check if already at goal
        if self.pos == self.goal_pos:
            return True  # Goal reached
        
        # Enhanced maze environment sensing
        maze_features = self.sense_maze_environment(global_theta, agent_positions)
        
        # Detect maze-specific failure patterns first
        pattern = self.detect_maze_failure_pattern(maze_features)
        
        # Check if should activate (more aggressive for maze solving)
        field_activity = abs(global_theta[self.pos] - self.last_field_value)
        should_activate = (
            field_activity > self.activation_threshold or
            maze_features["nearby_agents"] > 0 or
            maze_features["progress_toward_goal"] != 0 or
            maze_features["distance_to_goal"] <= 5 or  # Always active when close to goal
            pattern in ["dead_end", "excessive_backtracking", "local_oscillation"] or
            (self.timestep - self.last_progress_step) > 3  # More frequent activation
        )
        
        if not should_activate:
            self.active = False
            return False
        
        self.active = True
        
        # Update exploration mode based on maze patterns
        if pattern in ["dead_end", "excessive_backtracking", "local_oscillation"]:
            if not self.exploration_mode:
                self.exploration_mode = True
                self.exploration_steps_remaining = 3
        
        # Select vector for maze solving
        vector_entry, selection_method, confidence = self.select_maze_vector(pattern, maze_features)
        
        # Apply vector with maze modifications
        vector_invocation, outcome, accepted = self.apply_maze_vector(
            vector_entry, global_theta, global_theta_dot, maze_features, pattern
        )
        
        # Compute physics context
        physics_ctx = logger.compute_physics_context(
            global_theta, global_theta_dot, self.physics_cfg, self.pos
        )
        
        # Enhanced neighbor summary with maze context
        neighbor_summary = logger.compute_neighbor_summary(global_theta, self.pos)
        neighbor_summary.update({
            "goal_distance": maze_features["distance_to_goal"],
            "valid_neighbors": maze_features["valid_neighbors"],
            "nearby_agents": maze_features["nearby_agents"]
        })
        
        # Log action
        from utils.logging import CellLogEntry
        log_entry = CellLogEntry(
            cell_pos=self.pos,
            timestep=self.timestep,
            physics=physics_ctx,
            vector=vector_invocation,
            outcome=outcome,
            neighbors=neighbor_summary
        )
        
        logger.log_cell_action(log_entry)
        
        # Log maze-specific interactions
        if accepted:
            # Log progress toward goal
            logger.log_interaction(
                source_cell=self.pos,
                target_cell=self.goal_pos,
                interaction_type="goal_progress",
                strength=abs(maze_features["progress_toward_goal"]),
                timestep=self.timestep
            )
            
            # Log path marking for other agents
            if abs(outcome.delta_energy) > 0.05:
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = self.pos[0] + di, self.pos[1] + dj
                    if self._is_valid_position((ni, nj)):
                        logger.log_interaction(
                            source_cell=self.pos,
                            target_cell=(ni, nj),
                            interaction_type="path_marking",
                            strength=self.config.path_marking_strength,
                            timestep=self.timestep
                        )
        
        return accepted


class MultiAgentMazeSolver:
    """Multi-agent system specialized for maze solving."""
    
    def __init__(self, 
                 maze_params: MazeParams,
                 maze_layout: Array,
                 config: MazeAgentConfig = None):
        self.maze_params = maze_params
        self.maze_layout = maze_layout
        self.config = config or MazeAgentConfig()
        
        # Set start and goal positions
        h, w = maze_params.shape
        self.start_pos = maze_params.start_pos or (1, 1)
        self.goal_pos = maze_params.goal_pos or (h-2, w-2)
        
        # Initialize multi-agent system
        self.agent_system = MultiAgentGridSystem(maze_params.shape)
        
        # Replace agents with maze-solving agents
        self._initialize_maze_agents()
        
        # Maze solving state
        self.solved = False
        self.solution_path = []
        self.solving_steps = 0
        
    def _initialize_maze_agents(self):
        """Initialize maze-solving agents at strategic positions."""
        # Clear existing agents
        self.agent_system.agents.clear()
        
        # Create agents at free positions near start and interesting locations
        h, w = self.maze_layout.shape
        agent_positions = []
        
        # Always place agent at start
        if self.maze_layout[self.start_pos] == 0:
            agent_positions.append(self.start_pos)
        
        # Place agents at strategic positions (junctions, corridors)
        for i in range(1, h-1, 3):  # Sparse placement
            for j in range(1, w-1, 3):
                if self.maze_layout[i, j] == 0:  # Free space
                    # Check if it's a junction (3+ neighbors)
                    neighbors = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < h and 0 <= nj < w and self.maze_layout[ni, nj] == 0):
                            neighbors += 1
                    
                    if neighbors >= 3 or len(agent_positions) < 3:  # Junction or ensure minimum agents
                        agent_positions.append((i, j))
        
        # Create maze-solving agents
        for idx, pos in enumerate(agent_positions):
            agent = MazeGridCellAgent(
                pos, self.maze_layout.shape,
                self.agent_system.vector_registry,
                self.agent_system.cfg.physics,
                self.maze_layout,
                self.start_pos,
                self.goal_pos,
                self.config,
                f"maze_agent_{idx}"
            )
            self.agent_system.agents[pos] = agent
        
        print(f"Initialized {len(agent_positions)} maze-solving agents")
        print(f"Agent positions: {agent_positions}")
    
    def execute_agent_movement(self, agent, movement_probs, logger):
        """Execute agent movement based on computed probabilities"""
        # Select direction with highest probability
        selected_direction = max(movement_probs.items(), key=lambda x: x[1])[0]
        max_prob = movement_probs[selected_direction]
        
        if max_prob > 0.1:  # Movement threshold
            # Compute new position
            direction_deltas = {
                'up': (-1, 0), 'down': (1, 0),
                'left': (0, -1), 'right': (0, 1)
            }
            
            if selected_direction in direction_deltas:
                di, dj = direction_deltas[selected_direction]
                new_pos = (agent.pos[0] + di, agent.pos[1] + dj)
                
                # Check if new position is free (no other agents)
                if new_pos not in self.agent_system.agents:
                    # Move agent
                    old_pos = agent.pos
                    self.agent_system.agents[new_pos] = agent
                    del self.agent_system.agents[old_pos]
                    agent.pos = new_pos
                    
                    # Update agent's path history
                    agent.path_history.append(new_pos)
                    if len(agent.path_history) > 20:
                        agent.path_history.pop(0)
                    
                    # Log movement
                    logger.log_interaction(
                        source_cell=old_pos,
                        target_cell=new_pos, 
                        interaction_type="agent_movement",
                        strength=max_prob,
                        timestep=self.agent_system.timestep
                    )
                    
                    return True, new_pos, selected_direction
                else:
                    # Position occupied, log blocked movement
                    logger.log_interaction(
                        source_cell=agent.pos,
                        target_cell=new_pos,
                        interaction_type="movement_blocked",
                        strength=max_prob,
                        timestep=self.agent_system.timestep
                    )
        
        return False, agent.pos, None
    
    def _create_goal_gradient_field(self):
        """Create distance-based gradient field pointing toward goal."""
        h, w = self.maze_layout.shape
        
        # Use BFS to compute actual maze distances (not just Manhattan)
        distances = np.full((h, w), float('inf'))
        distances[self.goal_pos] = 0
        
        from collections import deque
        queue = deque([self.goal_pos])
        
        while queue:
            current_pos = queue.popleft()
            current_dist = distances[current_pos]
            
            # Check all 4 neighbors
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = current_pos[0] + di, current_pos[1] + dj
                
                # Valid position and not a wall
                if (0 <= ni < h and 0 <= nj < w and 
                    self.maze_layout[ni, nj] == 0 and 
                    distances[ni, nj] > current_dist + 1):
                    
                    distances[ni, nj] = current_dist + 1
                    queue.append((ni, nj))
        
        # Convert distances to gradient values (closer = higher value)
        max_dist = np.max(distances[distances != float('inf')])
        
        for i in range(h):
            for j in range(w):
                if self.maze_layout[i, j] == 0:  # Free space
                    if distances[i, j] != float('inf'):
                        # Gradient: 1.0 at goal, decreasing with distance
                        gradient_value = 1.0 - (distances[i, j] / max_dist) * 0.8
                        self.agent_system.global_theta[i, j] = gradient_value
                    else:
                        # Unreachable areas get negative value
                        self.agent_system.global_theta[i, j] = -0.5
    
    def solve_maze(self, max_steps: int = 100, log_dir: str = None) -> Dict:
        """Solve maze using multi-agent coordination."""
        if log_dir:
            run_dir = Path(log_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = create_run_directory("multi_agent_maze")
        
        logger = RunLogger(run_dir)
        
        print(f"Starting multi-agent maze solving...")
        print(f"Maze size: {self.maze_layout.shape}")
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        print(f"Agents: {len(self.agent_system.agents)}")
        
        # Initialize field with maze layout and goal gradient
        self.agent_system.global_theta = np.zeros_like(self.maze_layout, dtype=np.float32)
        
        # Create gradient field toward goal
        self._create_goal_gradient_field()
        
        # Mark walls as strong negative field values
        self.agent_system.global_theta[self.maze_layout == 1] = -2.0
        
        # Mark goal with very strong positive field
        self.agent_system.global_theta[self.goal_pos] = 2.0
        
        # Solve iteratively
        for step in range(max_steps):
            self.solving_steps = step + 1
            
            # Update global physics
            self.agent_system.timestep += 1
            
            # Refresh goal gradient periodically
            if step % 5 == 0:
                self._create_goal_gradient_field()
            
            # Get current agent positions
            agent_positions = {pos: agent for pos, agent in self.agent_system.agents.items()}
            
            # Step each maze agent
            stats = {
                "active_agents": 0,
                "successful_actions": 0,
                "agents_at_goal": 0,
                "total_exploration": 0
            }
            
            goal_reached = False
            movement_stats = {
                "agents_moved": 0,
                "movement_attempts": 0,
                "blocked_movements": 0
            }
            
            # Process each agent (need to use list() to avoid dict changing during iteration)
            for pos, agent in list(self.agent_system.agents.items()):
                # Step 1: Vector application (existing)
                accepted = agent.step_maze_solving(
                    self.agent_system.global_theta,
                    self.agent_system.global_theta_dot,
                    agent_positions,
                    logger
                )
                
                if agent.active:
                    stats["active_agents"] += 1
                    
                if accepted:
                    stats["successful_actions"] += 1
                
                # Step 2: Movement decision (NEW - every 3 steps)
                if step % 3 == 0 and agent.active:
                    # Get current maze features for movement computation
                    maze_features = agent.sense_maze_environment(
                        self.agent_system.global_theta, agent_positions
                    )
                    
                    # Compute movement direction
                    movement_probs, movement_grad = agent.compute_movement_direction(
                        self.agent_system.global_theta, maze_features
                    )
                    
                    # Execute movement
                    movement_stats["movement_attempts"] += 1
                    moved, new_pos, direction = self.execute_agent_movement(agent, movement_probs, logger)
                    
                    if moved:
                        movement_stats["agents_moved"] += 1
                        print(f"  Agent moved {pos} -> {new_pos} (direction: {direction})")
                        
                        # Check for goal reached
                        if new_pos == self.goal_pos:
                            stats["agents_at_goal"] += 1
                            goal_reached = True
                            self.solution_path = agent.path_history.copy()
                            print(f"  *** GOAL REACHED by agent at {new_pos}! ***")
                    else:
                        movement_stats["blocked_movements"] += 1
                
                # Check if agent is at goal (might have moved there)
                if agent.pos == self.goal_pos and not goal_reached:
                    stats["agents_at_goal"] += 1
                    goal_reached = True
                    self.solution_path = agent.path_history.copy()
                    print(f"  *** GOAL REACHED by agent at {agent.pos}! ***")
                
                if agent.exploration_mode:
                    stats["total_exploration"] += 1
            
            # Log global step with movement statistics
            global_energy = float(np.mean(self.agent_system.global_theta ** 2))
            logger.log_step({
                "step": step,
                "global_energy": global_energy,
                **stats,
                **movement_stats
            })
            
            # Check for solution
            if goal_reached:
                self.solved = True
                print(f"MAZE SOLVED at step {step + 1}!")
                break
            
            # Progress output
            if (step + 1) % 10 == 0 or stats["active_agents"] > 0:
                min_distance = min(
                    agent._manhattan_distance(agent.pos, self.goal_pos) 
                    for agent in self.agent_system.agents.values()
                )
                print(f"Step {step + 1:3d}: "
                      f"Active={stats['active_agents']:2d} "
                      f"Success={stats['successful_actions']:2d} "
                      f"AtGoal={stats['agents_at_goal']:2d} "
                      f"MinDist={min_distance:2d}")
        
        # Generate final summary
        logger.log_vector_stats_summary(self.agent_system.timestep)
        
        # Create result summary
        result = {
            "solved": self.solved,
            "steps_taken": self.solving_steps,
            "solution_path": self.solution_path,
            "agents_used": len(self.agent_system.agents),
            "final_positions": {pos: agent.pos for pos, agent in self.agent_system.agents.items()},
            "log_directory": str(run_dir)
        }
        
        print(f"\n=== Maze Solving Results ===")
        print(f"Solved: {result['solved']}")
        print(f"Steps taken: {result['steps_taken']}")
        if self.solved:
            print(f"Solution path length: {len(self.solution_path)}")
        print(f"Results saved to: {run_dir}")
        
        return result
    
    def visualize_solution(self, result: Dict, save_path: str = None):
        """Create visualization of maze solution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left: Maze with solution path
        maze_vis = self.maze_layout.copy()
        
        # Mark start and goal
        maze_vis[self.start_pos] = 0.3
        maze_vis[self.goal_pos] = 0.7
        
        # Mark solution path if found
        if self.solved and self.solution_path:
            for pos in self.solution_path:
                if pos != self.start_pos and pos != self.goal_pos:
                    maze_vis[pos] = 0.5
        
        # Mark final agent positions
        for agent in self.agent_system.agents.values():
            maze_vis[agent.pos] = 0.9
        
        ax1.imshow(maze_vis, cmap='RdYlBu_r')
        ax1.set_title(f"Multi-Agent Maze Solution\n{'SOLVED' if self.solved else 'UNSOLVED'}")
        ax1.axis('off')
        
        # Right: Field evolution
        field_vis = self.agent_system.global_theta
        im = ax2.imshow(field_vis, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title("Final SCFD Field State")
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        else:
            result_dir = Path(result["log_directory"])
            viz_path = result_dir / "maze_solution.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {viz_path}")
        
        plt.close()


__all__ = [
    "MazeAgentConfig", "MazeGridCellAgent", "MultiAgentMazeSolver"
]