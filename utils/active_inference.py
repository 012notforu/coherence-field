#!/usr/bin/env python3
"""
Active Inference for SCFD Systems

Drop-in Expected Free Energy (EFE) scoring for agent movement and parameter tuning.
Balances exploitation (task progress) with exploration (uncertainty reduction).

EFE = Risk + Ambiguity
- Risk: expected cost vs goals (PSNR loss, composition failure, distance to target)
- Ambiguity: uncertainty in the system (entropy, prediction error, validator uncertainty)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class EFEConfig:
    """Configuration for Expected Free Energy calculation."""
    
    # EFE balance weights
    risk_weight: float = 0.6          # Weight for exploitation (task progress)
    ambiguity_weight: float = 0.4     # Weight for exploration (uncertainty reduction)
    
    # Risk components
    goal_distance_weight: float = 0.4  # Weight for distance to goal/target
    energy_change_weight: float = 0.3  # Weight for field energy changes
    composition_fail_weight: float = 0.3  # Weight for composition failure risk
    
    # Ambiguity components  
    entropy_weight: float = 0.4        # Weight for local entropy
    gradient_variance_weight: float = 0.3  # Weight for gradient uncertainty
    prediction_error_weight: float = 0.3   # Weight for prediction errors
    
    # Normalization scales
    distance_scale: float = 1.0        # Typical distance scale
    energy_scale: float = 0.1          # Typical energy change scale
    entropy_scale: float = 2.0         # Typical entropy scale
    gradient_scale: float = 0.01       # Typical gradient magnitude scale
    
    # Behavior parameters
    temperature: float = 1.0           # Softmax temperature for action selection
    exploration_threshold: float = 0.6 # When ambiguity > risk, enter exploration mode


class ActiveInferenceEvaluator:
    """Computes Expected Free Energy for SCFD agent actions."""
    
    def __init__(self, config: EFEConfig = None):
        self.config = config or EFEConfig()
        self._composition_fail_predictor = None
        
    def compute_movement_efe(self, 
                           current_pos: Tuple[int, int],
                           candidate_directions: List[Tuple[int, int]],
                           field_state: np.ndarray,
                           field_gradient: np.ndarray,
                           target_info: Dict,
                           agent_context: Dict) -> np.ndarray:
        """
        Compute EFE scores for candidate movement directions.
        
        Args:
            current_pos: Current agent position (i, j)
            candidate_directions: List of (di, dj) direction vectors
            field_state: Current field state
            field_gradient: Field gradient array
            target_info: Goal/target information (position, desired value, etc.)
            agent_context: Agent-specific context (history, capabilities, etc.)
            
        Returns:
            Array of EFE scores for each direction (lower = better)
        """
        efe_scores = []
        
        for direction in candidate_directions:
            # Calculate candidate position
            new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            
            # Compute risk and ambiguity components
            risk = self._compute_risk(current_pos, new_pos, direction, field_state, 
                                    field_gradient, target_info, agent_context)
            
            ambiguity = self._compute_ambiguity(current_pos, new_pos, direction, 
                                              field_state, field_gradient, agent_context)
            
            # Combine into EFE
            efe = (self.config.risk_weight * risk + 
                   self.config.ambiguity_weight * ambiguity)
            
            efe_scores.append(efe)
        
        return np.array(efe_scores)
    
    def _compute_risk(self, current_pos: Tuple[int, int], new_pos: Tuple[int, int],
                     direction: Tuple[int, int], field_state: np.ndarray,
                     field_gradient: np.ndarray, target_info: Dict, 
                     agent_context: Dict) -> float:
        """Compute risk component of EFE."""
        
        risk_components = []
        
        # 1. Goal distance risk (how far from target)
        if "target_pos" in target_info:
            target_pos = target_info["target_pos"]
            current_dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            new_dist = np.linalg.norm(np.array(new_pos) - np.array(target_pos))
            
            # Risk increases if we move away from goal
            distance_risk = (new_dist - current_dist) / self.config.distance_scale
            risk_components.append(self.config.goal_distance_weight * distance_risk)
        
        # 2. Energy change risk (unstable field changes)
        if new_pos[0] < field_state.shape[0] and new_pos[1] < field_state.shape[1]:
            current_energy = field_state[current_pos]
            new_energy = field_state[new_pos]
            
            # Risk for large energy changes (instability)
            energy_change = abs(new_energy - current_energy) / self.config.energy_scale
            risk_components.append(self.config.energy_change_weight * energy_change)
        
        # 3. Composition failure risk
        comp_fail_risk = self._predict_composition_failure_risk(
            new_pos, field_state, field_gradient, agent_context
        )
        risk_components.append(self.config.composition_fail_weight * comp_fail_risk)
        
        return sum(risk_components) if risk_components else 0.0
    
    def _compute_ambiguity(self, current_pos: Tuple[int, int], new_pos: Tuple[int, int],
                          direction: Tuple[int, int], field_state: np.ndarray,
                          field_gradient: np.ndarray, agent_context: Dict) -> float:
        """Compute ambiguity component of EFE."""
        
        ambiguity_components = []
        
        # 1. Local entropy (uncertainty in field values)
        local_entropy = self._compute_local_entropy(new_pos, field_state)
        ambiguity_components.append(self.config.entropy_weight * local_entropy)
        
        # 2. Gradient variance (uncertainty in field dynamics)
        gradient_variance = self._compute_gradient_variance(new_pos, field_gradient)
        ambiguity_components.append(self.config.gradient_variance_weight * gradient_variance)
        
        # 3. Prediction error (model uncertainty)
        prediction_error = self._compute_prediction_error(
            current_pos, new_pos, direction, field_state, agent_context
        )
        ambiguity_components.append(self.config.prediction_error_weight * prediction_error)
        
        return sum(ambiguity_components) if ambiguity_components else 0.0
    
    def _compute_local_entropy(self, pos: Tuple[int, int], field_state: np.ndarray,
                              radius: int = 2) -> float:
        """Compute local entropy around position."""
        if not self._valid_position(pos, field_state.shape):
            return 1.0  # High uncertainty for out-of-bounds
        
        # Extract local neighborhood
        i, j = pos
        i_min, i_max = max(0, i-radius), min(field_state.shape[0], i+radius+1)
        j_min, j_max = max(0, j-radius), min(field_state.shape[1], j+radius+1)
        
        local_patch = field_state[i_min:i_max, j_min:j_max]
        
        # Compute histogram-based entropy
        # Use 10 bins, handle edge case for uniform values
        try:
            hist, _ = np.histogram(local_patch.flatten(), bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            return entropy / self.config.entropy_scale
        except:
            return 0.5  # Medium uncertainty if entropy calculation fails
    
    def _compute_gradient_variance(self, pos: Tuple[int, int], 
                                  field_gradient: np.ndarray, radius: int = 1) -> float:
        """Compute variance in local gradient magnitudes."""
        if not self._valid_position(pos, field_gradient.shape[:2]):
            return 1.0  # High uncertainty for out-of-bounds
        
        # Extract local gradient magnitudes
        i, j = pos
        i_min, i_max = max(0, i-radius), min(field_gradient.shape[0], i+radius+1)
        j_min, j_max = max(0, j-radius), min(field_gradient.shape[1], j+radius+1)
        
        local_gradients = field_gradient[i_min:i_max, j_min:j_max]
        
        # Compute gradient magnitudes
        if len(local_gradients.shape) == 3:  # (i, j, components)
            grad_magnitudes = np.linalg.norm(local_gradients, axis=2)
        else:  # Assume 2D scalar field
            grad_magnitudes = np.abs(local_gradients)
        
        # Return variance normalized by scale
        variance = np.var(grad_magnitudes.flatten())
        return variance / (self.config.gradient_scale ** 2)
    
    def _compute_prediction_error(self, current_pos: Tuple[int, int],
                                 new_pos: Tuple[int, int], direction: Tuple[int, int],
                                 field_state: np.ndarray, agent_context: Dict) -> float:
        """Compute one-step prediction error."""
        if not self._valid_position(new_pos, field_state.shape):
            return 1.0  # High uncertainty for invalid moves
        
        # Simple prediction: linear extrapolation from current gradient
        if "recent_moves" in agent_context and len(agent_context["recent_moves"]) > 0:
            # Use recent field changes to predict
            recent_change = agent_context["recent_moves"][-1].get("field_change", 0.0)
            predicted_value = field_state[current_pos] + recent_change
            actual_value = field_state[new_pos]
            
            prediction_error = abs(predicted_value - actual_value)
            return min(prediction_error, 1.0)  # Cap at 1.0
        
        return 0.3  # Medium uncertainty if no prediction history
    
    def _predict_composition_failure_risk(self, pos: Tuple[int, int],
                                         field_state: np.ndarray,
                                         field_gradient: np.ndarray,
                                         agent_context: Dict) -> float:
        """Predict probability of composition failure at position."""
        
        # Simple heuristic-based predictor (can be improved with ML)
        risk_factors = []
        
        if self._valid_position(pos, field_state.shape):
            # Factor 1: Field value extremes (boundary conditions)
            field_value = field_state[pos]
            if field_value < 0.01 or field_value > 0.99:
                risk_factors.append(0.3)
            
            # Factor 2: High gradient regions (edge instability)
            if self._valid_position(pos, field_gradient.shape[:2]):
                grad_mag = np.linalg.norm(field_gradient[pos]) if len(field_gradient.shape) > 2 else abs(field_gradient[pos])
                if grad_mag > 0.1:  # High gradient threshold
                    risk_factors.append(0.2)
            
            # Factor 3: Recent composition failures in neighborhood
            if "composition_failures" in agent_context:
                nearby_failures = sum(1 for fail_pos in agent_context["composition_failures"] 
                                    if np.linalg.norm(np.array(pos) - np.array(fail_pos)) < 3)
                if nearby_failures > 0:
                    risk_factors.append(min(0.4, nearby_failures * 0.1))
        else:
            risk_factors.append(1.0)  # Maximum risk for out-of-bounds
        
        return min(sum(risk_factors), 1.0)  # Cap at 100% risk
    
    def _valid_position(self, pos: Tuple[int, int], shape: Tuple[int, ...]) -> bool:
        """Check if position is valid within array bounds."""
        return (0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1])
    
    def select_action(self, efe_scores: np.ndarray) -> int:
        """Select action using EFE scores with softmax."""
        # Convert EFE to action probabilities (lower EFE = higher probability)
        neg_efe = -efe_scores / self.config.temperature
        probabilities = np.exp(neg_efe - np.max(neg_efe))  # Numerical stability
        probabilities /= np.sum(probabilities)
        
        # Sample action
        return np.random.choice(len(efe_scores), p=probabilities)
    
    def is_exploration_mode(self, recent_efe_scores: List[np.ndarray]) -> bool:
        """Determine if agent should be in exploration mode."""
        if not recent_efe_scores:
            return False
        
        # Calculate average ambiguity vs risk ratio
        total_ambiguity = 0.0
        total_risk = 0.0
        count = 0
        
        for scores in recent_efe_scores[-5:]:  # Last 5 decisions
            for score in scores:
                # This is simplified - in practice you'd decompose the scores
                ambiguity_estimate = score * self.config.ambiguity_weight
                risk_estimate = score * self.config.risk_weight
                
                total_ambiguity += ambiguity_estimate
                total_risk += risk_estimate
                count += 1
        
        if count == 0 or total_risk == 0:
            return False
        
        ambiguity_ratio = total_ambiguity / (total_ambiguity + total_risk)
        return ambiguity_ratio > self.config.exploration_threshold


def create_efe_movement_wrapper(base_agent_class):
    """
    Factory function to wrap existing agent classes with EFE-based movement.
    
    Usage:
        EFEAgent = create_efe_movement_wrapper(GridCellAgent)
        agent = EFEAgent(pos, grid_shape, vector_registry, physics_cfg)
    """
    
    class EFEEnhancedAgent(base_agent_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.efe_evaluator = ActiveInferenceEvaluator()
            self.efe_history = []
            self.composition_failures = []
            
        def step_with_efe(self, global_theta, global_theta_dot, neighbor_states, logger=None):
            """Enhanced step method using EFE scoring."""
            
            # Get available directions (keep existing collision logic)
            directions = self._get_available_directions(global_theta.shape)
            
            if not directions:
                return super().step_with_composition(global_theta, global_theta_dot, neighbor_states, logger)
            
            # Prepare context for EFE evaluation
            agent_context = {
                "recent_moves": getattr(self, "move_history", []),
                "composition_failures": self.composition_failures,
                "agent_id": self.agent_id
            }
            
            # Target info (adapt based on your task)
            target_info = self._get_target_info(global_theta)
            
            # Compute EFE scores
            efe_scores = self.efe_evaluator.compute_movement_efe(
                current_pos=self.cell_pos,
                candidate_directions=directions,
                field_state=global_theta,
                field_gradient=global_theta_dot,
                target_info=target_info,
                agent_context=agent_context
            )
            
            # Select action
            selected_idx = self.efe_evaluator.select_action(efe_scores)
            selected_direction = directions[selected_idx]
            
            # Store EFE history
            self.efe_history.append(efe_scores)
            if len(self.efe_history) > 10:  # Keep last 10
                self.efe_history.pop(0)
            
            # Execute movement (use existing movement logic)
            new_pos = (self.cell_pos[0] + selected_direction[0], 
                      self.cell_pos[1] + selected_direction[1])
            
            # Update position and continue with normal agent logic
            old_pos = self.cell_pos
            self.cell_pos = new_pos
            
            # Continue with composition/vector selection
            result = super().step_with_composition(global_theta, global_theta_dot, neighbor_states, logger)
            
            # Track composition failures for future EFE calculations
            if hasattr(result, 'accepted') and not result.accepted:
                self.composition_failures.append(new_pos)
                if len(self.composition_failures) > 20:  # Keep last 20
                    self.composition_failures.pop(0)
            
            return result
        
        def _get_available_directions(self, grid_shape):
            """Get valid movement directions (keep existing logic)."""
            directions = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    new_i = self.cell_pos[0] + di
                    new_j = self.cell_pos[1] + dj
                    if 0 <= new_i < grid_shape[0] and 0 <= new_j < grid_shape[1]:
                        directions.append((di, dj))
            return directions
        
        def _get_target_info(self, global_theta):
            """Get target information for risk calculation."""
            # Default implementation - override for specific tasks
            return {
                "target_pos": (global_theta.shape[0]//2, global_theta.shape[1]//2),  # Center
                "desired_value": 0.5
            }
    
    return EFEEnhancedAgent


# Quick integration test
if __name__ == "__main__":
    # Test EFE computation
    config = EFEConfig()
    evaluator = ActiveInferenceEvaluator(config)
    
    # Mock data
    field_state = np.random.random((10, 10))
    field_gradient = np.random.random((10, 10, 2))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    target_info = {"target_pos": (5, 5)}
    agent_context = {"recent_moves": [], "composition_failures": []}
    
    # Compute EFE
    efe_scores = evaluator.compute_movement_efe(
        current_pos=(3, 3),
        candidate_directions=directions,
        field_state=field_state,
        field_gradient=field_gradient,
        target_info=target_info,
        agent_context=agent_context
    )
    
    print(f"EFE scores: {efe_scores}")
    print(f"Selected action: {evaluator.select_action(efe_scores)}")
    print("Active inference module working!")