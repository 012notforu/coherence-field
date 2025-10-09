#!/usr/bin/env python3
"""
EFE Controller for SCFD Agents

Drop-in Expected Free Energy controller that enhances existing agent behavior.
Integrates with current agent.step() and composition systems.

Usage:
    # Wrap any existing agent
    enhanced_agent = EFEController(existing_agent)
    result = enhanced_agent.step_with_efe(global_theta, global_theta_dot, logger)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

from utils.active_inference import ActiveInferenceEvaluator, EFEConfig


@dataclass
class EFEControlConfig:
    """Configuration for EFE-enhanced agent control."""
    
    # When to use EFE vs standard behavior
    efe_activation_threshold: float = 0.3    # Activate EFE when ambiguity > threshold
    composition_efe_weight: float = 0.4      # How much EFE influences composition decisions
    movement_efe_weight: float = 0.6         # How much EFE influences movement
    
    # Risk components for SCFD context
    energy_stability_weight: float = 0.3     # Penalty for large energy changes
    progress_weight: float = 0.4             # Reward for task progress  
    composition_risk_weight: float = 0.3     # Risk of composition failure
    
    # Ambiguity components for SCFD context
    field_entropy_weight: float = 0.4        # Local field uncertainty
    gradient_variance_weight: float = 0.3    # Dynamics uncertainty
    composition_uncertainty_weight: float = 0.3  # Composition prediction uncertainty
    
    # Behavior tuning
    exploration_bonus: float = 0.1           # Extra reward for exploring uncertain areas
    exploitation_threshold: float = 0.7     # When to focus on exploitation
    adaptive_temperature: bool = True       # Adjust decision temperature based on context


class EFEController:
    """Active Inference controller for SCFD agents."""
    
    def __init__(self, base_agent, config: EFEControlConfig = None):
        self.base_agent = base_agent
        self.config = config or EFEControlConfig()
        
        # EFE evaluation components
        efe_config = EFEConfig(
            risk_weight=0.6,
            ambiguity_weight=0.4,
            temperature=1.0
        )
        self.efe_evaluator = ActiveInferenceEvaluator(efe_config)
        
        # State tracking for EFE
        self.recent_efe_scores = []
        self.composition_failure_history = []
        self.local_ambiguity_estimate = 0.5
        self.exploration_mode = False
        
        # Performance tracking
        self.efe_decisions = 0
        self.standard_decisions = 0
        self.composition_improvements = 0
    
    def step_with_efe(self, global_theta: np.ndarray, global_theta_dot: np.ndarray, 
                     logger=None, neighbor_states: Dict = None) -> Any:
        """Enhanced step using EFE principles."""
        
        # Update local context
        self._update_local_context(global_theta, global_theta_dot, neighbor_states)
        
        # Decide whether to use EFE enhancement
        if self._should_use_efe():
            return self._efe_enhanced_step(global_theta, global_theta_dot, logger, neighbor_states)
        else:
            # Use standard agent behavior
            self.standard_decisions += 1
            return self._standard_step(global_theta, global_theta_dot, logger, neighbor_states)
    
    def _should_use_efe(self) -> bool:
        """Decide when to activate EFE enhancement."""
        
        # Activate EFE when:
        # 1. Local ambiguity is high (uncertain situation)
        # 2. Recent composition failures (need better decisions)
        # 3. Exploration mode is active
        
        recent_failures = len([f for f in self.composition_failure_history[-5:] if not f])
        
        activation_signals = [
            self.local_ambiguity_estimate > self.config.efe_activation_threshold,
            recent_failures >= 2,  # Multiple recent failures
            self.exploration_mode,
            hasattr(self.base_agent, 'active') and self.base_agent.active  # Agent is active
        ]
        
        return any(activation_signals)
    
    def _efe_enhanced_step(self, global_theta: np.ndarray, global_theta_dot: np.ndarray,
                          logger, neighbor_states: Dict) -> Any:
        """Execute step with EFE enhancement."""
        
        self.efe_decisions += 1
        
        # Check if this is a composition agent
        if hasattr(self.base_agent, 'step_with_composition'):
            return self._efe_composition_step(global_theta, global_theta_dot, logger, neighbor_states)
        else:
            return self._efe_standard_agent_step(global_theta, global_theta_dot, logger, neighbor_states)
    
    def _efe_composition_step(self, global_theta: np.ndarray, global_theta_dot: np.ndarray,
                             logger, neighbor_states: Dict) -> Any:
        """EFE-enhanced composition step."""
        
        from benchmarks.vector_composition import CompositionContext
        
        # Get local features from base agent
        if hasattr(self.base_agent.base_agent, 'sense_maze_environment'):
            local_features = self.base_agent.base_agent.sense_maze_environment(global_theta, neighbor_states or {})
        else:
            local_features = self.base_agent.base_agent.sense_local_environment(global_theta)
        
        # Enhance context with EFE information
        enhanced_context = CompositionContext(
            current_field_state=global_theta,
            local_features=local_features,
            agent_history=getattr(self.base_agent.base_agent, 'recent_actions', []),
            neighbor_states=neighbor_states or {},
            timestep=getattr(self.base_agent.base_agent, 'timestep', 0),
            physics_context=local_features
        )
        
        # Evaluate composition options with EFE
        composition_options = self._evaluate_composition_options(enhanced_context, global_theta, global_theta_dot)
        
        # Select best composition based on EFE
        if composition_options and self._should_attempt_composition(composition_options):
            
            # Execute selected composition
            result = self.base_agent.step_with_composition(global_theta, global_theta_dot, neighbor_states or {}, logger)
            
            # Track composition success for future EFE calculations
            success = getattr(result, 'accepted', False) if hasattr(result, 'accepted') else True
            self.composition_failure_history.append(success)
            
            if success and len(self.composition_failure_history) > 1:
                prev_success_rate = np.mean(self.composition_failure_history[-10:])
                if prev_success_rate > np.mean(self.composition_failure_history[-20:-10]):
                    self.composition_improvements += 1
            
            return result
        else:
            # Skip composition, use base agent
            return self.base_agent.base_agent.step(global_theta, global_theta_dot, logger)
    
    def _efe_standard_agent_step(self, global_theta: np.ndarray, global_theta_dot: np.ndarray,
                                logger, neighbor_states: Dict) -> Any:
        """EFE-enhanced step for standard agents."""
        
        # For standard agents, EFE mainly influences activation and vector selection
        
        # Check if agent should activate based on EFE
        if hasattr(self.base_agent, 'active') and not self.base_agent.active:
            if self._efe_should_activate(global_theta, global_theta_dot):
                self.base_agent.active = True
        
        # Execute normal step
        return self.base_agent.step(global_theta, global_theta_dot, logger)
    
    def _standard_step(self, global_theta: np.ndarray, global_theta_dot: np.ndarray,
                      logger, neighbor_states: Dict) -> Any:
        """Execute standard step without EFE enhancement."""
        
        if hasattr(self.base_agent, 'step_with_composition'):
            return self.base_agent.step_with_composition(global_theta, global_theta_dot, neighbor_states or {}, logger)
        else:
            return self.base_agent.step(global_theta, global_theta_dot, logger)
    
    def _update_local_context(self, global_theta: np.ndarray, global_theta_dot: np.ndarray,
                             neighbor_states: Dict):
        """Update local ambiguity and context estimates."""
        
        agent_pos = getattr(self.base_agent, 'pos', getattr(self.base_agent, 'cell_pos', (0, 0)))
        
        # Update local ambiguity estimate
        self.local_ambiguity_estimate = self._compute_local_ambiguity(agent_pos, global_theta, global_theta_dot)
        
        # Update exploration mode
        self.exploration_mode = self._should_explore()
        
        # Trim history
        if len(self.composition_failure_history) > 20:
            self.composition_failure_history = self.composition_failure_history[-20:]
    
    def _compute_local_ambiguity(self, pos: Tuple[int, int], global_theta: np.ndarray,
                                global_theta_dot: np.ndarray) -> float:
        """Compute local ambiguity/uncertainty."""
        
        ambiguity_components = []
        
        # 1. Field entropy around position
        entropy = self.efe_evaluator._compute_local_entropy(pos, global_theta)
        ambiguity_components.append(self.config.field_entropy_weight * entropy)
        
        # 2. Gradient variance (dynamics uncertainty)
        if global_theta_dot.ndim >= 2:
            grad_variance = self.efe_evaluator._compute_gradient_variance(pos, global_theta_dot)
            ambiguity_components.append(self.config.gradient_variance_weight * grad_variance)
        
        # 3. Composition uncertainty (based on recent failures)
        recent_comp_failures = len([f for f in self.composition_failure_history[-5:] if not f])
        comp_uncertainty = min(1.0, recent_comp_failures / 3.0)  # 0-1 scale
        ambiguity_components.append(self.config.composition_uncertainty_weight * comp_uncertainty)
        
        return np.mean(ambiguity_components) if ambiguity_components else 0.5
    
    def _should_explore(self) -> bool:
        """Determine if agent should be in exploration mode."""
        
        # Explore when:
        # 1. High local ambiguity
        # 2. Stuck (repeated failures)
        # 3. Low recent progress
        
        recent_failures = len([f for f in self.composition_failure_history[-5:] if not f])
        
        exploration_signals = [
            self.local_ambiguity_estimate > 0.6,
            recent_failures >= 3,
            len(self.composition_failure_history) > 10 and np.mean(self.composition_failure_history[-10:]) < 0.3
        ]
        
        return any(exploration_signals)
    
    def _evaluate_composition_options(self, context, global_theta: np.ndarray,
                                     global_theta_dot: np.ndarray) -> List[Dict]:
        """Evaluate composition options using EFE."""
        
        if not hasattr(self.base_agent, 'composer'):
            return []
        
        # Get available compositions
        available_rules = list(self.base_agent.composer.composition_rules.values())
        
        composition_evaluations = []
        
        for rule in available_rules:
            # Estimate EFE for this composition
            risk = self._estimate_composition_risk(rule, context, global_theta)
            ambiguity = self._estimate_composition_ambiguity(rule, context, global_theta)
            
            efe_score = (self.config.composition_efe_weight * risk +
                        (1 - self.config.composition_efe_weight) * ambiguity)
            
            composition_evaluations.append({
                "rule": rule,
                "efe_score": efe_score,
                "risk": risk,
                "ambiguity": ambiguity
            })
        
        # Sort by EFE score (lower is better)
        composition_evaluations.sort(key=lambda x: x["efe_score"])
        
        return composition_evaluations
    
    def _estimate_composition_risk(self, rule, context, global_theta: np.ndarray) -> float:
        """Estimate risk of composition failure."""
        
        risk_factors = []
        
        # 1. Historical success rate for this rule
        if hasattr(rule, 'name') and hasattr(self.base_agent.composer, 'get_composition_analytics'):
            analytics = self.base_agent.composer.get_composition_analytics()
            if rule.name in analytics:
                success_rate = analytics[rule.name].get('success_rate', 0.5)
                risk_factors.append(1.0 - success_rate)
        
        # 2. Field conditions risk
        agent_pos = getattr(self.base_agent, 'pos', getattr(self.base_agent, 'cell_pos', (0, 0)))
        if (0 <= agent_pos[0] < global_theta.shape[0] and 
            0 <= agent_pos[1] < global_theta.shape[1]):
            
            field_value = global_theta[agent_pos]
            # Risk for extreme values
            if field_value < 0.01 or field_value > 0.99:
                risk_factors.append(0.3)
        
        # 3. Recent failure pattern
        recent_failures = len([f for f in self.composition_failure_history[-3:] if not f])
        if recent_failures >= 2:
            risk_factors.append(0.4)
        
        return np.mean(risk_factors) if risk_factors else 0.5
    
    def _estimate_composition_ambiguity(self, rule, context, global_theta: np.ndarray) -> float:
        """Estimate ambiguity reduction potential of composition."""
        
        # Compositions that work in uncertain areas reduce ambiguity
        ambiguity_reduction = 0.0
        
        # 1. If we're in high-ambiguity area, composition might help
        if self.local_ambiguity_estimate > 0.5:
            ambiguity_reduction += 0.3
        
        # 2. Certain composition types are better for uncertainty
        if hasattr(rule, 'composition_type'):
            if rule.composition_type == "adaptive":
                ambiguity_reduction += 0.2
            elif rule.composition_type == "sequential":
                ambiguity_reduction += 0.1
        
        # Return remaining ambiguity (lower is better)
        return max(0.0, self.local_ambiguity_estimate - ambiguity_reduction)
    
    def _should_attempt_composition(self, composition_options: List[Dict]) -> bool:
        """Decide whether to attempt composition based on EFE."""
        
        if not composition_options:
            return False
        
        best_option = composition_options[0]
        
        # Attempt composition if:
        # 1. EFE score is reasonable
        # 2. We're in exploration mode and ambiguity is high
        # 3. Recent standard behavior isn't working
        
        attempt_conditions = [
            best_option["efe_score"] < 0.7,  # Reasonable EFE score
            self.exploration_mode and best_option["ambiguity"] < self.local_ambiguity_estimate,
            len(self.composition_failure_history) > 0 and np.mean(self.composition_failure_history[-5:]) < 0.5
        ]
        
        return any(attempt_conditions)
    
    def _efe_should_activate(self, global_theta: np.ndarray, global_theta_dot: np.ndarray) -> bool:
        """Determine if agent should activate based on EFE."""
        
        agent_pos = getattr(self.base_agent, 'pos', getattr(self.base_agent, 'cell_pos', (0, 0)))
        
        # Activate if high ambiguity or significant field changes
        if (0 <= agent_pos[0] < global_theta.shape[0] and 
            0 <= agent_pos[1] < global_theta.shape[1]):
            
            current_value = global_theta[agent_pos]
            field_change = abs(global_theta_dot[agent_pos]) if global_theta_dot.ndim >= 2 else 0.0
            
            # Lower activation threshold in high-ambiguity areas
            dynamic_threshold = self.base_agent.activation_threshold * (1.0 - self.local_ambiguity_estimate * 0.5)
            
            return field_change > dynamic_threshold or self.local_ambiguity_estimate > 0.7
        
        return False
    
    def get_efe_stats(self) -> Dict[str, Any]:
        """Get EFE controller performance statistics."""
        
        total_decisions = self.efe_decisions + self.standard_decisions
        
        return {
            "total_decisions": total_decisions,
            "efe_decisions": self.efe_decisions,
            "efe_usage_rate": self.efe_decisions / max(total_decisions, 1),
            "current_ambiguity": self.local_ambiguity_estimate,
            "exploration_mode": self.exploration_mode,
            "composition_improvements": self.composition_improvements,
            "recent_composition_success_rate": np.mean(self.composition_failure_history[-10:]) if len(self.composition_failure_history) >= 10 else None
        }


def wrap_agent_with_efe(agent, config: EFEControlConfig = None):
    """Wrap any existing agent with EFE controller."""
    return EFEController(agent, config)


def wrap_system_with_efe(system, config: EFEControlConfig = None):
    """Wrap all agents in a system with EFE controllers."""
    
    if hasattr(system, 'agents') and isinstance(system.agents, dict):
        # MultiAgentGridSystem style
        for pos, agent in system.agents.items():
            system.agents[pos] = wrap_agent_with_efe(agent, config)
    
    elif hasattr(system, 'completion_agents') and isinstance(system.completion_agents, list):
        # ImageCompletionSystem style
        for i, agent in enumerate(system.completion_agents):
            system.completion_agents[i] = wrap_agent_with_efe(agent, config)
    
    return system


# Quick integration test
if __name__ == "__main__":
    print("EFE Controller for SCFD - Integration Test")
    
    # Mock agent for testing
    class MockAgent:
        def __init__(self):
            self.pos = (5, 5)
            self.active = False
            self.activation_threshold = 0.1
            self.timestep = 0
            
        def step(self, global_theta, global_theta_dot, logger):
            return {"accepted": True, "energy_change": 0.01}
    
    # Test EFE controller
    mock_agent = MockAgent()
    efe_controller = EFEController(mock_agent)
    
    # Mock field data
    global_theta = np.random.random((10, 10))
    global_theta_dot = np.random.random((10, 10))
    
    # Test step
    result = efe_controller.step_with_efe(global_theta, global_theta_dot)
    print(f"Step result: {result}")
    
    # Test stats
    stats = efe_controller.get_efe_stats()
    print(f"EFE stats: {stats}")
    
    print("EFE Controller working!")