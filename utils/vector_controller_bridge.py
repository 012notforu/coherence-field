"""
Vector-Controller Bridge for Policy Vectors

This module creates a bridge between the sensing router/controller system and 
vector selection/composition, enabling the controller to influence vector behavior
through policy vectors rather than just parameter adjustments.

Key Features:
1. Policy vector creation from controller decisions
2. Vector weight adjustment based on sensing outcomes
3. Composition rule modification from real-time feedback
4. Integration with existing vector registry and composer
5. Performance-based vector ranking and selection bias
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum
import time
import json
import numpy as np
from pathlib import Path

from utils.parameter_registry import get_global_registry, ParameterRegistry, ParameterRole
from utils.sensing_router import SensingRouter, SensingDecision, ConstraintPriority
from utils.event_bus import get_event_bus, EventType
from orchestrator.pipeline import VectorEntry, load_vector_registry
from benchmarks.vector_composition import VectorComposer, CompositionRule, CompositionContext
from benchmarks.multi_agent_grid import VectorRegistry

class PolicyVectorType(Enum):
    """Types of policy vectors that can be created from controller decisions."""
    COMPOSITION_WEIGHTS = "composition_weights"  # Adjust vector composition weights
    VECTOR_SELECTION_BIAS = "vector_selection_bias"  # Bias vector selection process
    ACCEPTANCE_MODULATION = "acceptance_modulation"  # Modulate metropolis acceptance
    PERFORMANCE_SCALING = "performance_scaling"  # Scale based on performance metrics
    SAFETY_ENFORCEMENT = "safety_enforcement"  # Enforce safety constraints

class ControllerInfluence(Enum):
    """Types of influence the controller can have on vector behavior."""
    PROMOTE = "promote"  # Increase usage/weight of vectors
    DEMOTE = "demote"    # Decrease usage/weight of vectors  
    BLOCK = "block"      # Prevent usage under certain conditions
    ADAPT = "adapt"      # Adapt parameters based on performance
    EXPLORE = "explore"  # Encourage exploration of new vectors

@dataclass
class PolicyVector:
    """A policy vector derived from controller decisions."""
    policy_id: str
    policy_type: PolicyVectorType
    source_decision: str  # ID of sensing decision that created this
    target_vectors: List[str]  # Vector IDs this policy affects
    target_roles: List[ParameterRole]  # Parameter roles this affects
    influence_type: ControllerInfluence
    weight_adjustments: Dict[str, float]  # vector_id -> weight multiplier
    parameter_biases: Dict[str, float]  # param_id -> bias value
    activation_conditions: Dict[str, Any]  # Conditions when this policy applies
    confidence: float  # 0-1 confidence in this policy
    lifetime_seconds: float  # How long this policy remains active
    created_at: float
    performance_history: List[float] = field(default_factory=list)
    
    def is_active(self) -> bool:
        """Check if this policy is still within its lifetime."""
        return time.time() - self.created_at < self.lifetime_seconds
    
    def applies_to_vector(self, vector_id: str) -> bool:
        """Check if this policy applies to a given vector."""
        return vector_id in self.target_vectors or not self.target_vectors
    
    def applies_to_context(self, context: Dict[str, Any]) -> bool:
        """Check if this policy applies to the current context."""
        if not self.activation_conditions:
            return True
        
        for condition_key, condition_value in self.activation_conditions.items():
            context_value = context.get(condition_key)
            if context_value is None:
                continue
                
            if isinstance(condition_value, str) and condition_value.startswith(">"):
                threshold = float(condition_value[1:])
                if context_value <= threshold:
                    return False
            elif isinstance(condition_value, str) and condition_value.startswith("<"):
                threshold = float(condition_value[1:])
                if context_value >= threshold:
                    return False
            elif context_value != condition_value:
                return False
        
        return True

@dataclass
class VectorPerformanceTracker:
    """Tracks vector performance for policy decisions."""
    vector_id: str
    physics_type: str
    success_rate: float = 0.0
    avg_energy_delta: float = 0.0
    recent_uses: List[float] = field(default_factory=list)  # timestamps
    recent_outcomes: List[bool] = field(default_factory=list)  # success/failure
    composition_compatibility: Dict[str, float] = field(default_factory=dict)  # other_vector -> compatibility
    constraint_violations: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def update_performance(self, success: bool, energy_delta: float, timestamp: float = None):
        """Update performance metrics with new outcome."""
        if timestamp is None:
            timestamp = time.time()
            
        self.recent_uses.append(timestamp)
        self.recent_outcomes.append(success)
        
        # Keep only recent history (last 100 uses or 1 hour)
        cutoff_time = timestamp - 3600  # 1 hour
        while (self.recent_uses and 
               (len(self.recent_uses) > 100 or self.recent_uses[0] < cutoff_time)):
            self.recent_uses.pop(0)
            self.recent_outcomes.pop(0)
        
        # Update success rate
        if self.recent_outcomes:
            self.success_rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
        
        # Update average energy delta with EMA
        alpha = 0.1
        if self.avg_energy_delta == 0.0:
            self.avg_energy_delta = energy_delta
        else:
            self.avg_energy_delta = alpha * energy_delta + (1 - alpha) * self.avg_energy_delta
        
        self.last_updated = timestamp

class VectorControllerBridge:
    """
    Bridge between sensing/controller system and vector selection/composition.
    
    Enables the controller to influence vector behavior through policy vectors,
    creating a feedback loop between real-time sensing and vector decisions.
    """
    
    def __init__(self, 
                 sensing_router: Optional[SensingRouter] = None,
                 vector_composer: Optional[VectorComposer] = None,
                 parameter_registry: Optional[ParameterRegistry] = None):
        
        self.sensing_router = sensing_router
        self.vector_composer = vector_composer
        self.registry = parameter_registry or get_global_registry()
        self.event_bus = get_event_bus()
        
        # Policy management
        self.active_policies: Dict[str, PolicyVector] = {}
        self.policy_history: List[PolicyVector] = []
        
        # Performance tracking
        self.vector_performance: Dict[str, VectorPerformanceTracker] = {}
        self.composition_performance: Dict[str, float] = {}  # rule_name -> success_rate
        
        # Integration state
        self.vector_registry: Optional[VectorRegistry] = None
        self.last_decision_time: float = 0.0
        self.policy_cleanup_interval: float = 60.0  # Clean up expired policies every minute
        self.last_cleanup: float = time.time()
        
        # Configuration
        self.max_active_policies = 50
        self.default_policy_lifetime = 300.0  # 5 minutes
        self.performance_window_size = 100
        
        # Subscribe to events
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """Set up event bus subscriptions for integration."""
        self.event_bus.subscribe(
            EventType.COMPOSITION_RESULT,
            self._on_composition_result,
            "VectorControllerBridge"
        )
        
        self.event_bus.subscribe(
            EventType.PARAMETER_UPDATED,
            self._on_parameter_update,
            "VectorControllerBridge"
        )
    
    def integrate_with_vector_system(self, vector_registry: VectorRegistry, vector_composer: VectorComposer):
        """Integrate with existing vector registry and composer."""
        self.vector_registry = vector_registry
        self.vector_composer = vector_composer
        
        # Initialize performance tracking for all vectors
        for vector_entry in vector_registry.base_registry:
            if vector_entry.vector_id not in self.vector_performance:
                self.vector_performance[vector_entry.vector_id] = VectorPerformanceTracker(
                    vector_id=vector_entry.vector_id,
                    physics_type=vector_entry.physics
                )
    
    def process_sensing_decision(self, decision: SensingDecision) -> List[PolicyVector]:
        """
        Process a sensing decision and create policy vectors as needed.
        
        This is the main entry point for controller influence on vector behavior.
        """
        policies_created = []
        
        # Clean up expired policies first
        self._cleanup_expired_policies()
        
        # Skip if no significant decisions were made
        if not decision.triggered_rules and not decision.constraint_violations:
            return policies_created
        
        # Create policy vectors based on decision type
        for rule_id in decision.triggered_rules:
            policy = self._create_policy_from_rule(rule_id, decision)
            if policy:
                policies_created.append(policy)
                self.active_policies[policy.policy_id] = policy
        
        # Handle constraint violations with safety policies
        if decision.constraint_violations:
            safety_policy = self._create_safety_policy(decision)
            if safety_policy:
                policies_created.append(safety_policy)
                self.active_policies[safety_policy.policy_id] = safety_policy
        
        # Performance-based policies
        if time.time() - self.last_decision_time > 30.0:  # Every 30 seconds
            perf_policies = self._create_performance_policies(decision)
            policies_created.extend(perf_policies)
            for policy in perf_policies:
                self.active_policies[policy.policy_id] = policy
        
        self.last_decision_time = time.time()
        
        # Log policy creation
        if policies_created:
            self._log_policy_creation(policies_created, decision)
        
        return policies_created
    
    def _create_policy_from_rule(self, rule_id: str, decision: SensingDecision) -> Optional[PolicyVector]:
        """Create a policy vector from a triggered sensing rule."""
        
        # Map common rule patterns to policy types
        policy_type = PolicyVectorType.VECTOR_SELECTION_BIAS
        influence_type = ControllerInfluence.ADAPT
        target_vectors = []
        weight_adjustments = {}
        activation_conditions = {}
        lifetime = self.default_policy_lifetime
        
        # Analyze rule ID to determine policy characteristics
        if "composition" in rule_id.lower():
            policy_type = PolicyVectorType.COMPOSITION_WEIGHTS
            # Adjust composition weights based on success rate
            success_rate = decision.signal_states.get("composition_success_rate", {}).get("norm", 0.5)
            if success_rate < 0.3:
                influence_type = ControllerInfluence.EXPLORE
                weight_adjustments = {"exploration_boost": 1.5}
                activation_conditions = {"composition_success_rate": "<0.4"}
            else:
                influence_type = ControllerInfluence.PROMOTE
                weight_adjustments = {"successful_boost": 1.2}
        
        elif "energy" in rule_id.lower():
            policy_type = PolicyVectorType.SAFETY_ENFORCEMENT
            influence_type = ControllerInfluence.BLOCK
            # Block risky vectors when energy is unstable
            energy_drift = decision.signal_states.get("energy_drift", {}).get("raw", 0.0)
            if abs(energy_drift) > 0.05:
                activation_conditions = {"energy_drift": f">{abs(energy_drift)}"}
                weight_adjustments = {"high_energy_penalty": 0.5}
                lifetime = 60.0  # Short-lived safety policy
        
        elif "cartpole" in rule_id.lower() or "cart_position" in rule_id.lower():
            policy_type = PolicyVectorType.PERFORMANCE_SCALING
            influence_type = ControllerInfluence.ADAPT
            # Get CartPole vectors
            if self.vector_registry:
                target_vectors = [v.vector_id for v in self.vector_registry.base_registry 
                                if "cartpole" in v.physics.lower()]
            
            cart_pos = decision.signal_states.get("cart_position", {}).get("norm", 0.5)
            if cart_pos > 0.8:  # Cart getting close to edge
                influence_type = ControllerInfluence.PROMOTE
                weight_adjustments = {"stabilization_boost": 1.8}
                activation_conditions = {"cart_position": ">0.7"}
        
        elif "noise" in rule_id.lower():
            policy_type = PolicyVectorType.ACCEPTANCE_MODULATION
            influence_type = ControllerInfluence.ADAPT
            # Adjust acceptance parameters for noisy conditions
            noise_level = decision.signal_states.get("sensor_noise_level", {}).get("norm", 0.5)
            if noise_level > 0.6:
                weight_adjustments = {"noise_tolerance": 1.3}
                activation_conditions = {"sensor_noise_level": ">0.5"}
        
        # Create policy
        policy_id = f"policy_{rule_id}_{int(time.time())}"
        confidence = decision.confidence_scores.get(rule_id, 0.5)
        
        return PolicyVector(
            policy_id=policy_id,
            policy_type=policy_type,
            source_decision=rule_id,
            target_vectors=target_vectors,
            target_roles=self._extract_target_roles(decision),
            influence_type=influence_type,
            weight_adjustments=weight_adjustments,
            parameter_biases={},
            activation_conditions=activation_conditions,
            confidence=confidence,
            lifetime_seconds=lifetime,
            created_at=time.time()
        )
    
    def _create_safety_policy(self, decision: SensingDecision) -> Optional[PolicyVector]:
        """Create a safety policy to handle constraint violations."""
        if not decision.constraint_violations:
            return None
        
        policy_id = f"safety_policy_{int(time.time())}"
        
        # Block risky operations when constraints are violated
        weight_adjustments = {"safety_penalty": 0.3}
        activation_conditions = {}
        
        # Set conditions based on specific violations
        for violation in decision.constraint_violations:
            if "energy" in violation.lower():
                activation_conditions["energy_stable"] = "false"
            elif "bounds" in violation.lower():
                activation_conditions["bounds_safe"] = "false"
        
        return PolicyVector(
            policy_id=policy_id,
            policy_type=PolicyVectorType.SAFETY_ENFORCEMENT,
            source_decision="constraint_violation",
            target_vectors=[],  # Apply to all vectors
            target_roles=[ParameterRole.ACCEPTANCE, ParameterRole.STABILITY],
            influence_type=ControllerInfluence.BLOCK,
            weight_adjustments=weight_adjustments,
            parameter_biases={},
            activation_conditions=activation_conditions,
            confidence=0.9,  # High confidence in safety
            lifetime_seconds=120.0,  # 2 minute safety block
            created_at=time.time()
        )
    
    def _create_performance_policies(self, decision: SensingDecision) -> List[PolicyVector]:
        """Create policies based on vector performance analysis."""
        policies = []
        
        if not self.vector_performance:
            return policies
        
        # Find best and worst performing vectors
        performance_scores = {
            vid: tracker.success_rate 
            for vid, tracker in self.vector_performance.items()
            if tracker.recent_outcomes
        }
        
        if len(performance_scores) < 2:
            return policies
        
        sorted_vectors = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        best_vectors = [vid for vid, score in sorted_vectors[:3] if score > 0.7]
        worst_vectors = [vid for vid, score in sorted_vectors[-3:] if score < 0.3]
        
        # Promote best performing vectors
        if best_vectors:
            policy_id = f"performance_boost_{int(time.time())}"
            policies.append(PolicyVector(
                policy_id=policy_id,
                policy_type=PolicyVectorType.VECTOR_SELECTION_BIAS,
                source_decision="performance_analysis",
                target_vectors=best_vectors,
                target_roles=[ParameterRole.COMPOSITION],
                influence_type=ControllerInfluence.PROMOTE,
                weight_adjustments={"performance_boost": 1.4},
                parameter_biases={},
                activation_conditions={},
                confidence=0.8,
                lifetime_seconds=600.0,  # 10 minute boost
                created_at=time.time()
            ))
        
        # Demote poor performing vectors
        if worst_vectors:
            policy_id = f"performance_penalty_{int(time.time())}"
            policies.append(PolicyVector(
                policy_id=policy_id,
                policy_type=PolicyVectorType.VECTOR_SELECTION_BIAS,
                source_decision="performance_analysis",
                target_vectors=worst_vectors,
                target_roles=[ParameterRole.COMPOSITION],
                influence_type=ControllerInfluence.DEMOTE,
                weight_adjustments={"performance_penalty": 0.6},
                parameter_biases={},
                activation_conditions={},
                confidence=0.7,
                lifetime_seconds=300.0,  # 5 minute penalty
                created_at=time.time()
            ))
        
        return policies
    
    def apply_policies_to_composition(self, composition_context: CompositionContext) -> Dict[str, float]:
        """
        Apply active policies to modify vector composition.
        
        Returns weight adjustments to apply to composition rules.
        """
        weight_adjustments = {}
        
        # Build context for policy evaluation
        context = {
            "timestep": composition_context.timestep,
            "physics_context": composition_context.physics_context,
        }
        
        # Add signal states if available
        if hasattr(composition_context, 'signal_states'):
            context.update(composition_context.signal_states)
        
        for policy in self.active_policies.values():
            if not policy.is_active() or not policy.applies_to_context(context):
                continue
            
            # Apply policy based on its type and influence
            if policy.policy_type == PolicyVectorType.COMPOSITION_WEIGHTS:
                self._apply_composition_weight_policy(policy, weight_adjustments)
            
            elif policy.policy_type == PolicyVectorType.VECTOR_SELECTION_BIAS:
                self._apply_selection_bias_policy(policy, weight_adjustments)
            
            elif policy.policy_type == PolicyVectorType.PERFORMANCE_SCALING:
                self._apply_performance_scaling_policy(policy, weight_adjustments)
            
            elif policy.policy_type == PolicyVectorType.SAFETY_ENFORCEMENT:
                self._apply_safety_enforcement_policy(policy, weight_adjustments)
        
        return weight_adjustments
    
    def _apply_composition_weight_policy(self, policy: PolicyVector, adjustments: Dict[str, float]):
        """Apply composition weight policy to weight adjustments."""
        for vector_id in policy.target_vectors:
            base_adjustment = policy.weight_adjustments.get("exploration_boost", 
                                                          policy.weight_adjustments.get("successful_boost", 1.0))
            confidence_multiplier = policy.confidence
            final_adjustment = 1.0 + (base_adjustment - 1.0) * confidence_multiplier
            
            if vector_id in adjustments:
                adjustments[vector_id] *= final_adjustment
            else:
                adjustments[vector_id] = final_adjustment
    
    def _apply_selection_bias_policy(self, policy: PolicyVector, adjustments: Dict[str, float]):
        """Apply selection bias policy to weight adjustments."""
        base_adjustment = 1.0
        
        if policy.influence_type == ControllerInfluence.PROMOTE:
            base_adjustment = policy.weight_adjustments.get("performance_boost", 1.2)
        elif policy.influence_type == ControllerInfluence.DEMOTE:
            base_adjustment = policy.weight_adjustments.get("performance_penalty", 0.8)
        
        for vector_id in policy.target_vectors:
            final_adjustment = 1.0 + (base_adjustment - 1.0) * policy.confidence
            adjustments[vector_id] = adjustments.get(vector_id, 1.0) * final_adjustment
    
    def _apply_performance_scaling_policy(self, policy: PolicyVector, adjustments: Dict[str, float]):
        """Apply performance scaling policy based on recent performance."""
        for vector_id in policy.target_vectors:
            if vector_id in self.vector_performance:
                performance = self.vector_performance[vector_id]
                # Scale adjustment based on success rate
                base_adjustment = policy.weight_adjustments.get("stabilization_boost", 1.0)
                performance_factor = min(performance.success_rate * 2.0, 1.0)  # Scale by success rate
                final_adjustment = 1.0 + (base_adjustment - 1.0) * performance_factor * policy.confidence
                adjustments[vector_id] = adjustments.get(vector_id, 1.0) * final_adjustment
    
    def _apply_safety_enforcement_policy(self, policy: PolicyVector, adjustments: Dict[str, float]):
        """Apply safety enforcement policy to prevent risky operations."""
        penalty = policy.weight_adjustments.get("safety_penalty", 0.5)
        
        if policy.target_vectors:
            # Apply to specific vectors
            for vector_id in policy.target_vectors:
                final_penalty = 1.0 - (1.0 - penalty) * policy.confidence
                adjustments[vector_id] = adjustments.get(vector_id, 1.0) * final_penalty
        else:
            # Apply to all vectors (global safety policy)
            global_penalty = 1.0 - (1.0 - penalty) * policy.confidence * 0.5  # Reduced global impact
            for vector_id in self.vector_performance.keys():
                adjustments[vector_id] = adjustments.get(vector_id, 1.0) * global_penalty
    
    def update_vector_performance(self, vector_id: str, success: bool, energy_delta: float, 
                                composition_rule: Optional[str] = None):
        """Update performance tracking for a vector."""
        if vector_id not in self.vector_performance:
            # Create new tracker
            physics_type = "unknown"
            if self.vector_registry:
                for entry in self.vector_registry.base_registry:
                    if entry.vector_id == vector_id:
                        physics_type = entry.physics
                        break
            
            self.vector_performance[vector_id] = VectorPerformanceTracker(
                vector_id=vector_id,
                physics_type=physics_type
            )
        
        self.vector_performance[vector_id].update_performance(success, energy_delta)
        
        # Update composition rule performance if provided
        if composition_rule:
            if composition_rule not in self.composition_performance:
                self.composition_performance[composition_rule] = 0.5
            
            # EMA update
            alpha = 0.1
            current_success = 1.0 if success else 0.0
            self.composition_performance[composition_rule] = (
                alpha * current_success + (1 - alpha) * self.composition_performance[composition_rule]
            )
    
    def _extract_target_roles(self, decision: SensingDecision) -> List[ParameterRole]:
        """Extract target parameter roles from a sensing decision."""
        roles = []
        
        # Map triggered rules to relevant roles
        for rule_id in decision.triggered_rules:
            if "composition" in rule_id.lower():
                roles.append(ParameterRole.COMPOSITION)
            elif "energy" in rule_id.lower():
                roles.extend([ParameterRole.STABILITY, ParameterRole.ACCEPTANCE])
            elif "control" in rule_id.lower():
                roles.append(ParameterRole.CONTROL_GAIN)
            elif "filter" in rule_id.lower():
                roles.append(ParameterRole.FILTERING)
        
        # Default to composition if no specific mapping
        if not roles:
            roles = [ParameterRole.COMPOSITION]
        
        return list(set(roles))  # Remove duplicates
    
    def _cleanup_expired_policies(self):
        """Remove expired policies and maintain policy limits."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.policy_cleanup_interval:
            return
        
        # Remove expired policies
        expired_policies = [
            policy_id for policy_id, policy in self.active_policies.items()
            if not policy.is_active()
        ]
        
        for policy_id in expired_policies:
            expired_policy = self.active_policies.pop(policy_id)
            self.policy_history.append(expired_policy)
        
        # Limit active policies
        if len(self.active_policies) > self.max_active_policies:
            # Remove oldest policies that aren't safety-critical
            non_safety = [
                (policy_id, policy) for policy_id, policy in self.active_policies.items()
                if policy.policy_type != PolicyVectorType.SAFETY_ENFORCEMENT
            ]
            non_safety.sort(key=lambda x: x[1].created_at)
            
            policies_to_remove = len(self.active_policies) - self.max_active_policies
            for policy_id, policy in non_safety[:policies_to_remove]:
                removed_policy = self.active_policies.pop(policy_id)
                self.policy_history.append(removed_policy)
        
        # Limit history size
        if len(self.policy_history) > 1000:
            self.policy_history = self.policy_history[-500:]
        
        self.last_cleanup = current_time
    
    def _log_policy_creation(self, policies: List[PolicyVector], decision: SensingDecision):
        """Log policy creation for debugging and analysis."""
        for policy in policies:
            print(f"Bridge: Created {policy.policy_type.value} policy {policy.policy_id}")
            print(f"  Source: {policy.source_decision}, Influence: {policy.influence_type.value}")
            print(f"  Targets: {len(policy.target_vectors)} vectors, Confidence: {policy.confidence:.2f}")
            if policy.weight_adjustments:
                print(f"  Adjustments: {policy.weight_adjustments}")
    
    def _on_composition_result(self, event):
        """Handle composition result events."""
        data = event.data
        vector_id = data.get("vector_id")
        success = data.get("success", False)
        energy_delta = data.get("energy_delta", 0.0)
        composition_rule = data.get("composition_rule")
        
        if vector_id:
            self.update_vector_performance(vector_id, success, energy_delta, composition_rule)
    
    def _on_parameter_update(self, event):
        """Handle parameter update events that might affect policies."""
        param_name = event.data.get("parameter_name", "")
        if "metropolis" in param_name or "acceptance" in param_name:
            # Clear acceptance-related policies when parameters change
            policies_to_remove = [
                policy_id for policy_id, policy in self.active_policies.items()
                if policy.policy_type == PolicyVectorType.ACCEPTANCE_MODULATION
            ]
            for policy_id in policies_to_remove:
                self.active_policies.pop(policy_id)
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current status of the vector-controller bridge."""
        active_by_type = {}
        for policy in self.active_policies.values():
            policy_type = policy.policy_type.value
            active_by_type[policy_type] = active_by_type.get(policy_type, 0) + 1
        
        performance_summary = {
            vector_id: {
                "success_rate": tracker.success_rate,
                "recent_uses": len(tracker.recent_uses),
                "avg_energy_delta": tracker.avg_energy_delta
            }
            for vector_id, tracker in self.vector_performance.items()
            if tracker.recent_uses
        }
        
        return {
            "active_policies": len(self.active_policies),
            "policies_by_type": active_by_type,
            "tracked_vectors": len(self.vector_performance),
            "performance_summary": performance_summary,
            "composition_performance": self.composition_performance,
            "last_decision_time": self.last_decision_time,
            "integration_active": self.vector_registry is not None
        }

# Integration helper functions

def create_bridge_for_system(sensing_router: SensingRouter, 
                           vector_registry: VectorRegistry,
                           vector_composer: VectorComposer) -> VectorControllerBridge:
    """
    Create a fully integrated vector-controller bridge for an existing system.
    
    This helper function sets up the bridge with all necessary integrations.
    """
    bridge = VectorControllerBridge(sensing_router=sensing_router)
    bridge.integrate_with_vector_system(vector_registry, vector_composer)
    
    return bridge

def apply_bridge_to_composition_context(bridge: VectorControllerBridge,
                                      context: CompositionContext) -> CompositionContext:
    """
    Apply bridge policies to a composition context.
    
    This modifies the context to include policy-driven adjustments.
    """
    if not bridge.active_policies:
        return context
    
    # Get weight adjustments from bridge
    weight_adjustments = bridge.apply_policies_to_composition(context)
    
    if weight_adjustments:
        # Add weight adjustments to context for composer to use
        if not hasattr(context, 'policy_weight_adjustments'):
            context.policy_weight_adjustments = {}
        context.policy_weight_adjustments.update(weight_adjustments)
    
    return context