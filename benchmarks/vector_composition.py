"""
Vector Composition Toolkit for Multi-Agent Systems

This module provides a framework for dynamically composing multiple vectors
to create sophisticated emergent behaviors beyond single vector selection.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path

from benchmarks.multi_agent_grid import VectorRegistry, VectorTag
from utils.logging import RunLogger, VectorInvocation, OutcomeMetrics
from utils.parameter_registry import get_global_registry
from utils.event_bus import get_event_bus, EventType

Array = np.ndarray

# Global SCFD configuration
SCFD_BOUNDARY_MODE = "reflecting"  # Single source of truth for boundary conditions
USE_SCFD_ACCEPTANCE = False  # Phase 1: Controller-only mode (sensing enabled, acceptance disabled)


@dataclass
class CompositionRule:
    """Defines how multiple vectors should be combined."""
    name: str
    vector_ids: List[str]
    weights: List[float]
    composition_type: str  # "linear", "conditional", "sequential", "adaptive"
    activation_condition: Optional[Callable] = None
    priority: float = 1.0
    success_history: List[bool] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.weights) != len(self.vector_ids):
            # Auto-generate equal weights if not provided
            self.weights = [1.0 / len(self.vector_ids)] * len(self.vector_ids)


@dataclass
class CompositionContext:
    """Context information for vector composition decisions."""
    current_field_state: Array
    local_features: Dict[str, float]
    agent_history: List[Dict]
    neighbor_states: Dict[Tuple[int, int], Dict]
    timestep: int
    physics_context: Dict[str, float]


class VectorComposer:
    """Core composition engine for combining multiple vectors."""
    
    def __init__(self, vector_registry: VectorRegistry, parameter_registry=None, controller_bridge=None):
        import threading
        import time
        
        self.registry = vector_registry
        self.composition_rules = {}
        self.adaptive_weights = {}
        self.performance_tracker = {}
        
        # Parameter registry integration
        self.parameter_registry = parameter_registry or get_global_registry()
        self.event_bus = get_event_bus()
        
        # Vector-controller bridge integration
        self.controller_bridge = controller_bridge
        
        # SCFD acceptance is now the single source of truth
        # All acceptance decisions use scfd_accept_update() with registry parameters
        
        # Phase 1: SCFD telemetry tracking for baseline validation
        self.scfd_telemetry = []
        
    def get_scfd_parameters_snapshot(self) -> Dict[str, float]:
        """
        Get snapshot of SCFD parameters with gamma stability guard.
        Called once per step to avoid mid-step async swaps.
        """
        try:
            alpha = self.parameter_registry.get_parameter("scfd.lagrangian.alpha").current_value
            beta = self.parameter_registry.get_parameter("scfd.lagrangian.beta").current_value  
            gamma = self.parameter_registry.get_parameter("scfd.lagrangian.gamma").current_value
            epsilon = self.parameter_registry.get_parameter("scfd.lagrangian.epsilon").current_value
            
            # Apply gamma stability guard: |γ| ≤ √(α·β)
            gamma_bound = np.sqrt(alpha * beta)
            gamma_clamped = np.clip(gamma, -gamma_bound, gamma_bound)
            
            if abs(gamma_clamped - gamma) > 1e-12:
                print(f"SCFD: Clamped gamma {gamma:.6f} → {gamma_clamped:.6f} (bound: ±{gamma_bound:.6f})")
            
            return {
                "alpha": alpha,
                "beta": beta, 
                "gamma": gamma_clamped,
                "epsilon": epsilon
            }
            
        except (AttributeError, TypeError):
            # Fallback to defaults if registry not available
            return {"alpha": 0.3, "beta": 0.2, "gamma": 0.1, "epsilon": 0.1}
    
    def record_scfd_telemetry(self, global_theta: Array, timestep: int) -> None:
        """
        Record SCFD telemetry for Phase 1 baseline validation.
        Tracks parameter evolution and sensing features without acceptance.
        """
        try:
            # Get current SCFD parameters
            scfd_params = self.get_scfd_parameters_snapshot()
            
            # Extract SCFD features for sensing
            scfd_features = self.extract_scfd_features_for_sensing(global_theta)
            
            # Record telemetry snapshot
            telemetry_snapshot = {
                "timestep": timestep,
                "scfd_mode": "sensing_only" if not USE_SCFD_ACCEPTANCE else "full_acceptance",
                "parameters": scfd_params,
                "features": scfd_features,
                "boundary_mode": SCFD_BOUNDARY_MODE
            }
            
            self.scfd_telemetry.append(telemetry_snapshot)
            
            # Keep last 1000 entries to prevent memory growth
            if len(self.scfd_telemetry) > 1000:
                self.scfd_telemetry.pop(0)
                
        except Exception as e:
            print(f"SCFD telemetry recording failed: {e}")
    
    def get_scfd_telemetry_summary(self) -> Dict[str, Any]:
        """Get summary of SCFD telemetry for Phase 1 validation."""
        if not self.scfd_telemetry:
            return {"status": "no_data"}
        
        recent_snapshots = self.scfd_telemetry[-10:]  # Last 10 entries
        
        # Extract parameter trends
        param_trends = {}
        for param in ["alpha", "beta", "gamma", "epsilon"]:
            values = [s["parameters"][param] for s in recent_snapshots]
            param_trends[param] = {
                "current": values[-1] if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "std": np.std(values) if len(values) > 1 else 0.0
            }
        
        # Extract feature trends
        feature_trends = {}
        for feature in ["mean_coherence", "mean_abs_curvature", "mean_cross_gradient"]:
            if feature in recent_snapshots[0]["features"]:
                values = [s["features"][feature] for s in recent_snapshots]
                feature_trends[feature] = {
                    "current": values[-1] if values else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                    "std": np.std(values) if len(values) > 1 else 0.0
                }
        
        return {
            "status": "active",
            "total_snapshots": len(self.scfd_telemetry),
            "mode": recent_snapshots[-1]["scfd_mode"],
            "boundary_mode": recent_snapshots[-1]["boundary_mode"],
            "parameter_trends": param_trends,
            "feature_trends": feature_trends
        }
        
        # Subscribe to parameter updates
        self.event_bus.subscribe(
            EventType.PARAMETER_UPDATED,
            self._on_parameter_update,
            "VectorComposer"
        )
        
        # Also subscribe to registry callbacks for redundancy
        if hasattr(self.parameter_registry, '_event_callbacks'):
            self.parameter_registry._event_callbacks.append(self._on_registry_callback)
        
        # Initialize with basic composition patterns
        self._initialize_default_compositions()
    
    def _initialize_metropolis_cache(self):
        """Initialize cache from registry values before subscribing to events."""
        import time
        
        # Explicit parameter IDs that match registry
        param_ids = {
            'temperature': 'scfd.accept.metropolis_temperature',
            'bounds_min': 'scfd.accept.bounds_min', 
            'bounds_max': 'scfd.accept.bounds_max'
        }
        
        missing_params = []
        
        with self._cache_lock:
            for param_name, param_id in param_ids.items():
                try:
                    # Get most specific scope available (defaults to GLOBAL)
                    param_entry = self.parameter_registry.get_parameter(param_id)
                    if param_entry is not None:
                        self._metropolis_cache[param_name] = param_entry.current_value
                        self._metropolis_cache['source_param_ids'][param_name] = param_id
                    else:
                        missing_params.append(param_id)
                except Exception as e:
                    missing_params.append(f"{param_id} (error: {e})")
            
            # Enforce invariants after initialization
            self._enforce_cache_invariants()
            self._metropolis_cache['last_update'] = time.time()
        
        if missing_params:
            print(f"WARN: VectorComposer missing metropolis parameters: {missing_params}, using fallbacks")
    
    def _enforce_cache_invariants(self):
        """Enforce metropolis parameter invariants locally."""
        # Ensure 0 < bounds_min < bounds_max < 1
        bounds_min = self._metropolis_cache['bounds_min']
        bounds_max = self._metropolis_cache['bounds_max']
        
        if bounds_min <= 0:
            self._metropolis_cache['bounds_min'] = 0.01
            print(f"WARN: Clamped bounds_min from {bounds_min} to 0.01")
            
        if bounds_max >= 1:
            self._metropolis_cache['bounds_max'] = 0.99
            print(f"WARN: Clamped bounds_max from {bounds_max} to 0.99")
            
        if self._metropolis_cache['bounds_min'] >= self._metropolis_cache['bounds_max']:
            self._metropolis_cache['bounds_min'] = 0.01
            self._metropolis_cache['bounds_max'] = 0.99
            print(f"ERROR: Bounds crossed (min >= max), auto-repaired to (0.01, 0.99)")
        
        # Ensure temperature > 0
        if self._metropolis_cache['temperature'] <= 0:
            old_temp = self._metropolis_cache['temperature']
            self._metropolis_cache['temperature'] = 1e-3
            print(f"WARN: Temperature was {old_temp}, nudged to 1e-3")
    
    def _on_parameter_update(self, event):
        """Handle parameter update events from event bus."""
        import time
        
        # Filter to only scfd.accept parameters
        param_name = event.data.get('parameter_name', '')
        if not param_name.startswith('scfd.accept.'):
            return
        
        # Map parameter ID to cache key
        id_to_key = {
            'scfd.accept.metropolis_temperature': 'temperature',
            'scfd.accept.bounds_min': 'bounds_min',
            'scfd.accept.bounds_max': 'bounds_max'
        }
        
        cache_key = id_to_key.get(param_name)
        if not cache_key:
            return
        
        # Thread-safe cache update
        with self._cache_lock:
            old_value = self._metropolis_cache[cache_key]
            new_value = event.data.get('new_value')
            
            if new_value is not None:
                self._metropolis_cache[cache_key] = new_value
                self._metropolis_cache['source_param_ids'][cache_key] = param_name
                self._metropolis_cache['last_update'] = time.time()
                
                # Enforce invariants after update
                self._enforce_cache_invariants()
                
                print(f"VectorComposer: Updated {cache_key} {old_value} -> {new_value}")
    
    def _on_registry_callback(self, param_id, old_value, new_value, scope):
        """Handle direct registry callback for redundancy."""
        if param_id.startswith('scfd.accept.'):
            # Confirm committed value from registry to handle trust region rejections
            confirmed_value = self.parameter_registry.get_parameter(param_id, scope=scope)
            if confirmed_value is not None:
                # Create synthetic event for consistent handling
                synthetic_event = type('Event', (), {
                    'data': {
                        'parameter_name': param_id,
                        'old_value': old_value,
                        'new_value': confirmed_value
                    }
                })()
                self._on_parameter_update(synthetic_event)
    
    def get_metropolis_temperature(self, context: Optional[CompositionContext] = None) -> Tuple[float, Dict[str, Any]]:
        """Get current metropolis temperature with provenance."""
        import time
        
        with self._cache_lock:
            # Check TTL and refresh if needed
            if time.time() - self._metropolis_cache['last_update'] > self._metropolis_cache['ttl']:
                self._refresh_cache_if_stale()
            
            temp = self._metropolis_cache['temperature']
            provenance = {
                'scope': self._metropolis_cache['scope'],
                'param_id': self._metropolis_cache['source_param_ids'].get('temperature', 'fallback'),
                'cache_age': time.time() - self._metropolis_cache['last_update'],
                'last_update_ts': self._metropolis_cache['last_update']
            }
            
            return temp, provenance
    
    def get_acceptance_bounds(self, context: Optional[CompositionContext] = None) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        """Get current acceptance bounds with provenance.""" 
        import time
        
        with self._cache_lock:
            # Check TTL and refresh if needed
            if time.time() - self._metropolis_cache['last_update'] > self._metropolis_cache['ttl']:
                self._refresh_cache_if_stale()
            
            bounds = (self._metropolis_cache['bounds_min'], self._metropolis_cache['bounds_max'])
            provenance = {
                'scope': self._metropolis_cache['scope'],
                'param_ids': {
                    'bounds_min': self._metropolis_cache['source_param_ids'].get('bounds_min', 'fallback'),
                    'bounds_max': self._metropolis_cache['source_param_ids'].get('bounds_max', 'fallback')
                },
                'cache_age': time.time() - self._metropolis_cache['last_update'],
                'last_update_ts': self._metropolis_cache['last_update']
            }
            
            return bounds, provenance
    
    def _refresh_cache_if_stale(self):
        """Refresh cache from registry if TTL expired."""
        import time
        
        try:
            # Re-read from registry
            param_ids = {
                'temperature': 'scfd.accept.metropolis_temperature',
                'bounds_min': 'scfd.accept.bounds_min',
                'bounds_max': 'scfd.accept.bounds_max'
            }
            
            for param_name, param_id in param_ids.items():
                param_entry = self.parameter_registry.get_parameter(param_id)
                if param_entry is not None:
                    self._metropolis_cache[param_name] = param_entry.current_value
                    self._metropolis_cache['source_param_ids'][param_name] = param_id
            
            self._enforce_cache_invariants()
            self._metropolis_cache['last_update'] = time.time()
            
        except Exception as e:
            print(f"WARN: Failed to refresh metropolis cache: {e}")
    
    def _initialize_default_compositions(self):
        """Set up basic vector composition patterns."""
        
        # Exploration + Stabilization combination
        try:
            exploration_vectors = [v.vector_id for v in self.registry.get_vectors_by_tag("explore")][:2]
            stabilization_vectors = [v.vector_id for v in self.registry.get_vectors_by_tag("stabilize")][:2]
        except (KeyError, AttributeError):
            # Handle test cases with incomplete registries
            exploration_vectors = []
            stabilization_vectors = []
        
        if exploration_vectors and stabilization_vectors:
            self.add_composition_rule(CompositionRule(
                name="explore_stabilize",
                vector_ids=exploration_vectors + stabilization_vectors,
                weights=[0.6, 0.4, 0.3, 0.7],  # Favor exploration slightly
                composition_type="adaptive",
                priority=0.8
            ))
        
        # High-energy field smoothing combination
        try:
            smoothing_vectors = [v.vector_id for v in self.registry.get_vectors_by_tag("smooth_front")][:2]
            alignment_vectors = [v.vector_id for v in self.registry.get_vectors_by_tag("align")][:2]
        except (KeyError, AttributeError):
            smoothing_vectors = []
            alignment_vectors = []
        
        if smoothing_vectors and alignment_vectors:
            self.add_composition_rule(CompositionRule(
                name="smooth_align",
                vector_ids=smoothing_vectors + alignment_vectors,
                weights=[0.7, 0.3, 0.4, 0.6],
                composition_type="conditional",
                priority=0.9
            ))
        
        # Meta-learning enhanced combination
        try:
            meta_vectors = [v.vector_id for v in self.registry.base_registry if "meta" in v.vector_id]
            if meta_vectors:
                # Combine meta vector with domain-specific vectors
                physics_vectors = []
                for domain in ["heat", "flow", "wave"]:
                    domain_vecs = [v.vector_id for v in self.registry.base_registry 
                                  if domain in v.vector_id and "smoke" in v.vector_id][:1]
                    physics_vectors.extend(domain_vecs)
        except (AttributeError, TypeError):
            meta_vectors = []
            physics_vectors = []
            
        if meta_vectors and physics_vectors:
            self.add_composition_rule(CompositionRule(
                name="meta_physics_blend",
                vector_ids=meta_vectors[:1] + physics_vectors[:3],
                weights=[0.5, 0.2, 0.2, 0.1],  # Meta vector dominates
                composition_type="sequential",
                priority=1.0
            ))
    
    def add_composition_rule(self, rule: CompositionRule):
        """Add a new composition rule to the toolkit."""
        self.composition_rules[rule.name] = rule
        self.adaptive_weights[rule.name] = rule.weights.copy()
        self.performance_tracker[rule.name] = {
            "total_uses": 0,
            "successful_uses": 0,
            "recent_performance": [],
            "weight_adjustments": []
        }
    
    def select_composition(self, context: CompositionContext) -> Optional[CompositionRule]:
        """Select the best composition rule for current context."""
        if not self.composition_rules:
            return None
        
        # Score each composition rule
        best_rule = None
        best_score = -1.0
        
        for rule_name, rule in self.composition_rules.items():
            score = self._score_composition_rule(rule, context)
            
            # Apply performance-based adjustment
            tracker = self.performance_tracker[rule_name]
            if tracker["total_uses"] > 0:
                success_rate = tracker["successful_uses"] / tracker["total_uses"]
                score *= (0.5 + success_rate)  # Boost successful compositions
            
            if score > best_score:
                best_score = score
                best_rule = rule
        
        return best_rule
    
    def _score_composition_rule(self, rule: CompositionRule, context: CompositionContext) -> float:
        """Score how well a composition rule fits the current context."""
        score = rule.priority
        
        # Check if all required vectors are available
        available_vectors = {v.vector_id for v in self.registry.base_registry}
        missing_vectors = set(rule.vector_ids) - available_vectors
        if missing_vectors:
            return 0.0  # Cannot use this composition
        
        # Context-based scoring
        features = context.local_features
        
        # Boost exploration compositions in low-activity areas
        if "explore" in rule.name and features.get("gradient_magnitude", 0) < 0.1:
            score += 0.3
        
        # Boost stabilization compositions in high-activity areas
        if "stabilize" in rule.name and features.get("gradient_magnitude", 0) > 0.5:
            score += 0.3
        
        # Boost meta compositions when coherence is in optimal range
        if "meta" in rule.name and 0.15 <= features.get("coherence", 0) <= 0.25:
            score += 0.4
        
        # Boost smoothing compositions when entropy is high
        if "smooth" in rule.name and features.get("entropy", 0) > 1.2:
            score += 0.2
        
        # Penalize recently failed compositions
        if len(rule.success_history) >= 3 and not any(rule.success_history[-3:]):
            score *= 0.5
        
        return score
    
    def _apply_weight_adjustments(self, rule: CompositionRule, adjustments: Dict[str, float]) -> CompositionRule:
        """Apply weight adjustments from controller bridge to composition rule."""
        # Create a copy of the rule with adjusted weights
        new_weights = rule.weights.copy()
        
        # Apply adjustments to specific vectors
        for i, vector_id in enumerate(rule.vector_ids):
            if vector_id in adjustments and i < len(new_weights):
                new_weights[i] *= adjustments[vector_id]
        
        # Apply global adjustments if no specific vector matches
        global_adjustment = adjustments.get("global", 1.0)
        if global_adjustment != 1.0:
            for i in range(len(new_weights)):
                new_weights[i] *= global_adjustment
        
        # Normalize weights to maintain stability
        total_weight = sum(new_weights)
        if total_weight > 0:
            new_weights = [w / total_weight for w in new_weights]
        
        # Create new rule with adjusted weights
        return CompositionRule(
            name=rule.name + "_adjusted",
            vector_ids=rule.vector_ids,
            weights=new_weights,
            composition_type=rule.composition_type,
            activation_condition=rule.activation_condition,
            priority=rule.priority,
            success_history=rule.success_history
        )
    
    def compose_vectors(self, 
                       rule: CompositionRule, 
                       context: CompositionContext,
                       global_theta: Array,
                       global_theta_dot: Array,
                       agent_pos: Tuple[int, int],
                       physics_cfg) -> Tuple[VectorInvocation, OutcomeMetrics, bool]:
        """Execute vector composition according to the specified rule."""
        
        # Phase 1: Record SCFD telemetry for baseline validation
        self.record_scfd_telemetry(global_theta, context.timestep)
        
        # Apply controller bridge policies if available
        if self.controller_bridge:
            weight_adjustments = self.controller_bridge.apply_policies_to_composition(context)
            if weight_adjustments:
                # Apply weight adjustments to rule
                rule = self._apply_weight_adjustments(rule, weight_adjustments)
        
        if rule.composition_type == "linear":
            return self._linear_composition(rule, context, global_theta, global_theta_dot, agent_pos, physics_cfg)
        elif rule.composition_type == "sequential":
            return self._sequential_composition(rule, context, global_theta, global_theta_dot, agent_pos, physics_cfg)
        elif rule.composition_type == "adaptive":
            return self._adaptive_composition(rule, context, global_theta, global_theta_dot, agent_pos, physics_cfg)
        elif rule.composition_type == "conditional":
            return self._conditional_composition(rule, context, global_theta, global_theta_dot, agent_pos, physics_cfg)
        else:
            # Fallback to linear
            return self._linear_composition(rule, context, global_theta, global_theta_dot, agent_pos, physics_cfg)
    
    def _linear_composition(self, 
                           rule: CompositionRule,
                           context: CompositionContext,
                           global_theta: Array,
                           global_theta_dot: Array,
                           agent_pos: Tuple[int, int],
                           physics_cfg) -> Tuple[VectorInvocation, OutcomeMetrics, bool]:
        """Linear weighted combination of vectors."""
        from engine.energy import total_energy_density, metropolis_accept
        
        i, j = agent_pos
        
        # Record initial state
        initial_field = global_theta[i, j]
        initial_energy = float(total_energy_density(global_theta, global_theta_dot, physics_cfg)[i, j])
        
        # Load and combine vectors
        combined_influence = 0.0
        valid_vectors = []
        
        # Ensure composition weights are normalized (handle mid-step updates)
        weights = rule.weights[:len(rule.vector_ids)]
        weights_sum = sum(weights)
        if weights_sum > 0:
            normalized_weights = [w / weights_sum for w in weights]
        else:
            normalized_weights = [1.0 / len(rule.vector_ids)] * len(rule.vector_ids)
        
        for vector_id, weight in zip(rule.vector_ids, normalized_weights):
            vector_entry = next((v for v in self.registry.base_registry if v.vector_id == vector_id), None)
            if vector_entry:
                try:
                    with open(vector_entry.path) as f:
                        data = json.load(f)
                    vector = np.array(data["vector"], dtype=np.float32)
                    
                    if len(vector) >= 1:
                        # Scale influence by weight and current weights
                        adaptive_weight = self.adaptive_weights[rule.name][len(valid_vectors)]
                        influence = float(vector[0]) * weight * adaptive_weight * 0.02
                        combined_influence += influence
                        valid_vectors.append(vector_id)
                except:
                    continue
        
        if not valid_vectors:
            # No valid vectors in composition
            return VectorInvocation(
                vector_id=f"composition_{rule.name}",
                reason="no_valid_vectors",
                confidence=0.0,
                selection_method="composition"
            ), OutcomeMetrics(0.0, False, 0.0, 0), False
        
        # Apply combined influence
        global_theta[i, j] += combined_influence
        
        # SCFD-based acceptance: Single source of truth for all acceptance decisions
        from engine.energy import scfd_accept_update, extract_scfd_features
        
        # Convert continuous field to symbols for SCFD evaluation
        # Simple discretization: map field values to {0,1,2} based on quantiles
        flat_field = global_theta.flatten()
        q33, q67 = np.percentile(flat_field, [33, 67])
        symbol_grid = np.zeros_like(global_theta, dtype=np.int32)
        symbol_grid[global_theta < q33] = 0
        symbol_grid[(global_theta >= q33) & (global_theta < q67)] = 1
        symbol_grid[global_theta >= q67] = 2
        
        # Get SCFD parameters snapshot (single source of truth with gamma guard)
        scfd_params = self.get_scfd_parameters_snapshot()
        alpha, beta, gamma, epsilon = scfd_params["alpha"], scfd_params["beta"], scfd_params["gamma"], scfd_params["epsilon"]
        
        # Test symbol change (simple mapping: decrease -> 0, stay -> 1, increase -> 2)
        old_symbol = symbol_grid[i, j]
        if combined_influence < -0.01:
            new_symbol = max(0, old_symbol - 1)
        elif combined_influence > 0.01:
            new_symbol = min(2, old_symbol + 1)
        else:
            new_symbol = old_symbol
        
        if USE_SCFD_ACCEPTANCE:
            # Phase 2+: Full SCFD acceptance
            accepted, delta_L, delta_S = scfd_accept_update(
                symbol_grid, i, j, new_symbol, alpha, beta, gamma, epsilon,
                use_metropolis=False,  # Pure SCFD acceptance
                boundary_mode=SCFD_BOUNDARY_MODE,  # Consistent boundary conditions
                delta_L_tolerance=1e-9  # Numerical robustness
            )
            delta_energy = delta_L
        else:
            # Phase 1: Controller-only mode with simple acceptance (sensing still enabled)
            # Compute SCFD features for sensing but use simple acceptance
            from engine.energy import total_energy_density
            
            new_energy = float(total_energy_density(global_theta, global_theta_dot, physics_cfg)[i, j])
            delta_energy = new_energy - initial_energy
            
            # Simple acceptance: favor energy decrease with some randomness
            if delta_energy < 0:
                accepted = True  # Energy decreases - always accept
            elif abs(delta_energy) < 0.1:
                accepted = np.random.random() < 0.5  # Small changes - 50% chance
            else:
                accepted = np.random.random() < 0.1  # Large increases - rarely accept
        
        if not accepted:
            # Revert change
            global_theta[i, j] = initial_field
            delta_energy = 0.0
            progress_made = 0.0
        else:
            progress_made = 1.0 if abs(delta_energy) > 0.01 else 0.0
        
        # Update performance tracking
        self._update_composition_performance(rule.name, accepted and progress_made > 0, delta_energy)
        
        # Update controller bridge with composition result
        if self.controller_bridge and valid_vectors:
            for vector_id in valid_vectors:
                self.controller_bridge.update_vector_performance(
                    vector_id, accepted and progress_made > 0, delta_energy, rule.name
                )
        
        # Create enhanced invocation record with acceptance snapshot
        vector_invocation = VectorInvocation(
            vector_id=f"composition_{rule.name}",
            reason=f"linear_composition_{len(valid_vectors)}_vectors",
            confidence=0.7,
            selection_method="composition"
        )
        
        # Add acceptance details to invocation for audit trail
        if hasattr(vector_invocation, 'metadata'):
            vector_invocation.metadata = getattr(vector_invocation, 'metadata', {})
        else:
            vector_invocation.metadata = {}
            
        vector_invocation.metadata.update({
            'acceptance_snapshot': {
                'temperature': temperature,
                'bounds': bounds,
                'temp_provenance': temp_provenance,
                'bounds_provenance': bounds_provenance,
                'acceptance_prob': acceptance_prob,
                'delta_energy': delta_energy,
                'reject_reason': None if accepted else 'metropolis_reject'
            },
            'vectors_used': [{'id': vid} for vid in valid_vectors],
            'weights_pre_norm': weights,
            'weights_post_norm': normalized_weights,
            'degraded_mode': not hasattr(self, 'parameter_registry') or self.parameter_registry is None
        })
        
        outcome = OutcomeMetrics(
            delta_energy=delta_energy,
            accepted=accepted,
            progress_made=progress_made,
            stuck_counter=0
        )
        
        return vector_invocation, outcome, accepted
    
    def _sequential_composition(self, 
                               rule: CompositionRule,
                               context: CompositionContext,
                               global_theta: Array,
                               global_theta_dot: Array,
                               agent_pos: Tuple[int, int],
                               physics_cfg) -> Tuple[VectorInvocation, OutcomeMetrics, bool]:
        """Apply vectors sequentially with intermediate acceptance testing."""
        from engine.energy import total_energy_density, metropolis_accept
        
        i, j = agent_pos
        initial_field = global_theta[i, j]
        initial_energy = float(total_energy_density(global_theta, global_theta_dot, physics_cfg)[i, j])
        
        total_influence = 0.0
        applied_vectors = []
        
        for vector_id, weight in zip(rule.vector_ids, rule.weights):
            vector_entry = next((v for v in self.registry.base_registry if v.vector_id == vector_id), None)
            if vector_entry:
                try:
                    with open(vector_entry.path) as f:
                        data = json.load(f)
                    vector = np.array(data["vector"], dtype=np.float32)
                    
                    if len(vector) >= 1:
                        # Apply vector influence
                        influence = float(vector[0]) * weight * 0.015
                        test_field = global_theta[i, j] + influence
                        
                        # Test intermediate acceptance
                        global_theta[i, j] = test_field
                        intermediate_energy = float(total_energy_density(global_theta, global_theta_dot, physics_cfg)[i, j])
                        delta_E = intermediate_energy - (initial_energy + total_influence)
                        
                        # Quick acceptance test
                        if abs(delta_E) < 0.1 or np.random.random() < 0.7:
                            total_influence += influence
                            applied_vectors.append(vector_id)
                        else:
                            # Revert this step
                            global_theta[i, j] = initial_field + total_influence
                except:
                    continue
        
        # Final SCFD acceptance test (single source of truth)
        if applied_vectors:
            from engine.energy import scfd_accept_update
            
            # Convert to symbol grid for SCFD acceptance
            flat_field = global_theta.flatten()
            q33, q67 = np.percentile(flat_field, [33, 67])
            symbol_grid = np.zeros_like(global_theta, dtype=np.int32)
            symbol_grid[global_theta < q33] = 0
            symbol_grid[(global_theta >= q33) & (global_theta < q67)] = 1
            symbol_grid[global_theta >= q67] = 2
            
            # Get SCFD parameters snapshot (consistent with linear composition)
            scfd_params = self.get_scfd_parameters_snapshot()
            alpha, beta, gamma, epsilon = scfd_params["alpha"], scfd_params["beta"], scfd_params["gamma"], scfd_params["epsilon"]
            
            # Determine symbol change
            old_symbol = symbol_grid[i, j]
            if total_influence < -0.01:
                new_symbol = max(0, old_symbol - 1)
            elif total_influence > 0.01:
                new_symbol = min(2, old_symbol + 1)
            else:
                new_symbol = old_symbol
            
            if USE_SCFD_ACCEPTANCE:
                # Phase 2+: Full SCFD acceptance
                accepted, delta_L, delta_S = scfd_accept_update(
                    symbol_grid, i, j, new_symbol, alpha, beta, gamma, epsilon,
                    use_metropolis=False, boundary_mode=SCFD_BOUNDARY_MODE, delta_L_tolerance=1e-9
                )
                delta_energy = delta_L
            else:
                # Phase 1: Controller-only mode with simple acceptance
                from engine.energy import total_energy_density
                
                final_energy = float(total_energy_density(global_theta, global_theta_dot, physics_cfg)[i, j])
                delta_energy = final_energy - initial_energy
                
                # Simple acceptance for Phase 1
                if delta_energy < 0:
                    accepted = True
                elif abs(delta_energy) < 0.1:
                    accepted = np.random.random() < 0.5
                else:
                    accepted = np.random.random() < 0.1
            
            if not accepted:
                global_theta[i, j] = initial_field
                delta_energy = 0.0
                progress_made = 0.0
            else:
                progress_made = 1.0 if abs(delta_energy) > 0.01 else 0.0
        else:
            accepted = False
            delta_energy = 0.0
            progress_made = 0.0
        
        self._update_composition_performance(rule.name, accepted and progress_made > 0, delta_energy)
        
        # Update controller bridge with composition result
        if self.controller_bridge and applied_vectors:
            for vector_id in applied_vectors:
                self.controller_bridge.update_vector_performance(
                    vector_id, accepted and progress_made > 0, delta_energy, rule.name
                )
        
        vector_invocation = VectorInvocation(
            vector_id=f"composition_{rule.name}",
            reason=f"sequential_composition_{len(applied_vectors)}_applied",
            confidence=0.8,
            selection_method="composition"
        )
        
        outcome = OutcomeMetrics(
            delta_energy=delta_energy,
            accepted=accepted,
            progress_made=progress_made,
            stuck_counter=0
        )
        
        return vector_invocation, outcome, accepted
    
    def _adaptive_composition(self, 
                             rule: CompositionRule,
                             context: CompositionContext,
                             global_theta: Array,
                             global_theta_dot: Array,
                             agent_pos: Tuple[int, int],
                             physics_cfg) -> Tuple[VectorInvocation, OutcomeMetrics, bool]:
        """Adaptive composition that adjusts weights based on performance."""
        # Adjust weights based on recent performance
        self._adapt_weights(rule.name, context)
        
        # Use linear composition with adapted weights
        return self._linear_composition(rule, context, global_theta, global_theta_dot, agent_pos, physics_cfg)
    
    def _conditional_composition(self, 
                                rule: CompositionRule,
                                context: CompositionContext,
                                global_theta: Array,
                                global_theta_dot: Array,
                                agent_pos: Tuple[int, int],
                                physics_cfg) -> Tuple[VectorInvocation, OutcomeMetrics, bool]:
        """Conditional composition based on local field conditions."""
        features = context.local_features
        
        # Select subset of vectors based on conditions
        selected_vectors = []
        selected_weights = []
        
        for vector_id, weight in zip(rule.vector_ids, rule.weights):
            should_include = True
            
            # Conditional logic based on vector type and context
            if "explore" in vector_id and features.get("gradient_magnitude", 0) > 0.3:
                should_include = False  # Don't explore in high-gradient areas
            elif "stabilize" in vector_id and features.get("coherence", 0) > 0.8:
                should_include = False  # Don't stabilize when already coherent
            elif "meta" in vector_id and not (0.1 <= features.get("coherence", 0) <= 0.3):
                should_include = False  # Meta vector has specific activation range
            
            if should_include:
                selected_vectors.append(vector_id)
                selected_weights.append(weight)
        
        if not selected_vectors:
            # Fallback to at least one vector
            selected_vectors = rule.vector_ids[:1]
            selected_weights = rule.weights[:1]
        
        # Create temporary rule with selected vectors
        temp_rule = CompositionRule(
            name=f"{rule.name}_conditional",
            vector_ids=selected_vectors,
            weights=selected_weights,
            composition_type="linear"
        )
        
        return self._linear_composition(temp_rule, context, global_theta, global_theta_dot, agent_pos, physics_cfg)
    
    def _adapt_weights(self, rule_name: str, context: CompositionContext):
        """Adapt composition weights based on performance history."""
        tracker = self.performance_tracker[rule_name]
        
        if tracker["total_uses"] < 5:
            return  # Need more data
        
        success_rate = tracker["successful_uses"] / tracker["total_uses"]
        
        # Adjust weights based on success rate
        if success_rate < 0.3:
            # Poor performance, try redistributing weights
            current_weights = self.adaptive_weights[rule_name]
            
            # Reduce weight of first vector, boost others
            adjustment = 0.1
            if current_weights[0] > adjustment:
                current_weights[0] -= adjustment
                # Distribute to others
                for i in range(1, len(current_weights)):
                    current_weights[i] += adjustment / (len(current_weights) - 1)
            
            tracker["weight_adjustments"].append({
                "timestep": context.timestep,
                "reason": "poor_performance",
                "adjustment": adjustment
            })
        
        elif success_rate > 0.8:
            # Good performance, try to optimize further
            features = context.local_features
            
            # Boost weights that seem to correlate with success
            if features.get("coherence", 0) > 0.2 and len(self.adaptive_weights[rule_name]) > 0:
                # Boost first vector (often meta or primary)
                self.adaptive_weights[rule_name][0] = min(1.0, self.adaptive_weights[rule_name][0] * 1.1)
    
    def _update_composition_performance(self, rule_name: str, success: bool, energy_change: float):
        """Update performance tracking for composition rules."""
        tracker = self.performance_tracker[rule_name]
        tracker["total_uses"] += 1
        
        if success:
            tracker["successful_uses"] += 1
        
        tracker["recent_performance"].append({
            "success": success,
            "energy_change": energy_change,
            "timestamp": tracker["total_uses"]
        })
        
        # Keep only recent performance data
        if len(tracker["recent_performance"]) > 20:
            tracker["recent_performance"].pop(0)
        
        # Update composition rule success history
        rule = self.composition_rules[rule_name]
        rule.success_history.append(success)
        if len(rule.success_history) > 10:
            rule.success_history.pop(0)
    
    def get_composition_analytics(self) -> Dict[str, Any]:
        """Get analytics on composition performance."""
        analytics = {}
        
        for rule_name, tracker in self.performance_tracker.items():
            if tracker["total_uses"] > 0:
                success_rate = tracker["successful_uses"] / tracker["total_uses"]
                avg_energy_change = np.mean([
                    p["energy_change"] for p in tracker["recent_performance"]
                ]) if tracker["recent_performance"] else 0.0
                
                analytics[rule_name] = {
                    "total_uses": tracker["total_uses"],
                    "success_rate": success_rate,
                    "avg_energy_change": avg_energy_change,
                    "current_weights": self.adaptive_weights[rule_name],
                    "weight_adjustments": len(tracker["weight_adjustments"])
                }
        
        return analytics
    
    def extract_scfd_features_for_sensing(self, global_theta: Array) -> Dict[str, float]:
        """
        Extract SCFD features from continuous field for sensing router.
        
        This method converts the continuous theta field to symbolic representation
        and computes the 3 core SCFD features for the sensing system.
        
        Args:
            global_theta: Continuous field state
            
        Returns:
            Dictionary with SCFD features: mean_coherence, mean_abs_curvature, mean_cross_gradient
        """
        try:
            from engine.energy import extract_scfd_features
            
            # Convert continuous field to symbols using adaptive quantiles
            flat_field = global_theta.flatten()
            if len(np.unique(flat_field)) <= 3:
                # Already discretized
                symbol_grid = global_theta.astype(np.int32)
            else:
                # Discretize using quantiles
                q33, q67 = np.percentile(flat_field, [33, 67])
                symbol_grid = np.zeros_like(global_theta, dtype=np.int32)
                symbol_grid[global_theta < q33] = 0
                symbol_grid[(global_theta >= q33) & (global_theta < q67)] = 1
                symbol_grid[global_theta >= q67] = 2
            
            # Extract SCFD features with consistent boundary conditions
            features = extract_scfd_features(symbol_grid, boundary_mode=SCFD_BOUNDARY_MODE)
            
            # Add composition-specific features
            features.update({
                'composition_activity': len([r for r in self.composition_rules if len(r.success_history) > 0]),
                'adaptive_weight_variance': np.var([
                    np.var(weights) for weights in self.adaptive_weights.values()
                ]) if self.adaptive_weights else 0.0
            })
            
            return features
            
        except ImportError:
            # Fallback if SCFD functions not available
            return {
                'mean_coherence': 0.5,
                'mean_abs_curvature': 0.2,
                'mean_cross_gradient': 0.0,
                'composition_activity': 0.0,
                'adaptive_weight_variance': 0.0
            }


class CompositeAgent:
    """Enhanced agent that uses vector composition instead of single vector selection."""
    
    def __init__(self, 
                 base_agent,  # GridCellAgent or MazeGridCellAgent
                 composer: VectorComposer):
        self.base_agent = base_agent
        self.composer = composer
        
        # Override base agent's vector selection
        self.composition_mode = True
        self.composition_history = []
    
    def step_with_composition(self, 
                             global_theta: Array, 
                             global_theta_dot: Array,
                             neighbor_states: Dict[Tuple[int, int], Dict],
                             logger: RunLogger) -> bool:
        """Execute agent step using vector composition."""
        
        # Use base agent's sensing
        if hasattr(self.base_agent, 'sense_maze_environment'):
            local_features = self.base_agent.sense_maze_environment(global_theta, neighbor_states)
        else:
            local_features = self.base_agent.sense_local_environment(global_theta)
        
        # Create composition context
        context = CompositionContext(
            current_field_state=global_theta,
            local_features=local_features,
            agent_history=self.base_agent.recent_actions,
            neighbor_states=neighbor_states,
            timestep=self.base_agent.timestep,
            physics_context=local_features
        )
        
        # Select composition rule
        selected_rule = self.composer.select_composition(context)
        
        if selected_rule is None:
            # Fallback to base agent behavior
            return self.base_agent.step(global_theta, global_theta_dot, logger)
        
        # Apply composition
        vector_invocation, outcome, accepted = self.composer.compose_vectors(
            selected_rule, context, global_theta, global_theta_dot, 
            self.base_agent.pos, self.base_agent.physics_cfg
        )
        
        # Log the composition
        physics_ctx = logger.compute_physics_context(
            global_theta, global_theta_dot, self.base_agent.physics_cfg, self.base_agent.pos
        )
        neighbor_summary = logger.compute_neighbor_summary(global_theta, self.base_agent.pos)
        
        from utils.logging import CellLogEntry
        log_entry = CellLogEntry(
            cell_pos=self.base_agent.pos,
            timestep=self.base_agent.timestep,
            physics=physics_ctx,
            vector=vector_invocation,
            outcome=outcome,
            neighbors=neighbor_summary
        )
        
        logger.log_cell_action(log_entry)
        
        # Track composition history
        self.composition_history.append({
            "rule_name": selected_rule.name,
            "vector_ids": selected_rule.vector_ids,
            "accepted": accepted,
            "delta_energy": outcome.delta_energy,
            "timestep": self.base_agent.timestep
        })
        
        # Update base agent state
        self.base_agent.timestep += 1
        if accepted:
            self.base_agent.last_progress_step = self.base_agent.timestep
            self.base_agent.stuck_counter = 0
        else:
            self.base_agent.stuck_counter += 1
        
        return accepted


__all__ = [
    "CompositionRule", "CompositionContext", "VectorComposer", "CompositeAgent"
]