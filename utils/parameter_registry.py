"""
Unified Parameter Registry - The Backbone of SCFD Integration

This registry serves as the canonical source of truth for all tunable parameters
across the SCFD system. Every component (vectors, controllers, EFE tuner, composition)
reads and writes through this registry with proper guardrails.

Key Design Principles:
1. Role-based parameter organization (not component-specific names)
2. Automatic conflict resolution and safety constraints  
3. Provenance tracking for audit and replay
4. Trust regions and cooldowns for stability
5. Coupling groups for coordinated parameter updates
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
import time
import math
import threading
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class ParameterRole(Enum):
    """Semantic roles for parameters - what they do, not where they come from."""
    ACCEPTANCE = "acceptance"           # Metropolis temperature, acceptance bounds
    CONTROL_GAIN = "control_gain"       # PID gains, control amplification  
    DIFFUSION = "diffusion"            # Alpha, diffusion coefficients
    SMOOTHING = "smoothing"            # Temporal smoothing, filtering
    STABILITY = "stability"            # Damping, energy bounds
    FILTER = "filter"                  # Noise filtering, signal conditioning
    SCHEDULE = "schedule"              # Time constants, adaptation rates
    MOVEMENT = "movement"              # Agent movement, navigation gains
    COMPOSITION = "composition"        # Vector composition weights
    THRESHOLD = "threshold"            # Activation and decision thresholds
    COUPLING = "coupling"              # Cross-field interactions, information flow
    UNKNOWN = "unknown"                # Quarantine for unrecognized parameters

class ParameterScope(Enum):
    """What the parameter applies to."""
    GLOBAL = "global"                  # Entire system
    REGION = "region"                  # Spatial region [x,y,w,h]
    CELL = "cell"                      # Single cell [i,j]  
    AGENT = "agent"                    # Single agent [k]
    VECTOR = "vector"                  # Single vector instance
    ROLE_GROUP = "role_group"          # Set of related parameters

class ParameterDType(Enum):
    """Parameter data type and scaling."""
    FLOAT_LINEAR = "float_linear"      # Linear scale
    FLOAT_LOG = "float_log"           # Logarithmic scale
    INTEGER = "integer"               # Discrete values
    BOOLEAN = "boolean"               # On/off flags
    PROBABILITY = "probability"       # [0,1] range
    WEIGHTS = "weights"               # Sum-to-one constraint

@dataclass
class ParameterConstraint:
    """Safety constraints for parameter values."""
    min_value: float
    max_value: float
    trust_region_pct: float = 0.1      # Max % change per update
    cooldown_seconds: float = 1.0      # Min time between updates
    invariant_check: Optional[Callable[[float], bool]] = None  # Custom validation
    rollback_on_violation: bool = True  # Auto-revert unsafe changes
    
    def validate(self, value: float, old_value: float, last_update_time: float, dtype: ParameterDType) -> Tuple[bool, str]:
        """Validate a proposed parameter change with dtype-aware logic."""
        current_time = time.time()
        
        # Check bounds first
        if not (self.min_value <= value <= self.max_value):
            return False, f"Value {value} outside bounds [{self.min_value}, {self.max_value}]"
        
        # Check trust region with dtype-aware logic
        if self._check_trust_region_violation(value, old_value, dtype):
            if dtype == ParameterDType.FLOAT_LOG:
                ratio = value / max(abs(old_value), 1e-10)
                return False, f"Log-scale change ratio {ratio:.3f} exceeds trust region {1+self.trust_region_pct}"
            else:
                pct_change = abs(value - old_value) / max(abs(old_value), 0.01 * (self.max_value - self.min_value))
                return False, f"Change {pct_change:.3f} exceeds trust region {self.trust_region_pct}"
        
        # Check cooldown
        if current_time - last_update_time < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (current_time - last_update_time)
            return False, f"Cooldown active: {remaining:.1f}s remaining"
        
        # Check custom invariant
        if self.invariant_check and not self.invariant_check(value):
            return False, "Custom invariant check failed"
            
        return True, "OK"
    
    def _check_trust_region_violation(self, value: float, old_value: float, dtype: ParameterDType) -> bool:
        """Check trust region violation with dtype-appropriate logic."""
        if dtype == ParameterDType.FLOAT_LOG:
            # For log-scale: multiplicative trust region
            if old_value <= 0:
                return False  # Can't do ratio check on zero/negative
            ratio = value / old_value
            return ratio > (1 + self.trust_region_pct) or ratio < (1 - self.trust_region_pct)
        else:
            # For linear scale: relative or absolute trust region (whichever is larger)
            abs_floor = 0.01 * (self.max_value - self.min_value)  # 1% of range as absolute floor
            denominator = max(abs(old_value), abs_floor)
            pct_change = abs(value - old_value) / denominator
            return pct_change > self.trust_region_pct

@dataclass
class GroupInvariant:
    """Invariant that applies to a group of parameters."""
    group_name: str
    check_func: Callable[[Dict[str, float]], Tuple[bool, str]]  # values -> (valid, reason)
    repair_func: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None  # Auto-repair if possible
    
class ParameterEntry:
    """Complete parameter specification with metadata."""
    
    def __init__(self, id: str, role: ParameterRole, scope: ParameterScope, dtype: ParameterDType,
                 constraints: ParameterConstraint, units: str = "", description: str = "",
                 coupling_group: Optional[str] = None, owner_hint: str = "system", current_value: float = 0.0):
        self.id = id
        self.role = role
        self.scope = scope
        self.dtype = dtype
        self.constraints = constraints
        self.units = units
        self.description = description
        self.coupling_group = coupling_group
        self.owner_hint = owner_hint
        
        # Runtime state
        self.current_value = current_value
        self.last_update_time = 0.0
        self.update_count = 0
        self.provenance: List[Dict[str, Any]] = []
        self._provenance_max_size = 20  # Ring buffer for provenance
    
    def update_value(self, new_value: float, writer: str, reason: str = "") -> Tuple[bool, str]:
        """Update parameter value with validation and provenance tracking."""
        # Validate the change
        is_valid, message = self.constraints.validate(
            new_value, self.current_value, self.last_update_time, self.dtype
        )
        
        if not is_valid:
            logger.warning(f"Parameter update rejected for {self.id}: {message}")
            return False, message
        
        # Apply dtype-specific rounding/clamping
        processed_value = self._process_value_by_dtype(new_value)
        
        # Record provenance (before applying change for event callback)
        old_value = self.current_value
        provenance_entry = {
            "timestamp": time.time(),
            "old_value": old_value,
            "new_value": processed_value,
            "writer": writer,
            "reason": reason,
            "update_count": self.update_count
        }
        
        # Maintain ring buffer for provenance
        self.provenance.append(provenance_entry)
        if len(self.provenance) > self._provenance_max_size:
            self.provenance = self.provenance[-self._provenance_max_size:]
        
        # Apply the change
        self.current_value = processed_value
        self.last_update_time = time.time()
        self.update_count += 1
        
        logger.info(f"Parameter {self.id} updated: {old_value:.6f} → {processed_value:.6f} by {writer} ({reason})")
        return True, "Updated successfully"
    
    def _process_value_by_dtype(self, value: float) -> float:
        """Apply dtype-specific processing to the value."""
        if self.dtype == ParameterDType.INTEGER:
            return float(round(value))
        elif self.dtype == ParameterDType.BOOLEAN:
            return 1.0 if value >= 0.5 else 0.0
        elif self.dtype == ParameterDType.PROBABILITY or self.dtype == ParameterDType.WEIGHTS:
            return max(0.0, min(1.0, value))  # Clamp to [0,1]
        else:
            return value  # FLOAT_LINEAR, FLOAT_LOG use value as-is

class ParameterRegistry:
    """Unified parameter registry - the backbone of SCFD integration."""
    
    def __init__(self):
        self._parameters: Dict[str, ParameterEntry] = {}
        self._role_index: Dict[ParameterRole, List[str]] = {}
        self._coupling_groups: Dict[str, List[str]] = {}
        self._group_invariants: Dict[str, GroupInvariant] = {}
        self._lock = threading.RLock()  # Thread-safe operations
        self._event_callbacks: List[Callable] = []
        
        # Role-level cooldowns to prevent flapping
        self._role_cooldowns: Dict[ParameterRole, float] = {}
        
        # Initialize with core SCFD parameters and group invariants
        self._initialize_core_parameters()
        self._initialize_group_invariants()
    
    def _initialize_group_invariants(self):
        """Set up group-level invariants for parameter coupling."""
        
        # Acceptance bounds: min < max with minimum gap
        def check_acceptance_bounds(values: Dict[str, float]) -> Tuple[bool, str]:
            min_val = values.get("scfd.accept.bounds_min", 0.01)
            max_val = values.get("scfd.accept.bounds_max", 0.99)
            min_gap = 0.1  # Minimum 10% gap
            if min_val >= max_val:
                return False, f"bounds_min ({min_val}) >= bounds_max ({max_val})"
            if (max_val - min_val) < min_gap:
                return False, f"Gap {max_val - min_val:.3f} < minimum {min_gap}"
            return True, "OK"
        
        def repair_acceptance_bounds(values: Dict[str, float]) -> Dict[str, float]:
            min_val = values.get("scfd.accept.bounds_min", 0.01)
            max_val = values.get("scfd.accept.bounds_max", 0.99)
            if min_val >= max_val:
                # Set to reasonable defaults with gap
                return {"scfd.accept.bounds_min": 0.01, "scfd.accept.bounds_max": 0.99}
            return values  # No repair needed
        
        self._group_invariants["acceptance_bounds"] = GroupInvariant(
            "acceptance_bounds", check_acceptance_bounds, repair_acceptance_bounds
        )
        
        # Composition weights: sum to 1 within tolerance
        def check_composition_weights(values: Dict[str, float]) -> Tuple[bool, str]:
            weight_sum = sum(v for k, v in values.items() if "composition.weight." in k)
            tolerance = 0.05  # 5% tolerance
            if abs(weight_sum - 1.0) > tolerance:
                return False, f"Weight sum {weight_sum:.3f} deviates from 1.0 by {abs(weight_sum - 1.0):.3f}"
            return True, "OK"
        
        def repair_composition_weights(values: Dict[str, float]) -> Dict[str, float]:
            weight_params = {k: v for k, v in values.items() if "composition.weight." in k}
            if not weight_params:
                return values
            
            # Normalize to sum=1
            total = sum(weight_params.values())
            if total > 0:
                normalized = {k: v / total for k, v in weight_params.items()}
                values.update(normalized)
            return values
        
        self._group_invariants["composition_weights"] = GroupInvariant(
            "composition_weights", check_composition_weights, repair_composition_weights
        )
    
    def _initialize_core_parameters(self):
        """Register core SCFD parameters with proper roles and constraints."""
        
        # Diffusion parameters
        self.register_parameter(ParameterEntry(
            id="scfd.physics.alpha",
            role=ParameterRole.DIFFUSION,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.01, 1.0, trust_region_pct=0.1, cooldown_seconds=2.0),
            units="1/s",
            description="SCFD coherence strength parameter",
            coupling_group="physics_core",
            current_value=0.1
        ))
        
        self.register_parameter(ParameterEntry(
            id="scfd.physics.gamma", 
            role=ParameterRole.DIFFUSION,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.01, 1.0, trust_region_pct=0.1, cooldown_seconds=2.0),
            units="1/s",
            description="SCFD wave propagation coefficient",
            coupling_group="physics_core",
            current_value=0.12
        ))
        
        self.register_parameter(ParameterEntry(
            id="scfd.physics.beta",
            role=ParameterRole.STABILITY,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.1, 5.0, trust_region_pct=0.15, cooldown_seconds=3.0),
            units="1/s",
            description="SCFD damping coefficient",
            coupling_group="physics_core",
            current_value=1.0
        ))
        
        # Acceptance parameters
        self.register_parameter(ParameterEntry(
            id="scfd.accept.metropolis_temperature",
            role=ParameterRole.ACCEPTANCE,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LOG,
            constraints=ParameterConstraint(0.1, 3.0, trust_region_pct=0.2, cooldown_seconds=5.0),
            units="energy",
            description="Metropolis acceptance temperature",
            current_value=1.2  # From EFE tuner analysis
        ))
        
        self.register_parameter(ParameterEntry(
            id="scfd.accept.bounds_min",
            role=ParameterRole.ACCEPTANCE,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.PROBABILITY,
            constraints=ParameterConstraint(0.001, 0.1, trust_region_pct=0.3, cooldown_seconds=10.0),
            units="probability",
            description="Minimum acceptance probability",
            coupling_group="acceptance_bounds",
            current_value=0.005  # From EFE tuner analysis
        ))
        
        self.register_parameter(ParameterEntry(
            id="scfd.accept.bounds_max",
            role=ParameterRole.ACCEPTANCE,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.PROBABILITY,
            constraints=ParameterConstraint(0.9, 0.999, trust_region_pct=0.3, cooldown_seconds=10.0),
            units="probability",
            description="Maximum acceptance probability",
            coupling_group="acceptance_bounds",
            current_value=0.98  # From EFE tuner analysis
        ))
        
        # Control parameters
        self.register_parameter(ParameterEntry(
            id="control.cartpole.gain_proportional",
            role=ParameterRole.CONTROL_GAIN,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.1, 10.0, trust_region_pct=0.2, cooldown_seconds=1.0),
            units="force/angle",
            description="CartPole proportional control gain",
            coupling_group="cartpole_gains",
            current_value=2.0
        ))
        
        self.register_parameter(ParameterEntry(
            id="control.cartpole.gain_derivative",
            role=ParameterRole.CONTROL_GAIN,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.1, 5.0, trust_region_pct=0.2, cooldown_seconds=1.0),
            units="force*s/angle",
            description="CartPole derivative control gain",
            coupling_group="cartpole_gains",
            current_value=0.8
        ))
        
        # Filter parameters
        self.register_parameter(ParameterEntry(
            id="sensing.filter.noise_strength",
            role=ParameterRole.FILTER,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.1, 1.0, trust_region_pct=0.15, cooldown_seconds=2.0),
            units="ratio",
            description="Noise filtering strength",
            current_value=0.6
        ))
        
        # Agent activation thresholds (fixed values from analysis)
        self.register_parameter(ParameterEntry(
            id="agent.activation.field_change_threshold",
            role=ParameterRole.THRESHOLD,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LOG,
            constraints=ParameterConstraint(0.001, 0.5, trust_region_pct=0.3, cooldown_seconds=5.0),
            units="field_units",
            description="Minimum field change to activate agent",
            current_value=0.01  # Fixed from 0.1
        ))
        
        self.register_parameter(ParameterEntry(
            id="agent.activation.gradient_threshold",
            role=ParameterRole.THRESHOLD,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LOG,
            constraints=ParameterConstraint(0.001, 1.0, trust_region_pct=0.3, cooldown_seconds=5.0),
            units="field_units/pixel",
            description="Minimum gradient magnitude to activate agent",
            current_value=0.02  # Fixed from 0.2
        ))
        
        # Composition weights (will be expanded dynamically)
        self.register_parameter(ParameterEntry(
            id="composition.weight.exploration",
            role=ParameterRole.COMPOSITION,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.WEIGHTS,
            constraints=ParameterConstraint(0.0, 1.0, trust_region_pct=0.2, cooldown_seconds=3.0),
            units="weight",
            description="Weight for exploration vectors in composition",
            coupling_group="composition_weights",
            current_value=0.6
        ))
        
        self.register_parameter(ParameterEntry(
            id="composition.weight.stabilization",
            role=ParameterRole.COMPOSITION,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.WEIGHTS,
            constraints=ParameterConstraint(0.0, 1.0, trust_region_pct=0.2, cooldown_seconds=3.0),
            units="weight",
            description="Weight for stabilization vectors in composition",
            coupling_group="composition_weights",
            current_value=0.4
        ))
        
        # Schedule parameters
        self.register_parameter(ParameterEntry(
            id="schedule.efe_update_interval",
            role=ParameterRole.SCHEDULE,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.INTEGER,
            constraints=ParameterConstraint(10, 200, trust_region_pct=0.5, cooldown_seconds=30.0),
            units="timesteps",
            description="Interval between EFE tuner updates",
            current_value=50
        ))
        
        # Pure SCFD Lagrangian parameters (from tech summary)
        self.register_parameter(ParameterEntry(
            id="scfd.lagrangian.alpha",
            role=ParameterRole.DIFFUSION,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.1, 0.9, trust_region_pct=0.1, cooldown_seconds=2.0),
            units="dimensionless",
            description="SCFD coherence weight (domain formation preference)",
            coupling_group="scfd_lagrangian",
            current_value=0.3  # From tech summary default
        ))
        
        self.register_parameter(ParameterEntry(
            id="scfd.lagrangian.beta",
            role=ParameterRole.SMOOTHING,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(0.1, 0.6, trust_region_pct=0.1, cooldown_seconds=2.0),
            units="dimensionless",
            description="SCFD curvature weight (interface smoothness)",
            coupling_group="scfd_lagrangian",
            current_value=0.2  # From tech summary default
        ))
        
        self.register_parameter(ParameterEntry(
            id="scfd.lagrangian.gamma",
            role=ParameterRole.COUPLING,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.FLOAT_LINEAR,
            constraints=ParameterConstraint(-0.3, 0.4, trust_region_pct=0.1, cooldown_seconds=2.0),
            units="dimensionless",
            description="SCFD cross-gradient coupling (information flow)",
            coupling_group="scfd_lagrangian",
            current_value=0.1  # From tech summary default
        ))
        
        self.register_parameter(ParameterEntry(
            id="scfd.lagrangian.epsilon",
            role=ParameterRole.ACCEPTANCE,
            scope=ParameterScope.GLOBAL,
            dtype=ParameterDType.PROBABILITY,
            constraints=ParameterConstraint(0.05, 0.2, trust_region_pct=0.1, cooldown_seconds=3.0),
            units="bits",
            description="SCFD entropy gate threshold",
            coupling_group="scfd_lagrangian",
            current_value=0.1  # From tech summary default
        ))
        
        logger.info(f"Initialized parameter registry with {len(self._parameters)} core parameters")
    
    def register_parameter(self, param: ParameterEntry) -> bool:
        """Register a new parameter in the registry."""
        with self._lock:
            if param.id in self._parameters:
                logger.warning(f"Parameter {param.id} already registered, skipping")
                return False
            
            self._parameters[param.id] = param
            
            # Update role index
            if param.role not in self._role_index:
                self._role_index[param.role] = []
            self._role_index[param.role].append(param.id)
            
            # Update coupling groups
            if param.coupling_group:
                if param.coupling_group not in self._coupling_groups:
                    self._coupling_groups[param.coupling_group] = []
                self._coupling_groups[param.coupling_group].append(param.id)
            
            logger.info(f"Registered parameter: {param.id} (role: {param.role.value})")
            return True
    
    def get_parameter(self, param_id: str) -> Optional[ParameterEntry]:
        """Get parameter by ID."""
        with self._lock:
            return self._parameters.get(param_id)
    
    def get_parameters_by_role(self, role: ParameterRole) -> List[ParameterEntry]:
        """Get all parameters with a specific role."""
        with self._lock:
            param_ids = self._role_index.get(role, [])
            return [self._parameters[pid] for pid in param_ids]
    
    def get_coupling_group(self, group_name: str) -> List[ParameterEntry]:
        """Get all parameters in a coupling group."""
        with self._lock:
            param_ids = self._coupling_groups.get(group_name, [])
            return [self._parameters[param_id] for param_id in param_ids]
    
    def update_parameter(self, param_id: str, new_value: float, writer: str, reason: str = "") -> Tuple[bool, str]:
        """Update a parameter value with validation and group invariant checking."""
        # Prepare event data outside the lock to avoid deadlock
        event_data = None
        
        with self._lock:
            param = self._parameters.get(param_id)
            if not param:
                return False, f"Parameter {param_id} not found"
            
            old_value = param.current_value
            success, message = param.update_value(new_value, writer, reason)
            
            if success:
                # Check group invariants if this param is in a coupling group
                if param.coupling_group and param.coupling_group in self._group_invariants:
                    group_values = {p.id: p.current_value for p in self.get_coupling_group(param.coupling_group)}
                    invariant = self._group_invariants[param.coupling_group]
                    is_valid, inv_message = invariant.check_func(group_values)
                    
                    if not is_valid:
                        logger.warning(f"Group invariant violation for {param.coupling_group}: {inv_message}")
                        
                        # Try auto-repair if available
                        if invariant.repair_func:
                            try:
                                repaired_values = invariant.repair_func(group_values)
                                for p_id, repaired_val in repaired_values.items():
                                    if p_id in self._parameters and p_id != param_id:
                                        self._parameters[p_id].current_value = repaired_val
                                        logger.info(f"Auto-repaired {p_id}: {group_values[p_id]:.6f} → {repaired_val:.6f}")
                            except Exception as e:
                                logger.error(f"Auto-repair failed for {param.coupling_group}: {e}")
                                # Revert the change
                                param.current_value = old_value
                                return False, f"Group invariant violation and repair failed: {inv_message}"
                
                # Prepare event data
                event_data = {
                    "param_id": param_id,
                    "old_value": old_value,
                    "new_value": new_value,
                    "writer": writer,
                    "reason": reason,
                    "role": param.role.value,
                    "coupling_group": param.coupling_group
                }
        
        # Fire callbacks outside the lock to prevent deadlock
        if success and event_data:
            for callback in self._event_callbacks[:]:  # Copy to avoid modification during iteration
                try:
                    callback("param.update", event_data)
                except Exception as e:
                    logger.error(f"Event callback failed: {e}")
        
        return success, message
    
    def update_role(self, role: ParameterRole, delta_pct: float, writer: str, reason: str = "") -> Dict[str, Tuple[bool, str]]:
        """Update all parameters in a role by a percentage delta with role-level cooldown."""
        current_time = time.time()
        
        # Check role-level cooldown
        last_role_update = self._role_cooldowns.get(role, 0.0)
        role_cooldown = 5.0  # 5 second cooldown between role updates
        if current_time - last_role_update < role_cooldown:
            remaining = role_cooldown - (current_time - last_role_update)
            return {"role_cooldown": (False, f"Role {role.value} cooldown active: {remaining:.1f}s remaining")}
        
        results = {}
        
        with self._lock:
            params_in_role = self.get_parameters_by_role(role)
            
            # For WEIGHTS dtype, handle special case to maintain group constraints
            if any(p.dtype == ParameterDType.WEIGHTS for p in params_in_role):
                results.update(self._update_weights_role(params_in_role, delta_pct, writer, reason))
            else:
                # Standard role update
                for param in params_in_role:
                    new_value = self._compute_role_delta(param, delta_pct)
                    success, message = param.update_value(new_value, writer, f"{reason} (role update)")
                    results[param.id] = (success, message)
            
            # Update role cooldown if any updates succeeded
            if any(success for success, _ in results.values()):
                self._role_cooldowns[role] = current_time
        
        return results
    
    def _compute_role_delta(self, param: ParameterEntry, delta_pct: float) -> float:
        """Compute new value for role update based on dtype."""
        if param.dtype == ParameterDType.INTEGER:
            delta = max(1, round(abs(param.current_value * delta_pct)))
            return param.current_value + (delta if delta_pct > 0 else -delta)
        elif param.dtype == ParameterDType.BOOLEAN:
            # Boolean toggle policy: small deltas don't change, large deltas toggle
            if abs(delta_pct) > 0.3:  # 30% threshold for toggle
                return 1.0 - param.current_value
            else:
                return param.current_value
        elif param.dtype == ParameterDType.FLOAT_LOG:
            # Multiplicative update for log scale
            multiplier = 1.0 + delta_pct
            return param.current_value * multiplier
        else:
            # Linear update for FLOAT_LINEAR, PROBABILITY
            return param.current_value * (1.0 + delta_pct)
    
    def _update_weights_role(self, params: List[ParameterEntry], delta_pct: float, writer: str, reason: str) -> Dict[str, Tuple[bool, str]]:
        """Update weights with automatic renormalization to maintain sum=1."""
        results = {}
        weight_params = [p for p in params if p.dtype == ParameterDType.WEIGHTS]
        
        if not weight_params:
            return results
        
        # Collect current weights by coupling group
        groups = {}
        for param in weight_params:
            group = param.coupling_group or "default"
            if group not in groups:
                groups[group] = []
            groups[group].append(param)
        
        # Update each coupling group separately
        for group_name, group_params in groups.items():
            current_weights = [p.current_value for p in group_params]
            
            # Apply delta to all weights
            new_weights = [w * (1.0 + delta_pct) for w in current_weights]
            
            # Renormalize to sum=1
            total = sum(new_weights)
            if total > 0:
                new_weights = [w / total for w in new_weights]
            else:
                # Fallback to equal weights
                new_weights = [1.0 / len(group_params)] * len(group_params)
            
            # Apply updates
            for param, new_weight in zip(group_params, new_weights):
                success, message = param.update_value(new_weight, writer, f"{reason} (weights role update)")
                results[param.id] = (success, message)
        
        return results
    
    def get_values_by_role(self, role: ParameterRole) -> Dict[str, float]:
        """Get current values for all parameters in a role."""
        with self._lock:
            return {param.id: param.current_value for param in self.get_parameters_by_role(role)}
    
    def get_all_values(self) -> Dict[str, float]:
        """Get current values for all parameters."""
        with self._lock:
            return {param_id: param.current_value for param_id, param in self._parameters.items()}
    
    def add_event_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for parameter update events."""
        self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[str, Dict], None]):
        """Remove callback for parameter update events."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    def save_state(self, filepath: Path):
        """Save registry state to file."""
        with self._lock:
            state = {
                "parameters": {
                    param_id: {
                        "current_value": param.current_value,
                        "update_count": param.update_count,
                        "last_update_time": param.last_update_time,
                        "provenance": param.provenance[-10:]  # Last 10 updates
                    }
                    for param_id, param in self._parameters.items()
                },
                "role_cooldowns": self._role_cooldowns,
                "timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Registry state saved to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load registry state from file with proper provenance handling."""
        with self._lock:
            try:
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                for param_id, param_state in state["parameters"].items():
                    if param_id in self._parameters:
                        param = self._parameters[param_id]
                        param.current_value = param_state["current_value"]
                        param.update_count = param_state["update_count"]
                        param.last_update_time = param_state["last_update_time"]
                        
                        # Replace provenance to avoid duplication
                        param.provenance = param_state["provenance"]
                
                # Load role cooldowns if available
                if "role_cooldowns" in state:
                    for role_str, cooldown_time in state["role_cooldowns"].items():
                        try:
                            role = ParameterRole(role_str)
                            self._role_cooldowns[role] = cooldown_time
                        except ValueError:
                            pass  # Skip unknown roles
                
                logger.info(f"Registry state loaded from {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load registry state: {e}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get registry diagnostics and health metrics."""
        with self._lock:
            total_params = len(self._parameters)
            updates_last_hour = sum(
                1 for param in self._parameters.values()
                if (time.time() - param.last_update_time) < 3600
            )
            
            role_counts = {}
            role_update_rates = {}
            for role, param_ids in self._role_index.items():
                role_counts[role.value] = len(param_ids)
                # Calculate hourly update rate for this role
                recent_updates = sum(
                    param.update_count for param_id in param_ids
                    if (param := self._parameters.get(param_id)) and 
                    (time.time() - param.last_update_time) < 3600
                )
                role_update_rates[role.value] = recent_updates
            
            # Check constraint margins for acceptance bounds
            constraint_margins = {}
            acceptance_params = self.get_parameters_by_role(ParameterRole.ACCEPTANCE)
            if len(acceptance_params) >= 2:
                bounds = {p.id: p.current_value for p in acceptance_params if "bounds" in p.id}
                if "scfd.accept.bounds_min" in bounds and "scfd.accept.bounds_max" in bounds:
                    margin = bounds["scfd.accept.bounds_max"] - bounds["scfd.accept.bounds_min"]
                    constraint_margins["acceptance_window"] = margin
            
            return {
                "total_parameters": total_params,
                "parameters_by_role": role_counts,
                "role_update_rates_last_hour": role_update_rates,
                "coupling_groups": len(self._coupling_groups),
                "group_invariants": len(self._group_invariants),
                "updates_last_hour": updates_last_hour,
                "event_callbacks": len(self._event_callbacks),
                "constraint_margins": constraint_margins,
                "active_role_cooldowns": len([r for r, t in self._role_cooldowns.items() 
                                            if time.time() - t < 10.0])
            }

# Global registry instance
_global_registry: Optional[ParameterRegistry] = None

def get_global_registry() -> ParameterRegistry:
    """Get the global parameter registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ParameterRegistry()
    return _global_registry

def register_parameter(param: ParameterEntry) -> bool:
    """Convenience function to register a parameter in the global registry."""
    return get_global_registry().register_parameter(param)

def update_parameter(param_id: str, new_value: float, writer: str, reason: str = "") -> Tuple[bool, str]:
    """Convenience function to update a parameter in the global registry."""
    return get_global_registry().update_parameter(param_id, new_value, writer, reason)

def get_parameter_value(param_id: str) -> Optional[float]:
    """Convenience function to get a parameter value from the global registry."""
    param = get_global_registry().get_parameter(param_id)
    return param.current_value if param else None

def get_role_values(role: ParameterRole) -> Dict[str, float]:
    """Convenience function to get values for all parameters in a role."""
    return get_global_registry().get_values_by_role(role)