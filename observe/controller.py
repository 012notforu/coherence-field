from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import time

import numpy as np


ALLOWED_NUDGES = {"T", "alpha", "gamma", "metropolis_temperature", "control_gain", "derivative_gain", "noise_filtering", "filter_strength"}


@dataclass
class SignalSpec:
    """Specification for a sensed signal."""
    name: str
    units: str = ""
    expected_range: Tuple[float, float] = (0.0, 1.0)
    normalization: str = "linear"  # "linear", "log", "none"
    smoothing_window: int = 10
    sampling_rate: Optional[float] = None


@dataclass 
class ParameterSpec:
    """Specification for a controllable parameter."""
    name: str
    role: str  # "diffusion", "accept_temp", "control_gain", etc.
    bounds: Tuple[float, float] = (-1.0, 1.0)
    safety_direction: str = "conservative"  # "conservative", "aggressive", "neutral"
    trust_region_pct: float = 0.1  # Max change per window as % of range
    cooldown_seconds: float = 3.0
    promotion_scope: str = "local"  # "local", "region", "global"


@dataclass
class PolicyRule:
    """Signal-to-parameter mapping rule."""
    name: str
    signal_conditions: Dict[str, Any]  # e.g., {"spectrum": ">2.0", "trend": "increasing"}
    target_role: str  # Role to nudge
    action_type: str  # "increase", "decrease", "target_value"
    magnitude: float = 0.1
    priority: int = 10  # Lower = higher priority
    domain_tags: List[str] = field(default_factory=list)  # ["CartPole", "heat", etc.]
    constraint_requirements: List[str] = field(default_factory=list)  # Must be satisfied first


@dataclass
class NudgeRecord:
    """Record of a parameter nudge with full audit trail."""
    timestamp: float
    parameter: str
    old_value: float
    new_value: float
    reason: str
    signals_used: Dict[str, float]
    constraint_margins: Dict[str, float]
    policy_rule: str


@dataclass
class ControllerDecision:
    nudges: Dict[str, float]
    safe_mode: bool = False


class MetadataDrivenController:
    """Enhanced controller with metadata-driven sensing and policy tables."""
    
    def __init__(self, 
                 signal_registry: Optional[Dict[str, SignalSpec]] = None,
                 parameter_registry: Optional[Dict[str, ParameterSpec]] = None,
                 policy_table: Optional[List[PolicyRule]] = None):
        
        # Initialize registries with defaults
        self.signal_registry = signal_registry or self._default_signal_registry()
        self.parameter_registry = parameter_registry or self._default_parameter_registry()
        self.policy_table = policy_table or self._default_policy_table()
        
        # Signal processing state
        self._signal_emas: Dict[str, float] = {}
        self._signal_trends: Dict[str, List[float]] = {}
        
        # Parameter state tracking
        self._parameter_cooldowns: Dict[str, float] = {}
        self._nudge_history: List[NudgeRecord] = []
        
        # Current parameter values (managed externally, tracked here)
        self._current_parameter_values: Dict[str, float] = {}
        
    def _default_signal_registry(self) -> Dict[str, SignalSpec]:
        """Default signal specifications."""
        return {
            "spectrum_width": SignalSpec("spectrum_width", "units", (0.0, 5.0), "linear", 10),
            "perplexity": SignalSpec("perplexity", "bits", (0.0, 5.0), "linear", 10),
            "horizon": SignalSpec("horizon", "steps", (0.0, 3.0), "linear", 10),
            "energy_drift": SignalSpec("energy_drift", "rate", (-0.1, 0.1), "linear", 5),
            "composition_success_rate": SignalSpec("composition_success_rate", "rate", (0.0, 1.0), "linear", 20),
            "cart_position": SignalSpec("cart_position", "m", (-3.0, 3.0), "linear", 5),
            "pole_angle": SignalSpec("pole_angle", "rad", (-0.3, 0.3), "linear", 5),
            "angular_velocity": SignalSpec("angular_velocity", "rad/s", (-5.0, 5.0), "linear", 5),
            "sensor_noise_level": SignalSpec("sensor_noise_level", "std", (0.0, 0.1), "linear", 10),
            "validator_margin": SignalSpec("validator_margin", "units", (0.0, 0.2), "linear", 15),
        }
    
    def _default_parameter_registry(self) -> Dict[str, ParameterSpec]:
        """Default parameter specifications."""
        return {
            "T": ParameterSpec("T", "diffusion", (-0.1, 0.1), "conservative", 0.1, 3.0, "local"),
            "alpha": ParameterSpec("alpha", "diffusion", (-0.1, 0.1), "conservative", 0.1, 3.0, "local"),
            "gamma": ParameterSpec("gamma", "horizon", (-0.1, 0.1), "conservative", 0.1, 3.0, "local"),
            "metropolis_temperature": ParameterSpec("metropolis_temperature", "accept_temp", (0.1, 3.0), "conservative", 0.2, 5.0, "global"),
            "control_gain": ParameterSpec("control_gain", "control_gain", (0.5, 5.0), "conservative", 0.15, 2.0, "local"),
            "filter_strength": ParameterSpec("filter_strength", "noise_filter", (0.1, 1.0), "conservative", 0.1, 4.0, "local"),
        }
    
    def _default_policy_table(self) -> List[PolicyRule]:
        """Default policy rules - start with legacy behavior."""
        return [
            # Legacy spectrum -> T mapping
            PolicyRule(
                name="spectrum_to_T",
                signal_conditions={"spectrum_width": "threshold_based"},
                target_role="diffusion",
                action_type="proportional_response",
                magnitude=0.05,
                priority=10,
                constraint_requirements=["energy_stable"]
            ),
            # Legacy perplexity -> alpha mapping  
            PolicyRule(
                name="perplexity_to_alpha",
                signal_conditions={"perplexity": "threshold_based"},
                target_role="diffusion", 
                action_type="proportional_response",
                magnitude=0.03,
                priority=10,
                constraint_requirements=["energy_stable"]
            ),
            # Legacy horizon -> gamma mapping
            PolicyRule(
                name="horizon_to_gamma",
                signal_conditions={"horizon": "threshold_based"},
                target_role="horizon",
                action_type="proportional_response", 
                magnitude=0.02,
                priority=10,
                constraint_requirements=["energy_stable"]
            ),
            # New: Composition success -> acceptance parameters
            PolicyRule(
                name="composition_failure_response",
                signal_conditions={"composition_success_rate": "<0.1"},
                target_role="accept_temp",
                action_type="increase",
                magnitude=0.1,
                priority=5,  # Higher priority than performance optimization
                constraint_requirements=[]
            ),
            # New: CartPole angle control
            PolicyRule(
                name="pole_angle_response", 
                signal_conditions={"pole_angle": ">0.15"},
                target_role="control_gain",
                action_type="increase",
                magnitude=0.2,
                priority=3,
                domain_tags=["CartPole"],
                constraint_requirements=["energy_stable"]
            ),
            # New: Sensor noise -> filtering
            PolicyRule(
                name="noise_filtering_response",
                signal_conditions={"sensor_noise_level": ">0.05"},
                target_role="noise_filter", 
                action_type="increase",
                magnitude=0.15,
                priority=8,
                constraint_requirements=[]
            ),
        ]
    
    def step(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Process metrics through policy table and generate parameter nudges."""
        current_time = time.time()
        
        # Update signal EMAs and trends
        self._update_signal_state(metrics)
        
        # Evaluate policy rules
        triggered_rules = self._evaluate_policy_rules(metrics)
        
        # Apply constraint checking (lexicographic)
        constraint_margins = self._check_constraints(metrics)
        
        # Generate parameter adjustments with arbitration
        adjustments = self._generate_parameter_adjustments(
            triggered_rules, constraint_margins, current_time
        )
        
        return adjustments
    
    def _update_signal_state(self, metrics: Dict[str, float]):
        """Update EMA and trend tracking for all signals."""
        for signal_name, value in metrics.items():
            if signal_name in self.signal_registry:
                spec = self.signal_registry[signal_name]
                
                # Update EMA
                if signal_name not in self._signal_emas:
                    self._signal_emas[signal_name] = value
                else:
                    # Use fixed alpha for now (could be spec-based)
                    alpha = 0.1
                    self._signal_emas[signal_name] = (
                        alpha * value + (1 - alpha) * self._signal_emas[signal_name]
                    )
                
                # Update trend tracking
                if signal_name not in self._signal_trends:
                    self._signal_trends[signal_name] = []
                self._signal_trends[signal_name].append(value)
                # Keep only recent values for trend analysis
                if len(self._signal_trends[signal_name]) > spec.smoothing_window:
                    self._signal_trends[signal_name].pop(0)
    
    def _evaluate_policy_rules(self, metrics: Dict[str, float]) -> List[PolicyRule]:
        """Evaluate which policy rules are triggered by current metrics."""
        triggered = []
        
        for rule in self.policy_table:
            if self._rule_triggered(rule, metrics):
                triggered.append(rule)
        
        # Sort by priority (lower number = higher priority)
        triggered.sort(key=lambda r: r.priority)
        return triggered
    
    def _rule_triggered(self, rule: PolicyRule, metrics: Dict[str, float]) -> bool:
        """Check if a policy rule is triggered by current conditions."""
        for signal_name, condition in rule.signal_conditions.items():
            if signal_name not in metrics:
                continue
                
            value = metrics[signal_name]
            ema_value = self._signal_emas.get(signal_name, value)
            
            if condition == "threshold_based":
                # Legacy threshold logic for spectrum/perplexity/horizon
                if signal_name in ["spectrum_width", "perplexity"]:
                    if ema_value > 2.0:  # surprise_threshold equivalent
                        return True
                elif signal_name == "horizon":
                    if ema_value < 2.0:  # surprise_threshold equivalent
                        return True
            elif isinstance(condition, str) and condition.startswith(">"):
                threshold = float(condition[1:])
                if value > threshold:
                    return True
            elif isinstance(condition, str) and condition.startswith("<"):
                threshold = float(condition[1:])
                if value < threshold:
                    return True
                    
        return False
    
    def _check_constraints(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Check constraint satisfaction and compute margins."""
        margins = {}
        
        # Energy stability constraint
        energy_drift = abs(metrics.get("energy_drift", 0.0))
        margins["energy_stable"] = max(0.0, 0.01 - energy_drift)  # >0 means satisfied
        
        # Composition success constraint  
        comp_success = metrics.get("composition_success_rate", 0.0)
        margins["composition_viable"] = max(0.0, comp_success - 0.05)  # >0 means satisfied
        
        # Validator margin constraint
        validator_margin = metrics.get("validator_margin", 0.1)
        margins["validator_safe"] = max(0.0, validator_margin - 0.02)  # >0 means satisfied
        
        return margins
    
    def _generate_parameter_adjustments(self, 
                                      triggered_rules: List[PolicyRule],
                                      constraint_margins: Dict[str, float], 
                                      current_time: float) -> Dict[str, float]:
        """Generate parameter adjustments with arbitration and constraints."""
        adjustments = {}
        
        for rule in triggered_rules:
            # Check constraint requirements
            if rule.constraint_requirements:
                constraints_satisfied = all(
                    constraint_margins.get(req, 0.0) > 0.0 
                    for req in rule.constraint_requirements
                )
                if not constraints_satisfied:
                    continue
            
            # Find parameters that match the target role
            target_params = [
                name for name, spec in self.parameter_registry.items()
                if spec.role == rule.target_role
            ]
            
            for param_name in target_params:
                # Check cooldown
                if self._parameter_on_cooldown(param_name, current_time):
                    continue
                
                # Compute adjustment based on action type
                adjustment = self._compute_adjustment(rule, param_name)
                
                if param_name in adjustments:
                    # Combine multiple adjustments (take minimum magnitude for safety)
                    if abs(adjustment) < abs(adjustments[param_name]):
                        adjustments[param_name] = adjustment
                else:
                    adjustments[param_name] = adjustment
                
                # Record the nudge for auditing
                self._record_nudge(param_name, adjustment, rule.name, current_time)
        
        return adjustments
    
    def _parameter_on_cooldown(self, param_name: str, current_time: float) -> bool:
        """Check if parameter is in cooldown period."""
        if param_name not in self._parameter_cooldowns:
            return False
        
        last_nudge = self._parameter_cooldowns[param_name]
        spec = self.parameter_registry[param_name]
        return (current_time - last_nudge) < spec.cooldown_seconds
    
    def _compute_adjustment(self, rule: PolicyRule, param_name: str) -> float:
        """Compute parameter adjustment based on rule and parameter spec."""
        spec = self.parameter_registry[param_name]
        
        if rule.action_type == "increase":
            base_adjustment = rule.magnitude
        elif rule.action_type == "decrease":
            base_adjustment = -rule.magnitude
        elif rule.action_type == "proportional_response":
            # Legacy behavior for spectrum/perplexity/horizon rules
            base_adjustment = rule.magnitude  # Already computed with sign
        else:
            base_adjustment = 0.0
        
        # Apply trust region constraints
        range_span = spec.bounds[1] - spec.bounds[0]
        max_change = spec.trust_region_pct * range_span
        
        # Clamp to trust region
        adjustment = np.clip(base_adjustment, -max_change, max_change)
        
        return float(adjustment)
    
    def _record_nudge(self, param_name: str, adjustment: float, rule_name: str, timestamp: float):
        """Record parameter nudge for audit trail."""
        current_value = self._current_parameter_values.get(param_name, 0.0)
        new_value = current_value + adjustment
        
        record = NudgeRecord(
            timestamp=timestamp,
            parameter=param_name,
            old_value=current_value,
            new_value=new_value,
            reason=rule_name,
            signals_used={},  # Would populate from rule evaluation
            constraint_margins={},  # Would populate from constraint check
            policy_rule=rule_name
        )
        
        self._nudge_history.append(record)
        self._parameter_cooldowns[param_name] = timestamp
        self._current_parameter_values[param_name] = new_value
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of metadata-driven controller."""
        return {
            "signal_registry_size": len(self.signal_registry),
            "parameter_registry_size": len(self.parameter_registry),
            "policy_rules_count": len(self.policy_table),
            "active_signals": len(self._signal_emas),
            "recent_nudges": len([r for r in self._nudge_history if time.time() - r.timestamp < 60]),
            "parameters_on_cooldown": len([
                name for name, last_time in self._parameter_cooldowns.items()
                if time.time() - last_time < self.parameter_registry[name].cooldown_seconds
            ]),
            "signal_emas": self._signal_emas.copy(),
            "current_parameter_values": self._current_parameter_values.copy()
        }


@dataclass
class GentleController:
    max_step: float = 0.1
    ema_tau: float = 100.0
    surprise_threshold: float = 2.0
    gains: Dict[str, float] = field(default_factory=lambda: {
        "spectrum": 0.05,
        "perplexity": 0.03,
        "horizon": 0.02,
    })

    def __post_init__(self) -> None:
        self._ema: Dict[str, float] = {}
        # Initialize metadata-driven backend
        self._metadata_controller = MetadataDrivenController()

    def _update_ema(self, key: str, value: float) -> float:
        prev = self._ema.get(key, value)
        decay = np.exp(-1.0 / max(self.ema_tau, 1.0))
        updated = decay * prev + (1.0 - decay) * value
        self._ema[key] = updated
        return updated

    def _clamp(self, value: float) -> float:
        return float(np.clip(value, -self.max_step, self.max_step))

    def step(self, metrics: Dict[str, float]) -> ControllerDecision:
        # Phase A: Preserve legacy behavior but route through metadata system
        spectrum = self._update_ema("spectrum", metrics.get("spectrum_width", 0.0))
        perplexity = self._update_ema("perplexity", metrics.get("perplexity", 0.0))
        horizon = self._update_ema("horizon", metrics.get("horizon", 0.0))
        drift = metrics.get("energy_drift", 0.0)
        
        # Original adjustments 
        legacy_adjustments = {
            "T": self._clamp(-self.gains["spectrum"] * (spectrum - self.surprise_threshold)),
            "alpha": self._clamp(-self.gains["perplexity"] * (perplexity - self.surprise_threshold)),
            "gamma": self._clamp(self.gains["horizon"] * (self.surprise_threshold - horizon)),
        }
        
        # Route through metadata controller for additional sensing
        enhanced_metrics = {
            **metrics,
            "spectrum_width": spectrum,  # Use EMA'd values
            "perplexity": perplexity,
            "horizon": horizon,
            "energy_drift": drift
        }
        
        metadata_adjustments = self._metadata_controller.step(enhanced_metrics)
        
        # Combine legacy and metadata-driven adjustments (legacy takes precedence for now)
        final_adjustments = {**metadata_adjustments}
        final_adjustments.update(legacy_adjustments)  # Legacy overrides
        
        self._validate_nudges(final_adjustments)
        
        # Safety override
        safe_mode = False
        if abs(drift) > 0.01:
            safe_mode = True
            final_adjustments = {key: 0.0 for key in ALLOWED_NUDGES}
        
        return ControllerDecision(nudges=final_adjustments, safe_mode=safe_mode)
    
    def get_sensing_status(self) -> Dict[str, Any]:
        """Get comprehensive status of sensing and policy system."""
        return {
            "legacy_status": {
                "ema_values": self._ema.copy(),
                "surprise_threshold": self.surprise_threshold,
                "gains": self.gains.copy()
            },
            "metadata_status": self._metadata_controller.get_status()
        }

    def _validate_nudges(self, nudges: Dict[str, float]) -> None:
        unknown = set(nudges) - ALLOWED_NUDGES
        if unknown:
            raise ValueError(f"Unsupported nudge parameters: {sorted(unknown)}")
