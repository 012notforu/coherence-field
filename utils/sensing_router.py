"""
Sensing Router - Role-Aware Signal Processing Engine

This module generalizes the GentleController into a comprehensive signal processing
engine that routes normalized metrics to parameter roles using the unified registry.

Key Features:
1. Automatic signal discovery and normalization with proper unit handling
2. Role-aware routing (not hardcoded parameter names) 
3. Lexicographic constraint handling (safety first)
4. EMA smoothing and two-sided CUSUM change detection
5. Policy-based decision making with priority arbitration
6. Integration with parameter registry for role-level updates
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import numpy as np
import logging
from collections import deque

from utils.parameter_registry import get_global_registry, ParameterRegistry, ParameterRole

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of signals the sensing system can process."""
    PHYSICS_ENERGY = "physics_energy"
    PHYSICS_GRADIENT = "physics_gradient"
    PHYSICS_SPECTRAL = "physics_spectral"
    PERFORMANCE_QUALITY = "performance_quality"
    CONTROL_STATE = "control_state"
    CONTROL_NOISE = "control_noise"
    SYSTEM_HEALTH = "system_health"
    VALIDATION_MARGIN = "validation_margin"

class ConstraintPriority(Enum):
    """Priority levels for constraint handling (lexicographic order)."""
    SAFETY = 1          # Energy stability, bounds ordering
    COMPOSITION = 2     # Vector composition viability
    PERFORMANCE = 3     # Quality metrics, convergence
    EFFICIENCY = 4      # Speed, resource usage

class EvalSpace(Enum):
    """Evaluation space for rule conditions."""
    RAW = "raw"         # Raw signal units
    NORMALIZED = "normalized"  # Min-max normalized [0,1]

@dataclass
class SignalSpec:
    """Specification for a signal channel."""
    signal_id: str
    signal_type: SignalType
    units: str
    expected_range: Tuple[float, float]  # Normal operating range (for normalization)
    physical_bounds: Tuple[float, float]  # Hard physical limits
    smoothing_alpha: float = 0.1
    cusum_sensitivity: float = 0.2
    roles_interested: List[ParameterRole] = field(default_factory=list)
    description: str = ""

@dataclass
class PolicyRule:
    """Decision rule mapping signals to role-level actions."""
    rule_id: str
    priority: int
    constraint_level: ConstraintPriority
    condition: str  # "signal_name > threshold"
    eval_space: EvalSpace
    target_role: ParameterRole
    action_magnitude: float  # Base adjustment (fraction)
    action_direction: str   # "increase", "decrease", "adaptive"
    cooldown_seconds: float = 5.0
    confidence_threshold: float = 0.7
    description: str = ""

@dataclass
class SensingDecision:
    """Result of sensing system decision making."""
    timestamp: float
    triggered_rules: List[str]
    role_adjustments: Dict[ParameterRole, float]
    constraint_violations: List[str]
    confidence_scores: Dict[str, float]
    signal_states: Dict[str, Dict[str, Any]]  # signal -> {"raw": val, "norm": val, "clamped": bool}
    rule_trace: List[Dict[str, Any]]
    reason: str
    safe_mode: bool = False

class SensingRouter:
    """Role-aware signal processing and decision engine."""
    
    def __init__(self, registry: Optional[ParameterRegistry] = None):
        self.registry = registry or get_global_registry()
        
        # Signal processing state
        self.signal_specs: Dict[str, SignalSpec] = {}
        self.signal_history: Dict[str, deque] = {}
        self.signal_ema: Dict[str, float] = {}
        self.signal_cusum_pos: Dict[str, float] = {}
        self.signal_cusum_neg: Dict[str, float] = {}
        self.signal_raw: Dict[str, float] = {}
        self.signal_normalized: Dict[str, float] = {}
        self.signal_clamp_count: Dict[str, int] = {}
        
        # Decision making state
        self.policy_rules: Dict[str, PolicyRule] = {}
        self.last_rule_action: Dict[str, float] = {}
        self.last_role_action: Dict[ParameterRole, float] = {}
        self.constraint_margins: Dict[str, float] = {}
        
        # Role-specific cooldowns
        self.role_cooldowns = {
            ParameterRole.ACCEPTANCE: 15.0,
            ParameterRole.STABILITY: 8.0,
            ParameterRole.CONTROL_GAIN: 1.5,
            ParameterRole.DIFFUSION: 5.0,
            ParameterRole.FILTER: 3.0,
            ParameterRole.THRESHOLD: 4.0,
            ParameterRole.COMPOSITION: 10.0
        }
        
        # System state
        self.safe_mode = False
        self.last_decision_time = 0.0
        
        # Initialize with standard signal catalog and policy rules
        self._initialize_signal_catalog()
        self._initialize_policy_rules()
    
    def _initialize_signal_catalog(self):
        """Initialize standard signal specifications."""
        
        # Physics signals
        self.register_signal(SignalSpec(
            signal_id="energy_drift",
            signal_type=SignalType.PHYSICS_ENERGY,
            units="energy/time",
            expected_range=(-0.02, 0.02),
            physical_bounds=(-0.5, 0.5),
            smoothing_alpha=0.2,
            cusum_sensitivity=0.3,
            roles_interested=[ParameterRole.STABILITY, ParameterRole.ACCEPTANCE],
            description="Rate of total energy change in system"
        ))
        
        self.register_signal(SignalSpec(
            signal_id="composition_success_rate",
            signal_type=SignalType.SYSTEM_HEALTH,
            units="rate",
            expected_range=(0.1, 0.6),
            physical_bounds=(0.0, 1.0),
            smoothing_alpha=0.08,
            cusum_sensitivity=0.25,
            roles_interested=[ParameterRole.ACCEPTANCE, ParameterRole.COMPOSITION],
            description="Vector composition success rate"
        ))
        
        self.register_signal(SignalSpec(
            signal_id="acceptance_margin",
            signal_type=SignalType.VALIDATION_MARGIN,
            units="probability",
            expected_range=(0.2, 0.8),
            physical_bounds=(0.0, 1.0),
            smoothing_alpha=0.1,
            cusum_sensitivity=0.2,
            roles_interested=[ParameterRole.ACCEPTANCE],
            description="Safety margin in acceptance probability window"
        ))
        
        # SCFD feature signals (the 3 core features)
        self.register_signal(SignalSpec(
            signal_id="mean_coherence",
            signal_type=SignalType.PHYSICS_ENERGY,
            units="fraction",
            expected_range=(0.2, 0.8),
            physical_bounds=(0.0, 1.0),
            smoothing_alpha=0.1,
            cusum_sensitivity=0.2,
            roles_interested=[ParameterRole.DIFFUSION],  # Maps to α influence
            description="SCFD mean coherence (domain formation)"
        ))
        
        self.register_signal(SignalSpec(
            signal_id="mean_abs_curvature",
            signal_type=SignalType.PHYSICS_GRADIENT,
            units="curvature",
            expected_range=(0.1, 0.5),
            physical_bounds=(0.0, 2.0),
            smoothing_alpha=0.1,
            cusum_sensitivity=0.2,
            roles_interested=[ParameterRole.SMOOTHING],  # Maps to β influence
            description="SCFD mean absolute curvature (interface smoothness)"
        ))
        
        self.register_signal(SignalSpec(
            signal_id="mean_cross_gradient",
            signal_type=SignalType.PHYSICS_SPECTRAL,
            units="gradient_product",
            expected_range=(-0.2, 0.2),
            physical_bounds=(-1.0, 1.0),
            smoothing_alpha=0.1,
            cusum_sensitivity=0.2,
            roles_interested=[ParameterRole.COUPLING],  # Maps to γ influence
            description="SCFD mean cross-gradient coupling (information flow)"
        ))
        
        logger.info(f"Initialized signal catalog with {len(self.signal_specs)} signals")
    
    def _initialize_policy_rules(self):
        """Initialize standard policy rules with proper unit handling."""
        
        # Safety-first rules (RAW space for physical thresholds)
        self.register_policy_rule(PolicyRule(
            rule_id="energy_stability_protection",
            priority=100,
            constraint_level=ConstraintPriority.SAFETY,
            condition="energy_drift > 0.05",
            eval_space=EvalSpace.RAW,
            target_role=ParameterRole.STABILITY,
            action_magnitude=0.1,
            action_direction="increase",
            cooldown_seconds=10.0,
            confidence_threshold=0.8,
            description="Increase stability when energy drift exceeds 5%"
        ))
        
        self.register_policy_rule(PolicyRule(
            rule_id="acceptance_bounds_protection",
            priority=95,
            constraint_level=ConstraintPriority.SAFETY,
            condition="acceptance_margin < 0.15",
            eval_space=EvalSpace.RAW,
            target_role=ParameterRole.ACCEPTANCE,
            action_magnitude=0.05,
            action_direction="adaptive",
            cooldown_seconds=15.0,
            confidence_threshold=0.9,
            description="Adjust acceptance when margin below 15%"
        ))
        
        # Composition viability rules
        self.register_policy_rule(PolicyRule(
            rule_id="composition_recovery",
            priority=80,
            constraint_level=ConstraintPriority.COMPOSITION,
            condition="composition_success_rate < 0.15",
            eval_space=EvalSpace.RAW,
            target_role=ParameterRole.ACCEPTANCE,
            action_magnitude=0.15,
            action_direction="increase",
            cooldown_seconds=8.0,
            confidence_threshold=0.7,
            description="Increase acceptance when composition success below 15%"
        ))
        
        # SCFD-specific policy rules
        self.register_policy_rule(PolicyRule(
            rule_id="scfd_low_coherence_response",
            priority=50,
            constraint_level=ConstraintPriority.PERFORMANCE,
            condition="mean_coherence < 0.3",
            eval_space=EvalSpace.RAW,
            target_role=ParameterRole.DIFFUSION,  # Increase α to favor domain formation
            action_magnitude=0.05,
            action_direction="increase",
            cooldown_seconds=5.0,
            confidence_threshold=0.7,
            description="Increase α when coherence is low to promote domain formation"
        ))
        
        self.register_policy_rule(PolicyRule(
            rule_id="scfd_high_curvature_response",
            priority=50,
            constraint_level=ConstraintPriority.PERFORMANCE,
            condition="mean_abs_curvature > 0.4",
            eval_space=EvalSpace.RAW,
            target_role=ParameterRole.SMOOTHING,  # Increase β to smooth interfaces
            action_magnitude=0.05,
            action_direction="increase",
            cooldown_seconds=5.0,
            confidence_threshold=0.7,
            description="Increase β when curvature is high to smooth interfaces"
        ))
        
        self.register_policy_rule(PolicyRule(
            rule_id="scfd_cross_gradient_modulation",
            priority=40,
            constraint_level=ConstraintPriority.PERFORMANCE,
            condition="mean_cross_gradient > 0.15",
            eval_space=EvalSpace.RAW,
            target_role=ParameterRole.COUPLING,  # Adjust γ for information flow
            action_magnitude=0.03,
            action_direction="adaptive",  # Can go either direction
            cooldown_seconds=7.0,
            confidence_threshold=0.6,
            description="Modulate γ based on cross-gradient coupling strength"
        ))
        
        logger.info(f"Initialized policy rules with {len(self.policy_rules)} rules")
    
    def register_signal(self, spec: SignalSpec):
        """Register a new signal specification."""
        self.signal_specs[spec.signal_id] = spec
        self.signal_history[spec.signal_id] = deque(maxlen=50)
        self.signal_ema[spec.signal_id] = 0.0
        self.signal_cusum_pos[spec.signal_id] = 0.0
        self.signal_cusum_neg[spec.signal_id] = 0.0
        self.signal_raw[spec.signal_id] = 0.0
        self.signal_normalized[spec.signal_id] = 0.0
        self.signal_clamp_count[spec.signal_id] = 0
        logger.info(f"Registered signal: {spec.signal_id}")
    
    def register_policy_rule(self, rule: PolicyRule):
        """Register a new policy rule."""
        self.policy_rules[rule.rule_id] = rule
        self.last_rule_action[rule.rule_id] = 0.0
        logger.info(f"Registered policy rule: {rule.rule_id} (priority {rule.priority}, {rule.eval_space.value} space)")
    
    def process_signals(self, raw_signals: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Process raw signals into smoothed and normalized values."""
        current_time = time.monotonic()  # Use monotonic time for cooldowns
        signal_states = {}
        
        for signal_id, raw_value in raw_signals.items():
            if signal_id not in self.signal_specs:
                logger.warning(f"Unknown signal {signal_id}, skipping")
                continue
            
            spec = self.signal_specs[signal_id]
            
            # Check for clamping and clamp to physical bounds
            clamped = raw_value < spec.physical_bounds[0] or raw_value > spec.physical_bounds[1]
            if clamped:
                self.signal_clamp_count[signal_id] += 1
            
            clamped_value = max(spec.physical_bounds[0], min(spec.physical_bounds[1], raw_value))
            
            # Add to history
            self.signal_history[signal_id].append((current_time, clamped_value))
            
            # Update EMA
            if len(self.signal_history[signal_id]) == 1:
                self.signal_ema[signal_id] = clamped_value
            else:
                alpha = spec.smoothing_alpha
                self.signal_ema[signal_id] = alpha * clamped_value + (1 - alpha) * self.signal_ema[signal_id]
            
            # Update two-sided CUSUM for change detection
            smoothed = self.signal_ema[signal_id]
            expected_center = (spec.expected_range[0] + spec.expected_range[1]) / 2
            range_width = spec.expected_range[1] - spec.expected_range[0]
            k = spec.cusum_sensitivity * range_width
            
            deviation = smoothed - expected_center
            self.signal_cusum_pos[signal_id] = max(0, self.signal_cusum_pos[signal_id] + deviation - k)
            self.signal_cusum_neg[signal_id] = max(0, self.signal_cusum_neg[signal_id] - deviation - k)
            
            # Store raw and normalized values
            self.signal_raw[signal_id] = smoothed
            self.signal_normalized[signal_id] = self._normalize_signal(signal_id, smoothed)
            
            # Package both spaces for decision making
            signal_states[signal_id] = {
                "raw": smoothed,
                "normalized": self.signal_normalized[signal_id],
                "cusum_pos": self.signal_cusum_pos[signal_id],
                "cusum_neg": self.signal_cusum_neg[signal_id],
                "clamped": clamped,
                "clamp_count": self.signal_clamp_count[signal_id]
            }
        
        return signal_states
    
    def _normalize_signal(self, signal_id: str, value: float) -> float:
        """Normalize signal value based on its specification."""
        spec = self.signal_specs[signal_id]
        min_val, max_val = spec.expected_range
        
        if max_val > min_val:
            return (value - min_val) / (max_val - min_val)
        else:
            return 0.5
    
    def evaluate_policies(self, signal_states: Dict[str, Dict[str, Any]]) -> SensingDecision:
        """Evaluate policy rules with lexicographic safety and proper constraint gating."""
        current_time = time.monotonic()
        
        # Update constraint margins in RAW units
        self._update_constraint_margins(signal_states)
        
        # Check if we should enter safe mode
        safety_margins = {k: v for k, v in self.constraint_margins.items() if "safety" in k}
        safe_mode = any(margin < 0.05 for margin in safety_margins.values())
        
        # Group rules by constraint level for lexicographic processing
        rules_by_level = {}
        for rule in self.policy_rules.values():
            level = rule.constraint_level
            if level not in rules_by_level:
                rules_by_level[level] = []
            rules_by_level[level].append(rule)
        
        # Sort each level by priority
        for level in rules_by_level:
            rules_by_level[level].sort(key=lambda r: -r.priority)
        
        # Process levels in lexicographic order
        triggered_rules = []
        role_adjustments: Dict[ParameterRole, float] = {}
        confidence_scores = {}
        rule_trace = []
        
        for level in sorted(rules_by_level.keys(), key=lambda x: x.value):
            for rule in rules_by_level[level]:
                # Check rule cooldown
                if current_time - self.last_rule_action[rule.rule_id] < rule.cooldown_seconds:
                    continue
                
                # Check role cooldown  
                role_cooldown = self.role_cooldowns.get(rule.target_role, 3.0)
                if rule.target_role in self.last_role_action:
                    if current_time - self.last_role_action[rule.target_role] < role_cooldown:
                        continue
                
                # Lexicographic constraint gating
                if not self._passes_constraint_gate(rule, safe_mode):
                    rule_trace.append({
                        "rule_id": rule.rule_id,
                        "status": "blocked_by_constraints",
                        "constraint_level": rule.constraint_level.value,
                        "safe_mode": safe_mode
                    })
                    continue
                
                # Evaluate condition in the specified space
                condition_met, confidence = self._evaluate_condition(rule, signal_states)
                confidence_scores[rule.rule_id] = confidence
                
                rule_trace.append({
                    "rule_id": rule.rule_id,
                    "condition": rule.condition,
                    "eval_space": rule.eval_space.value,
                    "condition_met": condition_met,
                    "confidence": confidence,
                    "threshold": rule.confidence_threshold,
                    "constraint_level": rule.constraint_level.value
                })
                
                if condition_met and confidence >= rule.confidence_threshold:
                    # Calculate adjustment magnitude
                    adjustment = self._calculate_adjustment(rule, signal_states, confidence)
                    
                    # Lexicographic role arbitration: higher constraint levels override lower ones
                    if rule.target_role in role_adjustments:
                        # Check if this rule has higher or equal constraint level
                        existing_level = None
                        for existing_rule_id in triggered_rules:
                            existing_rule = self.policy_rules[existing_rule_id]
                            if existing_rule.target_role == rule.target_role:
                                existing_level = existing_rule.constraint_level
                                break
                        
                        if existing_level and rule.constraint_level.value <= existing_level.value:
                            # Higher or equal constraint level - override
                            role_adjustments[rule.target_role] = adjustment
                            rule_trace[-1]["arbitration"] = f"override_level_{existing_level.value}"
                        else:
                            # Lower constraint level - skip
                            rule_trace[-1]["arbitration"] = "blocked_by_higher_level"
                            continue
                    else:
                        role_adjustments[rule.target_role] = adjustment
                        rule_trace[-1]["arbitration"] = "first"
                    
                    triggered_rules.append(rule.rule_id)
                    self.last_rule_action[rule.rule_id] = current_time
                    self.last_role_action[rule.target_role] = current_time
                    
                    rule_trace[-1]["adjustment"] = adjustment
                    rule_trace[-1]["status"] = "applied"
                    
                    logger.info(f"Rule triggered: {rule.rule_id} [{rule.eval_space.value}] -> "
                               f"{rule.target_role.value} adjustment {adjustment:+.4f} "
                               f"(confidence {confidence:.3f})")
        
        # Check for constraint violations
        constraint_violations = self._check_constraint_violations()
        
        # Generate explanation
        reason_parts = []
        if triggered_rules:
            rule_summary = {}
            for rule_id in triggered_rules:
                rule = self.policy_rules[rule_id]
                level = rule.constraint_level.value
                rule_summary[level] = rule_summary.get(level, 0) + 1
            reason_parts.append(f"Triggered {len(triggered_rules)} rules: {rule_summary}")
        
        if constraint_violations:
            reason_parts.append(f"{len(constraint_violations)} constraints violated")
        if safe_mode:
            reason_parts.append("SAFE MODE active")
        
        reason = "; ".join(reason_parts) if reason_parts else "No actions needed"
        
        decision = SensingDecision(
            timestamp=current_time,
            triggered_rules=triggered_rules,
            role_adjustments=role_adjustments,
            constraint_violations=constraint_violations,
            confidence_scores=confidence_scores,
            signal_states=signal_states.copy(),
            rule_trace=rule_trace,
            reason=reason,
            safe_mode=safe_mode
        )
        
        self.last_decision_time = current_time
        self.safe_mode = safe_mode
        return decision
    
    def _passes_constraint_gate(self, rule: PolicyRule, safe_mode: bool) -> bool:
        """Lexicographic constraint gating - safety first."""
        
        # In safe mode, only allow SAFETY and high-priority COMPOSITION rules
        if safe_mode:
            if rule.constraint_level == ConstraintPriority.SAFETY:
                return True
            elif rule.constraint_level == ConstraintPriority.COMPOSITION and rule.priority >= 80:
                return True
            else:
                return False
        
        # Check safety margins for all non-safety rules
        if rule.constraint_level != ConstraintPriority.SAFETY:
            safety_margin = self.constraint_margins.get("energy_safety", 1.0)
            if safety_margin < 0.1:
                return False
            
            bounds_margin = self.constraint_margins.get("bounds_safety", 1.0)
            if bounds_margin < 0.1:
                return False
        
        # Check composition margins for performance/efficiency rules
        if rule.constraint_level.value >= ConstraintPriority.PERFORMANCE.value:
            comp_margin = self.constraint_margins.get("composition_viable", 1.0)
            if comp_margin < 0.2:
                return False
        
        return True
    
    def _evaluate_condition(self, rule: PolicyRule, signal_states: Dict[str, Dict[str, Any]]) -> Tuple[bool, float]:
        """Evaluate a rule condition in the specified space with improved confidence scaling."""
        try:
            # Parse condition: "signal_name > threshold"
            parts = rule.condition.strip().split()
            if len(parts) != 3:
                return False, 0.0
            
            signal_name, operator, threshold_str = parts
            threshold = float(threshold_str)
            
            if signal_name not in signal_states:
                return False, 0.0
            
            # Get value in the specified evaluation space
            space_key = rule.eval_space.value
            if space_key not in signal_states[signal_name]:
                return False, 0.0
            
            signal_value = signal_states[signal_name][space_key]
            
            # Evaluate condition
            if operator == ">":
                condition_met = signal_value > threshold
                distance = signal_value - threshold
            elif operator == "<":
                condition_met = signal_value < threshold
                distance = threshold - signal_value
            else:
                return False, 0.0
            
            # Improved confidence scaling
            if condition_met:
                if rule.eval_space == EvalSpace.NORMALIZED:
                    # For normalized space, use fixed slope based on reasonable deviation
                    confidence = min(1.0, distance / 0.2)  # 20% normalized range for full confidence
                else:  # RAW space
                    # Scale by the signal's expected range width
                    spec = self.signal_specs.get(signal_name)
                    if spec:
                        range_width = spec.expected_range[1] - spec.expected_range[0]
                        confidence = min(1.0, distance / (0.1 * range_width))
                    else:
                        # Fallback for unknown signals
                        confidence = min(1.0, distance / max(0.01, 0.1 * abs(threshold)))
            else:
                confidence = 0.0
            
            confidence = max(0.0, confidence)
            return condition_met, confidence
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{rule.condition}': {e}")
            return False, 0.0
    
    def _calculate_adjustment(self, rule: PolicyRule, signal_states: Dict[str, Dict[str, Any]], confidence: float) -> float:
        """Calculate adjustment magnitude based on rule and current conditions."""
        base_magnitude = rule.action_magnitude
        scaled_magnitude = base_magnitude * confidence
        
        if rule.action_direction == "increase":
            return scaled_magnitude
        elif rule.action_direction == "decrease":
            return -scaled_magnitude
        elif rule.action_direction == "adaptive":
            # Determine direction based on signal deviation from center
            condition_parts = rule.condition.split()
            if len(condition_parts) >= 3:
                signal_name = condition_parts[0]
                if signal_name in signal_states:
                    signal_val = signal_states[signal_name]["normalized"]
                    # If signal is above center (0.5), decrease; if below, increase
                    if signal_val > 0.5:
                        return -scaled_magnitude
                    else:
                        return scaled_magnitude
            return scaled_magnitude  # Default to increase
        
        return 0.0
    
    def _update_constraint_margins(self, signal_states: Dict[str, Dict[str, Any]]):
        """Update constraint safety margins in raw units."""
        
        # Energy stability margin (raw units)
        if "energy_drift" in signal_states:
            current_drift = abs(signal_states["energy_drift"]["raw"])
            drift_threshold = 0.05
            self.constraint_margins["energy_safety"] = max(0.0, 1.0 - current_drift / drift_threshold)
        
        # Acceptance bounds safety margin
        if "acceptance_margin" in signal_states:
            margin_val = signal_states["acceptance_margin"]["raw"]
            margin_threshold = 0.15
            self.constraint_margins["bounds_safety"] = max(0.0, margin_val / margin_threshold)
        
        # Composition viability margin
        if "composition_success_rate" in signal_states:
            success_rate = signal_states["composition_success_rate"]["raw"]
            min_viable_rate = 0.1
            self.constraint_margins["composition_viable"] = max(0.0, success_rate / min_viable_rate)
        
        # Clamp all margins to [0, 1]
        for key in self.constraint_margins:
            self.constraint_margins[key] = max(0.0, min(1.0, self.constraint_margins[key]))
    
    def _check_constraint_violations(self) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        for constraint, margin in self.constraint_margins.items():
            if margin < 0.02:
                violations.append(f"{constraint}_critical")
            elif margin < 0.1:
                violations.append(f"{constraint}_warning")
        
        return violations
    
    def apply_role_adjustments(self, decision: SensingDecision, writer: str = "sensing_router") -> Dict[str, Dict[str, Any]]:
        """Apply role-level adjustments with detailed result tracking."""
        results = {}
        
        for role, adjustment_fraction in decision.role_adjustments.items():
            # Apply through registry
            role_results = self.registry.update_role(role, adjustment_fraction, writer, decision.reason)
            
            # Track detailed results
            successes = sum(1 for success, _ in role_results.values() if success)
            failures = {param_id: message for param_id, (success, message) in role_results.items() if not success}
            
            results[role.value] = {
                "success": successes > 0,
                "params_updated": successes,
                "params_failed": len(failures),
                "failure_reasons": failures
            }
            
            if successes > 0:
                adjustment_pct = adjustment_fraction * 100
                logger.info(f"Applied {adjustment_pct:+.1f}% adjustment to role {role.value} "
                           f"({successes} params updated)")
            
            if failures:
                logger.warning(f"Role {role.value} had {len(failures)} update failures: {list(failures.values())}")
        
        return results
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get sensing router diagnostics."""
        current_time = time.monotonic()
        
        # Fixed signal health iteration
        signal_summary = {}
        for signal_id in self.signal_specs.keys():
            signal_summary[signal_id] = {
                "raw": self.signal_raw.get(signal_id, 0.0),
                "normalized": self.signal_normalized.get(signal_id, 0.0),
                "cusum_pos": self.signal_cusum_pos.get(signal_id, 0.0),
                "cusum_neg": self.signal_cusum_neg.get(signal_id, 0.0),
                "cusum_activity": max(self.signal_cusum_pos.get(signal_id, 0.0),
                                    self.signal_cusum_neg.get(signal_id, 0.0)),
                "clamp_count": self.signal_clamp_count.get(signal_id, 0),
                "history_length": len(self.signal_history.get(signal_id, []))
            }
        
        # Rule activity (last minute)
        recent_rules = sum(1 for last_time in self.last_rule_action.values() 
                          if current_time - last_time < 60.0)
        
        # Role activity
        role_activity = {}
        for role, last_time in self.last_role_action.items():
            role_activity[role.value] = {
                "last_update_age": current_time - last_time,
                "cooldown_remaining": max(0, self.role_cooldowns.get(role, 3.0) - (current_time - last_time))
            }
        
        return {
            "signals_registered": len(self.signal_specs),
            "signals_active": len(signal_summary),
            "policy_rules": len(self.policy_rules),
            "rules_triggered_last_minute": recent_rules,
            "constraint_margins": self.constraint_margins.copy(),
            "safe_mode": self.safe_mode,
            "last_decision_age": current_time - self.last_decision_time,
            "signal_summary": signal_summary,
            "role_activity": role_activity
        }

# Global sensing router instance
_global_sensing_router: Optional[SensingRouter] = None

def get_global_sensing_router() -> SensingRouter:
    """Get the global sensing router instance."""
    global _global_sensing_router
    if _global_sensing_router is None:
        _global_sensing_router = SensingRouter()
    return _global_sensing_router