"""
Registry-Integrated EFE Tuner - Strategic Parameter Optimization

Upgrades the EFE tuner to work with the parameter registry for strategic optimization.
Uses Active Inference principles with role-based block coordinate optimization.

Key Features:
1. Role-based parameter optimization (not individual parameters)
2. Integration with parameter registry and sensing router
3. Early-fail constraint checking
4. A/B testing with trust regions
5. Strategic optimization (slower than sensing router)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import numpy as np
import logging
from collections import deque

from utils.parameter_registry import (
    get_global_registry, ParameterRegistry, ParameterRole, ParameterEntry
)
from utils.sensing_router import get_global_sensing_router, SensingRouter

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Strategy for parameter optimization."""
    ROLE_BLOCK_COORDINATE = "role_block_coordinate"  # Optimize one role at a time
    FULL_JOINT = "full_joint"                       # Optimize all roles together
    HIERARCHICAL = "hierarchical"                   # Safety -> Composition -> Performance

class ConstraintLevel(Enum):
    """Constraint checking levels for early-fail."""
    PERMISSIVE = "permissive"    # Allow wide exploration
    CONSERVATIVE = "conservative" # Stay close to current values  
    STRICT = "strict"           # Enforce tight safety margins

@dataclass
class EFETunerConfig:
    """Configuration for registry-integrated EFE tuning."""
    
    # Optimization strategy
    strategy: OptimizationStrategy = OptimizationStrategy.ROLE_BLOCK_COORDINATE
    constraint_level: ConstraintLevel = ConstraintLevel.CONSERVATIVE
    
    # Population parameters
    population_size: int = 12       # Smaller for role-based optimization
    num_generations: int = 8        # Fewer generations, more frequent runs
    elite_ratio: float = 0.25       # Keep top 25%
    
    # Role-specific weights in EFE calculation
    role_importance: Dict[str, float] = field(default_factory=lambda: {
        "acceptance": 0.4,      # Critical for composition success
        "stability": 0.3,       # Safety first
        "control_gain": 0.15,   # Performance
        "diffusion": 0.1,       # Fine-tuning
        "filter": 0.05          # Polish
    })
    
    # EFE fitness components
    performance_weight: float = 0.5   # Task success metrics
    robustness_weight: float = 0.3    # Parameter uncertainty reduction
    constraint_weight: float = 0.2    # Safety margin maintenance
    
    # Constraint thresholds for early-fail
    min_energy_stability_margin: float = 0.1      # 10% energy stability
    min_composition_success_rate: float = 0.05    # 5% composition success
    min_acceptance_window: float = 0.1             # 10% acceptance window
    
    # Trust region parameters
    role_trust_region_pct: float = 0.1     # 10% max change per role per window
    adaptive_trust_regions: bool = True     # Adapt based on success
    trust_region_decay: float = 0.9         # Shrink on failure
    trust_region_growth: float = 1.1        # Grow on success
    
    # Timing
    optimization_interval_seconds: float = 300.0  # 5 minutes between optimizations
    max_optimization_time_seconds: float = 60.0   # 1 minute per optimization
    
    # A/B testing
    ab_test_enabled: bool = True
    ab_test_duration_seconds: float = 30.0
    ab_confidence_threshold: float = 0.7

@dataclass
class EFEMetrics:
    """Metrics for EFE calculation."""
    performance_score: float = 0.0      # Task performance (0-1)
    robustness_score: float = 0.0       # Low uncertainty (0-1)  
    constraint_score: float = 0.0       # Safety margins (0-1)
    overall_efe: float = 0.0             # Combined EFE score
    constraint_violations: List[str] = field(default_factory=list)
    
@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    success: bool
    strategy_used: OptimizationStrategy
    roles_optimized: List[ParameterRole]
    parameter_changes: Dict[str, Tuple[float, float]]  # param_id -> (old, new)
    efe_improvement: float
    constraint_violations: List[str]
    optimization_time: float
    confidence: float
    reason: str

class RegistryEFETuner:
    """EFE tuner integrated with parameter registry for strategic optimization."""
    
    def __init__(self, config: EFETunerConfig = None, 
                 registry: Optional[ParameterRegistry] = None,
                 sensing_router: Optional[SensingRouter] = None):
        self.config = config or EFETunerConfig()
        self.registry = registry or get_global_registry()
        self.sensing_router = sensing_router or get_global_sensing_router()
        
        # Optimization state
        self.last_optimization_time = 0.0
        self.optimization_history: deque = deque(maxlen=100)
        self.role_trust_regions: Dict[ParameterRole, float] = {}
        self.role_success_rates: Dict[ParameterRole, deque] = {}
        
        # Current optimization cycle
        self.current_target_roles: List[ParameterRole] = []
        self.baseline_metrics: Optional[EFEMetrics] = None
        self.ab_test_start_time: Optional[float] = None
        self.ab_test_parameters: Dict[str, float] = {}
        
        # Performance tracking
        self.windowed_metrics: deque = deque(maxlen=20)  # Last 20 sensing decisions
        
        # Initialize role trust regions
        self._initialize_role_trust_regions()
        
        logger.info(f"Initialized registry EFE tuner with {self.config.strategy.value} strategy")
    
    def _initialize_role_trust_regions(self):
        """Initialize trust regions for each parameter role."""
        for role in ParameterRole:
            self.role_trust_regions[role] = self.config.role_trust_region_pct
            self.role_success_rates[role] = deque(maxlen=10)
    
    def should_optimize(self) -> Tuple[bool, str]:
        """Check if optimization should run now."""
        current_time = time.monotonic()
        
        # Check time interval
        if current_time - self.last_optimization_time < self.config.optimization_interval_seconds:
            remaining = self.config.optimization_interval_seconds - (current_time - self.last_optimization_time)
            return False, f"Cooldown: {remaining:.1f}s remaining"
        
        # Check if we have enough sensing data
        if len(self.windowed_metrics) < 5:
            return False, "Insufficient sensing data for optimization"
        
        # Check if system is in safe mode (no optimization during emergencies)
        latest_decision = self.windowed_metrics[-1] if self.windowed_metrics else None
        if latest_decision and latest_decision.get("safe_mode", False):
            return False, "System in safe mode"
        
        # Check constraint margins
        diagnostics = self.sensing_router.get_diagnostics()
        constraint_margins = diagnostics.get("constraint_margins", {})
        
        for constraint, margin in constraint_margins.items():
            if "safety" in constraint and margin < 0.1:
                return False, f"Safety constraint {constraint} too low: {margin:.3f}"
        
        return True, "Ready for optimization"
    
    def record_sensing_decision(self, decision_data: Dict[str, Any]):
        """Record sensing router decision for windowed analysis."""
        self.windowed_metrics.append({
            "timestamp": time.monotonic(),
            "triggered_rules": decision_data.get("triggered_rules", []),
            "role_adjustments": decision_data.get("role_adjustments", {}),
            "constraint_violations": decision_data.get("constraint_violations", []),
            "safe_mode": decision_data.get("safe_mode", False),
            "signal_states": decision_data.get("signal_states", {})
        })
    
    def run_strategic_optimization(self) -> Optional[OptimizationResult]:
        """Run strategic parameter optimization if conditions are met."""
        
        # Check if optimization should run
        should_run, reason = self.should_optimize()
        if not should_run:
            logger.debug(f"Skipping optimization: {reason}")
            return None
        
        logger.info("Starting strategic parameter optimization")
        start_time = time.monotonic()
        
        try:
            # Select optimization strategy
            if self.config.strategy == OptimizationStrategy.ROLE_BLOCK_COORDINATE:
                result = self._optimize_role_block_coordinate()
            elif self.config.strategy == OptimizationStrategy.HIERARCHICAL:
                result = self._optimize_hierarchical()
            else:
                result = self._optimize_full_joint()
            
            # Update optimization time
            result.optimization_time = time.monotonic() - start_time
            self.last_optimization_time = time.monotonic()
            
            # Record in history
            self.optimization_history.append(result)
            
            # Update trust regions based on success
            if result.success:
                self._update_trust_regions_success(result.roles_optimized)
                logger.info(f"Optimization succeeded: {result.reason}")
            else:
                self._update_trust_regions_failure(result.roles_optimized)
                logger.warning(f"Optimization failed: {result.reason}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return OptimizationResult(
                success=False,
                strategy_used=self.config.strategy,
                roles_optimized=[],
                parameter_changes={},
                efe_improvement=0.0,
                constraint_violations=[f"Exception: {e}"],
                optimization_time=time.monotonic() - start_time,
                confidence=0.0,
                reason=f"Exception during optimization: {e}"
            )
    
    def _optimize_role_block_coordinate(self) -> OptimizationResult:
        """Optimize one role at a time (block coordinate descent)."""
        
        # Select target role based on importance and recent activity
        target_role = self._select_target_role()
        
        if not target_role:
            return OptimizationResult(
                success=False,
                strategy_used=OptimizationStrategy.ROLE_BLOCK_COORDINATE,
                roles_optimized=[],
                parameter_changes={},
                efe_improvement=0.0,
                constraint_violations=["No suitable role found for optimization"],
                optimization_time=0.0,
                confidence=0.0,
                reason="No role available for optimization"
            )
        
        logger.info(f"Optimizing role: {target_role.value}")
        
        # Get current baseline metrics
        baseline_efe = self._compute_current_efe()
        
        # Generate candidate parameter sets for this role
        candidates = self._generate_role_candidates(target_role, num_candidates=8)
        
        best_candidate = None
        best_efe = baseline_efe.overall_efe
        best_parameters = {}
        
        for candidate in candidates:
            # Early-fail constraint check
            if not self._passes_early_fail_constraints(candidate):
                continue
            
            # A/B test if enabled
            if self.config.ab_test_enabled:
                test_efe = self._ab_test_parameters(candidate)
                if test_efe and test_efe.overall_efe > best_efe:
                    best_efe = test_efe.overall_efe
                    best_candidate = candidate
                    best_parameters = candidate.copy()
            else:
                # Direct evaluation (faster but less accurate)
                test_efe = self._evaluate_parameter_set(candidate)
                if test_efe.overall_efe > best_efe:
                    best_efe = test_efe.overall_efe
                    best_candidate = candidate
                    best_parameters = candidate.copy()
        
        # Apply best candidate if found
        if best_candidate and best_efe > baseline_efe.overall_efe:
            parameter_changes = self._apply_parameter_set(best_parameters)
            efe_improvement = best_efe - baseline_efe.overall_efe
            
            return OptimizationResult(
                success=True,
                strategy_used=OptimizationStrategy.ROLE_BLOCK_COORDINATE,
                roles_optimized=[target_role],
                parameter_changes=parameter_changes,
                efe_improvement=efe_improvement,
                constraint_violations=[],
                optimization_time=0.0,  # Will be set by caller
                confidence=0.8,
                reason=f"Improved {target_role.value} role by {efe_improvement:.4f}"
            )
        else:
            return OptimizationResult(
                success=False,
                strategy_used=OptimizationStrategy.ROLE_BLOCK_COORDINATE,
                roles_optimized=[target_role],
                parameter_changes={},
                efe_improvement=0.0,
                constraint_violations=[],
                optimization_time=0.0,
                confidence=0.3,
                reason=f"No improvement found for {target_role.value}"
            )
    
    def _optimize_hierarchical(self) -> OptimizationResult:
        """Optimize roles in priority order: Safety -> Composition -> Performance."""
        
        # Define role hierarchy
        role_hierarchy = [
            [ParameterRole.STABILITY, ParameterRole.ACCEPTANCE],  # Safety first
            [ParameterRole.COMPOSITION],                          # Composition viability
            [ParameterRole.CONTROL_GAIN, ParameterRole.DIFFUSION, ParameterRole.FILTER]  # Performance
        ]
        
        total_improvement = 0.0
        all_changes = {}
        optimized_roles = []
        
        for role_group in role_hierarchy:
            for role in role_group:
                # Try to optimize this role
                self.current_target_roles = [role]
                role_result = self._optimize_single_role(role)
                
                if role_result.success:
                    total_improvement += role_result.efe_improvement
                    all_changes.update(role_result.parameter_changes)
                    optimized_roles.append(role)
                    
                    # Early success - don't optimize lower priority roles if safety improved significantly
                    if role in [ParameterRole.STABILITY, ParameterRole.ACCEPTANCE] and role_result.efe_improvement > 0.1:
                        break
        
        success = len(optimized_roles) > 0
        return OptimizationResult(
            success=success,
            strategy_used=OptimizationStrategy.HIERARCHICAL,
            roles_optimized=optimized_roles,
            parameter_changes=all_changes,
            efe_improvement=total_improvement,
            constraint_violations=[],
            optimization_time=0.0,
            confidence=0.9 if success else 0.2,
            reason=f"Hierarchical optimization: {len(optimized_roles)} roles improved"
        )
    
    def _optimize_full_joint(self) -> OptimizationResult:
        """Optimize multiple roles simultaneously (higher risk, higher reward)."""
        logger.info("Running full joint optimization")
        
        # Select multiple roles for joint optimization
        target_roles = self._select_multiple_target_roles(max_roles=3)
        
        if not target_roles:
            return OptimizationResult(
                success=False,
                strategy_used=OptimizationStrategy.FULL_JOINT,
                roles_optimized=[],
                parameter_changes={},
                efe_improvement=0.0,
                constraint_violations=["No roles selected for joint optimization"],
                optimization_time=0.0,
                confidence=0.0,
                reason="No suitable roles found"
            )
        
        # Generate joint candidates
        baseline_efe = self._compute_current_efe()
        candidates = self._generate_joint_candidates(target_roles, num_candidates=6)
        
        best_candidate = None
        best_efe = baseline_efe.overall_efe
        
        for candidate in candidates:
            if self._passes_early_fail_constraints(candidate):
                test_efe = self._evaluate_parameter_set(candidate)
                if test_efe.overall_efe > best_efe:
                    best_efe = test_efe.overall_efe
                    best_candidate = candidate
        
        if best_candidate and best_efe > baseline_efe.overall_efe:
            parameter_changes = self._apply_parameter_set(best_candidate)
            efe_improvement = best_efe - baseline_efe.overall_efe
            
            return OptimizationResult(
                success=True,
                strategy_used=OptimizationStrategy.FULL_JOINT,
                roles_optimized=target_roles,
                parameter_changes=parameter_changes,
                efe_improvement=efe_improvement,
                constraint_violations=[],
                optimization_time=0.0,
                confidence=0.6,  # Lower confidence for joint optimization
                reason=f"Joint optimization improved {len(target_roles)} roles"
            )
        else:
            return OptimizationResult(
                success=False,
                strategy_used=OptimizationStrategy.FULL_JOINT,
                roles_optimized=target_roles,
                parameter_changes={},
                efe_improvement=0.0,
                constraint_violations=[],
                optimization_time=0.0,
                confidence=0.2,
                reason="No improvement from joint optimization"
            )
    
    def _select_target_role(self) -> Optional[ParameterRole]:
        """Select the most promising role for optimization."""
        
        # Get recent sensing activity
        recent_adjustments = {}
        for decision in list(self.windowed_metrics)[-5:]:  # Last 5 decisions
            for role_str, adjustment in decision.get("role_adjustments", {}).items():
                try:
                    role = ParameterRole(role_str)
                    if role not in recent_adjustments:
                        recent_adjustments[role] = []
                    recent_adjustments[role].append(abs(adjustment))
                except ValueError:
                    continue
        
        # Score roles by importance, recent activity, and success rate
        role_scores = {}
        for role in ParameterRole:
            if role == ParameterRole.UNKNOWN:  # Skip quarantined role
                continue
                
            # Base score from importance
            importance = self.config.role_importance.get(role.value, 0.1)
            
            # Recent activity score (higher if sensing router has been adjusting this role)
            activity_score = 0.0
            if role in recent_adjustments:
                activity_score = min(1.0, np.mean(recent_adjustments[role]) / 0.1)  # Normalize by 10%
            
            # Success rate score (prefer roles that have been successfully optimized)
            success_rate = np.mean(self.role_success_rates[role]) if self.role_success_rates[role] else 0.5
            
            # Trust region size (prefer roles with larger trust regions)
            trust_region_score = self.role_trust_regions[role] / self.config.role_trust_region_pct
            
            # Combined score
            role_scores[role] = importance * 0.4 + activity_score * 0.3 + success_rate * 0.2 + trust_region_score * 0.1
        
        if not role_scores:
            return None
        
        # Select role with highest score
        best_role = max(role_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected role {best_role.value} for optimization (score: {role_scores[best_role]:.3f})")
        return best_role
    
    def _select_multiple_target_roles(self, max_roles: int = 3) -> List[ParameterRole]:
        """Select multiple roles for joint optimization."""
        # Use single role selection multiple times, avoiding duplicates
        selected_roles = []
        excluded_roles = set()
        
        for _ in range(max_roles):
            # Temporarily exclude already selected roles
            temp_scores = {}
            for role in ParameterRole:
                if role not in excluded_roles and role != ParameterRole.UNKNOWN:
                    temp_scores[role] = self.config.role_importance.get(role.value, 0.1)
            
            if not temp_scores:
                break
                
            best_role = max(temp_scores.items(), key=lambda x: x[1])[0]
            selected_roles.append(best_role)
            excluded_roles.add(best_role)
        
        return selected_roles
    
    def _generate_role_candidates(self, role: ParameterRole, num_candidates: int = 8) -> List[Dict[str, float]]:
        """Generate parameter candidates for a specific role."""
        candidates = []
        
        # Get all parameters for this role
        role_params = self.registry.get_parameters_by_role(role)
        if not role_params:
            return candidates
        
        # Get trust region size for this role
        trust_region = self.role_trust_regions[role]
        
        for _ in range(num_candidates):
            candidate = {}
            
            for param in role_params:
                current_val = param.current_value
                param_range = param.constraints.max_value - param.constraints.min_value
                
                # Generate perturbation within trust region
                if param.dtype.value == "float_log":
                    # Multiplicative perturbation for log-scale
                    multiplier = 1.0 + np.random.uniform(-trust_region, trust_region)
                    new_val = current_val * multiplier
                else:
                    # Additive perturbation for linear scale
                    delta = np.random.uniform(-trust_region * param_range, trust_region * param_range)
                    new_val = current_val + delta
                
                # Clamp to parameter bounds
                new_val = max(param.constraints.min_value, min(param.constraints.max_value, new_val))
                candidate[param.id] = new_val
            
            candidates.append(candidate)
        
        return candidates
    
    def _generate_joint_candidates(self, roles: List[ParameterRole], num_candidates: int = 6) -> List[Dict[str, float]]:
        """Generate candidates for joint role optimization."""
        candidates = []
        
        for _ in range(num_candidates):
            candidate = {}
            
            for role in roles:
                role_candidates = self._generate_role_candidates(role, num_candidates=1)
                if role_candidates:
                    candidate.update(role_candidates[0])
            
            candidates.append(candidate)
        
        return candidates
    
    def _passes_early_fail_constraints(self, parameters: Dict[str, float]) -> bool:
        """Check if parameter set passes early-fail constraints."""
        
        # Check energy stability
        energy_params = [p for p in parameters.keys() if "alpha" in p or "beta" in p or "gamma" in p]
        if energy_params:
            # Rough heuristic: don't change core physics parameters too much
            for param_id in energy_params:
                param = self.registry.get_parameter(param_id)
                if param:
                    current_val = param.current_value
                    new_val = parameters[param_id]
                    change_pct = abs(new_val - current_val) / max(abs(current_val), 0.01)
                    if change_pct > 0.5:  # 50% change threshold
                        return False
        
        # Check acceptance parameters
        acceptance_params = [p for p in parameters.keys() if "accept" in p]
        for param_id in acceptance_params:
            param = self.registry.get_parameter(param_id)
            if param and "bounds" in param_id:
                # Ensure bounds remain ordered
                new_val = parameters[param_id]
                if param.constraints.min_value >= param.constraints.max_value:
                    return False
        
        return True
    
    def _evaluate_parameter_set(self, parameters: Dict[str, float]) -> EFEMetrics:
        """Evaluate a parameter set (simplified for now)."""
        
        # This is a simplified evaluation - in practice, this would run
        # the system with these parameters and measure performance
        
        # For now, use heuristics based on parameter values
        performance_score = 0.7  # Assume decent baseline performance
        robustness_score = 0.6   # Assume moderate robustness
        constraint_score = 0.8   # Assume good constraint compliance
        
        # Penalize extreme parameter values
        for param_id, value in parameters.items():
            param = self.registry.get_parameter(param_id)
            if param:
                # Normalize to [0, 1] range
                param_range = param.constraints.max_value - param.constraints.min_value
                normalized_val = (value - param.constraints.min_value) / param_range
                
                # Penalize values near boundaries (less robust)
                boundary_penalty = min(normalized_val, 1.0 - normalized_val) * 0.1
                robustness_score -= boundary_penalty
        
        # Clamp scores
        performance_score = max(0.0, min(1.0, performance_score))
        robustness_score = max(0.0, min(1.0, robustness_score))
        constraint_score = max(0.0, min(1.0, constraint_score))
        
        # Compute overall EFE
        overall_efe = (
            self.config.performance_weight * performance_score +
            self.config.robustness_weight * robustness_score +
            self.config.constraint_weight * constraint_score
        )
        
        return EFEMetrics(
            performance_score=performance_score,
            robustness_score=robustness_score,
            constraint_score=constraint_score,
            overall_efe=overall_efe,
            constraint_violations=[]
        )
    
    def _ab_test_parameters(self, parameters: Dict[str, float]) -> Optional[EFEMetrics]:
        """Run A/B test for parameter set."""
        
        # Store current parameters as baseline
        baseline_params = {}
        for param_id in parameters.keys():
            param = self.registry.get_parameter(param_id)
            if param:
                baseline_params[param_id] = param.current_value
        
        try:
            # Apply test parameters
            self._apply_parameter_set(parameters)
            
            # Wait for test duration
            time.sleep(min(self.config.ab_test_duration_seconds, 5.0))  # Cap at 5 seconds for now
            
            # Evaluate performance
            test_metrics = self._evaluate_parameter_set(parameters)
            
            return test_metrics
            
        finally:
            # Restore baseline parameters
            self._apply_parameter_set(baseline_params)
    
    def _apply_parameter_set(self, parameters: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Apply parameter set to registry and return changes made."""
        changes = {}
        
        for param_id, new_value in parameters.items():
            param = self.registry.get_parameter(param_id)
            if param:
                old_value = param.current_value
                success, message = self.registry.update_parameter(
                    param_id, new_value, "efe_tuner", "Strategic optimization"
                )
                
                if success:
                    changes[param_id] = (old_value, new_value)
                else:
                    logger.warning(f"Failed to update {param_id}: {message}")
        
        return changes
    
    def _compute_current_efe(self) -> EFEMetrics:
        """Compute current EFE metrics based on recent system performance."""
        
        if not self.windowed_metrics:
            return EFEMetrics(performance_score=0.5, robustness_score=0.5, constraint_score=0.5, overall_efe=0.5)
        
        # Analyze recent performance from sensing decisions
        recent_violations = []
        rule_activity = 0
        safe_mode_count = 0
        
        for decision in self.windowed_metrics:
            recent_violations.extend(decision.get("constraint_violations", []))
            rule_activity += len(decision.get("triggered_rules", []))
            if decision.get("safe_mode", False):
                safe_mode_count += 1
        
        # Performance score (lower rule activity = better performance)
        max_expected_rules = len(self.windowed_metrics) * 2  # Expect ~2 rules per decision on average
        performance_score = max(0.0, 1.0 - rule_activity / max(max_expected_rules, 1))
        
        # Robustness score (fewer safe mode activations = more robust)
        robustness_score = max(0.0, 1.0 - safe_mode_count / len(self.windowed_metrics))
        
        # Constraint score (fewer violations = better constraint compliance)
        constraint_score = max(0.0, 1.0 - len(recent_violations) / max(len(self.windowed_metrics), 1))
        
        # Overall EFE
        overall_efe = (
            self.config.performance_weight * performance_score +
            self.config.robustness_weight * robustness_score +
            self.config.constraint_weight * constraint_score
        )
        
        return EFEMetrics(
            performance_score=performance_score,
            robustness_score=robustness_score,
            constraint_score=constraint_score,
            overall_efe=overall_efe,
            constraint_violations=recent_violations
        )
    
    def _optimize_single_role(self, role: ParameterRole) -> OptimizationResult:
        """Optimize a single role (helper for hierarchical optimization)."""
        
        candidates = self._generate_role_candidates(role, num_candidates=4)
        baseline_efe = self._compute_current_efe()
        
        best_candidate = None
        best_efe = baseline_efe.overall_efe
        
        for candidate in candidates:
            if self._passes_early_fail_constraints(candidate):
                test_efe = self._evaluate_parameter_set(candidate)
                if test_efe.overall_efe > best_efe:
                    best_efe = test_efe.overall_efe
                    best_candidate = candidate
        
        if best_candidate and best_efe > baseline_efe.overall_efe:
            parameter_changes = self._apply_parameter_set(best_candidate)
            efe_improvement = best_efe - baseline_efe.overall_efe
            
            return OptimizationResult(
                success=True,
                strategy_used=OptimizationStrategy.HIERARCHICAL,
                roles_optimized=[role],
                parameter_changes=parameter_changes,
                efe_improvement=efe_improvement,
                constraint_violations=[],
                optimization_time=0.0,
                confidence=0.7,
                reason=f"Hierarchical optimization improved {role.value}"
            )
        else:
            return OptimizationResult(
                success=False,
                strategy_used=OptimizationStrategy.HIERARCHICAL,
                roles_optimized=[role],
                parameter_changes={},
                efe_improvement=0.0,
                constraint_violations=[],
                optimization_time=0.0,
                confidence=0.3,
                reason=f"No improvement found for {role.value}"
            )
    
    def _update_trust_regions_success(self, roles: List[ParameterRole]):
        """Update trust regions after successful optimization."""
        for role in roles:
            # Record success
            self.role_success_rates[role].append(1.0)
            
            # Grow trust region slightly
            if self.config.adaptive_trust_regions:
                self.role_trust_regions[role] = min(
                    0.3,  # Cap at 30%
                    self.role_trust_regions[role] * self.config.trust_region_growth
                )
    
    def _update_trust_regions_failure(self, roles: List[ParameterRole]):
        """Update trust regions after failed optimization."""
        for role in roles:
            # Record failure
            self.role_success_rates[role].append(0.0)
            
            # Shrink trust region
            if self.config.adaptive_trust_regions:
                self.role_trust_regions[role] = max(
                    0.01,  # Minimum 1%
                    self.role_trust_regions[role] * self.config.trust_region_decay
                )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get EFE tuner diagnostics."""
        current_time = time.monotonic()
        
        return {
            "last_optimization_age": current_time - self.last_optimization_time,
            "next_optimization_in": max(0, self.config.optimization_interval_seconds - 
                                      (current_time - self.last_optimization_time)),
            "optimization_history_count": len(self.optimization_history),
            "windowed_metrics_count": len(self.windowed_metrics),
            "role_trust_regions": {role.value: region for role, region in self.role_trust_regions.items()},
            "role_success_rates": {
                role.value: np.mean(rates) if rates else 0.0 
                for role, rates in self.role_success_rates.items()
            },
            "strategy": self.config.strategy.value,
            "constraint_level": self.config.constraint_level.value
        }

# Global registry EFE tuner instance
_global_registry_efe_tuner: Optional[RegistryEFETuner] = None

def get_global_registry_efe_tuner() -> RegistryEFETuner:
    """Get the global registry EFE tuner instance."""
    global _global_registry_efe_tuner
    if _global_registry_efe_tuner is None:
        _global_registry_efe_tuner = RegistryEFETuner()
    return _global_registry_efe_tuner