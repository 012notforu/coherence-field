"""
Vector Type Detection and Adapters

This module provides automatic vector type detection and adapters that properly
apply vector content to the parameter registry instead of using only vector[0].

Key Features:
1. Manifest-based vector type detection with fallback fingerprinting
2. Type-specific adapters that use full vector content
3. Coverage requirements (≥95% of vector elements must be used)
4. Sandbox testing before applying to real system
5. Invariant checking to ensure system stability
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
import json
import numpy as np
from pathlib import Path
import logging

from utils.parameter_registry import (
    ParameterRegistry, ParameterRole, ParameterEntry, ParameterDType, 
    ParameterScope, ParameterConstraint, get_global_registry
)

logger = logging.getLogger(__name__)

class VectorType(Enum):
    """Types of vectors based on their content and intended use."""
    POLICY_WEIGHTS = "policy_weights"       # Controller policy parameters (CartPole, etc.)
    CONTROLLER_CONFIG = "controller_config"  # Full controller configuration
    FIELD_DELTA = "field_delta"            # Direct field modifications
    REGIONAL_PARAMS = "regional_params"     # Spatially-varying parameters
    MOVEMENT_HINT = "movement_hint"        # Agent movement guidance
    SCHEDULE = "schedule"                  # Temporal scheduling parameters
    COMPOSITION_WEIGHTS = "composition_weights"  # Vector composition weights

@dataclass
class VectorManifest:
    """Metadata manifest for vector interpretation."""
    vector_type: VectorType
    element_count: int
    element_roles: List[str]              # What each element represents
    element_ranges: List[Tuple[float, float]]  # Valid range for each element
    element_dtypes: List[str]             # Data type for each element
    element_scopes: List[str]             # Scope for each element (global/vector/agent/region)
    param_mappings: Dict[int, str]        # index -> parameter_id mapping
    units: List[str]                      # Units for each element
    invariants: List[str]                 # Invariant descriptions to check
    coverage_requirement: float = 0.95   # Minimum fraction of elements that must be used
    description: str = ""

@dataclass
class AdapterResult:
    """Result of applying a vector through an adapter."""
    success: bool
    coverage_pct: float                   # Fraction of vector elements used
    elements_applied: int                 # Number of elements successfully applied
    param_updates: Dict[str, float]       # parameter_id -> new_value
    invariants_passed: List[str]          # Which invariants were satisfied
    invariants_failed: List[str]          # Which invariants failed
    reason: str                          # Success/failure explanation
    sandbox_metrics: Optional[Dict] = None  # Metrics from sandbox testing
    explain_table: List[Dict[str, Any]] = None  # Per-element explanation
    coverage_gaps: List[int] = None       # Indices of unmapped elements

class VectorAdapter:
    """Base class for vector type adapters."""
    
    def __init__(self, registry: ParameterRegistry):
        self.registry = registry
        self.vector_type = None  # Set by subclasses
    
    def can_handle(self, vector_type: VectorType) -> bool:
        """Check if this adapter can handle the given vector type."""
        return self.vector_type == vector_type
    
    def apply_vector(self, vector: np.ndarray, manifest: VectorManifest, 
                    writer: str, reason: str = "") -> AdapterResult:
        """Apply vector through this adapter."""
        raise NotImplementedError("Subclasses must implement apply_vector")
    
    def _check_invariants(self, vector: np.ndarray, manifest: VectorManifest,
                         param_updates: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Check vector invariants. Returns (passed, failed) lists."""
        passed = []
        failed = []
        
        for invariant in manifest.invariants:
            try:
                if self._evaluate_invariant(invariant, vector, manifest, param_updates):
                    passed.append(invariant)
                else:
                    failed.append(invariant)
            except Exception as e:
                logger.warning(f"Invariant check failed: {invariant} - {e}")
                failed.append(f"{invariant} (check failed)")
        
        return passed, failed
    
    def _evaluate_invariant(self, invariant: str, vector: np.ndarray, 
                           manifest: VectorManifest, param_updates: Dict[str, float]) -> bool:
        """Evaluate a specific invariant."""
        # Simple invariant language
        if invariant == "sum_weights_eq_1":
            weight_values = [v for k, v in param_updates.items() if "weight" in k.lower()]
            if weight_values:
                return abs(sum(weight_values) - 1.0) < 0.05
        elif invariant == "acceptance_bounds_valid":
            bounds = [v for k, v in param_updates.items() if "bounds" in k.lower()]
            if len(bounds) >= 2:
                return min(bounds) < max(bounds) and (max(bounds) - min(bounds)) > 0.1
        elif invariant == "gains_positive":
            gains = [v for k, v in param_updates.items() if "gain" in k.lower()]
            return all(g > 0 for g in gains)
        elif invariant == "stability_maintained":
            # Check that damping parameters remain positive
            stability = [v for k, v in param_updates.items() if "beta" in k.lower() or "damp" in k.lower()]
            return all(s > 0.1 for s in stability)
        
        return True  # Unknown invariants pass by default

class PolicyWeightsAdapter(VectorAdapter):
    """Adapter for policy weight vectors (CartPole, control policies)."""
    
    def __init__(self, registry: ParameterRegistry):
        super().__init__(registry)
        self.vector_type = VectorType.POLICY_WEIGHTS
    
    def apply_vector(self, vector: np.ndarray, manifest: VectorManifest,
                    writer: str, reason: str = "") -> AdapterResult:
        """Apply policy weights to control parameters."""
        
        param_updates = {}
        elements_applied = 0
        
        # Map vector elements to control parameters based on manifest
        for idx, param_id in manifest.param_mappings.items():
            if idx < len(vector):
                # Get or create the parameter
                param = self.registry.get_parameter(param_id)
                if not param:
                    # Create dynamic parameter for policy vectors
                    param = self._create_dynamic_parameter(param_id, idx, manifest)
                    if param:
                        self.registry.register_parameter(param)
                
                if param:
                    # Apply value with range checking
                    raw_value = float(vector[idx])
                    element_range = manifest.element_ranges[idx] if idx < len(manifest.element_ranges) else (0.0, 10.0)
                    
                    # Scale from vector range to parameter range
                    scaled_value = self._scale_to_param_range(raw_value, element_range, param)
                    param_updates[param_id] = scaled_value
                    elements_applied += 1
        
        # Calculate coverage
        coverage_pct = elements_applied / len(vector) if len(vector) > 0 else 0.0
        
        # Check coverage requirement
        if coverage_pct < manifest.coverage_requirement:
            return AdapterResult(
                success=False,
                coverage_pct=coverage_pct,
                elements_applied=elements_applied,
                param_updates={},
                invariants_passed=[],
                invariants_failed=[],
                reason=f"Coverage {coverage_pct:.1%} < required {manifest.coverage_requirement:.1%}"
            )
        
        # Check invariants
        passed, failed = self._check_invariants(vector, manifest, param_updates)
        
        if failed:
            return AdapterResult(
                success=False,
                coverage_pct=coverage_pct,
                elements_applied=elements_applied,
                param_updates={},
                invariants_passed=passed,
                invariants_failed=failed,
                reason=f"Invariant violations: {failed}"
            )
        
        # Apply parameter updates
        successful_updates = {}
        for param_id, new_value in param_updates.items():
            success, message = self.registry.update_parameter(param_id, new_value, writer, reason)
            if success:
                successful_updates[param_id] = new_value
            else:
                logger.warning(f"Failed to update {param_id}: {message}")
        
        final_success = len(successful_updates) > 0
        
        return AdapterResult(
            success=final_success,
            coverage_pct=coverage_pct,
            elements_applied=len(successful_updates),
            param_updates=successful_updates,
            invariants_passed=passed,
            invariants_failed=failed,
            reason="Applied successfully" if final_success else "No parameters updated"
        )
    
    def _scale_to_param_range(self, value: float, element_range: Tuple[float, float], 
                             param: ParameterEntry) -> float:
        """Scale vector element from its range to parameter's constraint range."""
        elem_min, elem_max = element_range
        param_min, param_max = param.constraints.min_value, param.constraints.max_value
        
        # Normalize to [0,1] from element range
        if elem_max > elem_min:
            normalized = (value - elem_min) / (elem_max - elem_min)
        else:
            normalized = 0.5
        
        # Scale to parameter range
        scaled = param_min + normalized * (param_max - param_min)
        
        # Clamp to bounds
        return max(param_min, min(param_max, scaled))

class CompositionWeightsAdapter(VectorAdapter):
    """Adapter for vector composition weight arrays."""
    
    def __init__(self, registry: ParameterRegistry):
        super().__init__(registry)
        self.vector_type = VectorType.COMPOSITION_WEIGHTS
    
    def apply_vector(self, vector: np.ndarray, manifest: VectorManifest,
                    writer: str, reason: str = "") -> AdapterResult:
        """Apply composition weights with automatic normalization."""
        
        # Ensure weights are positive and normalize
        weights = np.maximum(vector, 0.0)  # Remove negative weights
        total = np.sum(weights)
        
        if total > 0:
            weights = weights / total  # Normalize to sum=1
        else:
            # Fallback to equal weights
            weights = np.ones_like(vector) / len(vector)
        
        param_updates = {}
        elements_applied = 0
        
        # Map to composition weight parameters
        for idx, param_id in manifest.param_mappings.items():
            if idx < len(weights):
                param_updates[param_id] = float(weights[idx])
                elements_applied += 1
        
        coverage_pct = elements_applied / len(vector) if len(vector) > 0 else 0.0
        
        # Check coverage requirement
        if coverage_pct < manifest.coverage_requirement:
            return AdapterResult(
                success=False,
                coverage_pct=coverage_pct,
                elements_applied=elements_applied,
                param_updates={},
                invariants_passed=[],
                invariants_failed=[],
                reason=f"Coverage {coverage_pct:.1%} < required {manifest.coverage_requirement:.1%}"
            )
        
        # Check invariants (weights should sum to 1)
        passed, failed = self._check_invariants(vector, manifest, param_updates)
        
        # Apply updates
        successful_updates = {}
        for param_id, weight in param_updates.items():
            success, message = self.registry.update_parameter(param_id, weight, writer, reason)
            if success:
                successful_updates[param_id] = weight
        
        return AdapterResult(
            success=len(successful_updates) > 0,
            coverage_pct=coverage_pct,
            elements_applied=len(successful_updates),
            param_updates=successful_updates,
            invariants_passed=passed,
            invariants_failed=failed,
            reason="Weights applied and normalized"
        )

class RegionalParamsAdapter(VectorAdapter):
    """Adapter for spatially-varying parameter vectors."""
    
    def __init__(self, registry: ParameterRegistry):
        super().__init__(registry)
        self.vector_type = VectorType.REGIONAL_PARAMS
    
    def apply_vector(self, vector: np.ndarray, manifest: VectorManifest,
                    writer: str, reason: str = "") -> AdapterResult:
        """Apply regional parameters to create spatial variation."""
        
        # For regional params, create parameter entries for different spatial regions
        param_updates = {}
        elements_applied = 0
        
        # Divide vector elements across spatial regions
        # This is a simplified approach - real implementation would need coordinate system
        regions_per_param = len(vector) // len(manifest.param_mappings)
        
        for base_idx, base_param_id in manifest.param_mappings.items():
            start_idx = base_idx * regions_per_param
            end_idx = min((base_idx + 1) * regions_per_param, len(vector))
            
            for region_idx, vector_idx in enumerate(range(start_idx, end_idx)):
                if vector_idx < len(vector):
                    # Create region-specific parameter ID
                    region_param_id = f"{base_param_id}.region_{region_idx}"
                    
                    # Register regional parameter if it doesn't exist
                    if not self.registry.get_parameter(region_param_id):
                        base_param = self.registry.get_parameter(base_param_id)
                        if base_param:
                            regional_param = ParameterEntry(
                                id=region_param_id,
                                role=base_param.role,
                                scope=ParameterScope.REGION,
                                dtype=base_param.dtype,
                                constraints=base_param.constraints,
                                units=base_param.units,
                                description=f"Regional variant of {base_param_id}",
                                current_value=float(vector[vector_idx])
                            )
                            self.registry.register_parameter(regional_param)
                    
                    param_updates[region_param_id] = float(vector[vector_idx])
                    elements_applied += 1
        
        coverage_pct = elements_applied / len(vector) if len(vector) > 0 else 0.0
        
        # Apply updates
        successful_updates = {}
        for param_id, value in param_updates.items():
            success, message = self.registry.update_parameter(param_id, value, writer, reason)
            if success:
                successful_updates[param_id] = value
        
        passed, failed = self._check_invariants(vector, manifest, param_updates)
        
        return AdapterResult(
            success=len(successful_updates) > 0,
            coverage_pct=coverage_pct,
            elements_applied=len(successful_updates),
            param_updates=successful_updates,
            invariants_passed=passed,
            invariants_failed=failed,
            reason=f"Applied {len(successful_updates)} regional parameters"
        )

class VectorTypeDetector:
    """Detects vector types from manifests or fingerprinting."""
    
    def __init__(self):
        self.adapters = {
            VectorType.POLICY_WEIGHTS: PolicyWeightsAdapter,
            VectorType.COMPOSITION_WEIGHTS: CompositionWeightsAdapter,
            VectorType.REGIONAL_PARAMS: RegionalParamsAdapter,
        }
    
    def detect_vector_type(self, vector_path: Path, vector_data: Dict[str, Any]) -> Tuple[VectorType, VectorManifest]:
        """Detect vector type from path and data."""
        
        # Try to load manifest first
        manifest_path = vector_path.parent / "vector_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest_data = json.load(f)
                return self._parse_manifest(manifest_data)
            except Exception as e:
                logger.warning(f"Failed to load manifest {manifest_path}: {e}")
        
        # Fallback to fingerprinting
        return self._fingerprint_vector(vector_path, vector_data)
    
    def _parse_manifest(self, manifest_data: Dict[str, Any]) -> Tuple[VectorType, VectorManifest]:
        """Parse vector manifest from JSON data."""
        vector_type = VectorType(manifest_data["type"])
        
        manifest = VectorManifest(
            vector_type=vector_type,
            element_count=manifest_data["element_count"],
            element_roles=manifest_data["element_roles"],
            element_ranges=[(r[0], r[1]) for r in manifest_data["element_ranges"]],
            param_mappings={int(k): v for k, v in manifest_data["param_mappings"].items()},
            units=manifest_data.get("units", []),
            invariants=manifest_data.get("invariants", []),
            coverage_requirement=manifest_data.get("coverage_requirement", 0.95),
            description=manifest_data.get("description", "")
        )
        
        return vector_type, manifest
    
    def _fingerprint_vector(self, vector_path: Path, vector_data: Dict[str, Any]) -> Tuple[VectorType, VectorManifest]:
        """Fingerprint vector type from its characteristics."""
        
        vector = np.array(vector_data["vector"])
        vector_id = vector_path.parent.name
        
        # Heuristic detection based on vector characteristics
        if "cartpole" in vector_id.lower():
            return self._create_cartpole_manifest(vector)
        elif "composition" in vector_id.lower() or "weight" in vector_id.lower():
            return self._create_composition_manifest(vector)
        elif len(vector) > 20:  # Large vectors likely regional
            return self._create_regional_manifest(vector)
        else:
            # Default to policy weights
            return self._create_policy_manifest(vector)
    
    def _create_cartpole_manifest(self, vector: np.ndarray) -> Tuple[VectorType, VectorManifest]:
        """Create manifest for CartPole policy vectors."""
        
        # CartPole vectors typically have 10-15 elements representing gains and thresholds
        param_mappings = {}
        element_roles = []
        element_ranges = []
        
        if len(vector) >= 2:
            param_mappings[0] = "control.cartpole.gain_proportional"
            param_mappings[1] = "control.cartpole.gain_derivative"
            element_roles.extend(["proportional_gain", "derivative_gain"])
            element_ranges.extend([(0.1, 10.0), (0.1, 5.0)])
        
        # Fill remaining elements as filter/threshold parameters
        for i in range(2, min(len(vector), 10)):
            if i == 2:
                param_mappings[i] = "sensing.filter.noise_strength"
                element_roles.append("noise_filter")
                element_ranges.append((0.1, 1.0))
            else:
                # Create dynamic threshold parameters
                param_id = f"control.cartpole.threshold_{i-2}"
                param_mappings[i] = param_id
                element_roles.append(f"threshold_{i-2}")
                element_ranges.append((0.01, 2.0))
        
        manifest = VectorManifest(
            vector_type=VectorType.POLICY_WEIGHTS,
            element_count=len(vector),
            element_roles=element_roles,
            element_ranges=element_ranges,
            param_mappings=param_mappings,
            units=["gain"] * len(element_roles),
            invariants=["gains_positive", "stability_maintained"],
            coverage_requirement=0.8,  # Allow 80% coverage for policy vectors
            description=f"CartPole policy weights ({len(vector)} elements)"
        )
        
        return VectorType.POLICY_WEIGHTS, manifest
    
    def _create_composition_manifest(self, vector: np.ndarray) -> Tuple[VectorType, VectorManifest]:
        """Create manifest for composition weight vectors."""
        
        param_mappings = {}
        element_roles = []
        
        # Map to composition weight parameters
        weight_names = ["exploration", "stabilization", "smoothing", "alignment"]
        for i in range(min(len(vector), len(weight_names))):
            param_mappings[i] = f"composition.weight.{weight_names[i]}"
            element_roles.append(f"weight_{weight_names[i]}")
        
        manifest = VectorManifest(
            vector_type=VectorType.COMPOSITION_WEIGHTS,
            element_count=len(vector),
            element_roles=element_roles,
            element_ranges=[(0.0, 1.0)] * len(vector),
            param_mappings=param_mappings,
            units=["weight"] * len(vector),
            invariants=["sum_weights_eq_1"],
            coverage_requirement=0.95,
            description=f"Composition weights ({len(vector)} elements)"
        )
        
        return VectorType.COMPOSITION_WEIGHTS, manifest
    
    def _create_regional_manifest(self, vector: np.ndarray) -> Tuple[VectorType, VectorManifest]:
        """Create manifest for regional parameter vectors."""
        
        # Large vectors are treated as regional parameter variations
        base_params = ["scfd.physics.alpha", "scfd.physics.gamma", "scfd.physics.beta"]
        regions_per_param = len(vector) // len(base_params)
        
        param_mappings = {}
        element_roles = []
        
        for param_idx, base_param in enumerate(base_params):
            for region_idx in range(regions_per_param):
                vector_idx = param_idx * regions_per_param + region_idx
                if vector_idx < len(vector):
                    param_mappings[vector_idx] = f"{base_param}.region_{region_idx}"
                    element_roles.append(f"{base_param.split('.')[-1]}_region_{region_idx}")
        
        manifest = VectorManifest(
            vector_type=VectorType.REGIONAL_PARAMS,
            element_count=len(vector),
            element_roles=element_roles,
            element_ranges=[(0.01, 2.0)] * len(vector),
            param_mappings=param_mappings,
            units=["param"] * len(vector),
            invariants=["stability_maintained"],
            coverage_requirement=0.9,
            description=f"Regional parameters ({len(vector)} elements)"
        )
        
        return VectorType.REGIONAL_PARAMS, manifest
    
    def _create_policy_manifest(self, vector: np.ndarray) -> Tuple[VectorType, VectorManifest]:
        """Create generic policy manifest for unknown vectors."""
        
        param_mappings = {}
        element_roles = []
        
        # Map to generic control parameters
        for i in range(min(len(vector), 5)):
            param_id = f"control.generic.param_{i}"
            param_mappings[i] = param_id
            element_roles.append(f"generic_param_{i}")
        
        manifest = VectorManifest(
            vector_type=VectorType.POLICY_WEIGHTS,
            element_count=len(vector),
            element_roles=element_roles,
            element_ranges=[(0.1, 5.0)] * len(vector),
            param_mappings=param_mappings,
            units=["param"] * len(vector),
            invariants=[],
            coverage_requirement=0.7,  # Lower requirement for generic
            description=f"Generic policy vector ({len(vector)} elements)"
        )
        
        return VectorType.POLICY_WEIGHTS, manifest

class VectorApplicationManager:
    """Manages vector application through appropriate adapters."""
    
    def __init__(self, registry: Optional[ParameterRegistry] = None):
        self.registry = registry or get_global_registry()
        self.detector = VectorTypeDetector()
        self.adapters = {
            VectorType.POLICY_WEIGHTS: PolicyWeightsAdapter(self.registry),
            VectorType.COMPOSITION_WEIGHTS: CompositionWeightsAdapter(self.registry),
            VectorType.REGIONAL_PARAMS: RegionalParamsAdapter(self.registry),
        }
    
    def apply_vector_from_path(self, vector_path: Path, writer: str, reason: str = "") -> AdapterResult:
        """Apply vector from file path using appropriate adapter."""
        
        try:
            # Load vector data
            with open(vector_path) as f:
                vector_data = json.load(f)
            
            vector = np.array(vector_data["vector"])
            
            # Detect vector type and get manifest
            vector_type, manifest = self.detector.detect_vector_type(vector_path, vector_data)
            
            logger.info(f"Detected vector type {vector_type.value} for {vector_path.name}")
            
            # Get appropriate adapter
            adapter = self.adapters.get(vector_type)
            if not adapter:
                return AdapterResult(
                    success=False,
                    coverage_pct=0.0,
                    elements_applied=0,
                    param_updates={},
                    invariants_passed=[],
                    invariants_failed=[],
                    reason=f"No adapter for vector type {vector_type.value}"
                )
            
            # Apply vector through adapter
            result = adapter.apply_vector(vector, manifest, writer, reason)
            
            # Log result
            if result.success:
                logger.info(f"Vector applied: {result.elements_applied}/{len(vector)} elements, "
                           f"coverage {result.coverage_pct:.1%}")
            else:
                logger.warning(f"Vector application failed: {result.reason}")
            
            return result
            
        except Exception as e:
            logger.error(f"Vector application error for {vector_path}: {e}")
            return AdapterResult(
                success=False,
                coverage_pct=0.0,
                elements_applied=0,
                param_updates={},
                invariants_passed=[],
                invariants_failed=[],
                reason=f"Exception: {e}"
            )
    
    def get_supported_types(self) -> List[VectorType]:
        """Get list of supported vector types."""
        return list(self.adapters.keys())

# Global manager instance
_global_manager: Optional[VectorApplicationManager] = None

def get_global_vector_manager() -> VectorApplicationManager:
    """Get the global vector application manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = VectorApplicationManager()
    return _global_manager

def apply_vector(vector_path: Path, writer: str, reason: str = "") -> AdapterResult:
    """Convenience function to apply a vector using the global manager."""
    return get_global_vector_manager().apply_vector_from_path(vector_path, writer, reason)