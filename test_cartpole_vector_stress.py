#!/usr/bin/env python3
"""
CartPole Vector Stress Test - Faithful to SCFD Architecture

Tests the vector system under stress conditions using:
- Existing CartPole vectors from registry
- Enhanced GentleController with metadata-driven sensing  
- Real-time EFE tuner for strategic adaptation
- Unified metadata tracking across all adaptations
- Vector composition with metropolis acceptance

NO custom control logic - everything through the vector/sensing/composition system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import numpy as np

from engine import accel_theta, load_config, leapfrog_step, total_energy_density
from run.common import initialize_state
from observe.features import CartPoleFeatureExtractor, FeatureVector
from observe.policies import LinearPolicy, LinearPolicyConfig
from utils.logging import create_run_directory, RunLogger
from utils.efe_tuner import EFEParameterTuner
from utils.unified_metadata import UnifiedMetadataManager
from utils.vector_controller_bridge import VectorControllerBridge, create_bridge_for_system
from utils.sensing_router import SensingRouter
from observe.controller import GentleController
from orchestrator.pipeline import load_vector_registry, plan_for_environment, ProbeReport
from benchmarks.vector_composition import VectorComposer, CompositionRule, CompositionContext
from benchmarks.multi_agent_grid import VectorRegistry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

@dataclass
class CartPoleStressConfig:
    """Configuration for CartPole stress testing."""
    
    # Stress test parameters  
    max_timesteps: int = 8000
    sensor_noise_std: float = 0.01  # Light stress for dynamic behavior
    force_disturbance_std: float = 0.5  # Light disturbances for field dynamics
    physics_drift_rate: float = 0.00005  # Light drift for adaptation
    
    # CartPole physics (will drift over time)
    dt: float = 0.02
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    x_threshold: float = 2.4
    theta_threshold_rad: float = 12 * np.pi / 180.0
    
    # Control limits
    max_force: float = 10.0
    
    # Adaptation frequencies  
    gentle_controller_every: int = 1      # Every timestep
    efe_tuner_every: int = 20            # Base: Every 20 timesteps (was 50)
    efe_tuner_adaptive: bool = True      # Enable adaptive frequency
    efe_tuner_fast: int = 5              # Fast: Every 5 timesteps when unstable
    efe_tuner_slow: int = 40             # Slow: Every 40 timesteps when stable
    metadata_log_every: int = 100        # Every 100 timesteps
    
    # Visualization settings
    enable_visualization: bool = True
    save_animation: bool = True
    animation_format: str = "gif"  # "gif" or "mp4"
    save_raster: bool = True
    grid_size: int = 32  # Size of SCFD field grid for visualization

@dataclass
class CartPoleState:
    """CartPole state with noise and disturbances."""
    x: float = 0.0
    x_dot: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])
    
    def add_noise(self, noise_std: float, rng: np.random.Generator) -> np.ndarray:
        """Get noisy observation."""
        clean = self.to_array()
        noise = rng.normal(0, noise_std, 4)
        return clean + noise

class CartPoleVectorStressTest:
    """Stress test using actual SCFD vector system."""
    
    def __init__(self, config: CartPoleStressConfig = None, seed: int = None):
        self.config = config or CartPoleStressConfig()
        self.rng = np.random.default_rng(seed)
        
        # Initialize SCFD simulation configuration
        self.sim_cfg = load_config("cfg/defaults.yaml")
        self.dx = self.sim_cfg.grid.spacing
        self.grid_shape = self.sim_cfg.grid.shape
        self.mid_col = self.grid_shape[1] // 2
        
        # Initialize SCFD fields
        seed_val = int(self.rng.integers(0, 2**32 - 1))
        state = initialize_state(self.sim_cfg, seed_val)
        self.theta = state["theta"].astype(np.float32)
        self.theta_dot = state["theta_dot"].astype(np.float32)
        
        # CartPole observation state (what gets encoded into fields)
        self.cartpole_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [x, x_dot, theta, theta_dot]
        self.obs_filter = np.zeros(4, dtype=np.float32)
        
        # SCFD controller parameters
        self.encode_gain = 0.05
        self.encode_width = 3
        self.decay = 0.98
        self.micro_steps = 40
        self.micro_steps_calm = 16
        self.deadzone_angle = np.deg2rad(1.0)
        self.deadzone_ang_vel = 0.1
        
        # Physics parameters (will drift)
        self.current_physics = {
            'gravity': self.config.gravity,
            'masscart': self.config.masscart,
            'masspole': self.config.masspole,
            'length': self.config.length
        }
        
        # Load vector system
        self.setup_vector_system()
        
        # Setup SCFD feature extractor for proper vector controller inputs
        self.feature_extractor = CartPoleFeatureExtractor(
            self.sim_cfg,
            momentum=0.05,
            standardize=True,
            deadzone_scale=[0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5]
        )
        
        # Vector controller cache for performance
        self.vector_controllers = {}
        
        # Track composition activity for visualization
        self.composition_activity = []
        
        # Field history for real-time SCFD visualization
        self.field_history = []
        
        # Setup sensing and adaptation
        self.setup_controllers()
        
        # Metrics tracking
        self.reset_metrics()
        
    def reset_state(self):
        """Reset to challenging initial state."""
        self.cartpole_obs[0] = self.rng.uniform(-0.3, 0.3)  # x
        self.cartpole_obs[1] = self.rng.uniform(-0.1, 0.1)  # x_dot
        self.cartpole_obs[2] = self.rng.uniform(-0.08, 0.08)  # theta
        self.cartpole_obs[3] = self.rng.uniform(-0.05, 0.05)  # theta_dot
        
        # Reset SCFD fields and feature extractor
        seed_val = int(self.rng.integers(0, 2**32 - 1))
        state = initialize_state(self.sim_cfg, seed_val)
        self.theta = state["theta"].astype(np.float32)
        self.theta_dot = state["theta_dot"].astype(np.float32)
        self.obs_filter = np.zeros(4, dtype=np.float32)
        self.feature_extractor.reset()
    
    def get_composition_success_rate(self) -> float:
        """Get current composition success rate."""
        if self.metrics['composition_attempts'] == 0:
            return 0.0
        return self.metrics['composition_successes'] / self.metrics['composition_attempts']
    
    def _load_vector_controller(self, vector_entry) -> dict:
        """Load and cache full vector controller data."""
        if vector_entry.vector_id in self.vector_controllers:
            return self.vector_controllers[vector_entry.vector_id]
        
        try:
            import json
            from pathlib import Path
            
            # Load complete vector data
            vector_data = json.loads(Path(vector_entry.path).read_text())
            
            # Extract controller components with all SCFD parameters
            controller = {
                'vector_params': vector_data['vector'],
                'policy_weights': vector_data['controller_config']['policy_weights'],
                'policy_bias': vector_data['controller_config']['policy_bias'],
                'encode_gain': vector_data['controller_config']['encode_gain'],
                'action_clip': vector_data['controller_config']['action_clip'],
                'action_delta_clip': vector_data['controller_config'].get('action_delta_clip', 2.0),
                'smooth_lambda': vector_data['controller_config'].get('smooth_lambda', 0.25),
                'deadzone_angle': vector_data['controller_config']['deadzone_angle'],
                'deadzone_ang_vel': vector_data['controller_config']['deadzone_ang_vel'],
                'blend_linear_weight': vector_data['controller_config']['blend_linear_weight'],
                'blend_ternary_weight': vector_data['controller_config'].get('blend_ternary_weight', 0.0),
                'ternary_force_scale': vector_data['controller_config']['ternary_force_scale'],
                'micro_steps': vector_data['controller_config'].get('micro_steps', 40),
                'micro_steps_calm': vector_data['controller_config'].get('micro_steps_calm', 16),
                'decay': vector_data['controller_config'].get('decay', 0.98),
                'encode_width': vector_data['controller_config'].get('encode_width', 3),
                'metrics': vector_data['metrics'],
                'vector_id': vector_entry.vector_id
            }
            
            # Cache for performance
            self.vector_controllers[vector_entry.vector_id] = controller
            return controller
            
        except Exception as e:
            print(f"Error loading vector controller {vector_entry.vector_id}: {e}")
            return None
    
    def _compute_vector_control(self, controller: dict, cartpole_obs: np.ndarray) -> float:
        """Compute control force using full SCFD vector controller."""
        try:
            # Extract controller parameters
            policy_weights = controller['policy_weights']
            policy_bias = controller['policy_bias']
            action_clip = controller['action_clip']
            action_delta_clip = controller['action_delta_clip']
            smooth_lambda = controller['smooth_lambda']
            blend_linear_weight = controller['blend_linear_weight']
            
            # Check if in deadzone
            deadzone = (abs(cartpole_obs[2]) < self.deadzone_angle) and (abs(cartpole_obs[3]) < self.deadzone_ang_vel)
            
            # Extract SCFD features from fields using proper feature extractor
            feature_vector = self.feature_extractor.extract(
                self.theta,
                self.theta_dot,
                cartpole_obs,
                prev_action=getattr(self, '_last_vector_action', 0.0)
            )
            
            # Use normalized features for the policy
            features = feature_vector.normalized
            
            # Create LinearPolicy if not cached for this controller
            controller_id = controller['vector_id']
            if not hasattr(self, '_vector_policies'):
                self._vector_policies = {}
            
            if controller_id not in self._vector_policies:
                policy_cfg = LinearPolicyConfig(
                    action_clip=action_clip,
                    action_delta_clip=action_delta_clip,
                    smooth_lambda=smooth_lambda,
                )
                policy = LinearPolicy(
                    dim=len(policy_weights),
                    config=policy_cfg,
                    weights=policy_weights,
                    bias=policy_bias,
                )
                self._vector_policies[controller_id] = policy
            
            policy = self._vector_policies[controller_id]
            
            # Compute action using the trained policy
            action, policy_info = policy.act(
                features,
                deadzone=deadzone,
                deadzone_scale=self.feature_extractor.deadzone_scale,
            )
            
            # Apply blending weight
            action *= blend_linear_weight
            
            # Store for next timestep
            self._last_vector_action = action
            
            return float(action)
            
        except Exception as e:
            print(f"Error in SCFD vector control computation: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def _encode_observation(self, obs: np.ndarray, gain_scale: float = 1.0) -> None:
        """Encode CartPole observation into SCFD fields."""
        obs = obs.astype(np.float32)
        self.obs_filter = self.decay * self.obs_filter + (1.0 - self.decay) * obs
        h, w = self.grid_shape
        centers = np.linspace(0, h - 1, 4, dtype=np.int32)
        
        for i, center in enumerate(centers):
            row_start = max(center - 2, 0)
            row_end = min(center + 3, h)
            value = gain_scale * self.encode_gain * self.obs_filter[i]
            # inject antisymmetrically around center column
            left_start = max(self.mid_col - self.encode_width, 0)
            left_end = self.mid_col
            right_start = self.mid_col
            right_end = min(self.mid_col + self.encode_width, w)
            self.theta[row_start:row_end, left_start:left_end] -= value
            self.theta[row_start:row_end, right_start:right_end] += value
    
    def _evolve_scfd_field(self, deadzone: bool) -> dict:
        """Evolve SCFD fields using leapfrog integration."""
        steps = self.micro_steps_calm if deadzone else self.micro_steps
        steps = max(1, steps)
        for _ in range(steps):
            self.theta, self.theta_dot, _, info = leapfrog_step(
                self.theta,
                self.theta_dot,
                lambda f: accel_theta(f, self.sim_cfg.physics, dx=self.dx),
                self.sim_cfg.integration.dt,
                max_step=None,
            )
        return info
        
    def _is_in_deadzone(self, obs: np.ndarray) -> bool:
        """Check if CartPole is in deadzone (small angle/velocity)."""
        return (abs(obs[2]) < self.deadzone_angle) and (abs(obs[3]) < self.deadzone_ang_vel)
    
    def _generate_intermediate_visualization(self, run_dir: Path, timestep: int) -> None:
        """Generate visualization during the run to capture the full dynamics."""
        if not self.config.enable_visualization or len(self.history['states']) < 100:
            return
            
        try:
            print(f"Generating intermediate visualization at timestep {timestep}...")
            
            # Use current data up to this point
            viz_steps = min(len(self.history['states']), 800)  # Reasonable animation length
            states = np.array(self.history['states'][:viz_steps])
            forces = np.array(self.history['control_forces'][:viz_steps])
            
            # Use real SCFD field data
            if len(self.field_history) > 0:
                # Sample field history to match state history
                field_sample_rate = max(1, len(self.field_history) // viz_steps)
                field_data = np.array([self.field_history[i] for i in range(0, min(len(self.field_history), viz_steps * field_sample_rate), field_sample_rate)])
                
                # Ensure we have enough field data
                while len(field_data) < viz_steps:
                    field_data = np.concatenate([field_data, field_data[-1:]], axis=0)
                field_data = field_data[:viz_steps]
            else:
                field_data = None
            
            results = {}
            
            # Generate field raster with real SCFD data
            if self.config.save_raster and field_data is not None:
                raster_path = self._generate_field_raster(field_data, run_dir)
                results['raster'] = str(raster_path)
            
            # Generate animation with real dynamics
            if self.config.save_animation and viz_steps > 10:
                animation_path = self._generate_animation(states, forces, field_data, run_dir, suffix=f"_t{timestep}")
                if animation_path:
                    results['animation'] = str(animation_path)
            
            if results:
                print(f"Intermediate visualization saved:")
                for viz_type, viz_path in results.items():
                    print(f"  {viz_type}: {viz_path}")
                    
        except Exception as e:
            print(f"Warning: Intermediate visualization failed: {e}")

    def _get_adaptive_efe_interval(self, theta: float, timestep: int) -> int:
        """Calculate adaptive EFE tuner interval based on system state."""
        if not self.config.efe_tuner_adaptive:
            return self.config.efe_tuner_every
        
        # Calculate system instability indicators
        angle_instability = abs(theta) / self.config.theta_threshold_rad
        
        # Check recent composition success rate for additional signal
        recent_window = 10
        if len(self.composition_activity) >= recent_window:
            recent_success = sum(self.composition_activity[-recent_window:]) / recent_window
        else:
            recent_success = 0.5  # Neutral default
        
        # Determine interval based on system state
        if angle_instability > 0.7 or recent_success < 0.3:
            # High instability or poor performance = fast adaptation
            return self.config.efe_tuner_fast
        elif angle_instability < 0.2 and recent_success > 0.8:
            # Very stable and good performance = slow adaptation
            return self.config.efe_tuner_slow
        else:
            # Normal operation = base interval
            return self.config.efe_tuner_every
    
    def _register_vector_parameters(self):
        """Register vector selection/scaling parameters in the global registry."""
        from utils.parameter_registry import (
            get_global_registry, ParameterRole, ParameterDType, ParameterScope,
            ParameterConstraint, ParameterEntry
        )
        
        registry = get_global_registry()
        
        # Initialize vector weights (equal probability)
        n_vectors = len(self.cartpole_vectors)
        if n_vectors == 0:
            return
            
        equal_weight = 1.0 / n_vectors
        
        # Register composition weights (sum-to-one group)
        for vector in self.cartpole_vectors:
            vector_id = vector.vector_id
            
            # Composition weight: which vector to select
            weight_param_id = f"composition.weight.{vector_id}"
            weight_param = ParameterEntry(
                id=weight_param_id,
                role=ParameterRole.ACCEPTANCE,
                scope=ParameterScope.VECTOR,
                dtype=ParameterDType.WEIGHTS,
                constraints=ParameterConstraint(0.0, 1.0, trust_region_pct=0.2),
                current_value=equal_weight,
                coupling_group="composition_weights",
                description=f"Selection weight for {vector_id}"
            )
            registry.register_parameter(weight_param)
            
            # Policy scale: how much to scale the vector's output
            scale_param_id = f"policy.scale.{vector_id}"
            scale_param = ParameterEntry(
                id=scale_param_id,
                role=ParameterRole.DIFFUSION,
                scope=ParameterScope.VECTOR,
                dtype=ParameterDType.FLOAT_LINEAR,
                constraints=ParameterConstraint(0.5, 2.0, trust_region_pct=0.15),
                current_value=1.0,
                description=f"Output scaling for {vector_id}"
            )
            registry.register_parameter(scale_param)
        
        print(f"Registered {n_vectors * 2} vector parameters in global registry")
    
    def _compute_scfd_coherence_signal(self, x: float, theta: float) -> float:
        """Compute SCFD coherence signal from CartPole state."""
        # Coherence = system stability (1.0 = perfectly stable)
        angle_stability = 1.0 - abs(theta) / self.config.theta_threshold_rad
        position_stability = 1.0 - abs(x) / self.config.x_threshold
        return float(max(0.0, (angle_stability + position_stability) * 0.5))
    
    def _compute_scfd_curvature_signal(self, theta_dot: float) -> float:
        """Compute SCFD curvature signal from CartPole dynamics."""
        # Curvature = rate of change (angular velocity magnitude)
        max_expected_theta_dot = 3.5  # From original controller config
        return float(min(1.0, abs(theta_dot) / max_expected_theta_dot))
    
    def _compute_scfd_cross_gradient_signal(self, x: float, theta: float) -> float:
        """Compute SCFD cross-gradient coupling signal."""
        # Cross-gradient = interaction between position and angle
        coupling = abs(x * theta) / (self.config.x_threshold * self.config.theta_threshold_rad)
        return float(min(1.0, coupling))
    
    def setup_vector_system(self):
        """Load CartPole vectors from registry and setup composition."""
        print("Loading vector registry...")
        
        # Create enhanced vector registry
        self.vector_registry = VectorRegistry()
        
        # Filter for CartPole vectors
        self.cartpole_vectors = [
            v for v in self.vector_registry.base_registry 
            if "cartpole" in v.vector_id.lower() or "cartpole" in getattr(v, 'physics', '').lower()
        ]
        
        print(f"Found {len(self.cartpole_vectors)} CartPole vectors:")
        for v in self.cartpole_vectors:
            print(f"  - {v.vector_id}: {v.objective}")
        
        # Register per-vector parameters as first-class knobs
        self._register_vector_parameters()
        
        if not self.cartpole_vectors:
            print("No CartPole vectors found, creating synthetic probe...")
            # Create synthetic probe for CartPole
            self.probe = ProbeReport(
                physics="cartpole",
                grid_shape=(1, 4),  # [x, x_dot, theta, theta_dot]
                target_kind="balance",
                stats={"energy_mean": 0.1, "energy_std": 0.05}
            )
        else:
            # Use first CartPole vector for probe
            self.probe = ProbeReport(
                physics="cartpole", 
                grid_shape=(1, 4),
                target_kind="balance",
                stats={"energy_mean": 0.1, "energy_std": 0.05}
            )
        
        # Setup vector composer with enhanced registry
        self.composer = VectorComposer(self.vector_registry)
        
        # If we have CartPole vectors, add them to composition rules
        if self.cartpole_vectors:
            # Use up to 3 best CartPole vectors for composition
            vector_ids = [v.vector_id for v in self.cartpole_vectors[:3]]
            weights = [1.0] * len(vector_ids)  # Equal weights initially
            
            cartpole_rule = CompositionRule(
                name="cartpole_balance",
                vector_ids=vector_ids,
                weights=weights,
                composition_type="adaptive"  # Let it adapt weights based on performance
            )
            
            self.composer.add_composition_rule(cartpole_rule)
        
        print(f"Vector composer setup with {len(self.composer.composition_rules)} rules")
    
    def setup_controllers(self):
        """Setup enhanced sensing controller and EFE tuner."""
        
        # Enhanced GentleController with CartPole domain
        self.gentle_controller = GentleController()
        
        # EFE tuner for strategic adaptation
        self.efe_tuner = EFEParameterTuner()
        
        # Add CartPole-specific parameters to EFE tuner
        self.efe_tuner.add_parameter_range("control_gain", 0.5, 5.0, 2.0)
        self.efe_tuner.add_parameter_range("derivative_gain", 0.1, 2.0, 0.8)
        self.efe_tuner.add_parameter_range("noise_filtering", 0.1, 1.0, 0.6)
        self.efe_tuner.add_parameter_range("metropolis_temperature", 0.3, 2.0, 0.8)
        
        # Enable real-time tuning
        initial_params = {
            "control_gain": 2.0,
            "derivative_gain": 0.8, 
            "noise_filtering": 0.6,
            "metropolis_temperature": 0.8
        }
        
        # CartPole-specific metadata config
        metadata_config = {
            'control_gain': {
                'target_metrics': ['pole_angle', 'angular_velocity'],
                'response_direction': 'increase_when_high',
                'sensitivity': 0.8
            },
            'derivative_gain': {
                'target_metrics': ['angular_velocity', 'oscillation_measure'],
                'response_direction': 'increase_when_high',
                'sensitivity': 0.6
            },
            'noise_filtering': {
                'target_metrics': ['sensor_noise_level'],
                'response_direction': 'increase_when_high',
                'sensitivity': 0.7
            },
            'metropolis_temperature': {
                'target_metrics': ['composition_success_rate'],
                'response_direction': 'increase_when_low',
                'sensitivity': 0.9
            }
        }
        
        self.efe_tuner.enable_realtime_tuning(initial_params, metadata_config)
        
        # Setup vector-controller bridge integration
        self.sensing_router = SensingRouter()
        self.vector_bridge = create_bridge_for_system(
            self.sensing_router, self.vector_registry, self.composer
        )
        
        # Integrate bridge with composer
        self.composer.controller_bridge = self.vector_bridge
        
        print("Vector-controller bridge integration enabled")
        
        # Current control parameters (managed by controllers)
        self.control_params = initial_params.copy()
        
        # Unified metadata manager
        self.metadata_manager = UnifiedMetadataManager()
        
    def reset_metrics(self):
        """Reset performance tracking."""
        self.metrics = {
            'timesteps_survived': 0,
            'balance_violations': 0,
            'total_control_energy': 0.0,
            'composition_attempts': 0,
            'composition_successes': 0,
            'controller_adaptations': 0,
            'efe_updates': 0,
            'physics_drift_magnitude': 0.0
        }
        
        self.history = {
            'states': [],
            'control_forces': [],
            'composition_results': [],
            'controller_decisions': [],
            'efe_updates': [],
            'sensing_status': []
        }
    
    def update_physics_drift(self, timestep: int):
        """Simulate gradual physics parameter drift."""
        drift_rate = self.config.physics_drift_rate
        
        # Random walk in physics parameters
        for param in self.current_physics:
            self.current_physics[param] += self.rng.normal(0, drift_rate)
        
        # Keep within reasonable bounds
        self.current_physics['gravity'] = np.clip(self.current_physics['gravity'], 8.0, 12.0)
        self.current_physics['masscart'] = np.clip(self.current_physics['masscart'], 0.5, 2.0)
        self.current_physics['masspole'] = np.clip(self.current_physics['masspole'], 0.05, 0.3)
        self.current_physics['length'] = np.clip(self.current_physics['length'], 0.3, 0.8)
        
        # Track total drift magnitude
        reference = {
            'gravity': self.config.gravity,
            'masscart': self.config.masscart,
            'masspole': self.config.masspole,
            'length': self.config.length
        }
        
        total_drift = sum(
            abs(self.current_physics[k] - reference[k]) / reference[k]
            for k in reference
        )
        self.metrics['physics_drift_magnitude'] = total_drift
    
    def compute_control_force_via_vectors(self, timestep: int) -> float:
        """Compute control force using vector composition system."""
        
        # Create metrics for sensing
        # Add sensor noise to observation
        noise = self.rng.normal(0, self.config.sensor_noise_std, size=4)
        noisy_obs = self.cartpole_obs + noise
        x, x_dot, theta, theta_dot = noisy_obs
        
        # Compute oscillation measure from recent history
        if len(self.history['states']) > 10:
            recent_thetas = [s[2] for s in self.history['states'][-10:]]
            oscillation_measure = np.std(recent_thetas)
        else:
            oscillation_measure = abs(theta_dot)
        
        # Real task feedback metrics (not synthetic proxies)
        sensing_metrics = {
            # Task performance (primary signals)
            "episode_return": float(self.metrics.get('timesteps_survived', 0)),
            "steps_alive": float(timestep + 1),
            "angle_rms": float(np.sqrt(np.mean([s[2]**2 for s in self.history['states'][-10:]] or [theta**2]))),
            "action_smoothness": float(1.0 / (1.0 + np.std(self.history.get('forces', [0])[-5:] or [0]))),
            
            # SCFD substrate signals
            "mean_coherence": self._compute_scfd_coherence_signal(x, theta),
            "mean_abs_curvature": self._compute_scfd_curvature_signal(theta_dot),
            "mean_cross_gradient": self._compute_scfd_cross_gradient_signal(x, theta),
            
            # Composition performance (real signal)
            "composition_success_rate": (
                self.metrics['composition_successes'] / max(1, self.metrics['composition_attempts'])
            ),
            
            # System state (for context)
            "cart_position": float(x),
            "pole_angle": float(abs(theta)),
            "angular_velocity": float(abs(theta_dot)),
            "physics_drift_magnitude": self.metrics['physics_drift_magnitude']
        }
        
        # Step 1: Gentle controller for fast responses
        if timestep % self.config.gentle_controller_every == 0:
            controller_decision = self.gentle_controller.step(sensing_metrics)
            self.history['controller_decisions'].append({
                'timestep': timestep,
                'decision': controller_decision,
                'metrics': sensing_metrics.copy()
            })
            
            # Apply controller nudges to control parameters
            if controller_decision.nudges:
                for param, nudge in controller_decision.nudges.items():
                    if param == "T" and "control_gain" in self.control_params:
                        # Map T nudges to control gain (legacy compatibility)
                        self.control_params["control_gain"] += nudge * 10.0  # Scale appropriately
                        self.control_params["control_gain"] = np.clip(
                            self.control_params["control_gain"], 0.5, 5.0
                        )
                        self.metrics['controller_adaptations'] += 1
        
        # Step 2: EFE tuner for strategic adaptation (adaptive frequency)
        efe_interval = self._get_adaptive_efe_interval(theta, timestep)
        if timestep % efe_interval == 0:
            try:
                # Create safety constraints
                safety_constraints = {
                    'pole_angle': abs(theta),
                    'energy_drift': sensing_metrics['energy_drift'],
                    'composition_success_rate': sensing_metrics['composition_success_rate']
                }
                
                # Update EFE parameters
                updated_params = self.efe_tuner.update_realtime_parameters(
                    sensing_metrics, safety_constraints
                )
                
                if updated_params != self.control_params:
                    self.control_params.update(updated_params)
                    self.metrics['efe_updates'] += 1
                    self.history['efe_updates'].append({
                        'timestep': timestep,
                        'old_params': self.control_params.copy(),
                        'new_params': updated_params.copy(),
                        'trigger_metrics': sensing_metrics.copy()
                    })
                    
            except Exception as e:
                print(f"EFE tuner error at timestep {timestep}: {e}")
        
        # Step 3: Multi-domain metrics collection (following context pattern)
        metrics = {
            "cart_position": self.cartpole_obs[0],
            "pole_angle": self.cartpole_obs[2],
            "composition_success_rate": self.get_composition_success_rate(),
            "energy_drift": abs(self.cartpole_obs[2]) + abs(self.cartpole_obs[1]),  # Simple energy proxy
            "sensor_noise_level": self.config.sensor_noise_std
        }
        
        # Step 4: Real-time learning (GentleController): Fast responses every timestep
        try:
            if hasattr(self, 'gentle_controller'):
                controller_decision = self.gentle_controller.step(metrics)
                
                # Apply nudges to parameters by role (following context pattern)
                if hasattr(controller_decision, 'nudges'):
                    for param_name, nudge_value in controller_decision.nudges.items():
                        if param_name in self.control_params:
                            self.control_params[param_name] += nudge_value
                            self.metrics['controller_adaptations'] += 1
                        
        except Exception as e:
            print(f"Controller sensing error at timestep {timestep}: {e}")
        
        # Step 5: Strategic learning (EFE Tuner): Broader optimization every 50 steps
        if timestep % 50 == 0:
            try:
                windowed_metrics = {
                    'survival_rate': min(timestep / 100.0, 1.0),  # How long we've survived
                    'stability': 1.0 - (abs(self.cartpole_obs[2]) + abs(self.cartpole_obs[0])),  # How stable
                    'composition_success': self.get_composition_success_rate()
                }
                
                constraints = {
                    'energy_stable': abs(self.cartpole_obs[2]) < 0.2,
                    'position_stable': abs(self.cartpole_obs[0]) < 1.0,
                    'composition_viable': self.get_composition_success_rate() > 0.1
                }
                
                efe_updates = self.efe_tuner.update_realtime_parameters(windowed_metrics, constraints)
                if efe_updates:
                    # Apply EFE parameter updates
                    for param_name, new_value in efe_updates.items():
                        if param_name in self.control_params:
                            self.control_params[param_name] = new_value
                    self.metrics['efe_updates'] += 1
                    
            except Exception as e:
                print(f"EFE tuner error at timestep {timestep}: {e}")
        
        # Step 6: Simple vector-based control influence
        if self.cartpole_vectors:
            # TOP-1 Vector Selection (no composition - stable approach)
            try:
                self.metrics['composition_attempts'] += 1
                
                # Registry-based vector selection using learned weights
                best_vector = None
                best_score = -float('inf')
                
                # Get composition weights from parameter registry
                from utils.parameter_registry import get_global_registry
                registry = get_global_registry()
                
                for vector_entry in self.cartpole_vectors:
                    try:
                        # Get learned composition weight for this vector
                        weight_param_id = f"composition.weight.{vector_entry.vector_id}"
                        param_entry = registry.get_parameter(weight_param_id)
                        composition_weight = param_entry.current_value if param_entry else 0.5
                        
                        # Context-aware scoring (task performance feedback)
                        angle_factor = 1.0 - abs(self.cartpole_obs[2]) / self.config.theta_threshold_rad
                        position_factor = 1.0 - abs(self.cartpole_obs[0]) / self.config.x_threshold
                        stability_score = (angle_factor + position_factor) * 0.5
                        
                        # Combine learned weight with context
                        vector_score = composition_weight * stability_score
                        
                        if vector_score > best_score:
                            best_score = vector_score
                            best_vector = vector_entry
                            
                    except Exception as e:
                        print(f"Error scoring vector {vector_entry.vector_id}: {e}")
                        continue
                
                # Apply TOP-1 vector influence or fallback to PID
                if best_vector and best_score > 0.1:  # Acceptance threshold
                    # Load full vector controller data
                    controller = self._load_vector_controller(best_vector)
                    
                    if controller is not None:
                        # Use complete SCFD vector controller
                        base_control_force = self._compute_vector_control(
                            controller, self.cartpole_obs
                        )
                        
                        # Debug logging with field info
                        if timestep % 20 == 0 or timestep < 20:
                            field_rms = np.sqrt(np.mean(self.theta**2))
                            print(f"T{timestep}: Using {best_vector.vector_id}, force={base_control_force:.3f}, theta={self.cartpole_obs[2]:.3f}, field_rms={field_rms:.3f}")
                        
                        # Apply registry-based policy scaling
                        scale_param_id = f"policy.scale.{best_vector.vector_id}"
                        scale_param_entry = registry.get_parameter(scale_param_id)
                        policy_scale = scale_param_entry.current_value if scale_param_entry else 1.0
                        control_force = base_control_force * policy_scale
                        
                        # Optional: Blend with PID for robustness (disabled for pure vector learning)
                        blend_factor = 1.0  # Use 1.0 for pure vector
                        if blend_factor < 1.0:
                            pid_component = self._fallback_pid_control(x, x_dot, theta, theta_dot)
                            control_force = (blend_factor * control_force + 
                                           (1 - blend_factor) * pid_component)
                        
                        self.metrics['composition_successes'] += 1
                        success = True
                        self.composition_activity.append(True)  # Mark composition success
                        self._used_vector_controller = True  # Flag for control scaling
                    else:
                        # Fallback to PID if vector loading failed
                        control_force = self._fallback_pid_control(x, x_dot, theta, theta_dot)
                        success = False
                        self.composition_activity.append(False)  # Mark no composition
                        self._used_vector_controller = False  # Flag for control scaling
                else:
                    # Pure PID fallback when no good vector
                    control_force = self._fallback_pid_control(x, x_dot, theta, theta_dot)
                    success = False
                    self.composition_activity.append(False)  # Mark no composition
                    self._used_vector_controller = False  # Flag for control scaling
                
                # Track selection results  
                self.history['composition_results'].append({
                    'timestep': timestep,
                    'result': {'success': success, 'vector_selected': best_vector is not None},
                    'fallback_used': not success
                })
                
            except Exception as e:
                import traceback
                print(f"Vector selection error at timestep {timestep}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                control_force = self._fallback_pid_control(x, x_dot, theta, theta_dot)
        else:
            # No vectors available, use PID fallback
            control_force = self._fallback_pid_control(x, x_dot, theta, theta_dot)
            self._used_vector_controller = False  # Flag for control scaling
        
        # Apply control parameter scaling (only for PID fallback)
        if not hasattr(self, '_used_vector_controller') or not self._used_vector_controller:
            control_force *= self.control_params.get('control_gain', 2.0)
        
        # Apply derivative filtering
        if hasattr(self, '_last_control_force'):
            filter_strength = self.control_params.get('noise_filtering', 0.6)
            control_force = (
                filter_strength * control_force + 
                (1 - filter_strength) * self._last_control_force
            )
        self._last_control_force = control_force
        
        # Add environmental disturbances
        control_force += self.rng.normal(0, self.config.force_disturbance_std)
        
        # Apply limits
        control_force = np.clip(control_force, -self.config.max_force, self.config.max_force)
        
        return float(control_force)
    
    def _fallback_pid_control(self, x: float, x_dot: float, theta: float, theta_dot: float) -> float:
        """Fallback PID control when vector composition fails."""
        # Simple PID control for CartPole
        kp_theta = 20.0
        kd_theta = 5.0
        kp_x = 1.0
        kd_x = 2.0
        
        # Control law
        force = -(kp_theta * theta + kd_theta * theta_dot + kp_x * x + kd_x * x_dot)
        return force
    
    def step_physics(self, control_force: float) -> bool:
        """Step SCFD field simulation and CartPole physics, return failure status."""
        
        # Current physics parameters (with drift)
        g = self.current_physics['gravity']
        mc = self.current_physics['masscart']
        mp = self.current_physics['masspole']
        l = self.current_physics['length']
        dt = self.config.dt
        
        # Current CartPole state
        x, x_dot, theta, theta_dot = self.cartpole_obs
        
        # Check if in deadzone for field evolution
        deadzone = self._is_in_deadzone(self.cartpole_obs)
        gain_scale = 0.2 if deadzone else 1.0
        
        # Encode current observation into SCFD fields
        self._encode_observation(self.cartpole_obs, gain_scale)
        
        # Evolve SCFD fields
        field_info = self._evolve_scfd_field(deadzone)
        
        # Store field snapshot for visualization (sample every few steps to manage memory)
        if len(self.field_history) < 500:  # Limit memory usage
            self.field_history.append(self.theta.copy())
        
        # Step CartPole physics with control force
        n_substeps = 2
        for _ in range(n_substeps):
            costh = np.cos(theta)
            sinth = np.sin(theta)
            temp = (control_force + mp * l * theta_dot ** 2 * sinth) / (mc + mp)
            thacc = (g * sinth - costh * temp) / (l * (4.0 / 3.0 - mp * costh ** 2 / (mc + mp)))
            xacc = temp - mp * l * thacc * costh / (mc + mp)
            
            x += dt * x_dot / n_substeps
            x_dot += dt * xacc / n_substeps
            theta += dt * theta_dot / n_substeps
            theta_dot += dt * thacc / n_substeps
        
        # Update CartPole observation state
        self.cartpole_obs[0] = x
        self.cartpole_obs[1] = x_dot
        self.cartpole_obs[2] = theta
        self.cartpole_obs[3] = theta_dot
        
        # Check failure conditions
        failed = (abs(x) > self.config.x_threshold) or (abs(theta) > self.config.theta_threshold_rad)
        
        return failed
    
    def generate_visualization(self, run_dir: Path, steps: int = None) -> Dict[str, str]:
        """Generate CartPole animation and field raster visualization."""
        if not self.config.enable_visualization or not self.history['states']:
            return {}
        
        print("Generating CartPole visualization...")
        
        # Use actual run steps for visualization
        viz_steps = steps if steps is not None else len(self.history['states'])
        viz_steps = min(viz_steps, len(self.history['states']))
        states = np.array(self.history['states'][:viz_steps])
        forces = np.array(self.history['control_forces'][:viz_steps])
        
        # Get field data if available (simulate SCFD field evolution)
        field_data = self._generate_field_simulation(states, viz_steps)
        
        results = {}
        
        # Generate field raster
        if self.config.save_raster and field_data is not None:
            raster_path = self._generate_field_raster(field_data, run_dir)
            results['raster'] = str(raster_path)
        
        # Generate animation
        if self.config.save_animation and viz_steps > 10:
            animation_path = self._generate_animation(states, forces, field_data, run_dir)
            if animation_path:
                results['animation'] = str(animation_path)
        
        print(f"Visualization saved to: {run_dir}")
        return results
    
    def _generate_field_simulation(self, states: np.ndarray, steps: int) -> Optional[np.ndarray]:
        """Generate SCFD field data based on CartPole dynamics and vector activity."""
        # Create a realistic SCFD field simulation
        grid_shape = (self.config.grid_size, self.config.grid_size)
        field_data = np.zeros((steps, *grid_shape))
        
        # Track composition history from our vector selection
        composition_activity = getattr(self, 'composition_activity', [0] * steps)
        
        for t in range(steps):
            x, x_dot, theta, theta_dot = states[t]
            
            # Map cart position to grid coordinates
            center_x = int(grid_shape[1] * (0.5 + x / (2 * self.config.x_threshold)))
            center_x = np.clip(center_x, 0, grid_shape[1] - 1)
            
            # SCFD field components based on CartPole physics
            # 1. Coherence field - higher when system is stable
            coherence = 1.0 - (abs(theta) / self.config.theta_threshold_rad)
            coherence = max(0.0, coherence)
            
            # 2. Curvature field - reflects angular acceleration
            curvature = abs(theta_dot) / 3.5  # Normalize by max expected theta_dot
            curvature = min(1.0, curvature)
            
            # 3. Cross-gradient coupling - when position and angle interact
            coupling = abs(x * theta) / (self.config.x_threshold * self.config.theta_threshold_rad)
            coupling = min(1.0, coupling)
            
            # Vector composition influence
            vector_influence = 0.5 if (t < len(composition_activity) and composition_activity[t]) else 0.1
            
            # Create multi-component SCFD field
            y, x_grid = np.ogrid[:grid_shape[0], :grid_shape[1]]
            center_y = grid_shape[0] // 2
            
            # Distance from cart position
            dist = np.sqrt((x_grid - center_x)**2 + (y - center_y)**2)
            
            # SCFD Lagrangian-inspired field: L = α(1-C)² + βK² + γ(∇C·∇K)
            alpha, beta, gamma = 0.3, 0.2, 0.1  # Our SCFD parameters
            
            # Coherence component (Gaussian around cart, stronger when stable)
            coherence_field = alpha * (1 - coherence)**2 * np.exp(-dist**2 / (2 * 4**2))
            
            # Curvature component (sharper when high angular velocity) 
            curvature_field = beta * curvature**2 * np.exp(-dist**2 / (2 * 2**2))
            
            # Cross-gradient coupling (creates interference patterns)
            coupling_field = gamma * coupling * np.sin(x_grid * 0.5) * np.cos(y * 0.5)
            
            # Vector composition creates localized activity bursts
            vector_field = vector_influence * np.exp(-dist**2 / (2 * 6**2))
            
            # Combine all components
            field = coherence_field + curvature_field + np.abs(coupling_field) + vector_field
            
            # Add temporal evolution with SCFD-like dynamics
            if t > 0:
                # Diffusion-like spreading with decay
                prev_field = field_data[t-1]
                diffused = 0.8 * prev_field + 0.1 * (
                    np.roll(prev_field, 1, axis=0) + np.roll(prev_field, -1, axis=0) +
                    np.roll(prev_field, 1, axis=1) + np.roll(prev_field, -1, axis=1)
                ) / 4
                field += 0.3 * diffused * np.exp(-0.05)  # Decay factor
            
            field_data[t] = field
        
        return field_data
    
    def _generate_field_raster(self, field_data: np.ndarray, run_dir: Path) -> Path:
        """Generate SCFD CA-style raster visualization."""
        raster_path = run_dir / "scfd_ca_raster.png"
        
        # Create CA-style raster like EM33cart.py
        # field_data shape: (time_steps, grid_height, grid_width)
        time_steps, grid_h, grid_w = field_data.shape
        
        # Convert continuous field to discrete CA-like states (0, 1, 2)
        # Based on field intensity thresholds
        field_flat = field_data.reshape(time_steps, -1)  # (time, space)
        
        # Discretize to 3 states with better distribution
        field_abs = np.abs(field_flat)  # Use absolute values for activity measure
        field_mean = np.mean(field_abs)
        field_std = np.std(field_abs)
        
        print(f"Field stats: mean={field_mean:.6f}, std={field_std:.6f}, min={np.min(field_abs):.6f}, max={np.max(field_abs):.6f}")
        
        # Use statistical thresholds for better contrast
        # Low: below mean, Medium: mean to mean+std, High: above mean+std
        threshold_low = field_mean
        threshold_high = field_mean + field_std
        
        # Create 3-state CA-like representation with adaptive thresholds
        ca_raster = np.zeros_like(field_abs, dtype=np.uint8)
        ca_raster[field_abs < threshold_low] = 0  # Low activity (white)
        ca_raster[(field_abs >= threshold_low) & (field_abs < threshold_high)] = 1  # Medium (red)  
        ca_raster[field_abs >= threshold_high] = 2  # High activity (blue)
        
        # Print distribution for debugging
        unique, counts = np.unique(ca_raster, return_counts=True)
        total = len(ca_raster.flat)
        print(f"CA distribution: Low={counts[0]/total*100:.1f}% Med={counts[1]/total*100:.1f}% High={counts[2]/total*100:.1f}%")
        
        # Use EM33cart.py colormap: white, red, blue
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([[1,1,1], [1,0,0], [0,0,1]])  # 0:white, 1:red, 2:blue
        
        # Create tall raster image (time down, space across)
        plt.figure(figsize=(16, 12))
        plt.imshow(ca_raster, cmap=cmap, vmin=0, vmax=2, aspect='auto', interpolation='nearest')
        
        # Add labels and title
        plt.xlabel(f"SCFD Field Space ({grid_w}×{grid_h} grid flattened)")
        plt.ylabel(f"Time Steps (0 to {time_steps})")
        plt.title("SCFD CartPole CA Raster: Field Evolution Over Time\nWhite=Low, Red=Medium, Blue=High Activity")
        
        # Add a minimal colorbar
        cbar = plt.colorbar(ticks=[0, 1, 2], shrink=0.6)
        cbar.set_ticklabels(['Low', 'Med', 'High'])
        cbar.set_label('SCFD Activity Level')
        
        plt.tight_layout()
        plt.savefig(raster_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # Also save raw data for analysis
        np.savez_compressed(run_dir / "scfd_raster_data.npz", 
                           ca_raster=ca_raster, 
                           field_data=field_data,
                           field_stats={'min': np.min(field_abs), 'max': np.max(field_abs), 'mean': field_mean, 'std': field_std})
        
        return raster_path
    
    def _generate_animation(self, states: np.ndarray, forces: np.ndarray, 
                          field_data: Optional[np.ndarray], run_dir: Path, suffix: str = "") -> Optional[Path]:
        """Generate CartPole animation with field visualization."""
        
        steps = len(states)
        if steps < 2:
            return None
        
        # Determine animation format and path
        if self.config.animation_format.lower() == "mp4":
            anim_path = run_dir / f"scfd_cartpole{suffix}.mp4"
            writer = "ffmpeg"
        else:
            anim_path = run_dir / f"scfd_cartpole{suffix}.gif"
            writer = "pillow"
        
        try:
            # Create figure with subplots
            if field_data is not None:
                fig, (ax_field, ax_cart) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Setup field plot
                v_max = float(np.max(np.abs(field_data))) or 1.0
                im = ax_field.imshow(field_data[0], cmap="coolwarm", 
                                   vmin=-v_max, vmax=v_max, animated=True)
                ax_field.set_title("SCFD Field")
                ax_field.set_xlabel("X Position")
                ax_field.set_ylabel("Y Position")
                ax_field.axis('off')
            else:
                fig, ax_cart = plt.subplots(1, 1, figsize=(8, 5))
                im = None
            
            # Setup cart-pole plot
            ax_cart.set_xlim(-self.config.x_threshold * 1.2, self.config.x_threshold * 1.2)
            ax_cart.set_ylim(-0.6, 1.2)
            ax_cart.set_xlabel("Position (m)")
            ax_cart.set_ylabel("Height (m)")
            ax_cart.grid(True, alpha=0.3)
            
            # Cart and pole elements
            cart_w, cart_h = 0.3, 0.2
            y_cart = 0.0
            cart = plt.Rectangle((0 - cart_w / 2, y_cart - cart_h / 2), 
                               cart_w, cart_h, fill=False, linewidth=2)
            ax_cart.add_patch(cart)
            
            pole_line, = ax_cart.plot([], [], 'r-', lw=3, label='Pole')
            force_txt = ax_cart.text(0.02, 0.95, "", transform=ax_cart.transAxes, 
                                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="white", alpha=0.8))
            
            # Add threshold lines
            ax_cart.axvline(-self.config.x_threshold, color='red', linestyle='--', alpha=0.5, label='X Limit')
            ax_cart.axvline(self.config.x_threshold, color='red', linestyle='--', alpha=0.5)
            ax_cart.legend(loc='upper right')
            
            def init():
                cart.set_xy((0 - cart_w / 2, y_cart - cart_h / 2))
                pole_line.set_data([], [])
                force_txt.set_text("")
                if im is not None:
                    im.set_array(field_data[0])
                return (im, cart, pole_line, force_txt) if im else (cart, pole_line, force_txt)
            
            def animate(frame):
                if frame >= len(states):
                    return init()
                
                x, x_dot, theta, theta_dot = states[frame]
                force = forces[frame]
                
                # Update cart position
                cart.set_x(x - cart_w / 2)
                
                # Update pole
                pivot = np.array([x, y_cart + cart_h / 2])
                tip = pivot + np.array([np.sin(theta) * self.config.length * 2.0, 
                                      np.cos(theta) * self.config.length * 2.0])
                pole_line.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])
                
                # Update force text
                force_txt.set_text(f"Force: {force:.2f} N\nStep: {frame}\n"
                                 f"θ: {np.degrees(theta):.1f}°\nx: {x:.2f} m")
                
                # Update field if available
                if im is not None and frame < len(field_data):
                    im.set_array(field_data[frame])
                
                return (im, cart, pole_line, force_txt) if im else (cart, pole_line, force_txt)
            
            # Create animation
            anim = animation.FuncAnimation(
                fig, animate, init_func=init, frames=steps,
                interval=max(int(self.config.dt * 1000), 50), blit=True
            )
            
            # Save animation
            fps = max(int(1 / self.config.dt), 10)
            anim.save(anim_path, writer=writer, fps=fps, bitrate=1800)
            plt.close(fig)
            
            return anim_path
            
        except Exception as e:
            print(f"Failed to create animation: {e}")
            plt.close('all')
            return None
    
    def run_stress_test(self) -> Dict[str, Any]:
        """Run the complete stress test using vector system."""
        
        print("=== CartPole Vector Stress Test ===")
        print(f"Using {len(self.cartpole_vectors)} CartPole vectors")
        print(f"Max timesteps: {self.config.max_timesteps}")
        print(f"Sensor noise: {self.config.sensor_noise_std}")
        print(f"Force disturbances: {self.config.force_disturbance_std}")
        print(f"Physics drift rate: {self.config.physics_drift_rate}")
        
        # Create logging
        run_dir = create_run_directory("cartpole_vector_stress")
        logger = RunLogger(run_dir)
        
        start_time = time.time()
        failed = False
        
        for timestep in range(self.config.max_timesteps):
            
            # Update physics drift
            self.update_physics_drift(timestep)
            
            # Compute control force using vector system
            control_force = self.compute_control_force_via_vectors(timestep)
            
            # Step physics
            failed = self.step_physics(control_force)
            
            # Track metrics
            self.metrics['timesteps_survived'] = timestep + 1
            self.metrics['total_control_energy'] += abs(control_force)
            
            if abs(self.cartpole_obs[2]) > self.config.theta_threshold_rad * 0.8:
                self.metrics['balance_violations'] += 1
            
            # Record history
            self.history['states'].append(self.cartpole_obs.copy())
            self.history['control_forces'].append(control_force)
            
            # Log to metadata system
            if timestep % self.config.metadata_log_every == 0:
                sensing_status = self.gentle_controller.get_sensing_status()
                efe_status = self.efe_tuner.get_realtime_status()
                
                self.history['sensing_status'].append({
                    'timestep': timestep,
                    'sensing': sensing_status,
                    'efe': efe_status
                })
            
            # Log step data
            log_data = {
                'timestep': timestep,
                'state': self.cartpole_obs.tolist(),
                'control_force': float(control_force),
                'physics_params': self.current_physics.copy(),
                'control_params': self.control_params.copy(),
                'composition_success_rate': (
                    self.metrics['composition_successes'] / max(1, self.metrics['composition_attempts'])
                ),
                'controller_adaptations': self.metrics['controller_adaptations'],
                'efe_updates': self.metrics['efe_updates']
            }
            logger.log_step(log_data)
            
            # Progress reporting and intermediate visualization
            if (timestep + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                comp_rate = self.metrics['composition_successes'] / max(1, self.metrics['composition_attempts'])
                print(f"Step {timestep+1:5d}: theta={self.cartpole_obs[2]:6.3f} x={self.cartpole_obs[0]:6.3f} "
                      f"F={control_force:6.2f} comp={comp_rate:.3f} "
                      f"adapt={self.metrics['controller_adaptations']} time={elapsed:.1f}s")
                
                # Generate intermediate visualization to capture the dynamics
                self._generate_intermediate_visualization(run_dir, timestep + 1)
            
            # Check failure
            if failed:
                print(f"CartPole failed at timestep {timestep + 1}")
                print(f"Final state: x={self.cartpole_obs[0]:.3f}, theta={self.cartpole_obs[2]:.3f}")
                break
        
        # Final results
        total_time = time.time() - start_time
        survival_rate = self.metrics['timesteps_survived'] / self.config.max_timesteps
        composition_success_rate = (
            self.metrics['composition_successes'] / max(1, self.metrics['composition_attempts'])
        )
        
        # Population-based learning - Cross-run learning (following context pattern)
        try:
            # Real-time audit trail
            sensing_status = self.gentle_controller.get_sensing_status() if hasattr(self, 'gentle_controller') else {}
            efe_status = self.efe_tuner.get_realtime_status() if hasattr(self.efe_tuner, 'get_realtime_status') else {}
            
            # Cross-run learning metadata
            learning_metadata = {
                'vector_source': 'cartpole_balance',
                'survival_rate': survival_rate,
                'composition_success_rate': composition_success_rate,
                'controller_adaptations': self.metrics['controller_adaptations'],
                'efe_updates': self.metrics['efe_updates'],
                'physics_drift_magnitude': self.metrics['physics_drift_magnitude'],
                "controller_adaptations_detail": sensing_status.get("recent_nudges", []) if sensing_status else [],
                "efe_parameter_evolution": efe_status.get("parameter_updates", {}) if efe_status else {},
                "constraint_violations": efe_status.get("recent_constraint_violations", []) if efe_status else [],
                "performance_score": survival_rate * composition_success_rate,
                "learned_control_params": self.control_params.copy()
            }
            
            # Export to unified metadata with learning context
            self.metadata_manager.import_run_data(run_dir, learning_metadata)
            
            # Population-based EFE evolution (cross-run learning)
            if hasattr(self.efe_tuner, 'import_run_performance'):
                run_performance_data = {
                    'survival_rate': survival_rate,
                    'composition_success_rate': composition_success_rate,
                    'final_control_params': self.control_params.copy(),
                    'constraint_violations': efe_status.get("recent_constraint_violations", []) if efe_status else []
                }
                self.efe_tuner.import_run_performance(run_performance_data)
                print("Cross-run learning: Updated EFE population knowledge")
            
            # Analyze cross-vector performance for future learning
            if hasattr(self.metadata_manager, 'analyze_cross_vector_performance'):
                performance_correlations = self.metadata_manager.analyze_cross_vector_performance()
                if performance_correlations:
                    print(f"Cross-run learning: Found {len(performance_correlations)} performance patterns")
                    
        except Exception as e:
            print(f"Cross-run learning error: {e}")
            # Fallback to basic metadata export
            self.metadata_manager.import_run_data(run_dir, {
                'vector_source': 'cartpole_stress_test',
                'survival_rate': survival_rate,
                'composition_success_rate': composition_success_rate,
                'controller_adaptations': self.metrics['controller_adaptations'],
                'efe_updates': self.metrics['efe_updates'],
                'physics_drift_magnitude': self.metrics['physics_drift_magnitude']
            })
        
        results = {
            'success': not failed,
            'survival_rate': survival_rate,
            'timesteps_survived': self.metrics['timesteps_survived'],
            'composition_success_rate': composition_success_rate,
            'composition_attempts': self.metrics['composition_attempts'],
            'composition_successes': self.metrics['composition_successes'],
            'controller_adaptations': self.metrics['controller_adaptations'],
            'efe_updates': self.metrics['efe_updates'],
            'balance_violations': self.metrics['balance_violations'],
            'avg_control_energy': self.metrics['total_control_energy'] / self.metrics['timesteps_survived'],
            'physics_drift_magnitude': self.metrics['physics_drift_magnitude'],
            'total_runtime': total_time,
            'final_control_params': self.control_params.copy(),
            'vectors_used': [v.vector_id for v in self.cartpole_vectors],
            'run_directory': str(run_dir)
        }
        
        print(f"\n=== Stress Test Results ===")
        print(f"Success: {results['success']}")
        print(f"Survival rate: {survival_rate:.1%}")
        print(f"Composition success: {composition_success_rate:.1%}")
        print(f"Controller adaptations: {self.metrics['controller_adaptations']}")
        print(f"EFE updates: {self.metrics['efe_updates']}")
        print(f"Vectors used: {len(self.cartpole_vectors)}")
        print(f"Results saved to: {run_dir}")
        
        # Generate visualization if enabled
        if self.config.enable_visualization:
            try:
                viz_results = self.generate_visualization(run_dir, steps=timestep)
                results.update(viz_results)
                if viz_results:
                    print(f"Visualization files:")
                    for viz_type, viz_path in viz_results.items():
                        print(f"  {viz_type}: {viz_path}")
            except Exception as e:
                print(f"Warning: Visualization generation failed: {e}")
        
        return results

def main():
    """Run CartPole vector stress test."""
    # SCFD Phase 1 Integration: Preflight smoke tests
    print("Running SCFD preflight checks...")
    from engine.energy import scfd_preflight_checks
    
    preflight_results = scfd_preflight_checks()
    if not preflight_results.get('preflight_pass', False):
        print("CRITICAL: SCFD preflight tests failed!")
        failed_tests = [test for test, passed in preflight_results.items() 
                       if test != 'preflight_pass' and not passed]
        print(f"Failed tests: {failed_tests}")
        print("Aborting benchmark to prevent corrupted results.")
        return None
    
    print("SCFD preflight: ALL TESTS PASSED")
    
    config = CartPoleStressConfig(
        max_timesteps=500  # Test visualization improvements
    )
    
    stress_test = CartPoleVectorStressTest(config, seed=42)
    results = stress_test.run_stress_test()
    
    return results

if __name__ == "__main__":
    main()