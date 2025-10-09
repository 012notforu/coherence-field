#!/usr/bin/env python3
"""
Stress-Test CartPole with EFE Real-Time Parameter Tuning

Challenges the SCFD engine with:
- Environmental noise (sensor noise, force disturbances, wind)
- Parameter drift (changing physics over time) 
- Long-duration runs (10,000+ timesteps)
- Multiple objectives (balance + minimize energy + track setpoints)
- Real-time EFE tuner adaptation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import accel_theta, load_config, leapfrog_step, total_energy_density
from utils.logging import create_run_directory, RunLogger
from utils.efe_tuner import EFEParameterTuner
from observe.controller import GentleController, ControllerDecision

@dataclass
class StressTestConfig:
    """Configuration for stress-test CartPole."""
    
    # Basic CartPole physics
    dt: float = 0.01                    # Reduced for stability under stress
    n_substeps: int = 2                 # More substeps for accuracy
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    x_threshold: float = 3.0            # Wider limits for stress test
    theta_threshold_rad: float = 15 * np.pi / 180.0  # More lenient angle
    
    # Stress test parameters
    max_timesteps: int = 15000          # Long duration test
    sensor_noise_std: float = 0.02      # Noisy observations
    force_disturbance_std: float = 1.5  # Random force disturbances
    wind_strength: float = 0.8          # Constant horizontal bias
    
    # Parameter drift - physics changes over time
    mass_drift_rate: float = 0.00001    # Mass slowly changes
    length_drift_rate: float = 0.00005  # Length slowly changes
    gravity_drift_rate: float = 0.0001  # Gravity slowly changes
    
    # Multi-objective weights
    balance_weight: float = 0.6         # Primary: stay balanced
    energy_weight: float = 0.2          # Secondary: minimize control energy
    tracking_weight: float = 0.2        # Tertiary: track position setpoints
    
    # Control limits
    max_force: float = 15.0             # Increased force limit
    
    # SCFD integration
    scfd_grid_size: int = 32
    scfd_coupling_strength: float = 0.1

@dataclass 
class CartPoleState:
    """CartPole state with noise and disturbances."""
    x: float = 0.0          # Cart position
    x_dot: float = 0.0      # Cart velocity
    theta: float = 0.0      # Pole angle
    theta_dot: float = 0.0  # Pole angular velocity
    
    # Stress factors
    sensor_noise: np.ndarray = field(default_factory=lambda: np.zeros(4))
    force_disturbance: float = 0.0
    wind_bias: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])
    
    def add_noise(self, noise_std: float, rng: np.random.Generator):
        """Add sensor noise to observations."""
        self.sensor_noise = rng.normal(0, noise_std, 4)
    
    def get_noisy_observation(self) -> np.ndarray:
        """Get state observation with sensor noise."""
        clean_state = self.to_array()
        return clean_state + self.sensor_noise

class StressTestCartPole:
    """Stress-test CartPole with real-time EFE parameter tuning."""
    
    def __init__(self, config: StressTestConfig = None, seed: int = None):
        self.config = config or StressTestConfig()
        self.rng = np.random.default_rng(seed)
        
        # Initialize state
        self.state = CartPoleState()
        self.reset_state()
        
        # Physics parameters (these will drift over time)
        self.current_physics = {
            'mass_cart': self.config.masscart,
            'mass_pole': self.config.masspole,
            'length': self.config.length,
            'gravity': self.config.gravity
        }
        
        # SCFD integration
        self.initialize_scfd_field()
        
        # Multi-objective tracking
        self.position_setpoints = self.generate_position_setpoints()
        self.current_setpoint_idx = 0
        
        # Performance metrics
        self.reset_metrics()
        
        # EFE tuner for real-time adaptation
        self.setup_efe_tuner()
        
        # Gentle controller from observe module
        self.gentle_controller = GentleController(
            max_step=0.05,
            ema_tau=50.0,
            surprise_threshold=1.5
        )
        
    def reset_state(self):
        """Reset to a challenging initial state."""
        # Start with some initial perturbation
        self.state.x = self.rng.uniform(-0.5, 0.5)
        self.state.x_dot = self.rng.uniform(-0.2, 0.2)
        self.state.theta = self.rng.uniform(-0.1, 0.1)
        self.state.theta_dot = self.rng.uniform(-0.1, 0.1)
        
    def initialize_scfd_field(self):
        """Initialize SCFD field for coupling."""
        size = self.config.scfd_grid_size
        self.scfd_theta = np.zeros((size, size), dtype=np.float32)
        self.scfd_theta_dot = np.zeros((size, size), dtype=np.float32)
        
        # Initialize with some pattern related to CartPole state
        center = size // 2
        for i in range(size):
            for j in range(size):
                dx = i - center
                dy = j - center
                r = np.sqrt(dx*dx + dy*dy)
                self.scfd_theta[i, j] = np.exp(-r*r / 50.0) * 0.1
    
    def generate_position_setpoints(self) -> List[float]:
        """Generate challenging position setpoints to track."""
        timesteps = self.config.max_timesteps
        setpoints = []
        
        # Create a challenging trajectory
        for t in range(timesteps):
            # Sinusoidal with drift and noise
            base = 0.8 * np.sin(2 * np.pi * t / 2000)  # Slow sine wave
            drift = 0.3 * np.sin(2 * np.pi * t / 5000)  # Slower drift
            noise = 0.1 * np.sin(2 * np.pi * t / 300)   # Faster variation
            
            setpoint = base + drift + noise
            setpoint = np.clip(setpoint, -2.0, 2.0)  # Keep within bounds
            setpoints.append(setpoint)
        
        return setpoints
    
    def reset_metrics(self):
        """Reset performance tracking metrics."""
        self.metrics = {
            'timesteps_survived': 0,
            'total_control_energy': 0.0,
            'balance_violations': 0,
            'tracking_error_sum': 0.0,
            'successful_adaptations': 0,
            'parameter_updates': 0,
            'physics_drift_detected': 0,
            'noise_resistance_score': 0.0
        }
        
        self.history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'setpoints': [],
            'physics_params': [],
            'efe_updates': []
        }
    
    def setup_efe_tuner(self):
        """Setup EFE tuner for real-time parameter adaptation."""
        self.efe_tuner = EFEParameterTuner()
        
        # Add control parameters that affect CartPole performance
        self.efe_tuner.add_parameter_range("control_gain", 0.5, 5.0, 2.0)
        self.efe_tuner.add_parameter_range("derivative_gain", 0.1, 2.0, 0.8)
        self.efe_tuner.add_parameter_range("integral_gain", 0.0, 1.0, 0.2)
        self.efe_tuner.add_parameter_range("prediction_horizon", 1, 10, 5)
        self.efe_tuner.add_parameter_range("noise_filtering", 0.1, 1.0, 0.6)
        
        # Initial control parameters
        initial_params = {
            "control_gain": 2.0,
            "derivative_gain": 0.8,
            "integral_gain": 0.2,
            "prediction_horizon": 5,
            "noise_filtering": 0.6
        }
        
        # Metadata configuration for CartPole control
        metadata_config = {
            'control_gain': {
                'target_metrics': ['balance_stability', 'tracking_error'],
                'response_direction': 'increase_when_low',
                'sensitivity': 0.8
            },
            'derivative_gain': {
                'target_metrics': ['oscillation_control', 'noise_rejection'],
                'response_direction': 'increase_when_low', 
                'sensitivity': 0.6
            },
            'integral_gain': {
                'target_metrics': ['steady_state_error'],
                'response_direction': 'increase_when_low',
                'sensitivity': 0.4
            },
            'prediction_horizon': {
                'target_metrics': ['anticipation_performance'],
                'response_direction': 'increase_when_low',
                'sensitivity': 0.5
            },
            'noise_filtering': {
                'target_metrics': ['sensor_noise_impact'],
                'response_direction': 'increase_when_high',
                'sensitivity': 0.7
            }
        }
        
        self.efe_tuner.enable_realtime_tuning(initial_params, metadata_config)
        self.control_params = initial_params.copy()
        
    def update_physics_drift(self, timestep: int):
        """Simulate gradual changes in physics parameters."""
        t = timestep * self.config.dt
        
        # Gradual parameter drift
        self.current_physics['mass_cart'] += self.config.mass_drift_rate * self.rng.normal()
        self.current_physics['mass_pole'] += self.config.mass_drift_rate * 0.5 * self.rng.normal()
        self.current_physics['length'] += self.config.length_drift_rate * self.rng.normal()
        self.current_physics['gravity'] += self.config.gravity_drift_rate * self.rng.normal()
        
        # Add some periodic variations (simulating environmental changes)
        gravity_variation = 0.1 * np.sin(2 * np.pi * t / 1000)  # ~100 second cycles
        self.current_physics['gravity'] = self.config.gravity + gravity_variation
        
        # Clamp to reasonable bounds
        self.current_physics['mass_cart'] = np.clip(self.current_physics['mass_cart'], 0.5, 2.0)
        self.current_physics['mass_pole'] = np.clip(self.current_physics['mass_pole'], 0.05, 0.3)
        self.current_physics['length'] = np.clip(self.current_physics['length'], 0.2, 1.0)
        self.current_physics['gravity'] = np.clip(self.current_physics['gravity'], 8.0, 12.0)
    
    def compute_control_force(self, timestep: int) -> float:
        """Compute control force using current parameters and SCFD coupling."""
        
        # Get noisy observations
        obs = self.state.get_noisy_observation()
        x, x_dot, theta, theta_dot = obs
        
        # Current setpoint
        target_x = self.position_setpoints[min(timestep, len(self.position_setpoints)-1)]
        
        # Control errors
        position_error = x - target_x
        angle_error = theta  # Want theta = 0
        
        # PID-style control with EFE-tuned parameters
        proportional = (self.control_params['control_gain'] * angle_error + 
                       0.5 * self.control_params['control_gain'] * position_error)
        
        derivative = (self.control_params['derivative_gain'] * theta_dot +
                     0.3 * self.control_params['derivative_gain'] * x_dot)
        
        # Simple integral term (would need proper integral accumulation in practice)
        integral = self.control_params['integral_gain'] * position_error * 0.1
        
        # Base control force
        base_force = -(proportional + derivative + integral)
        
        # SCFD field coupling - use field state to adjust control
        center = self.config.scfd_grid_size // 2
        field_influence = self.scfd_theta[center, center] * self.config.scfd_coupling_strength
        
        control_force = base_force + field_influence
        
        # Add noise filtering
        if hasattr(self, '_last_control_force'):
            alpha = self.control_params['noise_filtering']
            control_force = alpha * control_force + (1 - alpha) * self._last_control_force
        self._last_control_force = control_force
        
        # Apply limits
        control_force = np.clip(control_force, -self.config.max_force, self.config.max_force)
        
        return control_force
        
    def step_physics(self, control_force: float) -> bool:
        """Step the CartPole physics with current parameters and disturbances."""
        
        # Add environmental disturbances
        total_force = control_force
        total_force += self.rng.normal(0, self.config.force_disturbance_std)  # Random disturbance
        total_force += self.config.wind_strength * (1 + 0.3 * np.sin(time.time()))  # Wind
        
        # Physics simulation using current (drifted) parameters
        g = self.current_physics['gravity']
        mc = self.current_physics['mass_cart'] 
        mp = self.current_physics['mass_pole']
        l = self.current_physics['length']
        dt = self.config.dt
        
        # Current state
        x, x_dot, theta, theta_dot = self.state.to_array()
        
        # CartPole dynamics
        for _ in range(self.config.n_substeps):
            costh = np.cos(theta)
            sinth = np.sin(theta)
            temp = (total_force + mp * l * theta_dot ** 2 * sinth) / (mc + mp)
            thacc = (g * sinth - costh * temp) / (l * (4.0 / 3.0 - mp * costh ** 2 / (mc + mp)))
            xacc = temp - mp * l * thacc * costh / (mc + mp)
            
            x = x + dt * x_dot / self.config.n_substeps
            x_dot = x_dot + dt * xacc / self.config.n_substeps
            theta = theta + dt * theta_dot / self.config.n_substeps
            theta_dot = theta_dot + dt * thacc / self.config.n_substeps
        
        # Update state
        self.state.x = x
        self.state.x_dot = x_dot
        self.state.theta = theta
        self.state.theta_dot = theta_dot
        
        # Add sensor noise
        self.state.add_noise(self.config.sensor_noise_std, self.rng)
        
        # Check failure conditions
        failed = (abs(x) > self.config.x_threshold) or (abs(theta) > self.config.theta_threshold_rad)
        
        return failed
    
    def update_scfd_field(self, timestep: int):
        """Update SCFD field based on CartPole state."""
        # Use some simple SCFD dynamics influenced by CartPole state
        center = self.config.scfd_grid_size // 2
        
        # Inject CartPole state into field
        self.scfd_theta[center, center] += 0.01 * self.state.theta
        self.scfd_theta[center-1, center] += 0.005 * self.state.x
        self.scfd_theta[center, center-1] += 0.005 * self.state.theta_dot
        
        # Simple field dynamics (diffusion)
        alpha = 0.1
        dt = self.config.dt
        
        # Compute field accelerations (simplified)
        laplacian = np.zeros_like(self.scfd_theta)
        laplacian[1:-1, 1:-1] = (
            self.scfd_theta[:-2, 1:-1] + self.scfd_theta[2:, 1:-1] +
            self.scfd_theta[1:-1, :-2] + self.scfd_theta[1:-1, 2:] -
            4 * self.scfd_theta[1:-1, 1:-1]
        )
        
        # Update field using simple dynamics
        accel = alpha * laplacian - 0.01 * self.scfd_theta  # Damping
        self.scfd_theta_dot += dt * accel
        self.scfd_theta += dt * self.scfd_theta_dot
        
        # Apply some decay to keep field bounded
        self.scfd_theta *= 0.999
        self.scfd_theta_dot *= 0.995
    
    def compute_performance_metrics(self, timestep: int, control_force: float) -> Dict[str, float]:
        """Compute multi-objective performance metrics."""
        
        # Current setpoint
        target_x = self.position_setpoints[min(timestep, len(self.position_setpoints)-1)]
        
        # Balance stability (lower is better)
        balance_stability = abs(self.state.theta) + 0.1 * abs(self.state.theta_dot)
        
        # Tracking error (lower is better)
        tracking_error = abs(self.state.x - target_x)
        
        # Control energy (lower is better)
        control_energy = abs(control_force) / self.config.max_force
        
        # Noise rejection (higher is better)
        noise_impact = np.linalg.norm(self.state.sensor_noise) / (4 * self.config.sensor_noise_std)
        sensor_noise_impact = min(1.0, noise_impact)
        
        # Oscillation control (lower is better) 
        if len(self.history['states']) > 10:
            recent_thetas = [s[2] for s in self.history['states'][-10:]]
            oscillation_control = np.std(recent_thetas)
        else:
            oscillation_control = abs(self.state.theta_dot)
        
        # Anticipation performance (based on how well control predicts)
        if len(self.history['states']) > 5:
            theta_trend = self.state.theta - self.history['states'][-5][2] 
            anticipation_performance = 1.0 / (1.0 + abs(theta_trend))
        else:
            anticipation_performance = 0.5
        
        # Steady state error 
        steady_state_error = tracking_error if timestep > 1000 else 0.0
        
        return {
            'balance_stability': balance_stability,
            'tracking_error': tracking_error, 
            'control_energy': control_energy,
            'sensor_noise_impact': sensor_noise_impact,
            'oscillation_control': oscillation_control,
            'anticipation_performance': anticipation_performance,
            'steady_state_error': steady_state_error
        }
    
    def update_efe_parameters(self, timestep: int, metrics: Dict[str, float]) -> bool:
        """Use EFE tuner to adapt control parameters in real-time."""
        
        # Only update every 50 timesteps to avoid thrashing
        if timestep % 50 != 0:
            return False
        
        # Create safety constraints
        safety_constraints = {
            'balance_stability': metrics['balance_stability'],
            'control_energy': metrics['control_energy']
        }
        
        # Update parameters
        try:
            updated_params = self.efe_tuner.update_realtime_parameters(metrics, safety_constraints)
            
            # Apply updated parameters
            if updated_params != self.control_params:
                self.control_params = updated_params.copy()
                self.metrics['parameter_updates'] += 1
                
                # Record the update
                self.history['efe_updates'].append({
                    'timestep': timestep,
                    'old_params': self.control_params.copy(),
                    'new_params': updated_params.copy(),
                    'trigger_metrics': metrics.copy()
                })
                
                return True
                
        except Exception as e:
            print(f"EFE tuner update failed at timestep {timestep}: {e}")
            
        return False
    
    def run_stress_test(self) -> Dict[str, any]:
        """Run the complete stress test."""
        
        print("=== CartPole Stress Test with EFE Real-Time Tuning ===")
        print(f"Max timesteps: {self.config.max_timesteps}")
        print(f"Sensor noise: {self.config.sensor_noise_std}")
        print(f"Force disturbances: {self.config.force_disturbance_std}")
        print(f"Physics drift enabled: mass={self.config.mass_drift_rate}, length={self.config.length_drift_rate}")
        
        # Create logging
        run_dir = create_run_directory("cartpole_stress_test")
        logger = RunLogger(run_dir)
        
        start_time = time.time()
        failed = False
        
        for timestep in range(self.config.max_timesteps):
            
            # Update physics parameters (drift)
            self.update_physics_drift(timestep)
            
            # Compute control force
            control_force = self.compute_control_force(timestep)
            
            # Step physics
            failed = self.step_physics(control_force)
            
            # Update SCFD field
            self.update_scfd_field(timestep)
            
            # Compute performance metrics
            perf_metrics = self.compute_performance_metrics(timestep, control_force)
            
            # EFE parameter updates
            params_updated = self.update_efe_parameters(timestep, perf_metrics)
            if params_updated:
                self.metrics['successful_adaptations'] += 1
            
            # Track metrics
            self.metrics['timesteps_survived'] = timestep + 1
            self.metrics['total_control_energy'] += abs(control_force)
            if abs(self.state.theta) > self.config.theta_threshold_rad * 0.8:
                self.metrics['balance_violations'] += 1
            
            target_x = self.position_setpoints[min(timestep, len(self.position_setpoints)-1)]
            self.metrics['tracking_error_sum'] += abs(self.state.x - target_x)
            
            # Record history
            self.history['states'].append(self.state.to_array().copy())
            self.history['actions'].append(control_force)
            self.history['setpoints'].append(target_x)
            self.history['physics_params'].append(self.current_physics.copy())
            
            # Log step
            log_data = {
                'timestep': timestep,
                'state': self.state.to_array().tolist(),
                'control_force': float(control_force),
                'target_position': float(target_x),
                'physics_params': self.current_physics.copy(),
                'performance_metrics': perf_metrics,
                'control_params': self.control_params.copy(),
                'efe_updated': params_updated
            }
            logger.log_step(log_data)
            
            # Progress reporting
            if (timestep + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Step {timestep+1:5d}: theta={self.state.theta:6.3f} x={self.state.x:6.3f} "
                      f"force={control_force:6.2f} adaptations={self.metrics['successful_adaptations']} "
                      f"time={elapsed:.1f}s")
            
            # Check failure
            if failed:
                print(f"CartPole failed at timestep {timestep + 1}")
                print(f"Final state: x={self.state.x:.3f}, theta={self.state.theta:.3f}")
                break
        
        # Final metrics
        total_time = time.time() - start_time
        survival_rate = self.metrics['timesteps_survived'] / self.config.max_timesteps
        avg_tracking_error = self.metrics['tracking_error_sum'] / self.metrics['timesteps_survived']
        avg_control_energy = self.metrics['total_control_energy'] / self.metrics['timesteps_survived']
        adaptation_rate = self.metrics['successful_adaptations'] / (self.metrics['timesteps_survived'] / 50)
        
        # EFE tuner status
        efe_status = self.efe_tuner.get_realtime_status()
        
        results = {
            'success': not failed,
            'survival_rate': survival_rate,
            'timesteps_survived': self.metrics['timesteps_survived'],
            'avg_tracking_error': avg_tracking_error,
            'avg_control_energy': avg_control_energy,
            'balance_violations': self.metrics['balance_violations'],
            'adaptation_rate': adaptation_rate,
            'parameter_updates': self.metrics['parameter_updates'],
            'total_runtime': total_time,
            'final_control_params': self.control_params.copy(),
            'efe_tuner_status': efe_status,
            'physics_drift_stats': {
                'final_mass_cart': self.current_physics['mass_cart'],
                'final_mass_pole': self.current_physics['mass_pole'], 
                'final_length': self.current_physics['length'],
                'final_gravity': self.current_physics['gravity']
            },
            'run_directory': str(run_dir)
        }
        
        # Save visualization
        self._save_stress_test_results(run_dir, results)
        
        print(f"\n=== Stress Test Results ===")
        print(f"Success: {results['success']}")
        print(f"Survival rate: {survival_rate:.1%}")
        print(f"Timesteps survived: {self.metrics['timesteps_survived']}")
        print(f"Average tracking error: {avg_tracking_error:.4f}")
        print(f"Average control energy: {avg_control_energy:.4f}")
        print(f"Balance violations: {self.metrics['balance_violations']}")
        print(f"EFE adaptations: {self.metrics['successful_adaptations']}")
        print(f"Adaptation rate: {adaptation_rate:.3f} updates/second")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Results saved to: {run_dir}")
        
        return results
    
    def _save_stress_test_results(self, run_dir: Path, results: Dict):
        """Save stress test visualization and analysis."""
        
        if not self.history['states']:
            return
        
        # Convert history to arrays
        states = np.array(self.history['states'])
        actions = np.array(self.history['actions'])
        setpoints = np.array(self.history['setpoints'])
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        timesteps = np.arange(len(states))
        
        # State trajectories
        axes[0, 0].plot(timesteps, states[:, 0], label='Cart Position', alpha=0.8)
        axes[0, 0].plot(timesteps, setpoints, label='Target Position', alpha=0.6, linestyle='--')
        axes[0, 0].set_title('Position Tracking')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(timesteps, np.degrees(states[:, 2]), label='Pole Angle', color='red', alpha=0.8)
        axes[0, 1].axhline(np.degrees(self.config.theta_threshold_rad), color='red', linestyle='--', alpha=0.5, label='Failure Threshold')
        axes[0, 1].axhline(-np.degrees(self.config.theta_threshold_rad), color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Pole Angle')
        axes[0, 1].set_ylabel('Angle (degrees)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Control forces
        axes[1, 0].plot(timesteps, actions, alpha=0.7, color='green')
        axes[1, 0].axhline(self.config.max_force, color='red', linestyle='--', alpha=0.5, label='Force Limit')
        axes[1, 0].axhline(-self.config.max_force, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Control Forces')
        axes[1, 0].set_ylabel('Force (N)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Velocities
        axes[1, 1].plot(timesteps, states[:, 1], label='Cart Velocity', alpha=0.8)
        axes[1, 1].plot(timesteps, states[:, 3], label='Pole Angular Velocity', alpha=0.8)
        axes[1, 1].set_title('Velocities')
        axes[1, 1].set_ylabel('Velocity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Physics parameter drift
        physics_history = np.array([p['mass_cart'] for p in self.history['physics_params']])
        axes[2, 0].plot(timesteps, physics_history, label='Cart Mass', alpha=0.8)
        physics_gravity = np.array([p['gravity'] for p in self.history['physics_params']])
        axes[2, 0].plot(timesteps, physics_gravity / 10, label='Gravity/10', alpha=0.8)  # Scaled for visibility
        axes[2, 0].set_title('Physics Parameter Drift')
        axes[2, 0].set_ylabel('Parameter Value')
        axes[2, 0].set_xlabel('Timestep')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # EFE parameter updates
        if self.history['efe_updates']:
            update_times = [u['timestep'] for u in self.history['efe_updates']]
            control_gains = [u['new_params']['control_gain'] for u in self.history['efe_updates']]
            axes[2, 1].scatter(update_times, control_gains, alpha=0.7, color='purple', s=20)
            axes[2, 1].set_title('EFE Parameter Adaptations')
            axes[2, 1].set_ylabel('Control Gain')
            axes[2, 1].set_xlabel('Timestep')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No EFE Updates', transform=axes[2, 1].transAxes, 
                           ha='center', va='center', fontsize=14)
            axes[2, 1].set_title('EFE Parameter Adaptations')
        
        plt.tight_layout()
        plt.savefig(run_dir / "stress_test_results.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Run CartPole stress test."""
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
    
    config = StressTestConfig(
        max_timesteps=12000,
        sensor_noise_std=0.025,
        force_disturbance_std=2.0,
        wind_strength=1.0
    )
    
    stress_test = StressTestCartPole(config, seed=42)
    results = stress_test.run_stress_test()
    
    return results

if __name__ == "__main__":
    main()