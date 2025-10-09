#!/usr/bin/env python3
"""
EFE-Based Auto-Tuner for SCFD Parameters

Uses Active Inference principles to optimize parameters by minimizing Expected Free Energy:
- Exploitation: Optimize task performance (PSNR, composition success, convergence)
- Exploration: Reduce parameter uncertainty and improve robustness

This addresses the 0% composition success rate by finding parameters that both work
and are robust across different conditions.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import copy

from benchmarks.image_completion import ImageCompletionConfig, ImageCompletionSystem


@dataclass
class EFETunerConfig:
    """Configuration for EFE-based parameter tuning."""
    
    # Population and evolution
    population_size: int = 20
    num_generations: int = 10
    elite_ratio: float = 0.2          # Top fraction to keep each generation
    mutation_rate: float = 0.1        # Parameter mutation probability
    mutation_strength: float = 0.05   # Size of parameter mutations
    
    # EFE fitness weights
    task_performance_weight: float = 0.5     # PSNR, convergence, speed
    composition_success_weight: float = 0.3  # Composition acceptance rate
    robustness_weight: float = 0.2          # Low ambiguity/uncertainty
    
    # Task performance sub-weights
    psnr_weight: float = 0.6
    convergence_weight: float = 0.2
    speed_weight: float = 0.2
    
    # Robustness measures
    uncertainty_measures: List[str] = None  # Will default to ["validator_margin", "outcome_variance"]
    
    # Termination criteria
    max_time_minutes: int = 60
    convergence_threshold: float = 0.01
    min_composition_success: float = 0.2    # Gate: must achieve 20% composition success
    
    def __post_init__(self):
        if self.uncertainty_measures is None:
            self.uncertainty_measures = ["validator_margin", "outcome_variance", "parameter_sensitivity"]


@dataclass
class ParameterRange:
    """Definition of parameter search space."""
    name: str
    min_val: float
    max_val: float
    current_val: float
    log_scale: bool = False  # Use log scale for wide ranges


class EFEParameterTuner:
    """Active Inference parameter tuner for SCFD systems."""
    
    def __init__(self, config: EFETunerConfig = None):
        self.config = config or EFETunerConfig()
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
        self.parameter_ranges = {}
        
        # Real-time tuning state
        self.realtime_mode = False
        self.current_parameters = {}
        self.parameter_emas = {}
        self.trust_region_size = 0.05  # 5% max change per window
        self.safety_margins = {}
        self.tuning_history = []
        
    def add_parameter_range(self, name: str, min_val: float, max_val: float, 
                           current_val: float = None, log_scale: bool = False):
        """Add a parameter to tune."""
        if current_val is None:
            current_val = (min_val + max_val) / 2
        
        self.parameter_ranges[name] = ParameterRange(
            name=name, min_val=min_val, max_val=max_val, 
            current_val=current_val, log_scale=log_scale
        )
    
    def setup_image_completion_tuning(self):
        """Setup parameter ranges for image completion tuning."""
        
        # Core ImageCompletionConfig parameters
        self.add_parameter_range("reconstruction_weight", 0.1, 1.0, 0.8)
        self.add_parameter_range("edge_preservation", 0.1, 1.0, 0.6)
        self.add_parameter_range("convergence_threshold", 0.001, 0.1, 0.01, log_scale=True)
        
        # Composition parameters (hypothetical - adapt to your actual composition config)
        self.add_parameter_range("composition_acceptance_threshold", 0.001, 0.1, 0.01, log_scale=True)
        self.add_parameter_range("composition_weight_adapt_rate", 0.01, 0.5, 0.1)
        self.add_parameter_range("physics_blend_weight", 0.1, 0.9, 0.5)
        self.add_parameter_range("texture_reconstruction_weight", 0.1, 0.9, 0.6)
        
        # Agent activation and movement
        self.add_parameter_range("activation_threshold", 0.001, 0.1, 0.01, log_scale=True)
        self.add_parameter_range("movement_temperature", 0.1, 5.0, 1.0)
        
        print(f"Setup {len(self.parameter_ranges)} parameters for tuning")
    
    def generate_individual(self) -> Dict[str, float]:
        """Generate a random parameter configuration."""
        individual = {}
        
        for param_name, param_range in self.parameter_ranges.items():
            if param_range.log_scale:
                # Log-uniform sampling
                log_min = np.log(param_range.min_val)
                log_max = np.log(param_range.max_val)
                log_val = np.random.uniform(log_min, log_max)
                individual[param_name] = np.exp(log_val)
            else:
                # Linear uniform sampling
                individual[param_name] = np.random.uniform(
                    param_range.min_val, param_range.max_val
                )
        
        return individual
    
    def mutate_individual(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Apply mutations to an individual."""
        mutated = copy.deepcopy(individual)
        
        for param_name, value in mutated.items():
            if np.random.random() < self.config.mutation_rate:
                param_range = self.parameter_ranges[param_name]
                
                if param_range.log_scale:
                    # Log-scale mutation
                    log_val = np.log(value)
                    log_mutation = np.random.normal(0, self.config.mutation_strength)
                    new_log_val = log_val + log_mutation
                    new_val = np.exp(new_log_val)
                else:
                    # Linear mutation
                    range_size = param_range.max_val - param_range.min_val
                    mutation = np.random.normal(0, self.config.mutation_strength * range_size)
                    new_val = value + mutation
                
                # Clamp to valid range
                mutated[param_name] = np.clip(new_val, param_range.min_val, param_range.max_val)
        
        return mutated
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Create offspring through crossover."""
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Uniform crossover
        for param_name in child1.keys():
            if np.random.random() < 0.5:
                child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def evaluate_fitness(self, individual: Dict[str, float], 
                        test_function: Callable[[Dict], Dict]) -> float:
        """
        Evaluate fitness using EFE principles.
        
        Args:
            individual: Parameter configuration to test
            test_function: Function that runs tests and returns results
            
        Returns:
            Fitness score (higher = better)
        """
        
        # Run tests with this parameter configuration
        try:
            results = test_function(individual)
        except Exception as e:
            print(f"Test failed for individual: {e}")
            return -1000.0  # Very poor fitness for failed tests
        
        # Extract key metrics
        composition_success_rate = results.get("composition_success_rate", 0.0)
        avg_psnr = results.get("avg_psnr", 0.0)
        convergence_rate = results.get("convergence_rate", 0.0)
        avg_processing_time = results.get("avg_processing_time", 100.0)
        
        # Gate: must achieve minimum composition success
        if composition_success_rate < self.config.min_composition_success:
            return -500.0 - (self.config.min_composition_success - composition_success_rate) * 1000
        
        # Task performance component
        psnr_score = avg_psnr / 20.0  # Normalize assuming 20 dB is excellent
        convergence_score = convergence_rate
        speed_score = max(0, 1.0 - avg_processing_time / 30.0)  # Penalize >30s processing
        
        task_performance = (
            self.config.psnr_weight * psnr_score +
            self.config.convergence_weight * convergence_score +
            self.config.speed_weight * speed_score
        )
        
        # Composition success component
        composition_score = composition_success_rate
        
        # Robustness/ambiguity component
        robustness_score = self._compute_robustness(individual, results)
        
        # Combine into overall fitness
        fitness = (
            self.config.task_performance_weight * task_performance +
            self.config.composition_success_weight * composition_score +
            self.config.robustness_weight * robustness_score
        )
        
        return fitness
    
    def _compute_robustness(self, individual: Dict[str, float], results: Dict) -> float:
        """Compute robustness score (lower ambiguity = higher robustness)."""
        robustness_components = []
        
        # 1. Validator margin: how close to failure boundaries
        if "validator_margins" in results:
            margins = results["validator_margins"]
            avg_margin = np.mean(margins) if margins else 0.0
            margin_score = min(1.0, avg_margin / 0.1)  # Normalize by 0.1 threshold
            robustness_components.append(margin_score)
        
        # 2. Outcome variance: consistency across different test cases
        if "psnr_variance" in results:
            variance = results["psnr_variance"]
            variance_score = max(0.0, 1.0 - variance / 10.0)  # Penalize high variance
            robustness_components.append(variance_score)
        
        # 3. Parameter sensitivity: small changes shouldn't cause large effects
        sensitivity_score = self._estimate_parameter_sensitivity(individual, results)
        robustness_components.append(sensitivity_score)
        
        return np.mean(robustness_components) if robustness_components else 0.5
    
    def _estimate_parameter_sensitivity(self, individual: Dict[str, float], 
                                       results: Dict) -> float:
        """Estimate sensitivity to parameter changes (cheap proxy)."""
        
        # Simple heuristic: parameters near boundaries are more sensitive
        sensitivity_penalties = []
        
        for param_name, value in individual.items():
            param_range = self.parameter_ranges[param_name]
            
            # Distance from boundaries (normalized)
            range_size = param_range.max_val - param_range.min_val
            dist_from_min = (value - param_range.min_val) / range_size
            dist_from_max = (param_range.max_val - value) / range_size
            
            min_distance = min(dist_from_min, dist_from_max)
            
            # Penalty for being too close to boundaries
            if min_distance < 0.1:
                sensitivity_penalties.append(0.1 - min_distance)
        
        # Convert penalties to score
        avg_penalty = np.mean(sensitivity_penalties) if sensitivity_penalties else 0.0
        return max(0.0, 1.0 - avg_penalty * 10.0)
    
    def evolve_population(self, population: List[Dict[str, float]], 
                         fitness_scores: List[float]) -> List[Dict[str, float]]:
        """Evolve population for next generation."""
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite individuals
        num_elite = int(self.config.population_size * self.config.elite_ratio)
        new_population = sorted_population[:num_elite].copy()
        
        # Generate offspring to fill remaining slots
        while len(new_population) < self.config.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_select(sorted_population, sorted_fitness)
            parent2 = self._tournament_select(sorted_population, sorted_fitness)
            
            # Create offspring
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate offspring
            child1 = self.mutate_individual(child1)
            child2 = self.mutate_individual(child2)
            
            # Add to new population
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.config.population_size]
    
    def _tournament_select(self, population: List[Dict[str, float]], 
                          fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, float]:
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def optimize(self, test_function: Callable[[Dict], Dict], 
                save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete optimization process.
        
        Args:
            test_function: Function that takes parameters and returns test results
            save_path: Optional path to save optimization results
            
        Returns:
            Dictionary with optimization results
        """
        
        print(f"Starting EFE parameter optimization")
        print(f"Population size: {self.config.population_size}")
        print(f"Generations: {self.config.num_generations}")
        print(f"Parameters: {list(self.parameter_ranges.keys())}")
        
        start_time = time.time()
        
        # Initialize population
        population = [self.generate_individual() for _ in range(self.config.population_size)]
        
        optimization_history = []
        
        for generation in range(self.config.num_generations):
            self.generation = generation
            generation_start = time.time()
            
            print(f"\nGeneration {generation + 1}/{self.config.num_generations}")
            
            # Evaluate fitness for all individuals
            fitness_scores = []
            for i, individual in enumerate(population):
                print(f"  Evaluating individual {i+1}/{len(population)}...", end=" ")
                
                fitness = self.evaluate_fitness(individual, test_function)
                fitness_scores.append(fitness)
                
                print(f"fitness: {fitness:.3f}")
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if self.best_individual is None or fitness_scores[best_idx] > max(self.fitness_history):
                self.best_individual = copy.deepcopy(population[best_idx])
            
            self.fitness_history.append(max(fitness_scores))
            
            # Record generation stats
            generation_stats = {
                "generation": generation,
                "best_fitness": max(fitness_scores),
                "avg_fitness": np.mean(fitness_scores),
                "worst_fitness": min(fitness_scores),
                "best_individual": copy.deepcopy(population[best_idx]),
                "generation_time": time.time() - generation_start
            }
            optimization_history.append(generation_stats)
            
            print(f"  Best fitness: {max(fitness_scores):.3f}")
            print(f"  Avg fitness: {np.mean(fitness_scores):.3f}")
            print(f"  Generation time: {generation_stats['generation_time']:.1f}s")
            
            # Check termination criteria
            total_time = (time.time() - start_time) / 60
            if total_time > self.config.max_time_minutes:
                print(f"Stopping: time limit reached ({total_time:.1f} minutes)")
                break
            
            if generation > 5:  # Check convergence after some generations
                recent_improvement = max(self.fitness_history[-5:]) - max(self.fitness_history[-10:-5])
                if recent_improvement < self.config.convergence_threshold:
                    print(f"Stopping: convergence reached (improvement < {self.config.convergence_threshold})")
                    break
            
            # Evolve population
            if generation < self.config.num_generations - 1:
                population = self.evolve_population(population, fitness_scores)
        
        # Compile results
        results = {
            "best_individual": self.best_individual,
            "best_fitness": max(self.fitness_history),
            "fitness_history": self.fitness_history,
            "optimization_history": optimization_history,
            "total_time_minutes": (time.time() - start_time) / 60,
            "total_generations": generation + 1
        }
        
        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {save_path}")
        
        print(f"\nOptimization complete!")
        print(f"Best fitness: {results['best_fitness']:.3f}")
        print(f"Best parameters: {self.best_individual}")
        
        return results
    
    def enable_realtime_tuning(self, initial_params: Dict[str, float], 
                             metadata_config: Dict = None):
        """Enable real-time parameter tuning with production-grade robustness."""
        self.realtime_mode = True
        self.current_parameters = copy.deepcopy(initial_params)
        
        # Initialize EMAs for each parameter with proper startup handling
        self.parameter_emas = {}
        for param_name, value in initial_params.items():
            self.parameter_emas[param_name] = {
                'value': value,
                'variance': 0.0,
                'sample_count': 1,  # Avoid division by zero
                'last_update': time.time()
            }
        
        # Metadata-driven parameter targeting (not string heuristics)
        self.metadata_config = metadata_config or {
            'metropolis_temperature': {
                'target_metrics': ['composition_success_rate'],
                'response_direction': 'increase_when_low',
                'sensitivity': 0.8
            },
            'acceptance_lower_bound': {
                'target_metrics': ['composition_success_rate'],
                'response_direction': 'decrease_when_low',
                'sensitivity': 0.5
            },
            'acceptance_upper_bound': {
                'target_metrics': ['composition_success_rate'],
                'response_direction': 'increase_when_low',
                'sensitivity': 0.3
            }
        }
        
        # Trust region floors - prevent zero-valued parameters from getting stuck
        self.trust_region_floors = {}
        for param_name in initial_params:
            if param_name in self.parameter_ranges:
                param_range = self.parameter_ranges[param_name]
                # Floor is 1% of parameter range or 0.001, whichever is larger
                range_span = param_range.max_val - param_range.min_val
                self.trust_region_floors[param_name] = max(0.001, range_span * 0.01)
            else:
                self.trust_region_floors[param_name] = 0.001
        
        # Cooldown/hysteresis to prevent thrashing
        self.cooldown_config = {
            'min_update_interval': 3.0,  # seconds
            'max_updates_per_window': 5,  # per 60s window
            'hysteresis_threshold': 0.15  # don't reverse within 15% of last change
        }
        
        self.update_history = []  # Track updates for cooldown logic
        
        # EFE thresholds - configurable and normalized
        self.efe_thresholds = {
            'risk_threshold': 0.3,       # Normalized [0,1]
            'ambiguity_threshold': 0.4,   # Normalized [0,1]
            'combined_threshold': 0.35    # For combined EFE score
        }
        
        # Per-metric windowing with EMAs
        self.metric_windows = {}
        self.metric_ema_alpha = 0.3  # Smoothing factor
        
        print(f"Real-time tuning enabled for {len(initial_params)} parameters")
        print(f"Trust region floors: {self.trust_region_floors}")
    
    def update_realtime_parameters(self, current_metrics: Dict[str, float], 
                                 safety_constraints: Dict[str, float] = None) -> Dict[str, float]:
        """Update parameters in real-time with full production robustness."""
        
        if not self.realtime_mode:
            raise RuntimeError("Real-time tuning not enabled. Call enable_realtime_tuning() first.")
        
        current_time = time.time()
        
        # Apply cooldown logic
        if not self._can_update_now(current_time):
            return self.current_parameters.copy()
        
        # Update metric windows with EMAs
        self._update_metric_windows(current_metrics)
        
        # Compute EFE components using windowed metrics
        risk_score = self._compute_realtime_risk(current_metrics)
        ambiguity_score = self._compute_realtime_ambiguity(current_metrics)
        combined_efe = risk_score + ambiguity_score
        
        # Check if EFE thresholds warrant parameter updates
        if combined_efe < self.efe_thresholds['combined_threshold']:
            return self.current_parameters.copy()
        
        # Compute parameter updates using metadata-driven targeting
        proposed_updates = self._compute_metadata_driven_updates(current_metrics, risk_score, ambiguity_score)
        
        # Apply trust region constraints with floors
        constrained_updates = self._apply_trust_region_with_floors(proposed_updates)
        
        # Lexicographic constraint handling with auto-revert
        final_updates, constraint_violations = self._apply_lexicographic_constraints(
            constrained_updates, safety_constraints or {}
        )
        
        # Update parameters and EMAs
        reason_tags = []
        for param_name, new_value in final_updates.items():
            old_value = self.current_parameters[param_name]
            self.current_parameters[param_name] = new_value
            
            # Update EMA
            ema_data = self.parameter_emas[param_name]
            delta = abs(new_value - ema_data['value'])
            ema_data['value'] = self.metric_ema_alpha * new_value + (1 - self.metric_ema_alpha) * ema_data['value']
            ema_data['variance'] = self.metric_ema_alpha * (delta ** 2) + (1 - self.metric_ema_alpha) * ema_data['variance']
            ema_data['sample_count'] += 1
            ema_data['last_update'] = current_time
            
            # Tag the reason for this update
            config = self.metadata_config.get(param_name, {})
            target_metrics = config.get('target_metrics', ['unknown'])
            reason_tags.append(f"{param_name}->{'_'.join(target_metrics)}")
        
        # Record update for cooldown tracking
        self.update_history.append({
            'timestamp': current_time,
            'updates': final_updates,
            'efe_scores': {'risk': risk_score, 'ambiguity': ambiguity_score, 'combined': combined_efe},
            'constraint_violations': constraint_violations,
            'reason_tags': reason_tags
        })
        
        # Auto-revert if too many constraint violations
        if len(constraint_violations) > len(final_updates) / 2:
            print(f"WARNING: Auto-reverting due to {len(constraint_violations)} constraint violations")
            self._auto_revert_parameters()
        
        return self.current_parameters.copy()
    
    def _can_update_now(self, current_time: float) -> bool:
        """Check cooldown constraints to prevent thrashing."""
        if not self.update_history:
            return True
        
        # Check minimum interval since last update
        last_update = self.update_history[-1]['timestamp']
        if current_time - last_update < self.cooldown_config['min_update_interval']:
            return False
        
        # Check update rate within window
        window_start = current_time - 60.0  # 60s window
        recent_updates = [h for h in self.update_history if h['timestamp'] > window_start]
        if len(recent_updates) >= self.cooldown_config['max_updates_per_window']:
            return False
        
        return True
    
    def _update_metric_windows(self, current_metrics: Dict[str, float]):
        """Update per-metric EMAs for windowed analysis."""
        current_time = time.time()
        
        for metric_name, value in current_metrics.items():
            if metric_name not in self.metric_windows:
                self.metric_windows[metric_name] = {
                    'ema': value,
                    'variance': 0.0,
                    'min_seen': value,
                    'max_seen': value,
                    'sample_count': 1
                }
            else:
                window = self.metric_windows[metric_name]
                delta = value - window['ema']
                window['ema'] += self.metric_ema_alpha * delta
                window['variance'] = (1 - self.metric_ema_alpha) * window['variance'] + self.metric_ema_alpha * (delta ** 2)
                window['min_seen'] = min(window['min_seen'], value)
                window['max_seen'] = max(window['max_seen'], value)
                window['sample_count'] += 1
    
    def _compute_realtime_risk(self, metrics: Dict[str, float]) -> float:
        """Compute normalized risk score [0,1] from current performance."""
        
        # Task performance risk (lower is worse)
        psnr = metrics.get('psnr', 0)
        expected_psnr = metrics.get('expected_psnr', 15.0)
        psnr_risk = max(0.0, 1.0 - (psnr / expected_psnr)) if expected_psnr > 0 else 1.0
        
        # Composition success risk (lower is worse)
        comp_success = metrics.get('composition_success_rate', 0)
        comp_risk = 1.0 - comp_success
        
        # Convergence risk
        convergence_rate = metrics.get('convergence_rate', 0.5)
        conv_risk = 1.0 - convergence_rate
        
        # Weighted combination
        total_risk = 0.4 * psnr_risk + 0.4 * comp_risk + 0.2 * conv_risk
        
        return min(1.0, max(0.0, total_risk))
    
    def _compute_realtime_ambiguity(self, metrics: Dict[str, float]) -> float:
        """Compute normalized ambiguity score [0,1] from uncertainty measures."""
        
        ambiguity_components = []
        
        # Validator margin ambiguity
        validator_margin = metrics.get('validator_margin', 0.05)
        margin_ambiguity = 1.0 - min(1.0, validator_margin / 0.1)  # Normalize to [0,1]
        ambiguity_components.append(margin_ambiguity)
        
        # Field entropy ambiguity
        field_entropy = metrics.get('field_entropy', 1.0)
        max_entropy = metrics.get('max_field_entropy', 2.0)
        entropy_ambiguity = field_entropy / max_entropy if max_entropy > 0 else 0.5
        ambiguity_components.append(entropy_ambiguity)
        
        # Parameter variance ambiguity
        param_variance = np.mean([ema['variance'] for ema in self.parameter_emas.values()])
        variance_ambiguity = min(1.0, param_variance / 0.1)  # Normalize to [0,1]
        ambiguity_components.append(variance_ambiguity)
        
        # Outcome variance from windowed metrics
        if 'psnr' in self.metric_windows:
            outcome_variance = self.metric_windows['psnr']['variance']
            outcome_ambiguity = min(1.0, outcome_variance / 4.0)  # Normalize to [0,1]
            ambiguity_components.append(outcome_ambiguity)
        
        return np.mean(ambiguity_components) if ambiguity_components else 0.5
    
    def _compute_metadata_driven_updates(self, metrics: Dict[str, float], 
                                       risk_score: float, ambiguity_score: float) -> Dict[str, float]:
        """Compute parameter updates using metadata configuration, not name heuristics."""
        
        proposed_updates = {}
        
        for param_name, current_value in self.current_parameters.items():
            if param_name not in self.metadata_config:
                continue
            
            config = self.metadata_config[param_name]
            target_metrics = config['target_metrics']
            response_direction = config['response_direction']
            sensitivity = config.get('sensitivity', 0.5)
            
            # Compute signal strength from target metrics
            signal_strength = 0.0
            for metric_name in target_metrics:
                if metric_name in metrics:
                    metric_value = metrics[metric_name]
                    
                    # Different response patterns
                    if response_direction == 'increase_when_low':
                        # Stronger signal when metric is lower
                        signal_strength += (1.0 - metric_value) if metric_value < 1.0 else 0.0
                    elif response_direction == 'decrease_when_high':
                        # Stronger signal when metric is higher
                        signal_strength += metric_value
                    elif response_direction == 'decrease_when_low':
                        # Decrease parameter when metric is low
                        signal_strength -= (1.0 - metric_value) if metric_value < 1.0 else 0.0
            
            # Scale by sensitivity and EFE components
            efe_multiplier = 0.5 * risk_score + 0.3 * ambiguity_score + 0.2
            update_magnitude = sensitivity * signal_strength * efe_multiplier
            
            # Convert to parameter change
            if param_name in self.parameter_ranges:
                param_range = self.parameter_ranges[param_name]
                range_span = param_range.max_val - param_range.min_val
                parameter_delta = update_magnitude * range_span * self.trust_region_size
                
                # Apply direction
                if response_direction in ['increase_when_low', 'increase_when_high']:
                    new_value = current_value + parameter_delta
                else:
                    new_value = current_value - parameter_delta
                
                # Clamp to bounds
                new_value = max(param_range.min_val, min(param_range.max_val, new_value))
                proposed_updates[param_name] = new_value
        
        return proposed_updates
    
    def _apply_trust_region_with_floors(self, proposed_updates: Dict[str, float]) -> Dict[str, float]:
        """Apply trust region constraints with floors for zero-valued parameters."""
        
        constrained_updates = {}
        
        for param_name, proposed_value in proposed_updates.items():
            current_value = self.current_parameters[param_name]
            
            # Compute maximum allowed change
            if param_name in self.parameter_ranges:
                param_range = self.parameter_ranges[param_name]
                range_span = param_range.max_val - param_range.min_val
                max_change = self.trust_region_size * range_span
            else:
                max_change = self.trust_region_size * abs(current_value) if current_value != 0 else 0.01
            
            # Apply trust region floor
            floor_value = self.trust_region_floors.get(param_name, 0.001)
            if abs(current_value) < floor_value:
                # Use larger trust region for near-zero parameters
                max_change = max(max_change, floor_value)
            
            # Constrain the change
            change = proposed_value - current_value
            if abs(change) > max_change:
                change = np.sign(change) * max_change
            
            constrained_updates[param_name] = current_value + change
        
        return constrained_updates
    
    def _apply_lexicographic_constraints(self, proposed_updates: Dict[str, float], 
                                       safety_constraints: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        """Apply lexicographic constraint handling with violation tracking."""
        
        violations = []
        final_updates = proposed_updates.copy()
        
        # Check each safety constraint
        for constraint_name, threshold in safety_constraints.items():
            # Mock constraint checking - adapt to your actual constraint system
            if constraint_name == 'composition_success_rate' and threshold < 0.05:
                # Violation: composition success too low
                violations.append(f"composition_success_rate_{threshold:.3f}_too_low")
                
                # Conservative adjustment: reduce changes
                for param_name in final_updates:
                    current_value = self.current_parameters[param_name]
                    change = final_updates[param_name] - current_value
                    final_updates[param_name] = current_value + 0.5 * change
            
            elif constraint_name == 'energy_drift' and threshold > 0.01:
                # Violation: energy drift too high
                violations.append(f"energy_drift_{threshold:.3f}_too_high")
                
                # Conservative adjustment: smaller temperature changes
                if 'metropolis_temperature' in final_updates:
                    current_temp = self.current_parameters['metropolis_temperature']
                    temp_change = final_updates['metropolis_temperature'] - current_temp
                    final_updates['metropolis_temperature'] = current_temp + 0.3 * temp_change
        
        return final_updates, violations
    
    def _auto_revert_parameters(self):
        """Auto-revert parameters to last stable state."""
        if len(self.update_history) < 2:
            return
        
        # Find last update without violations
        stable_state = None
        for history_entry in reversed(self.update_history[:-1]):
            if not history_entry.get('constraint_violations', []):
                stable_state = history_entry['updates']
                break
        
        if stable_state:
            self.current_parameters.update(stable_state)
            print(f"Auto-reverted {len(stable_state)} parameters to stable state")
    
    def get_realtime_status(self) -> Dict[str, Any]:
        """Get comprehensive status of real-time tuning system."""
        
        if not self.realtime_mode:
            return {"status": "disabled"}
        
        current_time = time.time()
        
        # EMA statistics
        ema_stats = {}
        for param_name, ema_data in self.parameter_emas.items():
            ema_stats[param_name] = {
                "current_value": self.current_parameters[param_name],
                "ema_value": ema_data['value'],
                "variance": ema_data['variance'],
                "sample_count": ema_data['sample_count'],
                "stability": 1.0 / (1.0 + ema_data['variance'])  # Higher is more stable
            }
        
        # Recent update statistics
        recent_updates = [h for h in self.update_history if current_time - h['timestamp'] < 300.0]  # 5min window
        
        status = {
            "status": "active",
            "realtime_mode": True,
            "parameter_count": len(self.current_parameters),
            "total_updates": len(self.update_history),
            "recent_updates": len(recent_updates),
            "ema_statistics": ema_stats,
            "trust_region_size": self.trust_region_size,
            "cooldown_active": not self._can_update_now(current_time),
            "last_update": self.update_history[-1]['timestamp'] if self.update_history else None
        }
        
        if recent_updates:
            recent_violations = sum(len(h.get('constraint_violations', [])) for h in recent_updates)
            status["recent_constraint_violations"] = recent_violations
            status["violation_rate"] = recent_violations / len(recent_updates)
        
        return status


def create_image_completion_test_function(test_configs: List[Dict], test_images: Dict) -> Callable:
    """Create test function for image completion parameter tuning."""
    
    def test_function(parameters: Dict[str, float]) -> Dict[str, Any]:
        """Run image completion tests with given parameters."""
        
        results = {
            "psnr_scores": [],
            "composition_success_rates": [],
            "convergence_rates": [],
            "processing_times": [],
            "validator_margins": []
        }
        
        for config_dict in test_configs:
            for image_name, image in test_images.items():
                
                # Create config with tuned parameters
                config = ImageCompletionConfig(**config_dict)
                
                # Apply tuned parameters (adapt field names as needed)
                if "reconstruction_weight" in parameters:
                    config.reconstruction_weight = parameters["reconstruction_weight"]
                if "edge_preservation" in parameters:
                    config.edge_preservation = parameters["edge_preservation"]
                if "convergence_threshold" in parameters:
                    config.convergence_threshold = parameters["convergence_threshold"]
                
                try:
                    start_time = time.time()
                    
                    # Run test
                    system = ImageCompletionSystem(image, config)
                    test_results = system.reconstruct_image()
                    
                    processing_time = time.time() - start_time
                    
                    # Extract metrics
                    results["psnr_scores"].append(test_results.get("final_psnr", 0))
                    results["processing_times"].append(processing_time)
                    results["convergence_rates"].append(1.0 if test_results.get("converged", False) else 0.0)
                    
                    # Extract composition metrics
                    analytics = test_results.get("composition_analytics", {})
                    success_rates = []
                    for comp_data in analytics.values():
                        success_rates.append(comp_data.get("success_rate", 0.0))
                    
                    avg_comp_success = np.mean(success_rates) if success_rates else 0.0
                    results["composition_success_rates"].append(avg_comp_success)
                    
                    # Mock validator margin (adapt to your actual validator)
                    results["validator_margins"].append(0.05)  # Placeholder
                    
                except Exception as e:
                    print(f"Test failed: {e}")
                    # Add poor results for failed tests
                    results["psnr_scores"].append(0.0)
                    results["composition_success_rates"].append(0.0)
                    results["convergence_rates"].append(0.0)
                    results["processing_times"].append(100.0)
                    results["validator_margins"].append(0.0)
        
        # Compute summary statistics
        summary = {
            "avg_psnr": np.mean(results["psnr_scores"]),
            "psnr_variance": np.var(results["psnr_scores"]),
            "composition_success_rate": np.mean(results["composition_success_rates"]),
            "convergence_rate": np.mean(results["convergence_rates"]),
            "avg_processing_time": np.mean(results["processing_times"]),
            "validator_margins": results["validator_margins"]
        }
        
        return summary
    
    return test_function


# Quick test
if __name__ == "__main__":
    # Test the tuner setup
    tuner = EFEParameterTuner()
    tuner.setup_image_completion_tuning()
    
    # Generate and evaluate a test individual
    individual = tuner.generate_individual()
    print(f"Generated individual: {individual}")
    
    # Mock test function
    def mock_test(params):
        return {
            "avg_psnr": 12.0 + np.random.normal(0, 2),
            "composition_success_rate": 0.1 + np.random.random() * 0.3,
            "convergence_rate": 0.5 + np.random.random() * 0.5,
            "avg_processing_time": 10 + np.random.random() * 5,
            "psnr_variance": 1.0 + np.random.random(),
            "validator_margins": [0.05] * 10
        }
    
    fitness = tuner.evaluate_fitness(individual, mock_test)
    print(f"Fitness: {fitness:.3f}")
    print("EFE tuner working!")


def create_realtime_tuner_for_composition():
    """Create a real-time tuner specifically for composition acceptance issues."""
    tuner = EFEParameterTuner()
    
    # Focus on composition-critical parameters
    tuner.add_parameter_range("metropolis_temperature", 0.5, 2.0, 1.2)
    tuner.add_parameter_range("acceptance_lower_bound", 0.001, 0.02, 0.005)
    tuner.add_parameter_range("acceptance_upper_bound", 0.95, 0.999, 0.98)
    
    # Enable real-time mode with better defaults
    initial_params = {
        "metropolis_temperature": 1.2,  # From auto-tuner analysis
        "acceptance_lower_bound": 0.005,
        "acceptance_upper_bound": 0.98
    }
    
    # Custom metadata config for composition parameters
    metadata_config = {
        'metropolis_temperature': {
            'target_metrics': ['composition_success_rate', 'energy_drift'],
            'response_direction': 'increase_when_low',
            'sensitivity': 0.7
        },
        'acceptance_lower_bound': {
            'target_metrics': ['composition_success_rate'],
            'response_direction': 'decrease_when_low',
            'sensitivity': 0.8
        },
        'acceptance_upper_bound': {
            'target_metrics': ['composition_success_rate'],
            'response_direction': 'increase_when_low',
            'sensitivity': 0.6
        }
    }
    
    tuner.enable_realtime_tuning(initial_params, metadata_config)
    return tuner