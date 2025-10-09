"""
Image Completion Test Framework

Uses SCFD multi-agent system with vector composition to reconstruct damaged images.
Agents work collaboratively to fill in missing regions using field dynamics.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from benchmarks.multi_agent_grid import MultiAgentGridSystem, GridCellAgent
from benchmarks.vector_composition import VectorComposer, CompositeAgent, CompositionRule
from utils.logging import create_run_directory, RunLogger

Array = np.ndarray


@dataclass
class ImageCompletionConfig:
    """Configuration for image completion tasks."""
    damage_type: str = "random_holes"  # "random_holes", "strips", "blocks", "noise"
    damage_ratio: float = 0.3  # Fraction of image to damage
    reconstruction_weight: float = 0.8  # Weight for reconstruction vs smoothing
    edge_preservation: float = 0.6  # How much to preserve edges
    max_iterations: int = 200
    convergence_threshold: float = 0.01


class ImageCompletionAgent(GridCellAgent):
    """Specialized agent for image completion tasks."""
    
    def __init__(self, 
                 cell_pos: Tuple[int, int],
                 grid_shape: Tuple[int, int],
                 vector_registry,
                 physics_cfg,
                 target_image: Array,
                 damage_mask: Array,
                 config: ImageCompletionConfig,
                 agent_id: str = None):
        super().__init__(cell_pos, grid_shape, vector_registry, physics_cfg, agent_id)
        
        self.target_image = target_image
        self.damage_mask = damage_mask  # 1 = damaged (needs reconstruction), 0 = intact
        self.config = config
        
        # Image-specific state
        self.local_contrast = 0.0
        self.edge_strength = 0.0
        self.reconstruction_confidence = 0.0
        
        # Override activation for image completion
        self.activation_threshold = 0.02  # More sensitive for image work
        
    def sense_image_environment(self, current_image: Array) -> Dict[str, float]:
        """Enhanced environment sensing for image completion."""
        # Get base features
        base_features = self.sense_local_environment(current_image)
        
        i, j = self.pos
        h, w = self.grid_shape
        
        # Image-specific features
        center_value = current_image[i, j]
        target_value = self.target_image[i, j]
        is_damaged = self.damage_mask[i, j] > 0.5
        
        # Local contrast analysis
        neighborhood = []
        damaged_neighbors = 0
        intact_neighbors = 0
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i + di) % h, (j + dj) % w
                neighborhood.append(current_image[ni, nj])
                
                if self.damage_mask[ni, nj] > 0.5:
                    damaged_neighbors += 1
                else:
                    intact_neighbors += 1
        
        local_contrast = float(np.std(neighborhood))
        edge_strength = float(np.max(neighborhood) - np.min(neighborhood))
        
        # Reconstruction quality metrics
        if not is_damaged:
            reconstruction_error = 0.0  # Intact pixel
            reconstruction_confidence = 1.0
        else:
            reconstruction_error = float(abs(center_value - target_value))
            reconstruction_confidence = max(0.0, 1.0 - reconstruction_error)
        
        # Information from intact neighbors
        intact_neighbor_mean = 0.0
        if intact_neighbors > 0:
            intact_values = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = (i + di) % h, (j + dj) % w
                    if self.damage_mask[ni, nj] < 0.5:  # Intact
                        intact_values.append(current_image[ni, nj])
            
            if intact_values:
                intact_neighbor_mean = float(np.mean(intact_values))
        
        image_features = {
            **base_features,
            "is_damaged": float(is_damaged),
            "reconstruction_error": reconstruction_error,
            "reconstruction_confidence": reconstruction_confidence,
            "local_contrast": local_contrast,
            "edge_strength": edge_strength,
            "damaged_neighbors": damaged_neighbors,
            "intact_neighbors": intact_neighbors,
            "intact_neighbor_mean": intact_neighbor_mean,
            "target_value": float(target_value),
            "current_value": float(center_value),
            "needs_reconstruction": float(is_damaged and reconstruction_error > 0.1)
        }
        
        # Update internal state
        self.local_contrast = local_contrast
        self.edge_strength = edge_strength
        self.reconstruction_confidence = reconstruction_confidence
        
        return image_features
    
    def detect_image_failure_pattern(self, image_features: Dict[str, float]) -> str:
        """Detect image-specific failure patterns."""
        if image_features.get("needs_reconstruction", 0) > 0.5:
            if image_features.get("intact_neighbors", 0) < 2:
                return "isolated_damage"
            elif image_features.get("edge_strength", 0) > 0.5:
                return "edge_reconstruction"
            elif image_features.get("local_contrast", 0) < 0.1:
                return "smooth_region"
            else:
                return "texture_reconstruction"
        
        if image_features.get("reconstruction_error", 0) > 0.2:
            return "poor_reconstruction"
        
        if image_features.get("local_contrast", 0) > 0.8:
            return "high_contrast_region"
        
        return "reconstruction_complete"


class ImageCompletionSystem:
    """Multi-agent system for collaborative image completion."""
    
    def __init__(self, target_image: Array, config: ImageCompletionConfig = None):
        self.target_image = target_image.astype(np.float32)
        self.config = config or ImageCompletionConfig()
        self.grid_shape = target_image.shape
        
        # Create damage mask
        self.damage_mask = self._create_damage_mask()
        
        # Initialize damaged image
        self.current_image = self._apply_damage()
        
        # Create base multi-agent system
        self.agent_system = MultiAgentGridSystem(self.grid_shape)
        
        # Create image completion composer with specialized rules
        self.composer = VectorComposer(self.agent_system.vector_registry)
        self._setup_image_completion_compositions()
        
        # Replace with image completion agents
        self.completion_agents = {}
        self._create_image_completion_agents()
        
        # Tracking
        self.reconstruction_history = []
        self.convergence_metrics = []
        
    def _create_damage_mask(self) -> Array:
        """Create damage mask based on configuration."""
        h, w = self.grid_shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        if self.config.damage_type == "random_holes":
            # Random scattered damage
            total_pixels = h * w
            damage_pixels = int(total_pixels * self.config.damage_ratio)
            
            damaged_indices = np.random.choice(total_pixels, damage_pixels, replace=False)
            damaged_coords = np.unravel_index(damaged_indices, (h, w))
            mask[damaged_coords] = 1.0
            
        elif self.config.damage_type == "strips":
            # Horizontal strips
            strip_width = max(1, int(h * 0.1))
            num_strips = int(self.config.damage_ratio * h / strip_width)
            
            for _ in range(num_strips):
                start_row = np.random.randint(0, h - strip_width)
                mask[start_row:start_row + strip_width, :] = 1.0
                
        elif self.config.damage_type == "blocks":
            # Square blocks
            block_size = max(3, int(min(h, w) * 0.15))
            num_blocks = int((h * w * self.config.damage_ratio) / (block_size * block_size))
            
            for _ in range(num_blocks):
                start_i = np.random.randint(0, h - block_size)
                start_j = np.random.randint(0, w - block_size)
                mask[start_i:start_i + block_size, start_j:start_j + block_size] = 1.0
                
        elif self.config.damage_type == "noise":
            # Salt and pepper noise
            mask = np.random.random((h, w)) < self.config.damage_ratio
            mask = mask.astype(np.float32)
        
        return mask
    
    def _apply_damage(self) -> Array:
        """Apply damage to target image."""
        damaged_image = self.target_image.copy()
        
        # Set damaged pixels to random values or zero
        if self.config.damage_type == "noise":
            # Salt and pepper noise
            damaged_image[self.damage_mask > 0.5] = np.random.choice([0.0, 1.0], 
                                                                     size=np.sum(self.damage_mask > 0.5))
        else:
            # Set to random values
            damaged_image[self.damage_mask > 0.5] = np.random.random(np.sum(self.damage_mask > 0.5))
        
        return damaged_image
    
    def _setup_image_completion_compositions(self):
        """Setup vector compositions for image completion."""
        
        # 1. Smoothing + Reconstruction composition
        smooth_vectors = [v.vector_id for v in self.composer.registry.get_vectors_by_tag("smooth_front")]
        align_vectors = [v.vector_id for v in self.composer.registry.get_vectors_by_tag("align")]
        
        if smooth_vectors and align_vectors:
            self.composer.add_composition_rule(CompositionRule(
                name="smooth_reconstruction",
                vector_ids=smooth_vectors[:2] + align_vectors[:2],
                weights=[0.4, 0.3, 0.2, 0.1],
                composition_type="conditional",
                priority=1.2
            ))
        
        # 2. Edge preservation composition
        stabilize_vectors = [v.vector_id for v in self.composer.registry.get_vectors_by_tag("stabilize")]
        if len(stabilize_vectors) >= 2:
            self.composer.add_composition_rule(CompositionRule(
                name="edge_preservation",
                vector_ids=stabilize_vectors[:3],
                weights=[0.5, 0.3, 0.2],
                composition_type="adaptive",
                priority=1.0
            ))
        
        # 3. Meta-enhanced reconstruction
        meta_vectors = [v.vector_id for v in self.composer.registry.base_registry if "meta" in v.vector_id]
        texture_vectors = [v.vector_id for v in self.composer.registry.base_registry 
                          if "gray_scott" in v.vector_id or "heat" in v.vector_id][:2]
        
        if meta_vectors and texture_vectors:
            self.composer.add_composition_rule(CompositionRule(
                name="meta_texture_reconstruction",
                vector_ids=meta_vectors[:1] + texture_vectors[:2],
                weights=[0.6, 0.25, 0.15],
                composition_type="sequential",
                priority=1.4
            ))
        
        # 4. Fine detail composition
        detail_vectors = [v.vector_id for v in self.composer.registry.base_registry 
                         if "quick" in v.vector_id or "precise" in v.vector_id][:3]
        if len(detail_vectors) >= 2:
            self.composer.add_composition_rule(CompositionRule(
                name="fine_detail",
                vector_ids=detail_vectors[:3],
                weights=[0.5, 0.3, 0.2],
                composition_type="linear",
                priority=0.9
            ))
    
    def _create_image_completion_agents(self):
        """Create image completion agents."""
        # Create agents at damaged regions and boundaries
        h, w = self.grid_shape
        
        # Place agents at damaged pixels and their neighbors
        agent_positions = set()
        
        for i in range(h):
            for j in range(w):
                if self.damage_mask[i, j] > 0.5:  # Damaged pixel
                    agent_positions.add((i, j))
                    
                    # Add neighbors of damaged pixels
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                agent_positions.add((ni, nj))
        
        # Limit number of agents for performance
        max_agents = min(len(agent_positions), 50)
        if len(agent_positions) > max_agents:
            agent_positions = set(list(agent_positions)[:max_agents])
        
        print(f"Creating {len(agent_positions)} image completion agents")
        
        for pos in agent_positions:
            base_agent = ImageCompletionAgent(
                pos, self.grid_shape, 
                self.agent_system.vector_registry,
                self.agent_system.cfg.physics,
                self.target_image, self.damage_mask, self.config,
                f"completion_agent_{pos[0]}_{pos[1]}"
            )
            
            # Create composite agent
            composite_agent = CompositeAgent(base_agent, self.composer)
            self.completion_agents[pos] = composite_agent
    
    def reconstruct_image(self, max_iterations: int = None) -> Dict:
        """Run image reconstruction using multi-agent composition system."""
        max_iterations = max_iterations or self.config.max_iterations
        
        # Create logging directory
        run_dir = create_run_directory("image_completion")
        logger = RunLogger(run_dir)
        
        print(f"Starting image reconstruction...")
        print(f"Image size: {self.grid_shape}")
        print(f"Damage type: {self.config.damage_type}")
        print(f"Damage ratio: {self.config.damage_ratio:.3f}")
        print(f"Agents: {len(self.completion_agents)}")
        print(f"Compositions: {len(self.composer.composition_rules)}")
        
        # Initialize field with current image
        self.agent_system.global_theta = self.current_image.copy()
        self.agent_system.global_theta_dot = np.zeros_like(self.current_image)
        
        # Track metrics
        iteration = 0
        converged = False
        
        for iteration in range(max_iterations):
            # Update global physics
            from engine import accel_theta
            from engine.integrators import leapfrog_step
            
            self.agent_system.global_theta, self.agent_system.global_theta_dot, _, _ = leapfrog_step(
                self.agent_system.global_theta,
                self.agent_system.global_theta_dot,
                lambda f: accel_theta(f, self.agent_system.cfg.physics, dx=1.0),
                self.agent_system.cfg.integration.dt
            )
            
            # Step completion agents
            stats = {
                "active_agents": 0,
                "successful_reconstructions": 0,
                "composition_usage": {},
                "reconstruction_error": 0.0
            }
            
            neighbor_states = {}  # Simple version
            total_reconstruction_error = 0.0
            damaged_pixels = 0
            
            for pos, composite_agent in self.completion_agents.items():
                base_agent = composite_agent.base_agent
                
                # Get image features
                image_features = base_agent.sense_image_environment(self.agent_system.global_theta)
                
                # Step with composition
                accepted = composite_agent.step_with_composition(
                    self.agent_system.global_theta,
                    self.agent_system.global_theta_dot,
                    neighbor_states,
                    logger
                )
                
                if base_agent.active:
                    stats["active_agents"] += 1
                
                if accepted and image_features.get("is_damaged", 0) > 0.5:
                    stats["successful_reconstructions"] += 1
                
                # Track composition usage
                if composite_agent.composition_history:
                    last_comp = composite_agent.composition_history[-1]
                    rule_name = last_comp["rule_name"]
                    stats["composition_usage"][rule_name] = stats["composition_usage"].get(rule_name, 0) + 1
                
                # Track reconstruction error for damaged pixels
                if self.damage_mask[pos] > 0.5:
                    error = image_features.get("reconstruction_error", 0)
                    total_reconstruction_error += error
                    damaged_pixels += 1
            
            # Compute overall reconstruction error
            if damaged_pixels > 0:
                avg_reconstruction_error = total_reconstruction_error / damaged_pixels
                stats["reconstruction_error"] = avg_reconstruction_error
            
            # Update current image from field
            self.current_image = self.agent_system.global_theta.copy()
            
            # Preserve intact pixels
            intact_mask = self.damage_mask < 0.5
            self.current_image[intact_mask] = self.target_image[intact_mask]
            self.agent_system.global_theta[intact_mask] = self.target_image[intact_mask]
            
            # Log iteration
            step_data = {
                "iteration": iteration,
                "global_energy": float(np.mean(self.agent_system.global_theta ** 2)),
                **stats
            }
            logger.log_step(step_data)
            
            # Track convergence
            self.convergence_metrics.append(stats["reconstruction_error"])
            
            # Check convergence
            if (iteration > 10 and 
                stats["reconstruction_error"] < self.config.convergence_threshold):
                converged = True
                print(f"Converged at iteration {iteration + 1}")
                break
            
            # Progress reporting
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1:3d}: Active={stats['active_agents']} "
                      f"Reconstructed={stats['successful_reconstructions']} "
                      f"Error={stats['reconstruction_error']:.4f}")
        
        # Compute final metrics
        final_mse = float(np.mean((self.current_image - self.target_image) ** 2))
        final_psnr = -10 * np.log10(final_mse) if final_mse > 0 else float('inf')
        
        # Compute reconstruction quality on damaged regions only
        damaged_mse = float(np.mean((self.current_image[self.damage_mask > 0.5] - 
                                   self.target_image[self.damage_mask > 0.5]) ** 2))
        damaged_psnr = -10 * np.log10(damaged_mse) if damaged_mse > 0 else float('inf')
        
        results = {
            "converged": converged,
            "iterations": iteration + 1,
            "final_mse": final_mse,
            "final_psnr": final_psnr,
            "damaged_mse": damaged_mse,
            "damaged_psnr": damaged_psnr,
            "reconstruction_error": stats.get("reconstruction_error", 0),
            "composition_analytics": self.composer.get_composition_analytics(),
            "run_directory": str(run_dir)
        }
        
        # Save images
        self._save_reconstruction_results(run_dir, results)
        
        print(f"\\n=== Image Reconstruction Results ===")
        print(f"Converged: {converged}")
        print(f"Iterations: {iteration + 1}")
        print(f"Final MSE: {final_mse:.6f}")
        print(f"Final PSNR: {final_psnr:.2f} dB")
        print(f"Damaged region PSNR: {damaged_psnr:.2f} dB")
        
        return results
    
    def _save_reconstruction_results(self, run_dir: Path, results: Dict):
        """Save reconstruction images and results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(self.target_image, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Damage mask
        axes[0, 1].imshow(self.damage_mask, cmap='Reds', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Damage Mask ({self.config.damage_type})')
        axes[0, 1].axis('off')
        
        # Damaged image
        axes[0, 2].imshow(self._apply_damage(), cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('Damaged Image')
        axes[0, 2].axis('off')
        
        # Reconstructed image
        axes[1, 0].imshow(self.current_image, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Reconstructed (PSNR: {results["final_psnr"]:.1f} dB)')
        axes[1, 0].axis('off')
        
        # Error map
        error_map = np.abs(self.current_image - self.target_image)
        axes[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=np.max(error_map))
        axes[1, 1].set_title('Reconstruction Error')
        axes[1, 1].axis('off')
        
        # Convergence plot
        if self.convergence_metrics:
            axes[1, 2].plot(self.convergence_metrics)
            axes[1, 2].set_title('Reconstruction Error vs Iteration')
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Reconstruction Error')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(run_dir / "reconstruction_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Results saved to: {run_dir}")


def create_test_images() -> Dict[str, Array]:
    """Create test images for completion experiments."""
    test_images = {}
    
    # 1. Simple geometric pattern
    geometric = np.zeros((32, 32))
    geometric[8:24, 8:24] = 1.0  # Square
    geometric[12:20, 12:20] = 0.0  # Hole in square
    test_images["geometric"] = geometric
    
    # 2. Gradient pattern
    gradient = np.zeros((32, 32))
    for i in range(32):
        gradient[i, :] = i / 31.0
    test_images["gradient"] = gradient
    
    # 3. Checkerboard pattern
    checkerboard = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            if (i // 4 + j // 4) % 2 == 0:
                checkerboard[i, j] = 1.0
    test_images["checkerboard"] = checkerboard
    
    # 4. Circular pattern
    circular = np.zeros((32, 32))
    center = 16
    for i in range(32):
        for j in range(32):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if 8 <= dist <= 12:
                circular[i, j] = 1.0
    test_images["circular"] = circular
    
    return test_images


__all__ = [
    "ImageCompletionConfig", "ImageCompletionAgent", "ImageCompletionSystem", "create_test_images"
]