"""Maze solving benchmark using SCFD adaptive meta-learning workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import accel_theta, load_config, total_energy_density
from engine.integrators import leapfrog_step

Array = np.ndarray


@dataclass
class MazeParams:
    shape: Tuple[int, int] = (32, 32)
    wall_density: float = 0.3
    start_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None
    dt: float = 0.1
    init_seed: int = 0


@dataclass 
class MazeSCFDConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    encode_gain: float = 1.0
    encode_decay: float = 0.9
    movement_threshold: float = 0.1
    exploration_bonus: float = 0.2
    theta_clip: float = 3.0


class AdaptiveMazeSolver:
    """Adaptive maze solver with multi-generation vector selection."""
    
    def __init__(self, params: MazeParams, maze_layout: Array, vector_registry: List) -> None:
        self.params = params
        self.maze = maze_layout.astype(np.float32)
        self.registry = vector_registry
        self.rng = np.random.default_rng(params.init_seed)
        
        # Set start/goal positions
        h, w = params.shape
        self.start_pos = params.start_pos or (1, 1)
        self.goal_pos = params.goal_pos or (h-2, w-2)  # Hidden from controller
        
        # Multi-generation tracking
        self.generation = 0
        self.vector_history = []
        self.performance_history = []
        self.failure_patterns = {}
        
        self.reset()
        
    def reset(self) -> None:
        """Reset for new solving attempt."""
        self.current_pos = self.start_pos
        self.path_history = [self.start_pos]
        self.visited_cells = set([self.start_pos])
        self.solved = False
        self.steps_taken = 0
        self.stuck_counter = 0
        self.last_progress_step = 0
        
        # Environment observation (no goal knowledge)
        self.environment_obs = self._create_environment_observation()
        
    def _create_environment_observation(self) -> Array:
        """Create pure environment observation without goal knowledge."""
        obs = np.zeros_like(self.maze)
        
        # Wall structure
        obs += self.maze  # 1 = wall, 0 = free
        
        # Current agent position  
        obs[self.current_pos] = 0.5
        
        # Visited regions (exploration tracking)
        for pos in self.visited_cells:
            obs[pos] = 0.3
            
        # Local structure analysis (corners, edges, openings)
        h, w = self.maze.shape
        y, x = self.current_pos
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and self.maze[ny, nx] == 0:
                    distance = abs(dy) + abs(dx)
                    obs[ny, nx] += 0.1 / max(1, distance)  # Distance-weighted free space
        
        return obs
    
    def _detect_failure_pattern(self) -> str:
        """Analyze current failure pattern for vector re-selection."""
        if self.stuck_counter > 10:
            return "stuck_in_loop"
        elif self.steps_taken - self.last_progress_step > 15:
            return "no_recent_progress"
        elif len(set(self.path_history[-10:])) < 3:  # Repeating positions
            return "local_oscillation"
        elif self.stuck_counter > 5 and len(self.visited_cells) < 3:
            return "no_exploration"
        else:
            return "making_progress"
    
    def _should_switch_vector(self) -> bool:
        """Determine if current vector should be abandoned."""
        pattern = self._detect_failure_pattern()
        
        # Switch if clearly failing
        if pattern in ["stuck_in_loop", "no_exploration", "local_oscillation"]:
            return True
            
        # Switch if no progress for too long
        if self.steps_taken - self.last_progress_step > 30:
            return True
            
        return False
    
    def _select_next_vector(self) -> Dict:
        """Select next vector based on failure analysis and history."""
        from orchestrator.pipeline import plan_for_environment
        
        # Create factory for current state
        def state_factory():
            # Return a mock environment that represents current solving state
            class MockMazeState:
                def __init__(self, obs, failure_pattern):
                    self.observation = obs
                    self.failure_type = failure_pattern
                    self.steps_taken = self.steps_taken
                    
                def step(self):
                    return {"observation": self.observation, "failure": self.failure_type}
                    
            return MockMazeState(self.environment_obs, self._detect_failure_pattern())
        
        # Filter out recently failed vectors
        available_registry = [
            vec for vec in self.registry 
            if vec.vector_id not in [vh["vector_entry"].vector_id for vh in self.vector_history[-2:]]
        ]
        
        if not available_registry:
            available_registry = self.registry  # Reset if all tried
            
        # Use orchestrator to select based on current state
        try:
            plan = plan_for_environment(state_factory, registry=available_registry, steps=20)
            selected = plan.steps[0].vector
        except:
            # Fallback to random selection
            selected = self.rng.choice(available_registry)
            
        return {
            "vector_entry": selected,
            "selection_reason": self._detect_failure_pattern(),
            "generation": self.generation,
        }
    
    def _apply_vector_to_config(self, vector: np.ndarray) -> MazeSCFDConfig:
        """Apply vector to maze configuration (heuristic mapping)."""
        base_config = MazeSCFDConfig()
        
        if len(vector) >= 4:
            return MazeSCFDConfig(
                encode_gain=float(np.clip(vector[0], 0.1, 3.0)),
                encode_decay=float(np.clip(vector[1], 0.5, 0.99)),
                movement_threshold=float(np.clip(vector[2], 0.01, 0.5)),
                exploration_bonus=float(np.clip(vector[3], 0.0, 1.0)),
            )
        return base_config
    
    def _create_scfd_controller(self, config: MazeSCFDConfig):
        """Create SCFD controller with given configuration."""
        return MazeSCFDController(config, self.params.shape)
    
    def run_generation(self, max_steps: int = 100) -> Dict[str, object]:
        """Run one generation with current vector."""
        # Select vector for this generation
        vector_selection = self._select_next_vector()
        vector_entry = vector_selection["vector_entry"]
        
        print(f"Generation {self.generation}: Trying vector {vector_entry.vector_id}")
        print(f"  Reason: {vector_selection['selection_reason']}")
        
        # Load and apply vector
        import json
        with open(vector_entry.path) as f:
            data = json.load(f)
        vector = np.array(data["vector"], dtype=np.float32)
        
        config = self._apply_vector_to_config(vector)
        controller = self._create_scfd_controller(config)
        
        # Track generation performance
        gen_start_pos = self.current_pos
        gen_start_visited = len(self.visited_cells)
        generation_metrics = []
        
        # Run steps for this generation
        for step in range(max_steps):
            # Update environment observation
            self.environment_obs = self._create_environment_observation()
            
            # Get SCFD guidance (no goal knowledge)
            control = controller.step(self.environment_obs)
            guidance = control["path_guidance"]
            
            # Move based on guidance
            old_pos = self.current_pos
            self._make_move(guidance)
            
            # Track progress
            if self.current_pos != old_pos:
                self.last_progress_step = self.steps_taken
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
                
            self.steps_taken += 1
            
            # Check termination conditions
            if self.current_pos == self.goal_pos:
                self.solved = True
                break
                
            if self._should_switch_vector():
                print(f"  Switching vector after {step+1} steps")
                break
                
            # Record metrics
            distance_to_goal = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
            generation_metrics.append({
                "step": step,
                "distance_to_goal": float(distance_to_goal),
                "exploration_count": len(self.visited_cells),
                "current_pos": self.current_pos,
            })
        
        # Evaluate generation performance
        progress_made = len(self.visited_cells) - gen_start_visited
        distance_progress = (
            abs(gen_start_pos[0] - self.goal_pos[0]) + abs(gen_start_pos[1] - self.goal_pos[1])
        ) - (
            abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
        )
        
        performance = {
            "vector_id": vector_entry.vector_id,
            "vector_physics": vector_entry.physics,
            "progress_made": progress_made,
            "distance_progress": distance_progress,
            "steps_used": len(generation_metrics),
            "final_pos": self.current_pos,
            "selection_reason": vector_selection["selection_reason"],
        }
        
        self.vector_history.append(vector_selection)
        self.performance_history.append(performance)
        self.generation += 1
        
        return {
            "generation": self.generation - 1,
            "performance": performance,
            "metrics": generation_metrics,
            "solved": self.solved,
        }
    
    def _make_move(self, guidance: Array) -> None:
        """Make movement decision based on SCFD guidance."""
        y, x = self.current_pos
        h, w = self.maze.shape
        
        # Sample possible moves
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        move_scores = []
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and self.maze[ny, nx] == 0):
                # Base score from SCFD guidance
                score = guidance[ny, nx]
                
                # Exploration bonus for unvisited cells
                if (ny, nx) not in self.visited_cells:
                    score += 0.5
                    
                move_scores.append(score)
            else:
                move_scores.append(-np.inf)
        
        # Choose best move
        if max(move_scores) > -np.inf:
            best_idx = np.argmax(move_scores)
            dy, dx = directions[best_idx]
            new_pos = (y + dy, x + dx)
            
            self.current_pos = new_pos
            self.path_history.append(new_pos)
            self.visited_cells.add(new_pos)
    
    def solve_adaptively(self, max_generations: int = 5, steps_per_generation: int = 100) -> Dict[str, object]:
        """Solve maze using adaptive multi-generation approach."""
        print(f"Starting adaptive maze solving: {max_generations} generations, {steps_per_generation} steps each")
        
        all_results = []
        
        for gen in range(max_generations):
            if self.solved:
                break
                
            result = self.run_generation(steps_per_generation)
            all_results.append(result)
            
            print(f"Generation {gen} completed: {result['performance']['vector_id']}")
            print(f"  Progress: {result['performance']['progress_made']} new cells")
            print(f"  Distance progress: {result['performance']['distance_progress']}")
            
        return {
            "solved": self.solved,
            "generations_used": len(all_results),
            "final_path_length": len(self.path_history),
            "total_steps": self.steps_taken,
            "vector_sequence": [r["performance"]["vector_id"] for r in all_results],
            "generation_results": all_results,
            "path_history": self.path_history,
        }
    
    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        """Generate adaptive solving visualization."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left: maze with adaptive path colored by generation
        maze_vis = self.maze.copy()
        
        # Color path by generation
        gen_results = history["generation_results"]
        colors = plt.cm.Set1(np.linspace(0, 1, len(gen_results)))
        
        for i, gen_result in enumerate(gen_results):
            gen_metrics = gen_result["metrics"]
            for metric in gen_metrics:
                pos = metric["current_pos"]
                maze_vis[pos] = 0.3 + 0.1 * i  # Different shade per generation
        
        maze_vis[self.start_pos] = 0.8  # Start
        maze_vis[self.goal_pos] = 0.2   # Goal
        
        axes[0].imshow(maze_vis, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0].set_title(f"Adaptive Solution ({len(gen_results)} vectors used)")
        axes[0].axis('off')
        
        # Right: vector performance over generations
        if gen_results:
            vectors = [r["performance"]["vector_id"] for r in gen_results]
            progress = [r["performance"]["progress_made"] for r in gen_results]
            
            axes[1].bar(range(len(vectors)), progress)
            axes[1].set_xlabel("Generation")
            axes[1].set_ylabel("Exploration Progress")
            axes[1].set_title("Vector Performance by Generation")
            axes[1].set_xticks(range(len(vectors)))
            axes[1].set_xticklabels([v[:15] for v in vectors], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save files
        viz_path = out_path / "adaptive_maze_solution.png"
        fig.savefig(viz_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        
        history_path = out_path / "adaptive_maze_history.npz"
        np.savez_compressed(history_path, **history)
        
        return {
            "visualization": str(viz_path),
            "history": str(history_path),
        }


def generate_maze(shape: Tuple[int, int], wall_density: float = 0.3, seed: int = 0) -> Array:
    """Generate random maze layout."""
    rng = np.random.default_rng(seed)
    h, w = shape
    
    maze = np.ones((h, w), dtype=np.float32)
    maze[0, :] = 1  # Border walls
    maze[-1, :] = 1
    maze[:, 0] = 1  
    maze[:, -1] = 1
    
    # Random interior
    interior = rng.random((h-2, w-2)) < wall_density
    maze[1:-1, 1:-1] = interior.astype(np.float32)
    
    # Ensure start/goal are free
    maze[1, 1] = 0
    maze[h-2, w-2] = 0
    
    return maze


# Keep the simple controller for compatibility
class MazeSCFDController:
    """Simple SCFD controller for maze navigation."""
    
    def __init__(self, params: MazeSCFDConfig, grid_shape: Tuple[int, int]) -> None:
        self.cfg = params
        self.sim_cfg = load_config(params.scfd_cfg_path)
        self.dx = self.sim_cfg.grid.spacing
        self.theta = np.zeros(grid_shape, dtype=np.float32)
        self.theta_dot = np.zeros_like(self.theta)
        
    def step(self, observation: Array) -> Dict[str, Array]:
        """Update SCFD field based on environment observation."""
        # Inject observation into SCFD field
        inject = self.cfg.encode_gain * observation
        inject = np.clip(inject, -self.cfg.theta_clip, self.cfg.theta_clip)
        self.theta += inject
        self.theta = np.clip(self.theta, -self.cfg.theta_clip, self.cfg.theta_clip)
        
        # Evolve SCFD dynamics
        self.theta, self.theta_dot, _, _ = leapfrog_step(
            self.theta,
            self.theta_dot,
            lambda f: accel_theta(f, self.sim_cfg.physics, dx=self.dx),
            self.sim_cfg.integration.dt,
        )
        
        # Generate guidance
        guidance = np.tanh(self.theta)
        
        return {"path_guidance": guidance, "theta": self.theta.copy()}


__all__ = [
    "MazeParams",
    "MazeSCFDConfig",
    "MazeSCFDController", 
    "AdaptiveMazeSolver",
    "generate_maze",
]