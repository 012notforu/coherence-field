"""
Example: decide engine by env metadata and replay a vector.
Policy: if task has 'wave' or 'flow' tags -> SCFD; if 'lenia' -> EM; else default SCFD.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

def choose_engine(env_meta: dict) -> str:
    """Choose engine based on environment metadata."""
    task = env_meta.get("task", "").lower()
    tags = env_meta.get("tags", [])
    
    # Convert tags to lowercase for comparison
    tags_lower = [tag.lower() for tag in tags]
    
    # Engine selection logic
    if "lenia" in tags_lower:
        return "em"
    elif "wave" in tags_lower or "flow" in tags_lower:
        return "scfd"
    elif "cartpole" in task or "control" in tags_lower:
        return "scfd"
    else:
        return "scfd"  # default

def run_episode(controller: str, steps: int, episodes: int) -> float:
    """Run an episode using the specified controller."""
    if controller == "scfd":
        # Check if we have a trained vector available
        vector_path = Path("runs/cartpole_cma/best_vector.json")
        if vector_path.exists():
            cmd = [
                sys.executable, "-m", "benchmarks.run_cartpole",
                "--controller", "scfd",
                "--vector", str(vector_path),
                "--steps", str(steps),
                "--episodes", str(episodes),
                "--viz", "none"
            ]
        else:
            cmd = [
                sys.executable, "-m", "benchmarks.run_cartpole",
                "--controller", "scfd",
                "--steps", str(steps),
                "--episodes", str(episodes),
                "--viz", "none"
            ]
    else:  # em
        cmd = [
            sys.executable, "-m", "benchmarks.run_cartpole",
            "--controller", "em",
            "--steps", str(steps),
            "--viz", "none"
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Parse basic success - in real implementation would extract metrics
            return float(steps)  # Simplified return value
        else:
            print(f"Warning: Command failed: {result.stderr}")
            return 0.0
    except subprocess.TimeoutExpired:
        print("Warning: Command timed out")
        return 0.0
    except Exception as e:
        print(f"Warning: Command failed with exception: {e}")
        return 0.0

def main():
    # Example environment metadata scenarios
    scenarios = [
        {"task": "cartpole", "tags": ["control", "inverted-pendulum"]},
        {"task": "flow", "tags": ["fluid", "navier-stokes"]}, 
        {"task": "pattern", "tags": ["lenia", "cellular-automata"]},
        {"task": "wave", "tags": ["wave", "acoustics"]},
    ]
    
    for i, env_meta in enumerate(scenarios):
        print(f"\nScenario {i+1}: {env_meta}")
        engine = choose_engine(env_meta)
        print(f"Selected engine: {engine}")
        
        # Only run cartpole scenario since others may not be immediately available
        if env_meta["task"] == "cartpole":
            reward = run_episode(controller=engine, steps=1000, episodes=1)
            print(f"Result: {reward}")
        else:
            print("Result: skipped (demo scenario)")

if __name__ == "__main__":
    main()