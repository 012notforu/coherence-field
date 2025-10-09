# SCFD Research Installation & Usage

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/012notforu/coherence-field.git
   cd coherence-field
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Run the CartPole demo:**
   ```bash
   python test_cartpole_vector_stress.py
   ```

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- PyTorch (for advanced features)
- All dependencies listed in `pyproject.toml`

## System Components

### Core SCFD Engine
- **`engine/`** - Symbolic Coherent Field Dynamics physics engine
- **`cfg/defaults.yaml`** - SCFD simulation configuration

### Advanced Control System
- **`utils/efe_tuner.py`** - Expected Free Energy parameter tuner
- **`utils/vector_controller_bridge.py`** - Vector-controller integration
- **`utils/parameter_registry.py`** - Dynamic parameter management
- **`utils/unified_metadata.py`** - Comprehensive metadata tracking

### Vector System
- **`orchestrator/pipeline.py`** - Vector discovery and environment planning
- **`runs/*/best_vector.json`** - Trained SCFD vectors across domains
- **`benchmarks/vector_composition.py`** - Vector composition rules

### Observation & Control
- **`observe/features.py`** - SCFD field feature extraction
- **`observe/policies/`** - Linear policy implementations
- **`observe/controller.py`** - GentleController for sensing

## Usage Examples

### Basic CartPole Control
```bash
python test_cartpole_vector_stress.py
```

### Run with different stress levels
Edit the configuration in `test_cartpole_vector_stress.py`:
```python
config = CartPoleStressConfig(
    max_timesteps=2000,
    sensor_noise_std=0.01,      # Light stress
    force_disturbance_std=0.5,  # Moderate disturbances
    physics_drift_rate=0.00005  # Slow parameter drift
)
```

### Access Trained Vectors
The system auto-discovers vectors in `runs/*/best_vector.json`:
- `cartpole_cma` - High-performance CartPole control (5000 timesteps)
- `gray_scott_cma_stripes` - Gray-Scott pattern formation
- `heat_diffusion_cma_hotcorner` - Heat transfer control
- `flow_cylinder_cma_large` - Fluid dynamics control
- `wave_field_cma_large` - Wave propagation control

## Output

Successful runs generate:
- **Visualization**: `scfd_cartpole.gif` (animated CartPole control)
- **Field Analysis**: `scfd_ca_raster.png` (SCFD field evolution)
- **Logs**: Complete run metadata in `logs.jsonl`
- **Results**: Performance metrics and vector usage

## Configuration

Key configuration in `cfg/defaults.yaml`:
- **Grid**: 128x128 spatial resolution
- **Physics**: SCFD dynamics parameters  
- **Integration**: Symplectic time stepping
- **Scheduling**: Poisson process control

## Advanced Features

- **Vector Composition**: Multiple vectors can be combined using Metropolis acceptance
- **Real-time Adaptation**: EFE tuner adjusts parameters during execution
- **Field Visualization**: CA-style raster plots show SCFD field evolution
- **Cross-domain Vectors**: Same framework works for heat, flow, waves, and control