# SCFD Research Quick Start

## 30-Second Demo

```bash
git clone https://github.com/012notforu/coherence-field.git
cd coherence-field
pip install -e .
python test_cartpole_vector_stress.py
```

This runs the complete SCFD system demonstrating:
- **Vector discovery**: Auto-loads CartPole control vectors
- **Real-time adaptation**: EFE tuner adjusts parameters during execution  
- **Field dynamics**: SCFD fields evolve in response to CartPole motion
- **Robust control**: Maintains balance under noise and disturbances

## Expected Output

```
🚀 SCFD Research Demo Starting...
Running SCFD preflight checks...
SCFD preflight: ALL TESTS PASSED
Loading vector registry...
Found 6 vectors across 5 domains
=== CartPole Vector Stress Test ===
Using 2 CartPole vectors
Max timesteps: 500
T0: Using cartpole_cma, force=-0.206, theta=0.000, field_rms=0.005
T20: Using cartpole_cma, force=-0.381, theta=0.001, field_rms=0.005
...
Success: True
Survival rate: 100.0%
Composition success: 100.0%
```

## Generated Files

After running, check `runs/[timestamp]_cartpole_vector_stress/`:
- **`scfd_cartpole.gif`**: Animated CartPole + field visualization
- **`scfd_ca_raster.png`**: Field evolution raster (white/red/blue activity)
- **`logs.jsonl`**: Complete run metadata and performance metrics

## What Just Happened?

1. **Vector Discovery**: System found CartPole vectors in `runs/cartpole_cma*/`
2. **Field Initialization**: Created 128×128 SCFD field with random initial conditions  
3. **Real-time Control**: Vector controllers guided field evolution for CartPole balancing
4. **Adaptation**: EFE tuner adjusted parameters based on performance feedback
5. **Visualization**: Generated animations showing coupled field-control dynamics

The system demonstrates **production-ready SCFD control** with no hardcoded parameters - everything learned and adapted in real-time.