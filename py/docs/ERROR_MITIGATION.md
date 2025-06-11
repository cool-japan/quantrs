# Quantum Error Mitigation in QuantRS2

Quantum error mitigation techniques help reduce the impact of noise in quantum computations without requiring full error correction. This module provides several mitigation strategies, with Zero-Noise Extrapolation (ZNE) being the primary implemented method.

## Features

### Implemented
- **Zero-Noise Extrapolation (ZNE)**: Extrapolate results to the zero-noise limit
  - Multiple noise scaling methods (global/local folding, pulse stretching, digital repetition)
  - Various extrapolation methods (linear, polynomial, exponential, Richardson, adaptive)
  - Bootstrap error estimation
  - Observable expectation value calculation

### Planned
- **Probabilistic Error Cancellation (PEC)**: Cancel errors using quasi-probability decomposition
- **Virtual Distillation**: Purify quantum states using multiple circuit copies
- **Symmetry Verification**: Detect and mitigate errors using symmetry constraints
- **Measurement Error Mitigation**: Already available in the measurement module

## Zero-Noise Extrapolation (ZNE)

ZNE works by intentionally amplifying the noise in a quantum circuit by known scale factors, measuring the observable at each noise level, and then extrapolating back to the zero-noise limit.

### Basic Usage

```python
from quantrs2 import Circuit
from quantrs2.mitigation import (
    ZNEConfig, Observable, ZeroNoiseExtrapolation
)

# Create a quantum circuit
circuit = Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)

# Configure ZNE
config = ZNEConfig(
    scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
    scaling_method="global",
    extrapolation_method="richardson",
    bootstrap_samples=100
)

# Create ZNE executor
zne = ZeroNoiseExtrapolation(config)

# Define observable to measure
observable = Observable.zz(0, 1)  # Measure Z₀Z₁ correlation

# In practice, you would:
# 1. Fold circuits at each scale factor
# 2. Execute on quantum hardware
# 3. Collect measurement results
# 4. Extrapolate to zero noise

# Example with synthetic data
data = [(1.0, 0.95), (1.5, 0.90), (2.0, 0.85), (2.5, 0.80), (3.0, 0.75)]
result = zne.extrapolate(data)
print(f"Mitigated value: {result.mitigated_value:.4f}")
print(f"Error estimate: ±{result.error_estimate:.4f}")
```

### Configuration Options

#### Noise Scaling Methods
- `"global"`: Fold the entire circuit uniformly (G → GG†G)
- `"local"`: Fold individual gates with different weights
- `"pulse"`: Stretch pulse durations (for pulse-level control)
- `"digital"`: Repeat digital gates

#### Extrapolation Methods
- `"linear"`: Simple linear extrapolation
- `"polynomial2"`, `"polynomial3"`: Polynomial fitting of specified order
- `"exponential"`: Exponential decay model
- `"richardson"`: Richardson extrapolation (recommended)
- `"adaptive"`: Automatically choose best fitting method

### Circuit Folding

Circuit folding is the primary method for scaling noise in digital quantum circuits:

```python
from quantrs2.mitigation import CircuitFolding

# Global folding: G → GG†G
folded = CircuitFolding.fold_global(circuit, scale_factor=3.0)

# Local folding with custom gate weights
weights = [1.0, 2.0, 0.5]  # Different scaling for each gate
folded = CircuitFolding.fold_local(circuit, scale_factor=2.0, gate_weights=weights)
```

### Observable Definition

Define observables as weighted sums of Pauli strings:

```python
from quantrs2.mitigation import Observable

# Single qubit observables
z0 = Observable.z(0)          # Z on qubit 0
z1 = Observable.z(1)          # Z on qubit 1

# Two-qubit correlation
zz = Observable.zz(0, 1)      # Z₀Z₁

# Custom observable: 0.5 * X₀Y₁ + 0.3 * Z₂
custom = Observable(
    [(0, "X"), (1, "Y")], 
    coefficient=0.5
)

# Calculate expectation value from measurements
expectation = observable.expectation_value(measurement_result)
```

### Extrapolation Fitting

Use different fitting methods directly:

```python
from quantrs2.mitigation import ExtrapolationFitting
import numpy as np

# Your noisy data
scale_factors = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
expectation_values = np.array([0.95, 0.90, 0.86, 0.82, 0.78])

# Try different fits
linear_result = ExtrapolationFitting.fit_linear(scale_factors, expectation_values)
poly_result = ExtrapolationFitting.fit_polynomial(scale_factors, expectation_values, order=2)
exp_result = ExtrapolationFitting.fit_exponential(scale_factors, expectation_values)
richardson_result = ExtrapolationFitting.fit_richardson(scale_factors, expectation_values)

# Adaptive fitting automatically selects best method
adaptive_result = ExtrapolationFitting.fit_adaptive(scale_factors, expectation_values)

print(f"Linear: {linear_result.mitigated_value:.4f} (R²={linear_result.r_squared:.4f})")
print(f"Richardson: {richardson_result.mitigated_value:.4f}")
```

## Complete Example: VQE with Error Mitigation

```python
import numpy as np
from quantrs2 import Circuit
from quantrs2.mitigation import ZNEConfig, Observable, ZeroNoiseExtrapolation
from quantrs2.measurement import MeasurementSampler

def create_vqe_ansatz(theta):
    """Create a simple VQE ansatz."""
    circuit = Circuit(2)
    circuit.ry(0, theta[0])
    circuit.ry(1, theta[1])
    circuit.cnot(0, 1)
    circuit.ry(0, theta[2])
    circuit.ry(1, theta[3])
    return circuit

def run_vqe_with_mitigation(theta, backend=None):
    """Run VQE with ZNE error mitigation."""
    # Create ansatz
    circuit = create_vqe_ansatz(theta)
    
    # Define Hamiltonian (H = Z₀Z₁ + 0.5*(X₀ + X₁))
    observables = [
        (Observable.zz(0, 1), 1.0),
        (Observable([(0, "X")], 1.0), 0.5),
        (Observable([(1, "X")], 1.0), 0.5),
    ]
    
    # Configure ZNE
    zne_config = ZNEConfig(
        scale_factors=[1.0, 1.5, 2.0, 2.5],
        scaling_method="global",
        extrapolation_method="richardson"
    )
    zne = ZeroNoiseExtrapolation(zne_config)
    
    # Measure each observable with ZNE
    total_energy = 0.0
    
    for obs, coeff in observables:
        # Collect measurements at different noise scales
        measurements = []
        
        for scale in zne_config.scale_factors:
            # Fold circuit
            folded_circuit = zne.fold_circuit(circuit, scale)
            
            # Execute (simulation for demo)
            result = folded_circuit.run()
            sampler = MeasurementSampler()
            measurement = sampler.sample_counts(result, shots=8192)
            measurements.append((scale, measurement))
        
        # Extrapolate to zero noise
        mitigated = zne.mitigate_observable(obs, measurements)
        total_energy += coeff * mitigated.mitigated_value
    
    return total_energy

# Optimize VQE parameters
theta = np.array([0.1, 0.2, 0.3, 0.4])
energy = run_vqe_with_mitigation(theta)
print(f"Energy with error mitigation: {energy:.6f}")
```

## Best Practices

1. **Scale Factor Selection**
   - Use at least 3-5 scale factors for reliable extrapolation
   - Include scale factor 1.0 (unmodified circuit)
   - Don't use very large scale factors (typically ≤ 5)

2. **Extrapolation Method**
   - Richardson extrapolation is generally most reliable
   - Use adaptive fitting to automatically select best method
   - Linear extrapolation works well for small noise levels

3. **Bootstrap Error Estimation**
   - Enable bootstrap sampling for error bars
   - Use 100-1000 bootstrap samples
   - Provides confidence intervals on mitigated values

4. **Observable Selection**
   - Define observables that match your problem
   - Use symmetries to simplify measurements
   - Consider measuring multiple observables simultaneously

5. **Circuit Folding**
   - Global folding is simpler and often sufficient
   - Local folding allows fine-grained control
   - Ensure folded circuits remain within coherence time

## Performance Considerations

- ZNE requires multiple circuit executions (one per scale factor)
- Total shots = base_shots × number_of_scale_factors
- Bootstrap error estimation adds computational overhead
- Richardson extrapolation is more expensive than linear

## Limitations

1. **Current Implementation**
   - Circuit folding returns placeholder circuits (full implementation pending)
   - PEC, Virtual Distillation, and Symmetry Verification are placeholders
   - Observable measurements limited to Pauli Z basis

2. **Fundamental Limitations**
   - ZNE assumes noise can be amplified uniformly
   - Works best for noise that scales predictably
   - Cannot mitigate coherent errors effectively
   - Requires good statistics (many shots)

## Future Enhancements

1. **Full Circuit Folding**
   - Implement actual gate folding once Circuit API supports it
   - Add support for partial folding
   - Optimize folding strategies

2. **Additional Mitigation Methods**
   - Probabilistic Error Cancellation (PEC)
   - Virtual Distillation / Error Suppression by Decoherence
   - Symmetry-based verification and post-selection
   - Clifford Data Regression (CDR)

3. **Advanced Features**
   - Automatic scale factor selection
   - Multi-observable optimization
   - Integration with variational algorithms
   - Hardware-specific calibration

## References

1. [Temme et al., "Error mitigation for short-depth quantum circuits" (2017)](https://arxiv.org/abs/1612.02058)
2. [Li & Benjamin, "Efficient variational quantum simulator incorporating active error minimization" (2017)](https://arxiv.org/abs/1611.09301)
3. [Kandala et al., "Error mitigation extends the computational reach of a noisy quantum processor" (2019)](https://arxiv.org/abs/1805.04492)
4. [Giurgica-Tiron et al., "Digital zero noise extrapolation for quantum error mitigation" (2020)](https://arxiv.org/abs/2005.10921)