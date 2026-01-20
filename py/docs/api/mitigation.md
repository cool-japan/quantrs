# Error Mitigation API Reference

Complete reference for quantum error mitigation techniques in QuantRS2, essential for improving results on noisy quantum devices.

## Overview

Error mitigation techniques help improve quantum algorithm results without full quantum error correction. These methods are crucial for NISQ (Noisy Intermediate-Scale Quantum) devices where perfect error correction isn't feasible.

```python
import quantrs2
from quantrs2.mitigation import (
    zero_noise_extrapolation,
    readout_error_mitigation,
    symmetry_verification,
    richardson_extrapolation
)

# Create and run circuit with error mitigation
circuit = quantrs2.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Apply error mitigation
result = circuit.run()
mitigated_result = zero_noise_extrapolation(circuit, noise_levels=[0.01, 0.02, 0.03])
```

## Error Mitigation Categories

### Pre-Execution Techniques
- **Circuit optimization**: Reduce noise exposure
- **Pulse optimization**: Hardware-level improvements
- **Dynamical decoupling**: Protect qubits during idle times

### Post-Execution Techniques
- **Zero-noise extrapolation**: Extrapolate to zero noise limit
- **Readout error mitigation**: Correct measurement errors
- **Symmetry verification**: Use known symmetries to detect errors

### Hybrid Techniques
- **Virtual distillation**: Combine multiple circuit executions
- **Error amplification**: Magnify errors for better extrapolation

## Zero-Noise Extrapolation

### Basic Zero-Noise Extrapolation

#### `zero_noise_extrapolation(circuit, noise_levels, extrapolation_method='linear')`

Extrapolate measurement results to the zero-noise limit by running circuits at different noise levels.

**Parameters:**
- `circuit` (Circuit): Quantum circuit to execute
- `noise_levels` (List[float]): List of noise amplification factors
- `extrapolation_method` (str): Extrapolation method ('linear', 'exponential', 'polynomial')

**Returns:**
- `MitigatedResult`: Extrapolated measurement results

**Example:**
```python
import quantrs2
from quantrs2.mitigation import zero_noise_extrapolation

# Create Bell state circuit
circuit = quantrs2.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Apply zero-noise extrapolation
noise_levels = [1.0, 1.5, 2.0, 2.5]  # Noise amplification factors
mitigated_result = zero_noise_extrapolation(
    circuit, 
    noise_levels=noise_levels,
    extrapolation_method='exponential'
)

print(f"Original result: {circuit.run().state_probabilities()}")
print(f"Mitigated result: {mitigated_result.expectation_values}")
```

### Advanced Extrapolation Methods

#### `richardson_extrapolation(results, noise_levels, order=2)`

Apply Richardson extrapolation for higher-order error cancellation.

**Parameters:**
- `results` (List[SimulationResult]): Results at different noise levels
- `noise_levels` (List[float]): Corresponding noise levels
- `order` (int): Order of Richardson extrapolation

**Returns:**
- `MitigatedResult`: Higher-order extrapolated result

**Example:**
```python
from quantrs2.mitigation import richardson_extrapolation

# Collect results at different noise levels
results = []
noise_levels = [1.0, 1.2, 1.4, 1.6]

for noise in noise_levels:
    noisy_circuit = add_noise_to_circuit(circuit, noise)
    result = noisy_circuit.run()
    results.append(result)

# Apply Richardson extrapolation
mitigated = richardson_extrapolation(
    results, 
    noise_levels, 
    order=3  # Higher order for better accuracy
)
```

#### `exponential_extrapolation(results, noise_levels, fit_function='exponential')`

Fit exponential decay model and extrapolate to zero noise.

**Parameters:**
- `results` (List[SimulationResult]): Measurement results
- `noise_levels` (List[float]): Noise amplification factors
- `fit_function` (str): Function type ('exponential', 'power', 'polynomial')

**Example:**
```python
from quantrs2.mitigation import exponential_extrapolation
import numpy as np

# Generate noise levels exponentially
noise_levels = [1.0 + 0.1 * i for i in range(5)]
results = [run_with_noise(circuit, noise) for noise in noise_levels]

mitigated = exponential_extrapolation(
    results, 
    noise_levels,
    fit_function='exponential'
)

print(f"Fit quality: R² = {mitigated.fit_quality}")
print(f"Extrapolated expectation: {mitigated.expectation_values}")
```

## Readout Error Mitigation

### Calibration Matrix Method

#### `create_readout_calibration_matrix(num_qubits, backend=None)`

Create calibration matrix by measuring all computational basis states.

**Parameters:**
- `num_qubits` (int): Number of qubits to calibrate
- `backend` (Optional[Backend]): Quantum device backend

**Returns:**
- `np.ndarray`: Readout calibration matrix

**Example:**
```python
from quantrs2.mitigation import (
    create_readout_calibration_matrix,
    apply_readout_correction
)

# Create calibration matrix
num_qubits = 3
calibration_matrix = create_readout_calibration_matrix(num_qubits)

print(f"Calibration matrix shape: {calibration_matrix.shape}")
print(f"Matrix conditioning: {np.linalg.cond(calibration_matrix)}")

# Apply to measurement results
circuit = quantrs2.Circuit(num_qubits)
# ... build circuit ...
result = circuit.run()

corrected_probs = apply_readout_correction(
    result.state_probabilities(),
    calibration_matrix
)
```

#### `apply_readout_correction(measurement_counts, calibration_matrix)`

Apply readout error correction using calibration matrix.

**Parameters:**
- `measurement_counts` (Dict[str, int] or Dict[str, float]): Raw measurement counts or probabilities
- `calibration_matrix` (np.ndarray): Calibration matrix from basis state measurements

**Returns:**
- `Dict[str, float]`: Corrected measurement probabilities

**Example:**
```python
# Raw noisy measurements
raw_counts = {'000': 485, '001': 12, '010': 8, '111': 495}

# Convert to probabilities
total_shots = sum(raw_counts.values())
raw_probs = {state: count/total_shots for state, count in raw_counts.items()}

# Apply correction
corrected_probs = apply_readout_correction(raw_probs, calibration_matrix)

print("Readout error correction:")
for state in ['000', '111']:
    print(f"  {state}: {raw_probs.get(state, 0):.3f} → {corrected_probs.get(state, 0):.3f}")
```

### Least Squares Readout Correction

#### `least_squares_readout_correction(counts, calibration_matrix, regularization=1e-3)`

Apply readout correction using regularized least squares.

**Parameters:**
- `counts` (Dict[str, int]): Measurement counts
- `calibration_matrix` (np.ndarray): Calibration matrix
- `regularization` (float): Regularization parameter to avoid ill-conditioning

**Returns:**
- `Dict[str, float]`: Corrected probabilities

**Example:**
```python
from quantrs2.mitigation import least_squares_readout_correction

# For poorly conditioned calibration matrices
corrected = least_squares_readout_correction(
    raw_counts,
    calibration_matrix,
    regularization=1e-2  # Higher regularization for stability
)
```

### Tensored Readout Correction

#### `tensored_readout_correction(counts, single_qubit_matrices)`

Apply readout correction assuming independent single-qubit errors.

**Parameters:**
- `counts` (Dict[str, int]): Measurement counts
- `single_qubit_matrices` (List[np.ndarray]): List of 2×2 single-qubit calibration matrices

**Returns:**
- `Dict[str, float]`: Corrected probabilities

**Example:**
```python
from quantrs2.mitigation import (
    calibrate_single_qubit_readout,
    tensored_readout_correction
)

# Calibrate each qubit independently
single_qubit_matrices = []
for qubit in range(num_qubits):
    matrix = calibrate_single_qubit_readout(qubit)
    single_qubit_matrices.append(matrix)

# Apply tensored correction (scales better)
corrected = tensored_readout_correction(raw_counts, single_qubit_matrices)
```

## Symmetry Verification

### Symmetry-Based Error Detection

#### `verify_measurement_symmetries(result, symmetry_operators, tolerance=0.1)`

Verify measurement results respect known symmetries of the quantum state.

**Parameters:**
- `result` (SimulationResult): Measurement results to verify
- `symmetry_operators` (List[str]): List of symmetry operators (e.g., ['X0', 'Z1*Z2'])
- `tolerance` (float): Allowed violation threshold

**Returns:**
- `SymmetryVerificationResult`: Verification results and detected violations

**Example:**
```python
from quantrs2.mitigation import verify_measurement_symmetries

# Bell state should have certain symmetries
circuit = quantrs2.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
result = circuit.run()

# Define expected symmetries
symmetries = [
    'Z0*Z1',    # Bell state has Z⊗Z symmetry
    'X0*X1'     # Also X⊗X symmetry
]

verification = verify_measurement_symmetries(
    result, 
    symmetries, 
    tolerance=0.05
)

print(f"Symmetry violations: {verification.violations}")
print(f"Overall symmetry score: {verification.symmetry_score:.3f}")

if verification.violations:
    print("⚠️ High error detected - consider repeating experiment")
else:
    print("✅ Symmetries preserved - good measurement quality")
```

#### `parity_symmetry_check(result, parity_groups)`

Check parity conservation for specific qubit groups.

**Parameters:**
- `result` (SimulationResult): Measurement results
- `parity_groups` (List[List[int]]): Groups of qubits with conserved parity

**Returns:**
- `ParityCheckResult`: Parity conservation analysis

**Example:**
```python
from quantrs2.mitigation import parity_symmetry_check

# For GHZ state, all qubits should have same parity
ghz_circuit = quantrs2.Circuit(3)
ghz_circuit.h(0)
ghz_circuit.cx(0, 1)
ghz_circuit.cx(1, 2)
ghz_circuit.measure_all()
result = ghz_circuit.run()

# Check parity conservation
parity_check = parity_symmetry_check(
    result,
    parity_groups=[[0, 1, 2]]  # All qubits should have same parity
)

print(f"Parity violation probability: {parity_check.violation_probability:.3f}")
```

## Virtual Distillation

### Virtual Distillation Method

#### `virtual_distillation(circuits, num_copies=2, measurement_strategy='bell')`

Improve circuit fidelity by running multiple copies and post-selecting.

**Parameters:**
- `circuits` (List[Circuit] or Circuit): Circuit(s) to virtualize
- `num_copies` (int): Number of virtual copies
- `measurement_strategy` (str): Strategy for copy measurements ('bell', 'swap', 'parity')

**Returns:**
- `VirtualDistillationResult`: Distilled measurement results

**Example:**
```python
from quantrs2.mitigation import virtual_distillation

# Create algorithm circuit
algorithm_circuit = quantrs2.Circuit(2)
algorithm_circuit.h(0)
algorithm_circuit.ry(1, np.pi/3)
algorithm_circuit.cx(0, 1)
algorithm_circuit.measure_all()

# Apply virtual distillation
distilled_result = virtual_distillation(
    algorithm_circuit,
    num_copies=3,
    measurement_strategy='bell'
)

print(f"Original fidelity estimate: {algorithm_circuit.run().fidelity:.3f}")
print(f"Distilled fidelity: {distilled_result.effective_fidelity:.3f}")
print(f"Success probability: {distilled_result.success_probability:.3f}")
```

#### `probabilistic_error_cancellation(circuit, error_maps, sampling_overhead=1.0)`

Cancel errors using probabilistic error maps.

**Parameters:**
- `circuit` (Circuit): Target circuit
- `error_maps` (Dict): Mapping from ideal to noisy operations
- `sampling_overhead` (float): Factor for additional sampling

**Example:**
```python
from quantrs2.mitigation import probabilistic_error_cancellation

# Define error model
error_maps = {
    'cx': {'depolarizing_rate': 0.01, 'coherent_error': 0.005},
    'single_qubit': {'depolarizing_rate': 0.001}
}

# Apply probabilistic error cancellation
pec_result = probabilistic_error_cancellation(
    circuit,
    error_maps,
    sampling_overhead=2.0  # Run 2x more shots for better statistics
)
```

## Dynamical Decoupling

### Dynamical Decoupling Sequences

#### `apply_dynamical_decoupling(circuit, idle_qubits, sequence_type='XY4')`

Apply dynamical decoupling during idle periods to suppress decoherence.

**Parameters:**
- `circuit` (Circuit): Circuit to modify
- `idle_qubits` (List[int]): Qubits idle during operations
- `sequence_type` (str): DD sequence ('XY4', 'CPMG', 'UDD', 'Knill')

**Returns:**
- `Circuit`: Modified circuit with DD sequences

**Example:**
```python
from quantrs2.mitigation import apply_dynamical_decoupling

# Original circuit with idle periods
circuit = quantrs2.Circuit(4)
circuit.h(0)
circuit.cx(0, 1)
# Qubits 2 and 3 are idle during this time
circuit.ry(0, np.pi/4)
circuit.cx(1, 2)

# Add dynamical decoupling to idle qubits
dd_circuit = apply_dynamical_decoupling(
    circuit,
    idle_qubits=[2, 3],
    sequence_type='XY4'
)

print(f"Original gates: {circuit.gate_count}")
print(f"With DD gates: {dd_circuit.gate_count}")
```

#### `xy4_sequence(circuit, qubit, duration_gates=4)`

Apply XY4 dynamical decoupling sequence.

**Parameters:**
- `circuit` (Circuit): Circuit to modify
- `qubit` (int): Target qubit for DD
- `duration_gates` (int): Number of gate periods to protect

**Example:**
```python
from quantrs2.mitigation import xy4_sequence

# Apply XY4 to protect qubit during idle time
circuit = quantrs2.Circuit(3)
circuit.h(0)
# Qubit 1 idle for 8 gate periods
xy4_sequence(circuit, qubit=1, duration_gates=8)
circuit.cx(0, 2)
```

#### `cpmg_sequence(circuit, qubit, num_pulses=4)`

Apply Carr-Purcell-Meiboom-Gill (CPMG) sequence.

**Parameters:**
- `circuit` (Circuit): Circuit to modify
- `qubit` (int): Target qubit
- `num_pulses` (int): Number of π pulses in sequence

## Circuit-Level Mitigation

### Error Transparent Gates

#### `compile_error_transparent(circuit, error_model)`

Compile circuit to be transparent to specific error types.

**Parameters:**
- `circuit` (Circuit): Original circuit
- `error_model` (ErrorModel): Target error model

**Returns:**
- `Circuit`: Error-transparent compiled circuit

**Example:**
```python
from quantrs2.mitigation import compile_error_transparent
from quantrs2.noise import DepolarizingErrorModel

# Define error model
error_model = DepolarizingErrorModel(
    single_qubit_rate=0.001,
    two_qubit_rate=0.01
)

# Compile for error transparency
transparent_circuit = compile_error_transparent(circuit, error_model)
```

### Pauli Twirling

#### `apply_pauli_twirling(circuit, gate_types=['cx'], twirl_probability=1.0)`

Apply Pauli twirling to convert coherent errors to incoherent.

**Parameters:**
- `circuit` (Circuit): Circuit to modify
- `gate_types` (List[str]): Gate types to twirl
- `twirl_probability` (float): Probability of applying twirl

**Returns:**
- `Circuit`: Twirled circuit

**Example:**
```python
from quantrs2.mitigation import apply_pauli_twirling

# Twirl CNOT gates to reduce coherent errors
twirled_circuit = apply_pauli_twirling(
    circuit,
    gate_types=['cx', 'ccx'],
    twirl_probability=0.75
)
```

## Composite Mitigation Strategies

### Multi-Level Error Mitigation

#### `comprehensive_error_mitigation(circuit, mitigation_config)`

Apply multiple error mitigation techniques in sequence.

**Parameters:**
- `circuit` (Circuit): Target circuit
- `mitigation_config` (MitigationConfig): Configuration for mitigation strategies

**Returns:**
- `ComprehensiveMitigationResult`: Results from all mitigation techniques

**Example:**
```python
from quantrs2.mitigation import (
    comprehensive_error_mitigation,
    MitigationConfig
)

# Configure multi-level mitigation
config = MitigationConfig(
    zero_noise_extrapolation={
        'noise_levels': [1.0, 1.3, 1.6, 2.0],
        'extrapolation_method': 'exponential'
    },
    readout_correction={
        'method': 'least_squares',
        'regularization': 1e-3
    },
    symmetry_verification={
        'symmetries': ['Z0*Z1', 'X0*X1'],
        'tolerance': 0.05
    },
    dynamical_decoupling={
        'sequence_type': 'XY4',
        'apply_to_idle': True
    }
)

# Apply comprehensive mitigation
mitigated = comprehensive_error_mitigation(circuit, config)

print(f"Original fidelity: {mitigated.original_fidelity:.3f}")
print(f"Final fidelity: {mitigated.final_fidelity:.3f}")
print(f"Improvement factor: {mitigated.improvement_factor:.2f}x")
```

### Adaptive Error Mitigation

#### `adaptive_error_mitigation(circuit, error_threshold=0.1, max_iterations=5)`

Automatically adapt mitigation strategy based on error analysis.

**Parameters:**
- `circuit` (Circuit): Target circuit
- `error_threshold` (float): Acceptable error level
- `max_iterations` (int): Maximum mitigation iterations

**Returns:**
- `AdaptiveMitigationResult`: Optimally mitigated results

**Example:**
```python
from quantrs2.mitigation import adaptive_error_mitigation

# Automatically optimize mitigation strategy
adaptive_result = adaptive_error_mitigation(
    circuit,
    error_threshold=0.05,  # 5% error tolerance
    max_iterations=3
)

print(f"Mitigation strategy used: {adaptive_result.strategy}")
print(f"Final error estimate: {adaptive_result.error_estimate:.4f}")
print(f"Computational overhead: {adaptive_result.overhead:.1f}x")
```

## Result Classes

### MitigatedResult Class

```python
class MitigatedResult:
    """Results from error mitigation techniques."""
    
    @property
    def expectation_values(self) -> Dict[str, float]:
        """Mitigated expectation values."""
        pass
    
    @property
    def error_bars(self) -> Dict[str, float]:
        """Error estimates for mitigated results."""
        pass
    
    @property
    def mitigation_overhead(self) -> float:
        """Computational overhead factor."""
        pass
    
    @property
    def fit_quality(self) -> float:
        """Quality of extrapolation fit (R²)."""
        pass
    
    def plot_extrapolation(self, observable: str = None):
        """Plot extrapolation curve."""
        pass
```

### SymmetryVerificationResult Class

```python
class SymmetryVerificationResult:
    """Results from symmetry verification."""
    
    @property
    def violations(self) -> List[str]:
        """List of violated symmetries."""
        pass
    
    @property
    def symmetry_score(self) -> float:
        """Overall symmetry preservation score."""
        pass
    
    @property
    def violation_probabilities(self) -> Dict[str, float]:
        """Violation probability for each symmetry."""
        pass
    
    def is_reliable(self, threshold: float = 0.1) -> bool:
        """Check if results are reliable given violations."""
        pass
```

## Error Models

### Built-in Error Models

#### Depolarizing Noise

```python
from quantrs2.mitigation import DepolarizingErrorModel

error_model = DepolarizingErrorModel(
    single_qubit_rate=0.001,    # 0.1% per single-qubit gate
    two_qubit_rate=0.01,        # 1% per two-qubit gate
    readout_error=0.02          # 2% readout error
)
```

#### Amplitude Damping

```python
from quantrs2.mitigation import AmplitudeDampingModel

error_model = AmplitudeDampingModel(
    t1_time=100e-6,      # 100 μs T1 time
    gate_time=50e-9      # 50 ns gate time
)
```

#### Correlated Noise

```python
from quantrs2.mitigation import CorrelatedNoiseModel

error_model = CorrelatedNoiseModel(
    crosstalk_matrix=crosstalk_rates,
    correlation_length=2  # Errors correlated over 2 qubits
)
```

## Benchmarking and Validation

### Mitigation Benchmarks

#### `benchmark_mitigation_methods(circuits, methods, noise_model)`

Compare different mitigation methods on benchmark circuits.

**Parameters:**
- `circuits` (List[Circuit]): Test circuits
- `methods` (List[str]): Mitigation methods to compare
- `noise_model` (ErrorModel): Noise model for testing

**Returns:**
- `BenchmarkResults`: Comparative performance analysis

**Example:**
```python
from quantrs2.mitigation import benchmark_mitigation_methods

# Define test circuits
test_circuits = [
    create_bell_state_circuit(),
    create_ghz_state_circuit(3),
    create_qft_circuit(3)
]

# Compare mitigation methods
benchmark = benchmark_mitigation_methods(
    circuits=test_circuits,
    methods=['zne', 'readout_correction', 'virtual_distillation'],
    noise_model=error_model
)

print(f"Best method overall: {benchmark.best_method}")
print(f"Average fidelity improvement: {benchmark.avg_improvement:.2f}x")
```

### Validation Tools

#### `validate_mitigation_result(original_result, mitigated_result, known_ideal=None)`

Validate mitigation results for consistency and improvement.

**Parameters:**
- `original_result` (SimulationResult): Original noisy results
- `mitigated_result` (MitigatedResult): Mitigated results
- `known_ideal` (Optional[SimulationResult]): Known ideal result for comparison

**Returns:**
- `ValidationReport`: Validation analysis

## Best Practices

### Choosing Mitigation Methods

```python
def recommend_mitigation_strategy(circuit, noise_model, resources):
    """Recommend optimal mitigation strategy."""
    
    recommendations = []
    
    # For shallow circuits with readout errors
    if circuit.depth < 10:
        recommendations.append('readout_error_mitigation')
    
    # For circuits with known symmetries
    if has_symmetries(circuit):
        recommendations.append('symmetry_verification')
    
    # For high-fidelity requirements
    if resources.computational_budget > 5:
        recommendations.append('zero_noise_extrapolation')
    
    # For circuits with idle periods
    if has_idle_periods(circuit):
        recommendations.append('dynamical_decoupling')
    
    return recommendations
```

### Error Analysis

```python
def analyze_mitigation_effectiveness(original, mitigated):
    """Analyze mitigation effectiveness."""
    
    # Calculate improvement metrics
    fidelity_improvement = mitigated.fidelity / original.fidelity
    error_reduction = (original.error_rate - mitigated.error_rate) / original.error_rate
    
    # Check for over-mitigation
    if mitigated.error_bars > 0.1:
        print("⚠️ Warning: Large error bars suggest over-mitigation")
    
    # Validate physical constraints
    if any(prob < 0 for prob in mitigated.probabilities.values()):
        print("⚠️ Warning: Negative probabilities indicate fitting issues")
    
    return {
        'fidelity_improvement': fidelity_improvement,
        'error_reduction': error_reduction,
        'overhead': mitigated.computational_overhead
    }
```

## Performance Considerations

### Computational Overhead

Different mitigation methods have varying computational costs:

| Method | Overhead Factor | Best For |
|--------|----------------|----------|
| Readout correction | 1.1x | All circuits |
| Dynamical decoupling | 1.5x | Idle-heavy circuits |
| Zero-noise extrapolation | 3-5x | High-accuracy needs |
| Virtual distillation | 4-8x | Small, high-value circuits |

### Memory Requirements

```python
# Estimate memory requirements for mitigation
def estimate_mitigation_memory(circuit, method):
    base_memory = 2 ** circuit.num_qubits * 16  # bytes for state vector
    
    overhead_factors = {
        'readout_correction': 1.2,
        'zero_noise_extrapolation': 3.0,
        'virtual_distillation': 2.0
    }
    
    return base_memory * overhead_factors.get(method, 1.0)
```

## Related Documentation

- [Hardware Optimization](../tutorials/beginner/04-hardware-optimization.md) - Circuit optimization for NISQ devices
- [Noise Models](noise.md) - Quantum noise simulation
- [Core API](core.md) - Basic circuit operations
- [Performance Guide](../user-guide/performance.md) - Optimization strategies

## Examples

### Complete Mitigation Pipeline

```python
import quantrs2
from quantrs2.mitigation import *

def complete_mitigation_example():
    """Complete error mitigation pipeline example."""
    
    # 1. Create algorithm circuit
    circuit = quantrs2.Circuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()
    
    # 2. Characterize readout errors
    calib_matrix = create_readout_calibration_matrix(3)
    
    # 3. Run with zero-noise extrapolation
    noise_levels = [1.0, 1.3, 1.6, 2.0]
    zne_result = zero_noise_extrapolation(circuit, noise_levels)
    
    # 4. Apply readout correction
    corrected_probs = apply_readout_correction(
        zne_result.state_probabilities(),
        calib_matrix
    )
    
    # 5. Verify symmetries
    symmetries = ['Z0*Z1*Z2']  # GHZ state parity
    verification = verify_measurement_symmetries(
        zne_result, 
        symmetries
    )
    
    # 6. Report results
    print(f"Original result: {circuit.run().state_probabilities()}")
    print(f"Mitigated result: {corrected_probs}")
    print(f"Symmetry violations: {verification.violations}")
    print(f"Computational overhead: {zne_result.mitigation_overhead:.1f}x")
    
    return corrected_probs

# Run complete pipeline
final_result = complete_mitigation_example()
```

---

*Error mitigation is essential for near-term quantum computing.* Start with [readout error correction](../examples/mitigation/readout_correction.py) and [zero-noise extrapolation](../examples/mitigation/zero_noise_extrapolation.py) examples!