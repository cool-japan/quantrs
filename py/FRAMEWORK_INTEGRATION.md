# Framework Integration Guide

**QuantRS2-Py v0.1.0-beta.3**

This guide covers seamless integration between QuantRS2 and other quantum computing frameworks.

## Supported Frameworks

QuantRS2 provides bidirectional conversion with:

- **Qiskit** (IBM Quantum)
- **Cirq** (Google Quantum AI)
- **PennyLane** (Xanadu) - via enhanced plugin

## Installation

```bash
# Install QuantRS2 with framework support
pip install quantrs2

# Install optional framework dependencies
pip install qiskit         # For Qiskit support
pip install cirq           # For Cirq support
pip install pennylane      # For PennyLane support
```

## Quick Start

### Qiskit → QuantRS2

```python
from qiskit import QuantumCircuit
from quantrs2 import convert_from_qiskit

# Create Qiskit circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Convert to QuantRS2
quantrs_circuit = convert_from_qiskit(qc)

# Run on QuantRS2
result = quantrs_circuit.run()
print(result.probabilities())
```

### Cirq → QuantRS2

```python
import cirq
from quantrs2 import convert_from_cirq

# Create Cirq circuit
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit()
circuit.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])

# Convert to QuantRS2
quantrs_circuit = convert_from_cirq(circuit)

# Run on QuantRS2
result = quantrs_circuit.run()
print(result.probabilities())
```

## Advanced Usage

### Qiskit Converter

#### Full Conversion Control

```python
from quantrs2.qiskit_converter import QiskitConverter

# Create converter with strict mode
converter = QiskitConverter(strict_mode=False)

# Convert with optimization
quantrs_circuit, stats = converter.from_qiskit(
    qiskit_circuit,
    optimize=True
)

# Check conversion statistics
print(f"Original gates: {stats.original_gates}")
print(f"Converted gates: {stats.converted_gates}")
print(f"Decomposed gates: {stats.decomposed_gates}")
print(f"Success: {stats.success}")

if stats.warnings:
    for warning in stats.warnings:
        print(f"Warning: {warning}")
```

#### QASM Support

```python
# Import from QASM string
qasm_str = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""

quantrs_circuit, stats = converter.from_qasm(qasm_str)

# Export to QASM
qasm_output = converter.to_qasm(quantrs_circuit)
print(qasm_output)
```

#### Gate Mapping

Supported Qiskit gates:

| Qiskit Gate | QuantRS2 Gate | Notes |
|-------------|---------------|-------|
| `h` | `h` | Hadamard |
| `x` | `x` | Pauli-X |
| `y` | `y` | Pauli-Y |
| `z` | `z` | Pauli-Z |
| `s` | `s` | S gate |
| `sdg` | `sdg` | S-dagger |
| `t` | `t` | T gate |
| `tdg` | `tdg` | T-dagger |
| `rx` | `rx` | X-rotation |
| `ry` | `ry` | Y-rotation |
| `rz` | `rz` | Z-rotation |
| `cx`, `cnot` | `cnot` | CNOT |
| `cy` | `cy` | Controlled-Y |
| `cz` | `cz` | Controlled-Z |
| `swap` | `swap` | SWAP |
| `ccx`, `toffoli` | `toffoli` | Toffoli |
| `cswap`, `fredkin` | `cswap` | Fredkin |
| `u1` | → `rz` | Decomposed |
| `u2` | → `rz`, `ry`, `rz` | Decomposed |
| `u3` | → `rz`, `ry`, `rz` | Decomposed |

### Cirq Converter

#### Full Conversion Control

```python
from quantrs2.cirq_converter import CirqConverter

# Create converter
converter = CirqConverter(strict_mode=False)

# Convert circuit
quantrs_circuit, stats = converter.from_cirq(cirq_circuit)

# Check statistics
print(f"Original operations: {stats.original_operations}")
print(f"Converted operations: {stats.converted_operations}")
print(f"Number of moments: {stats.num_moments}")
```

#### Powered Gate Handling

```python
import cirq

qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit()

# Cirq supports powered gates
circuit.append([
    cirq.X(qubits[0]) ** 0.5,  # √X gate
    cirq.Z(qubits[1]) ** 0.5,  # S gate
    cirq.Z(qubits[1]) ** 0.25, # T gate
])

# Convert (powered gates are handled automatically)
quantrs_circuit, stats = converter.from_cirq(circuit)
```

#### Supported Cirq Operations

| Cirq Operation | QuantRS2 Gate | Notes |
|----------------|---------------|-------|
| `HPowGate` | `h` | Hadamard (exp=1.0) |
| `XPowGate` | `x`, `sx` | X or √X based on exp |
| `YPowGate` | `y` | Pauli-Y (exp=1.0) |
| `ZPowGate` | `z`, `s`, `t` | Based on exponent |
| `rx`, `Rx` | `rx` | X-rotation |
| `ry`, `Ry` | `ry` | Y-rotation |
| `rz`, `Rz` | `rz` | Z-rotation |
| `CNotPowGate` | `cnot` | CNOT |
| `CZPowGate` | `cz` | Controlled-Z |
| `SwapPowGate` | `swap` | SWAP |
| `CCXPowGate` | `toffoli` | Toffoli |
| `CSwapGate` | `cswap` | Fredkin |

## Circuit Equivalence Verification

### Qiskit Circuits

```python
from quantrs2.qiskit_converter import QiskitConverter

converter = QiskitConverter()

# Verify two Qiskit circuits are equivalent
equivalent, fidelity = converter.verify_equivalence(
    circuit1,
    circuit2,
    tolerance=1e-6
)

print(f"Equivalent: {equivalent}")
print(f"Fidelity: {fidelity:.10f}")
```

### Cirq Circuits

```python
from quantrs2.cirq_converter import CirqConverter

converter = CirqConverter()

# Verify two Cirq circuits are equivalent
equivalent, fidelity = converter.verify_equivalence(
    circuit1,
    circuit2,
    tolerance=1e-6
)

print(f"Equivalent: {equivalent}")
print(f"Fidelity: {fidelity:.10f}")
```

## Performance Comparison

Use the benchmarking suite to compare frameworks:

```python
from quantrs2.benchmarking import PerformanceBenchmark, BenchmarkType

# Initialize benchmark
benchmark = PerformanceBenchmark()

# Run comparison across frameworks
results = benchmark.run_benchmark(
    BenchmarkType.GHZ_STATE,
    n_qubits=10,
    num_runs=5
)

# Print summary
benchmark.print_summary()

# Export results
benchmark.export_results("./benchmark_results/")
```

## Migration Strategies

### From Qiskit

```python
# 1. Convert existing Qiskit code
from qiskit import QuantumCircuit
from quantrs2 import convert_from_qiskit

# Your existing Qiskit code
qc = QuantumCircuit(5)
# ... build circuit ...

# Simply convert and run on QuantRS2
quantrs_circuit = convert_from_qiskit(qc, optimize=True)
result = quantrs_circuit.run(use_gpu=True)  # GPU acceleration!
```

### From Cirq

```python
# 1. Convert existing Cirq code
import cirq
from quantrs2 import convert_from_cirq

# Your existing Cirq code
qubits = cirq.LineQubit.range(5)
circuit = cirq.Circuit()
# ... build circuit ...

# Convert and run on QuantRS2
quantrs_circuit = convert_from_cirq(circuit)
result = quantrs_circuit.run(use_gpu=True)
```

### Hybrid Approach

```python
# Use best of both worlds
from qiskit import QuantumCircuit
from quantrs2 import convert_from_qiskit

# Build circuit in Qiskit (familiar API)
qc = QuantumCircuit(20)
# ... complex circuit construction ...

# Run on QuantRS2 (better performance)
quantrs_circuit = convert_from_qiskit(qc, optimize=True)
result = quantrs_circuit.run(use_gpu=True)
```

## Best Practices

### 1. Check Conversion Statistics

```python
quantrs_circuit, stats = converter.from_qiskit(qc)

if not stats.success:
    print(f"Warning: {len(stats.unsupported_gates)} unsupported gates")
    print(stats.unsupported_gates)

if stats.decomposed_gates > 0:
    print(f"Note: {stats.decomposed_gates} gates were decomposed")
```

### 2. Use Optimization

```python
# Enable optimization during conversion
quantrs_circuit, stats = converter.from_qiskit(
    qc,
    optimize=True  # Simplify circuit structure
)
```

### 3. Handle Errors Gracefully

```python
try:
    quantrs_circuit = convert_from_qiskit(qc, strict=False)
    result = quantrs_circuit.run()
except Exception as e:
    print(f"Conversion or execution failed: {e}")
    # Fallback to original framework
```

### 4. Verify Critical Circuits

```python
# For production code, verify equivalence
from qiskit import Aer, execute

# Run on both frameworks
qiskit_result = execute(qc, Aer.get_backend('statevector_simulator')).result()
quantrs_result = quantrs_circuit.run()

# Compare state vectors
import numpy as np
qiskit_sv = qiskit_result.get_statevector()
quantrs_probs = quantrs_result.probabilities()

# Verify (approximately) equal
assert np.allclose(np.abs(qiskit_sv)**2, quantrs_probs, atol=1e-6)
```

## Troubleshooting

### Common Issues

**Issue: Unsupported gate warning**
```
Warning: Unsupported gate: custom_gate
```
**Solution**: Check if the gate can be decomposed to basic gates, or implement custom handling.

**Issue: Measurement gates skipped**
```
Warning: Skipping measurement - QuantRS2 measures at simulation time
```
**Solution**: This is expected. QuantRS2 handles measurements differently. Use `result.probabilities()` or `result.state_probabilities()`.

**Issue: Parametric circuits**
```
Error: Parameter must be resolved
```
**Solution**: Bind parameters before conversion:
```python
qc_bound = qc.bind_parameters({theta: 1.5})
quantrs_circuit = convert_from_qiskit(qc_bound)
```

### Debug Mode

```python
# Enable strict mode to catch issues
converter = QiskitConverter(strict_mode=True)

try:
    circuit, stats = converter.from_qiskit(qc)
except ValueError as e:
    print(f"Conversion error: {e}")
    # Handle error
```

## Examples

Complete examples are available in:
- `examples/advanced/framework_interop_demo.py` - Comprehensive demo
- `examples/basic/qiskit_to_quantrs2.py` - Simple Qiskit conversion
- `examples/basic/cirq_to_quantrs2.py` - Simple Cirq conversion

## Performance Tips

1. **Use GPU acceleration**: `circuit.run(use_gpu=True)`
2. **Enable optimization**: `convert_from_qiskit(qc, optimize=True)`
3. **Batch operations**: Convert multiple circuits at once
4. **Profile performance**: Use benchmarking suite to compare

## Contributing

To add support for additional frameworks:

1. Create a converter module: `quantrs2/framework_converter.py`
2. Implement gate mapping
3. Add conversion statistics tracking
4. Write tests and examples
5. Submit a pull request

## Support

- Documentation: See README.md and inline docstrings
- Examples: Check `examples/advanced/` directory
- Issues: Report at GitHub repository
- Benchmarks: Run `python -m quantrs2.benchmarking`

---

**Last Updated**: 2025-11-17
**QuantRS2 Version**: v0.1.0-beta.3
