# Advanced Tutorial: Quantum Error Correction with QuantRS2

## Overview

Quantum error correction (QEC) is essential for building fault-tolerant quantum computers. This tutorial demonstrates how to implement and analyze quantum error correction codes using QuantRS2-Py.

## Prerequisites

- Understanding of quantum gates and circuits
- Familiarity with quantum noise models
- Knowledge of linear algebra
- Completion of intermediate tutorials

## Topics Covered

1. Bit-flip and phase-flip codes
2. Shor's 9-qubit code
3. Steane code (7-qubit)
4. Surface codes
5. Error syndrome detection
6. Logical qubit operations

## 1. Three-Qubit Bit-Flip Code

The simplest quantum error correction code protects against single bit-flip errors.

```python
import quantrs2 as qr
import numpy as np

def create_bit_flip_encoder():
    """Create a circuit that encodes one logical qubit into three physical qubits."""
    circuit = qr.PyCircuit(3)

    # Encoding circuit
    circuit.cnot(0, 1)
    circuit.cnot(0, 2)

    return circuit

def create_bit_flip_syndrome_measurement():
    """Measure error syndromes for bit-flip code."""
    circuit = qr.PyCircuit(5)  # 3 data + 2 ancilla qubits

    # Syndrome measurement
    circuit.cnot(0, 3)
    circuit.cnot(1, 3)
    circuit.cnot(1, 4)
    circuit.cnot(2, 4)

    # Measure ancilla qubits
    circuit.measure(3)
    circuit.measure(4)

    return circuit

def apply_bit_flip_correction(syndrome):
    """Determine which qubit to correct based on syndrome."""
    syndrome_map = {
        '00': None,  # No error
        '01': 2,     # Error on qubit 2
        '10': 0,     # Error on qubit 0
        '11': 1      # Error on qubit 1
    }
    return syndrome_map.get(syndrome)

# Example usage
encoder = create_bit_flip_encoder()
print("Bit-flip encoder created")

# Prepare a state to encode
state_prep = qr.PyCircuit(1)
state_prep.ry(0, np.pi/4)  # Arbitrary state
```

## 2. Phase-Flip Code

Similar to the bit-flip code, but protects against phase errors.

```python
def create_phase_flip_encoder():
    """Create a circuit that encodes against phase-flip errors."""
    circuit = qr.PyCircuit(3)

    # Apply Hadamards to convert to X-basis
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)

    # Encoding in X-basis
    circuit.cnot(0, 1)
    circuit.cnot(0, 2)

    # Convert back to Z-basis
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)

    return circuit
```

## 3. Shor's 9-Qubit Code

Corrects both bit-flip and phase-flip errors simultaneously.

```python
def create_shor_encoder():
    """Implement Shor's 9-qubit error correction code."""
    circuit = qr.PyCircuit(9)

    # First level: protect against phase flips
    circuit.cnot(0, 3)
    circuit.cnot(0, 6)

    # Apply Hadamards
    for i in range(3):
        circuit.h(i * 3)

    # Second level: protect against bit flips (for each of 3 blocks)
    for i in range(3):
        base = i * 3
        circuit.cnot(base, base + 1)
        circuit.cnot(base, base + 2)

    return circuit

def create_shor_syndrome_circuit():
    """Create syndrome measurement circuit for Shor code."""
    circuit = qr.PyCircuit(17)  # 9 data + 8 ancilla qubits

    # Measure bit-flip syndromes (6 measurements)
    for i in range(3):
        base = i * 3
        ancilla_base = 9 + i * 2

        circuit.cnot(base, ancilla_base)
        circuit.cnot(base + 1, ancilla_base)
        circuit.cnot(base + 1, ancilla_base + 1)
        circuit.cnot(base + 2, ancilla_base + 1)

    # Measure phase-flip syndromes (2 measurements)
    phase_ancilla = 15
    for i in range(3):
        circuit.h(i * 3)

    circuit.cnot(0, phase_ancilla)
    circuit.cnot(3, phase_ancilla)
    circuit.cnot(3, phase_ancilla + 1)
    circuit.cnot(6, phase_ancilla + 1)

    for i in range(3):
        circuit.h(i * 3)

    # Measure all ancilla qubits
    for i in range(9, 17):
        circuit.measure(i)

    return circuit
```

## 4. Steane Code (7-Qubit)

More efficient than Shor's code, protecting against arbitrary single-qubit errors.

```python
def create_steane_encoder():
    """Implement Steane's 7-qubit error correction code."""
    circuit = qr.PyCircuit(7)

    # Steane code generator matrix encoding
    # |0⟩_L = |0000000⟩ + |1010101⟩ + |0110011⟩ + |1100110⟩
    #        + |0001111⟩ + |1011010⟩ + |0111100⟩ + |1101001⟩ (divided by 2√2)

    # Encoding circuit implementation
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(0, 3)
    circuit.cnot(0, 5)
    circuit.cnot(0, 6)

    circuit.h(2)
    circuit.cnot(2, 1)
    circuit.cnot(2, 3)
    circuit.cnot(2, 4)
    circuit.cnot(2, 6)

    circuit.h(4)
    circuit.cnot(4, 1)
    circuit.cnot(4, 5)
    circuit.cnot(4, 6)

    return circuit

def steane_syndrome_extraction():
    """Extract error syndromes for Steane code."""
    circuit = qr.PyCircuit(13)  # 7 data + 6 ancilla qubits

    # X-stabilizer measurements
    for i in range(3):
        ancilla = 7 + i
        circuit.h(ancilla)
        # Apply CNOTs according to parity check matrix
        for j in range(7):
            if parity_check_x[i][j]:
                circuit.cnot(ancilla, j)
        circuit.h(ancilla)
        circuit.measure(ancilla)

    # Z-stabilizer measurements
    for i in range(3):
        ancilla = 10 + i
        # Apply CNOTs according to parity check matrix
        for j in range(7):
            if parity_check_z[i][j]:
                circuit.cnot(j, ancilla)
        circuit.measure(ancilla)

    return circuit

# Parity check matrices for Steane code
parity_check_x = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
]

parity_check_z = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
]
```

## 5. Surface Codes (Introduction)

Surface codes are topological codes suitable for 2D qubit layouts.

```python
def create_surface_code_layout(distance):
    """
    Create a surface code with given code distance.

    Args:
        distance: Code distance (determines error correction capability)

    Returns:
        Circuit with surface code layout
    """
    # Number of data qubits
    n_data = distance * distance
    # Number of syndrome qubits
    n_syndrome = (distance - 1) * (distance - 1) + (distance - 1) * distance

    total_qubits = n_data + n_syndrome
    circuit = qr.PyCircuit(total_qubits)

    # Initialize the lattice structure
    # X-stabilizers and Z-stabilizers are arranged in checkerboard pattern

    return circuit

def surface_code_syndrome_round(circuit, distance):
    """Perform one round of syndrome measurement for surface code."""
    # Apply measurements in proper order to maintain commutation
    # X-stabilizers first, then Z-stabilizers
    pass
```

## 6. Practical Error Correction Pipeline

Combining encoding, error injection, syndrome measurement, and correction:

```python
import quantrs2 as qr
from quantrs2 import mitigation

class ErrorCorrectionPipeline:
    """Complete pipeline for quantum error correction."""

    def __init__(self, code_type='steane'):
        self.code_type = code_type
        self.encoder = self._create_encoder()
        self.syndrome_circuit = self._create_syndrome_circuit()

    def _create_encoder(self):
        if self.code_type == 'bitflip':
            return create_bit_flip_encoder()
        elif self.code_type == 'steane':
            return create_steane_encoder()
        elif self.code_type == 'shor':
            return create_shor_encoder()
        else:
            raise ValueError(f"Unknown code type: {self.code_type}")

    def _create_syndrome_circuit(self):
        if self.code_type == 'bitflip':
            return create_bit_flip_syndrome_measurement()
        elif self.code_type == 'steane':
            return steane_syndrome_extraction()
        elif self.code_type == 'shor':
            return create_shor_syndrome_circuit()
        else:
            raise ValueError(f"Unknown code type: {self.code_type}")

    def encode(self, logical_state):
        """Encode a logical qubit into physical qubits."""
        # Apply encoding circuit to the logical state
        pass

    def inject_errors(self, circuit, error_rate=0.01):
        """Inject random errors for testing."""
        from quantrs2 import utils
        # Inject random bit-flip and phase-flip errors
        pass

    def measure_syndrome(self, state):
        """Measure error syndromes."""
        result = self.syndrome_circuit.run()
        return result.measurements()

    def decode_syndrome(self, syndrome):
        """Determine correction operations from syndrome."""
        # Implement syndrome decoding logic
        pass

    def apply_correction(self, state, corrections):
        """Apply correction operations."""
        # Apply the necessary Pauli operators
        pass

# Example usage
pipeline = ErrorCorrectionPipeline(code_type='steane')

# Prepare logical state
logical_circuit = qr.PyCircuit(1)
logical_circuit.ry(0, np.pi/4)
logical_state = logical_circuit.run()

# Encode
encoded_state = pipeline.encode(logical_state)

# Inject errors
noisy_state = pipeline.inject_errors(encoded_state, error_rate=0.01)

# Measure syndrome
syndrome = pipeline.measure_syndrome(noisy_state)

# Decode and correct
corrections = pipeline.decode_syndrome(syndrome)
corrected_state = pipeline.apply_correction(noisy_state, corrections)

print(f"Original fidelity: {calculate_fidelity(encoded_state, noisy_state):.4f}")
print(f"Corrected fidelity: {calculate_fidelity(encoded_state, corrected_state):.4f}")
```

## 7. Benchmarking Error Correction

```python
def benchmark_error_correction(code_type, error_rates, num_trials=100):
    """
    Benchmark error correction performance.

    Args:
        code_type: Type of error correction code
        error_rates: List of error rates to test
        num_trials: Number of trials per error rate

    Returns:
        Dictionary with benchmark results
    """
    results = {
        'error_rates': error_rates,
        'logical_error_rates': [],
        'fidelities': []
    }

    pipeline = ErrorCorrectionPipeline(code_type=code_type)

    for error_rate in error_rates:
        logical_errors = 0
        fidelities = []

        for _ in range(num_trials):
            # Prepare random logical state
            logical_circuit = qr.PyCircuit(1)
            logical_circuit.ry(0, np.random.uniform(0, 2*np.pi))
            logical_circuit.rz(0, np.random.uniform(0, 2*np.pi))
            logical_state = logical_circuit.run()

            # Encode and inject errors
            encoded_state = pipeline.encode(logical_state)
            noisy_state = pipeline.inject_errors(encoded_state, error_rate)

            # Correct errors
            syndrome = pipeline.measure_syndrome(noisy_state)
            corrections = pipeline.decode_syndrome(syndrome)
            corrected_state = pipeline.apply_correction(noisy_state, corrections)

            # Calculate fidelity
            fidelity = calculate_fidelity(encoded_state, corrected_state)
            fidelities.append(fidelity)

            # Check for logical errors
            if fidelity < 0.9:  # Threshold for logical error
                logical_errors += 1

        results['logical_error_rates'].append(logical_errors / num_trials)
        results['fidelities'].append(np.mean(fidelities))

    return results

# Run benchmark
error_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
results = benchmark_error_correction('steane', error_rates, num_trials=100)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results['error_rates'], results['logical_error_rates'], 'o-')
plt.xlabel('Physical Error Rate')
plt.ylabel('Logical Error Rate')
plt.title('Error Correction Performance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(results['error_rates'], results['fidelities'], 'o-')
plt.xlabel('Physical Error Rate')
plt.ylabel('Average Fidelity')
plt.title('State Fidelity vs Error Rate')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 8. Integration with Quantum Algorithms

```python
def protected_vqe_example():
    """Run VQE with error correction."""
    from quantrs2 import ml, advanced_algorithms

    # Define Hamiltonian (e.g., H2 molecule)
    hamiltonian = advanced_algorithms.create_h2_hamiltonian()

    # Create error-corrected VQE ansatz
    def error_corrected_ansatz(params):
        # Logical circuit
        logical_circuit = qr.PyCircuit(2)
        logical_circuit.ry(0, params[0])
        logical_circuit.ry(1, params[1])
        logical_circuit.cnot(0, 1)

        # Encode with Steane code
        encoder = create_steane_encoder()
        # ... encoding logic

        return encoded_circuit

    # Run VQE with error correction
    vqe = ml.VQE(hamiltonian, error_corrected_ansatz)
    result = vqe.optimize()

    print(f"Ground state energy: {result['energy']:.6f}")
    print(f"Optimization iterations: {result['iterations']}")

# Run protected VQE
protected_vqe_example()
```

## Exercises

1. Implement a complete bit-flip code with error injection and correction
2. Compare the performance of different error correction codes
3. Design a custom error correction code for a specific noise model
4. Implement fault-tolerant gates for the Steane code
5. Create a visualization of syndrome measurement patterns

## Further Reading

- Nielsen & Chuang, "Quantum Computation and Quantum Information", Chapter 10
- Preskill, "Quantum Computation" lecture notes
- Surface Codes: https://arxiv.org/abs/1208.0928
- Steane Code: https://arxiv.org/abs/quant-ph/9605021

## Next Steps

- Advanced Tutorial: Fault-Tolerant Quantum Computing
- Advanced Tutorial: Topological Quantum Computing
- Research: Surface Code Implementations on Real Hardware
