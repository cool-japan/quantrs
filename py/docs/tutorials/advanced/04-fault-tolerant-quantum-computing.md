# Advanced Tutorial: Fault-Tolerant Quantum Computing

## Overview

This expert-level tutorial covers fault-tolerant quantum computing (FTQC), the theoretical and practical framework for building scalable, error-resilient quantum computers. You'll learn how to implement fault-tolerant gates, magic state distillation, and resource estimation using QuantRS2-Py.

## Prerequisites

- Deep understanding of quantum error correction
- Familiarity with stabilizer formalism
- Knowledge of Clifford gates and non-Clifford gates
- Understanding of the threshold theorem
- Completion of quantum error correction tutorial

## Topics Covered

1. Transversal Gates and Fault Tolerance
2. Magic State Distillation
3. Fault-Tolerant Gate Synthesis
4. The Threshold Theorem
5. Resource Estimation for FTQC
6. Lattice Surgery Operations
7. Fault-Tolerant Quantum Algorithms

## 1. Transversal Gates

Transversal gates apply a gate to each physical qubit independently, preventing error propagation.

### Theory

A gate `U` is transversal for a code if it can be implemented as:
```
U_L = ⊗ᵢ Uᵢ
```
where each `Uᵢ` acts on a single physical qubit.

### Implementation

```python
import quantrs2 as qr
import numpy as np

class TransversalGates:
    """
    Implement transversal gates for fault-tolerant quantum computing.

    Transversal gates prevent error propagation across code blocks.
    """

    def __init__(self, code_type='steane'):
        self.code_type = code_type
        self.n_physical = self._get_physical_qubits()

    def _get_physical_qubits(self):
        """Get number of physical qubits for error correction code."""
        code_sizes = {
            'bitflip': 3,
            'steane': 7,
            'shor': 9,
            'surface': 17  # Distance-3 surface code
        }
        return code_sizes.get(self.code_type, 7)

    def transversal_cnot(self, control_block, target_block):
        """
        Implement transversal CNOT between two encoded blocks.

        For most CSS codes, logical CNOT can be implemented
        transversally by applying physical CNOTs bitwise.

        Args:
            control_block: Circuit with control logical qubit
            target_block: Circuit with target logical qubit

        Returns:
            Combined circuit with transversal CNOT
        """
        total_qubits = control_block.num_qubits() + target_block.num_qubits()
        circuit = qr.PyCircuit(total_qubits)

        # Copy control block operations
        for op in control_block.get_operations():
            circuit.add_operation(op)

        # Copy target block operations
        offset = control_block.num_qubits()
        for op in target_block.get_operations():
            shifted_op = op.shift_qubits(offset)
            circuit.add_operation(shifted_op)

        # Apply transversal CNOT
        for i in range(self.n_physical):
            circuit.cnot(i, offset + i)

        return circuit

    def transversal_hadamard(self, encoded_block):
        """
        Implement transversal Hadamard on encoded qubit.

        For CSS codes like Steane, Hadamard is transversal.
        """
        circuit = encoded_block.copy()

        # Apply Hadamard to each physical qubit
        for i in range(self.n_physical):
            circuit.h(i)

        return circuit

    def transversal_phase(self, encoded_block):
        """
        Implement transversal S (phase) gate.

        For Steane code, S gate is transversal.
        """
        circuit = encoded_block.copy()

        # Apply S gate to each physical qubit
        for i in range(self.n_physical):
            circuit.s(i)

        return circuit

    def transversal_pauli(self, encoded_block, pauli_type, qubit_idx=None):
        """
        Apply transversal Pauli gates.

        Args:
            encoded_block: Encoded qubit circuit
            pauli_type: 'X', 'Y', or 'Z'
            qubit_idx: Specific qubit index (None for all)

        Returns:
            Circuit with transversal Pauli
        """
        circuit = encoded_block.copy()

        qubits = [qubit_idx] if qubit_idx is not None else range(self.n_physical)

        for i in qubits:
            if pauli_type == 'X':
                circuit.x(i)
            elif pauli_type == 'Y':
                circuit.y(i)
            elif pauli_type == 'Z':
                circuit.z(i)

        return circuit

    def check_fault_tolerance(self, gate_circuit):
        """
        Verify fault tolerance property of a gate implementation.

        A gate is fault-tolerant if a single error in the circuit
        causes at most one error in the output code block.

        Returns:
            Boolean indicating fault tolerance
        """
        # Simplified check: verify transversality
        # In practice, need detailed error propagation analysis

        operations = gate_circuit.get_operations()

        # Check if each operation acts on distinct qubits
        qubit_sets = []
        for op in operations:
            qubits = op.get_qubits()
            for qs in qubit_sets:
                if len(set(qubits) & set(qs)) > 0:
                    # Operations share qubits - not transversal
                    return False
            qubit_sets.append(qubits)

        return True

# Example transversal gates
print("Transversal Gates for Fault-Tolerant QC")
print("=" * 50)

# Create encoded qubits (using Steane code)
from quantrs2 import advanced_algorithms

# Prepare logical |0⟩ and |+⟩ states
steane_encoder = advanced_algorithms.create_steane_encoder()
logical_zero = steane_encoder.encode_zero()
logical_plus = steane_encoder.encode_plus()

# Transversal operations
tg = TransversalGates(code_type='steane')

# Transversal Hadamard
print("\n1. Transversal Hadamard")
h_circuit = tg.transversal_hadamard(logical_zero)
print(f"   Applied H to {tg.n_physical} physical qubits")

# Transversal CNOT
print("\n2. Transversal CNOT")
cnot_circuit = tg.transversal_cnot(logical_zero, logical_plus)
print(f"   Applied CNOT between two {tg.n_physical}-qubit blocks")

# Check fault tolerance
is_ft = tg.check_fault_tolerance(h_circuit)
print(f"\n3. Fault Tolerance Check: {is_ft}")
```

## 2. Magic State Distillation

Non-Clifford gates like T require magic states for fault-tolerant implementation.

```python
class MagicStateDistillation:
    """
    Implement magic state distillation for fault-tolerant non-Clifford gates.

    Magic states enable universal quantum computing with Clifford gates
    plus state injection.
    """

    def __init__(self, protocol='15-to-1'):
        self.protocol = protocol
        self.protocols = {
            '15-to-1': self.fifteen_to_one_distillation,
            '5-to-1': self.five_to_one_distillation,
            'bravyi-kitaev': self.bravyi_kitaev_distillation
        }

    def prepare_noisy_magic_state(self, error_rate=0.1):
        """
        Prepare noisy |T⟩ state.

        |T⟩ = (|0⟩ + e^(iπ/4)|1⟩) / √2
        """
        circuit = qr.PyCircuit(1)

        # Prepare |+⟩
        circuit.h(0)

        # Apply T gate
        circuit.t(0)

        # Add noise
        if np.random.random() < error_rate:
            # Apply random Pauli error
            pauli = np.random.choice(['x', 'y', 'z'])
            if pauli == 'x':
                circuit.x(0)
            elif pauli == 'y':
                circuit.y(0)
            elif pauli == 'z':
                circuit.z(0)

        return circuit

    def fifteen_to_one_distillation(self, noisy_states):
        """
        15-to-1 magic state distillation protocol.

        Takes 15 noisy |T⟩ states and produces 1 higher-fidelity |T⟩ state.

        Args:
            noisy_states: List of 15 noisy magic state circuits

        Returns:
            Distilled magic state circuit and success probability
        """
        if len(noisy_states) != 15:
            raise ValueError("15-to-1 protocol requires exactly 15 input states")

        # Create circuit with 15 qubits
        circuit = qr.PyCircuit(15)

        # Prepare each noisy state
        for i, state in enumerate(noisy_states):
            # Copy state preparation to qubit i
            for op in state.get_operations():
                shifted_op = op.shift_qubits(i)
                circuit.add_operation(shifted_op)

        # Distillation circuit
        # Step 1: Apply verification circuit
        # Simplified version - full protocol requires stabilizer measurements

        # CNOT cascade for error detection
        for i in range(7):
            circuit.cnot(i, i + 7)

        # Measure ancilla qubits to detect errors
        for i in range(7, 14):
            circuit.measure(i)

        # Post-selection: keep only if all measurements are 0
        result = circuit.run(shots=1000)
        measurements = result.measurements()

        # Count successful distillations
        success_count = sum(
            1 for m in measurements
            if all(m[i] == '0' for i in range(7, 14))
        )

        success_prob = success_count / len(measurements)

        # The distilled state is in qubit 14
        return circuit, success_prob

    def five_to_one_distillation(self, noisy_states):
        """
        5-to-1 magic state distillation (simpler, lower overhead).

        Args:
            noisy_states: List of 5 noisy magic states

        Returns:
            Distilled magic state and success probability
        """
        if len(noisy_states) != 5:
            raise ValueError("5-to-1 protocol requires exactly 5 input states")

        circuit = qr.PyCircuit(5)

        # Prepare noisy states
        for i, state in enumerate(noisy_states):
            for op in state.get_operations():
                shifted_op = op.shift_qubits(i)
                circuit.add_operation(shifted_op)

        # Distillation circuit
        # Apply controlled operations for error detection
        circuit.cnot(0, 3)
        circuit.cnot(1, 3)
        circuit.cnot(0, 4)
        circuit.cnot(2, 4)

        # Measure ancillas
        circuit.measure(3)
        circuit.measure(4)

        # Evaluate success
        result = circuit.run(shots=1000)
        measurements = result.measurements()

        success_count = sum(
            1 for m in measurements
            if m[3] == '0' and m[4] == '0'
        )

        success_prob = success_count / len(measurements)

        return circuit, success_prob

    def iterative_distillation(self, initial_error_rate, target_error_rate,
                               max_rounds=10):
        """
        Perform iterative magic state distillation.

        Args:
            initial_error_rate: Starting error rate
            target_error_rate: Desired final error rate
            max_rounds: Maximum distillation rounds

        Returns:
            Number of rounds needed and final error rate
        """
        current_error = initial_error_rate
        rounds = 0

        while current_error > target_error_rate and rounds < max_rounds:
            # Prepare noisy states
            if self.protocol == '15-to-1':
                noisy_states = [
                    self.prepare_noisy_magic_state(current_error)
                    for _ in range(15)
                ]
                _, success_prob = self.fifteen_to_one_distillation(noisy_states)
            else:  # 5-to-1
                noisy_states = [
                    self.prepare_noisy_magic_state(current_error)
                    for _ in range(5)
                ]
                _, success_prob = self.five_to_one_distillation(noisy_states)

            # Update error rate (simplified model)
            # Real error rate depends on detailed noise analysis
            current_error = current_error ** 2 * (1 - success_prob)
            rounds += 1

            print(f"Round {rounds}: Error rate = {current_error:.6f}, "
                  f"Success prob = {success_prob:.3f}")

        return rounds, current_error

    def resource_requirements(self, initial_error, target_error):
        """
        Estimate resource requirements for magic state distillation.

        Returns:
            Dictionary with resource estimates
        """
        rounds, final_error = self.iterative_distillation(
            initial_error, target_error
        )

        if self.protocol == '15-to-1':
            states_per_round = 15
        else:
            states_per_round = 5

        total_states = states_per_round ** rounds

        return {
            'rounds': rounds,
            'total_input_states': total_states,
            'final_error_rate': final_error,
            'protocol': self.protocol
        }

# Example magic state distillation
print("\nMagic State Distillation")
print("=" * 50)

msd = MagicStateDistillation(protocol='5-to-1')

# Prepare noisy magic states
print("\n1. Preparing 5 noisy |T⟩ states (10% error rate)")
noisy_states = [msd.prepare_noisy_magic_state(0.1) for _ in range(5)]

# Distill
print("\n2. Performing 5-to-1 distillation")
distilled_circuit, success_prob = msd.five_to_one_distillation(noisy_states)
print(f"   Success probability: {success_prob:.3f}")

# Resource estimation
print("\n3. Resource Requirements")
resources = msd.resource_requirements(
    initial_error=0.1,
    target_error=0.001
)
print(f"   Distillation rounds: {resources['rounds']}")
print(f"   Total input states: {resources['total_input_states']}")
print(f"   Final error rate: {resources['final_error_rate']:.6f}")
```

## 3. Fault-Tolerant Gate Synthesis

Decompose arbitrary gates into fault-tolerant gate sequences.

```python
class FaultTolerantSynthesis:
    """
    Synthesize arbitrary single-qubit gates into fault-tolerant sequences.

    Uses Clifford+T gates with magic state distillation.
    """

    def __init__(self, epsilon=1e-3):
        """
        Initialize synthesis.

        Args:
            epsilon: Target approximation error
        """
        self.epsilon = epsilon
        self.clifford_gates = ['H', 'S', 'CNOT']
        self.t_gate_cost = 1000  # Relative cost of T gate

    def solovay_kitaev_synthesis(self, target_unitary, n=5):
        """
        Approximate arbitrary single-qubit unitary using Solovay-Kitaev algorithm.

        Args:
            target_unitary: 2x2 unitary matrix
            n: Recursion depth

        Returns:
            Gate sequence approximating target
        """
        # Base case: find best single Clifford+T gate
        if n == 0:
            return self._find_best_gate(target_unitary)

        # Recursive case
        # Find sequence G_n-1 approximating U
        g_prev = self.solovay_kitaev_synthesis(target_unitary, n - 1)

        # Compute error: U * G†_n-1
        u_error = target_unitary @ g_prev.conjugate().T

        # Factor error as V W V† W†
        v, w = self._group_commutator_factor(u_error)

        # Recursively approximate V and W
        v_seq = self.solovay_kitaev_synthesis(v, n - 1)
        w_seq = self.solovay_kitaev_synthesis(w, n - 1)

        # Combine: G_n = V W V† W† G_n-1
        sequence = (v_seq + w_seq +
                   self._dagger(v_seq) + self._dagger(w_seq) +
                   g_prev)

        return sequence

    def _find_best_gate(self, target):
        """Find best single Clifford+T gate approximation."""
        # Simple greedy search over small gate combinations
        # In practice, use precomputed lookup table

        gates = {
            'I': np.eye(2, dtype=complex),
            'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            'S': np.array([[1, 0], [0, 1j]]),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi/4)]])
        }

        best_gate = 'I'
        best_error = np.inf

        for name, gate in gates.items():
            error = np.linalg.norm(target - gate)
            if error < best_error:
                best_error = error
                best_gate = name

        return [best_gate]

    def _group_commutator_factor(self, u):
        """Factor unitary as group commutator V W V† W†."""
        # Simplified - real implementation requires geometric techniques
        # For demonstration, return approximate factors

        # Simple factorization using eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(u)

        # Construct V and W
        angle = np.angle(eigenvalues[0]) / 4
        v = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
        w = v.T

        return v, w

    def _dagger(self, sequence):
        """Compute Hermitian conjugate of gate sequence."""
        # Reverse and conjugate each gate
        dagger_seq = []
        for gate in reversed(sequence):
            if gate == 'H':
                dagger_seq.append('H')  # H = H†
            elif gate == 'S':
                dagger_seq.append('S†')
            elif gate == 'T':
                dagger_seq.append('T†')
            else:
                dagger_seq.append(gate + '†')

        return dagger_seq

    def count_t_gates(self, sequence):
        """Count number of T gates in sequence."""
        return sum(1 for gate in sequence if gate == 'T' or gate == 'T†')

    def estimate_cost(self, sequence):
        """
        Estimate cost of gate sequence.

        Cost is dominated by T gates due to magic state distillation overhead.
        """
        t_count = self.count_t_gates(sequence)
        clifford_count = len(sequence) - t_count

        # T gates are ~1000x more expensive than Cliffords
        total_cost = clifford_count + t_count * self.t_gate_cost

        return {
            'total_gates': len(sequence),
            't_gates': t_count,
            'clifford_gates': clifford_count,
            'total_cost': total_cost,
            'circuit_depth': len(sequence)  # Simplified
        }

    def optimize_t_count(self, sequence):
        """
        Optimize gate sequence to reduce T-count.

        Uses circuit identities and T-gate commutation rules.
        """
        # Simple optimization: cancel adjacent T and T† gates
        optimized = []

        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1:
                if ((sequence[i] == 'T' and sequence[i+1] == 'T†') or
                    (sequence[i] == 'T†' and sequence[i+1] == 'T')):
                    # Cancel T and T†
                    i += 2
                    continue

            optimized.append(sequence[i])
            i += 1

        return optimized

# Example fault-tolerant synthesis
print("\nFault-Tolerant Gate Synthesis")
print("=" * 50)

fts = FaultTolerantSynthesis(epsilon=1e-3)

# Target gate: Arbitrary rotation
theta = np.pi / 7
target_unitary = np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2), np.cos(theta/2)]
])

print(f"\n1. Synthesizing rotation by {theta:.3f} rad")

# Synthesize using Solovay-Kitaev
sequence = fts.solovay_kitaev_synthesis(target_unitary, n=3)
print(f"   Gate sequence: {sequence}")

# Optimize
optimized_sequence = fts.optimize_t_count(sequence)
print(f"   Optimized sequence: {optimized_sequence}")

# Estimate cost
cost = fts.estimate_cost(optimized_sequence)
print(f"\n2. Resource Estimation")
print(f"   Total gates: {cost['total_gates']}")
print(f"   T gates: {cost['t_gates']}")
print(f"   Relative cost: {cost['total_cost']:.0f}")
```

## 4. The Threshold Theorem

Understanding and verifying the fault-tolerance threshold.

```python
class ThresholdAnalysis:
    """
    Analyze and estimate fault-tolerance threshold.

    The threshold theorem states that if physical error rates
    are below a threshold, logical error rates decrease
    exponentially with code distance.
    """

    def __init__(self, code_distance=3):
        self.code_distance = code_distance

    def logical_error_rate(self, physical_error, code_distance):
        """
        Estimate logical error rate from physical error rate.

        For surface codes:
        p_L ≈ C * (p/p_th)^((d+1)/2)

        where:
        - p_L is logical error rate
        - p is physical error rate
        - p_th is threshold (~0.01 for surface codes)
        - d is code distance
        - C is a constant
        """
        p_threshold = 0.01  # Surface code threshold
        C = 0.1  # Empirical constant

        if physical_error >= p_threshold:
            # Above threshold - errors increase
            return 1.0  # Logical qubit fails

        # Below threshold - errors decrease exponentially
        p_logical = C * ((physical_error / p_threshold) **
                        ((code_distance + 1) / 2))

        return min(p_logical, 1.0)

    def required_code_distance(self, physical_error, target_logical_error):
        """
        Calculate required code distance for target logical error rate.

        Args:
            physical_error: Physical qubit error rate
            target_logical_error: Desired logical error rate

        Returns:
            Minimum code distance
        """
        p_threshold = 0.01
        C = 0.1

        if physical_error >= p_threshold:
            return np.inf  # Cannot achieve fault tolerance

        # Solve for d in: p_L = C * (p/p_th)^((d+1)/2)
        # d = 2 * log(p_L/C) / log(p/p_th) - 1

        log_ratio = np.log(target_logical_error / C) / np.log(
            physical_error / p_threshold
        )

        distance = 2 * log_ratio - 1

        return max(3, int(np.ceil(distance)))  # Minimum distance is 3

    def physical_resources(self, num_logical_qubits, code_distance):
        """
        Estimate physical qubit requirements.

        For surface codes:
        n_physical ≈ 2 * d^2 per logical qubit
        """
        qubits_per_logical = 2 * code_distance ** 2

        total_physical_qubits = num_logical_qubits * qubits_per_logical

        # Add overhead for magic state distillation
        magic_state_factories = int(np.ceil(num_logical_qubits / 10))
        magic_state_qubits = magic_state_factories * 15 * code_distance ** 2

        return {
            'logical_qubits': num_logical_qubits,
            'code_distance': code_distance,
            'physical_qubits_data': total_physical_qubits,
            'physical_qubits_magic': magic_state_qubits,
            'total_physical_qubits': total_physical_qubits + magic_state_qubits
        }

    def analyze_threshold_crossing(self, physical_errors, code_distances):
        """
        Analyze threshold behavior across different parameters.

        Returns:
            DataFrame with logical error rates
        """
        import pandas as pd

        results = []

        for p in physical_errors:
            for d in code_distances:
                p_logical = self.logical_error_rate(p, d)

                results.append({
                    'physical_error': p,
                    'code_distance': d,
                    'logical_error': p_logical,
                    'below_threshold': p < 0.01
                })

        return pd.DataFrame(results)

    def visualize_threshold(self):
        """Visualize threshold behavior."""
        import matplotlib.pyplot as plt

        physical_errors = np.logspace(-4, -1, 50)
        distances = [3, 5, 7, 9]

        plt.figure(figsize=(10, 6))

        for d in distances:
            logical_errors = [
                self.logical_error_rate(p, d)
                for p in physical_errors
            ]
            plt.loglog(physical_errors, logical_errors,
                      label=f'd = {d}')

        # Plot threshold
        plt.axvline(x=0.01, color='r', linestyle='--',
                   label='Threshold (≈1%)')

        plt.xlabel('Physical Error Rate')
        plt.ylabel('Logical Error Rate')
        plt.title('Fault-Tolerance Threshold Behavior')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example threshold analysis
print("\nThreshold Theorem Analysis")
print("=" * 50)

ta = ThresholdAnalysis()

# Example 1: Logical error rate
physical_error = 0.001  # 0.1%
code_distance = 5

p_logical = ta.logical_error_rate(physical_error, code_distance)
print(f"\n1. Logical Error Rate")
print(f"   Physical error: {physical_error:.4f}")
print(f"   Code distance: {code_distance}")
print(f"   Logical error: {p_logical:.6f}")

# Example 2: Required code distance
target_logical = 1e-10
required_d = ta.required_code_distance(physical_error, target_logical)
print(f"\n2. Required Code Distance")
print(f"   Target logical error: {target_logical:.2e}")
print(f"   Required distance: {required_d}")

# Example 3: Physical resources
num_logical = 100
resources = ta.physical_resources(num_logical, required_d)
print(f"\n3. Physical Resource Requirements")
print(f"   Logical qubits: {resources['logical_qubits']}")
print(f"   Physical qubits (data): {resources['physical_qubits_data']:,}")
print(f"   Physical qubits (magic): {resources['physical_qubits_magic']:,}")
print(f"   Total physical qubits: {resources['total_physical_qubits']:,}")
```

## 5. Resource Estimation for FTQC

Estimate resources needed for fault-tolerant quantum algorithms.

```python
class FTQCResourceEstimator:
    """
    Comprehensive resource estimation for fault-tolerant quantum computing.

    Estimates:
    - Physical qubit count
    - Circuit depth and runtime
    - T-gate count and magic state requirements
    - Error budget allocation
    """

    def __init__(self, physical_error_rate=0.001):
        self.physical_error_rate = physical_error_rate
        self.t_gate_time = 1e-3  # seconds (magic state distillation time)
        self.clifford_time = 1e-6  # seconds (surface code cycle time)

    def estimate_algorithm_resources(self, algorithm_spec):
        """
        Estimate resources for a quantum algorithm.

        Args:
            algorithm_spec: Dictionary with algorithm parameters
                - logical_qubits: Number of logical qubits
                - t_gates: Number of T gates
                - clifford_gates: Number of Clifford gates
                - circuit_depth: Logical circuit depth
                - target_error: Target total error probability

        Returns:
            Resource estimates
        """
        # Calculate required code distance
        ta = ThresholdAnalysis()
        code_distance = ta.required_code_distance(
            self.physical_error_rate,
            algorithm_spec['target_error'] / algorithm_spec['t_gates']
        )

        # Physical qubit requirements
        resources = ta.physical_resources(
            algorithm_spec['logical_qubits'],
            code_distance
        )

        # Time requirements
        t_gate_time_total = algorithm_spec['t_gates'] * self.t_gate_time
        clifford_time_total = (algorithm_spec['clifford_gates'] *
                              self.clifford_time)

        total_time = t_gate_time_total + clifford_time_total

        # Magic state requirements
        magic_states_needed = algorithm_spec['t_gates']

        # Distillation rounds for magic states
        msd = MagicStateDistillation(protocol='15-to-1')
        magic_resources = msd.resource_requirements(
            self.physical_error_rate,
            algorithm_spec['target_error'] / algorithm_spec['t_gates']
        )

        return {
            **resources,
            'code_distance': code_distance,
            'total_runtime_seconds': total_time,
            'total_runtime_hours': total_time / 3600,
            'magic_states_needed': magic_states_needed,
            'magic_distillation_rounds': magic_resources['rounds'],
            'raw_magic_states': magic_resources['total_input_states']
        }

    def compare_algorithms(self, algorithms):
        """
        Compare resource requirements of multiple algorithms.

        Args:
            algorithms: Dictionary of algorithm specifications

        Returns:
            Comparison DataFrame
        """
        import pandas as pd

        results = []

        for name, spec in algorithms.items():
            resources = self.estimate_algorithm_resources(spec)
            results.append({
                'Algorithm': name,
                'Logical Qubits': resources['logical_qubits'],
                'Physical Qubits': resources['total_physical_qubits'],
                'Code Distance': resources['code_distance'],
                'Runtime (hours)': resources['total_runtime_hours'],
                'T-gates': spec['t_gates'],
                'Magic States': resources['magic_states_needed']
            })

        return pd.DataFrame(results)

# Example resource estimation
print("\nFTQC Resource Estimation")
print("=" * 50)

estimator = FTQCResourceEstimator(physical_error_rate=0.001)

# Example: Shor's algorithm for factoring 2048-bit RSA
shor_spec = {
    'logical_qubits': 4096,
    't_gates': 10**12,
    'clifford_gates': 10**11,
    'circuit_depth': 10**9,
    'target_error': 0.01
}

print("\n1. Shor's Algorithm (2048-bit RSA)")
resources = estimator.estimate_algorithm_resources(shor_spec)
print(f"   Logical qubits: {resources['logical_qubits']:,}")
print(f"   Physical qubits: {resources['total_physical_qubits']:,}")
print(f"   Code distance: {resources['code_distance']}")
print(f"   Runtime: {resources['total_runtime_hours']:.2f} hours")
print(f"   Magic states: {resources['magic_states_needed']:,}")

# Compare multiple algorithms
algorithms = {
    'Grover (2^40 items)': {
        'logical_qubits': 40,
        't_gates': 10**6,
        'clifford_gates': 10**7,
        'circuit_depth': 10**6,
        'target_error': 0.01
    },
    'Quantum Chemistry (100 qubits)': {
        'logical_qubits': 100,
        't_gates': 10**8,
        'clifford_gates': 10**9,
        'circuit_depth': 10**7,
        'target_error': 0.001
    },
    'Shor (2048-bit)': shor_spec
}

print("\n2. Algorithm Comparison")
comparison = estimator.compare_algorithms(algorithms)
print(comparison.to_string(index=False))
```

## Exercises

1. Implement fault-tolerant Toffoli gate using magic states
2. Design custom distillation protocol for Y-rotations
3. Analyze threshold for concatenated codes
4. Estimate resources for quantum simulation algorithm
5. Implement lattice surgery operations

## Further Reading

- **Fault Tolerance**: Gottesman, "Stabilizer Codes and Quantum Error Correction" (1997)
- **Magic States**: Bravyi & Kitaev, "Universal quantum computation with ideal Clifford gates" (2005)
- **Threshold**: Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012)
- **Resource Estimation**: Gidney & Ekerå, "How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits" (2019)

## Next Steps

- Advanced Tutorial: Topological Quantum Computing
- Research: Implementing fault-tolerant quantum algorithms
- Practice: Resource estimation for your algorithms
