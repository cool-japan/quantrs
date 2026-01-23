"""
Property-Based Testing Framework for QuantRS2

This module provides property-based testing capabilities for quantum circuits,
algorithms, and other quantum computing components using hypothesis.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from hypothesis import given, strategies as st, assume, note, settings, Verbosity
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, invariant, initialize
import pytest

try:
    from hypothesis import HealthCheck
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


# Quantum-specific strategies
@st.composite
def qubit_indices(draw, max_qubits: int = 10):
    """Strategy for generating valid qubit indices."""
    n_qubits = draw(st.integers(min_value=1, max_value=max_qubits))
    qubit = draw(st.integers(min_value=0, max_value=n_qubits - 1))
    return n_qubits, qubit


@st.composite
def qubit_pairs(draw, max_qubits: int = 10):
    """Strategy for generating valid qubit pairs for two-qubit gates."""
    n_qubits = draw(st.integers(min_value=2, max_value=max_qubits))
    control = draw(st.integers(min_value=0, max_value=n_qubits - 1))
    target = draw(st.integers(min_value=0, max_value=n_qubits - 1))
    assume(control != target)  # Control and target must be different
    return n_qubits, control, target


@st.composite
def rotation_angles(draw):
    """Strategy for generating rotation angles."""
    return draw(st.floats(
        min_value=-4 * np.pi,
        max_value=4 * np.pi,
        allow_nan=False,
        allow_infinity=False
    ))


@st.composite
def complex_amplitudes(draw):
    """Strategy for generating complex amplitudes."""
    real = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    imag = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return complex(real, imag)


@st.composite
def quantum_states(draw, n_qubits: int):
    """Strategy for generating normalized quantum state vectors."""
    dim = 2 ** n_qubits
    amplitudes = draw(st.lists(complex_amplitudes(), min_size=dim, max_size=dim))
    
    # Normalize the state
    state = np.array(amplitudes, dtype=complex)
    norm = np.linalg.norm(state)
    assume(norm > 1e-10)  # Avoid zero states
    
    return state / norm


@st.composite
def unitary_matrices(draw, n_qubits: int):
    """Strategy for generating unitary matrices."""
    dim = 2 ** n_qubits
    
    # Generate random complex matrix
    real_part = draw(st.lists(
        st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False), 
                min_size=dim, max_size=dim),
        min_size=dim, max_size=dim
    ))
    
    imag_part = draw(st.lists(
        st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
                min_size=dim, max_size=dim),
        min_size=dim, max_size=dim
    ))
    
    matrix = np.array(real_part) + 1j * np.array(imag_part)
    
    # Make it unitary using QR decomposition
    Q, R = np.linalg.qr(matrix)
    # Ensure determinant is 1 (special unitary)
    Q = Q * (np.linalg.det(Q) ** (-1/dim))
    
    return Q


@st.composite
def hermitian_matrices(draw, n_qubits: int):
    """Strategy for generating Hermitian matrices (observables)."""
    dim = 2 ** n_qubits
    
    # Generate random complex matrix
    real_part = draw(st.lists(
        st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
                min_size=dim, max_size=dim),
        min_size=dim, max_size=dim
    ))
    
    imag_part = draw(st.lists(
        st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
                min_size=dim, max_size=dim),
        min_size=dim, max_size=dim
    ))
    
    matrix = np.array(real_part) + 1j * np.array(imag_part)
    
    # Make it Hermitian
    hermitian = (matrix + matrix.conj().T) / 2
    
    return hermitian


@st.composite 
def gate_sequences(draw, max_qubits: int = 5, max_depth: int = 20):
    """Strategy for generating sequences of quantum gates."""
    n_qubits = draw(st.integers(min_value=1, max_value=max_qubits))
    depth = draw(st.integers(min_value=1, max_value=max_depth))
    
    gates = []
    for _ in range(depth):
        gate_type = draw(st.sampled_from([
            'h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cnot', 'cz'
        ]))
        
        if gate_type in ['h', 'x', 'y', 'z', 's', 't']:
            # Single-qubit gates
            qubit = draw(st.integers(min_value=0, max_value=n_qubits - 1))
            gates.append((gate_type, [qubit]))
        
        elif gate_type in ['rx', 'ry', 'rz']:
            # Rotation gates
            qubit = draw(st.integers(min_value=0, max_value=n_qubits - 1))
            angle = draw(rotation_angles())
            gates.append((gate_type, [qubit, angle]))
        
        elif gate_type in ['cnot', 'cz'] and n_qubits >= 2:
            # Two-qubit gates
            control = draw(st.integers(min_value=0, max_value=n_qubits - 1))
            target = draw(st.integers(min_value=0, max_value=n_qubits - 1))
            assume(control != target)
            gates.append((gate_type, [control, target]))
    
    return n_qubits, gates


class QuantumProperties:
    """Collection of quantum computing properties for testing."""
    
    @staticmethod
    def is_normalized(state: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a quantum state is normalized."""
        norm = np.linalg.norm(state)
        return abs(norm - 1.0) < tolerance
    
    @staticmethod
    def is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a matrix is unitary."""
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        identity = np.eye(matrix.shape[0])
        product = matrix @ matrix.conj().T
        return np.allclose(product, identity, atol=tolerance)
    
    @staticmethod
    def is_hermitian(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if a matrix is Hermitian."""
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        return np.allclose(matrix, matrix.conj().T, atol=tolerance)
    
    @staticmethod
    def preserves_probabilities(state1: np.ndarray, state2: np.ndarray, 
                              tolerance: float = 1e-10) -> bool:
        """Check if transformation preserves probability (norm)."""
        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)
        return abs(norm1 - norm2) < tolerance
    
    @staticmethod
    def commutes(matrix1: np.ndarray, matrix2: np.ndarray, 
                tolerance: float = 1e-10) -> bool:
        """Check if two matrices commute."""
        commutator = matrix1 @ matrix2 - matrix2 @ matrix1
        return np.allclose(commutator, np.zeros_like(commutator), atol=tolerance)
    
    @staticmethod
    def is_valid_probability_distribution(probs: Dict[str, float], 
                                        tolerance: float = 1e-10) -> bool:
        """Check if probability distribution is valid."""
        # All probabilities non-negative
        if any(p < -tolerance for p in probs.values()):
            return False
        
        # Probabilities sum to 1
        total = sum(probs.values())
        return abs(total - 1.0) < tolerance
    
    @staticmethod
    def circuit_depth_invariant(original_depth: int, optimized_depth: int) -> bool:
        """Circuit optimization should not increase depth unnecessarily."""
        # Optimized circuit should not be significantly deeper
        return optimized_depth <= original_depth * 2  # Allow some flexibility
    
    @staticmethod
    def gate_count_invariant(original_count: int, optimized_count: int) -> bool:
        """Circuit optimization should not increase gate count."""
        return optimized_count <= original_count


class QuantumCircuitStateMachine(RuleBasedStateMachine):
    """Stateful testing for quantum circuits."""

    # Bundle must be a class variable for @rule decorators to reference it
    circuits = Bundle('circuits')

    def __init__(self):
        super().__init__()
        self.max_qubits = 8
    
    @initialize()
    def init_circuit(self):
        """Initialize with an empty circuit."""
        if not _NATIVE_AVAILABLE:
            return
        
        n_qubits = 2  # Start simple
        self.current_circuit = _quantrs2.PyCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.gate_history = []
    
    @rule(target=circuits)
    def create_circuit(self, n_qubits=st.integers(min_value=1, max_value=8)):
        """Create a new quantum circuit."""
        if not _NATIVE_AVAILABLE:
            return st.just(None)
        
        circuit = _quantrs2.PyCircuit(n_qubits)
        return circuit
    
    @rule(circuit=circuits, qubit=st.integers(min_value=0, max_value=7))
    def add_hadamard(self, circuit, qubit):
        """Add Hadamard gate to circuit."""
        if circuit is None:
            return
        
        assume(qubit < circuit.n_qubits)
        circuit.h(qubit)
        note(f"Added H gate on qubit {qubit}")
    
    @rule(circuit=circuits, control=st.integers(min_value=0, max_value=7),
          target_qubit=st.integers(min_value=0, max_value=7))
    def add_cnot(self, circuit, control, target_qubit):
        """Add CNOT gate to circuit."""
        if circuit is None:
            return

        assume(circuit.n_qubits >= 2)
        assume(control < circuit.n_qubits)
        assume(target_qubit < circuit.n_qubits)
        assume(control != target_qubit)

        circuit.cnot(control, target_qubit)
        note(f"Added CNOT gate: control={control}, target={target_qubit}")
    
    @rule(circuit=circuits, qubit=st.integers(min_value=0, max_value=7), 
          angle=rotation_angles())
    def add_rotation(self, circuit, qubit, angle):
        """Add rotation gate to circuit."""
        if circuit is None:
            return
        
        assume(qubit < circuit.n_qubits)
        
        gate_type = st.sampled_from(['rx', 'ry', 'rz']).example()
        if gate_type == 'rx':
            circuit.rx(qubit, angle)
        elif gate_type == 'ry':
            circuit.ry(qubit, angle)
        else:
            circuit.rz(qubit, angle)
        
        note(f"Added {gate_type.upper()} gate on qubit {qubit} with angle {angle:.3f}")
    
    @invariant()
    def circuit_is_valid(self):
        """Circuit should always be in valid state."""
        if not _NATIVE_AVAILABLE:
            return
        
        # Basic validity checks
        if hasattr(self, 'current_circuit'):
            circuit = self.current_circuit
            assert circuit.n_qubits > 0
            assert circuit.gate_count() >= 0
            assert circuit.depth() >= 0


def create_quantum_property_test(property_func: Callable) -> Callable:
    """Decorator to create quantum property tests."""
    def test_wrapper(*args, **kwargs):
        if not HYPOTHESIS_AVAILABLE:
            pytest.skip("Hypothesis not available for property testing")
        return property_func(*args, **kwargs)
    
    return test_wrapper


# Property test examples
@create_quantum_property_test
@given(n_qubits=st.integers(min_value=1, max_value=6))
def test_circuit_creation_property(n_qubits):
    """Property: Circuits can be created with any positive number of qubits."""
    if not _NATIVE_AVAILABLE:
        return
    
    circuit = _quantrs2.PyCircuit(n_qubits)
    assert circuit.n_qubits == n_qubits
    assert circuit.gate_count() == 0
    assert circuit.depth() == 0


@create_quantum_property_test  
@given(data=st.data())
def test_single_qubit_gate_properties(data):
    """Property: Single-qubit gates preserve normalization."""
    if not _NATIVE_AVAILABLE:
        return
    
    n_qubits, qubit = data.draw(qubit_indices(max_qubits=4))
    gate_type = data.draw(st.sampled_from(['h', 'x', 'y', 'z', 's', 't']))
    
    circuit = _quantrs2.PyCircuit(n_qubits)
    
    # Add gate
    if gate_type == 'h':
        circuit.h(qubit)
    elif gate_type == 'x':
        circuit.x(qubit)
    elif gate_type == 'y':
        circuit.y(qubit)
    elif gate_type == 'z':
        circuit.z(qubit)
    elif gate_type == 's':
        circuit.s(qubit)
    elif gate_type == 't':
        circuit.t(qubit)
    
    # Run circuit
    result = circuit.run()
    probs = result.state_probabilities()
    
    # Property: Probabilities should form valid distribution
    assert QuantumProperties.is_valid_probability_distribution(probs)


@create_quantum_property_test
@given(data=st.data())
def test_rotation_gate_properties(data):
    """Property: Rotation gates preserve normalization and are reversible."""
    if not _NATIVE_AVAILABLE:
        return
    
    n_qubits, qubit = data.draw(qubit_indices(max_qubits=3))
    angle = data.draw(rotation_angles())
    gate_type = data.draw(st.sampled_from(['rx', 'ry', 'rz']))
    
    # Forward rotation
    circuit1 = _quantrs2.PyCircuit(n_qubits)
    if gate_type == 'rx':
        circuit1.rx(qubit, angle)
    elif gate_type == 'ry':
        circuit1.ry(qubit, angle)
    else:
        circuit1.rz(qubit, angle)
    
    result1 = circuit1.run()
    probs1 = result1.state_probabilities()
    
    # Property: Valid probability distribution
    assert QuantumProperties.is_valid_probability_distribution(probs1)
    
    # Reverse rotation (should return to original state)
    circuit2 = _quantrs2.PyCircuit(n_qubits)
    if gate_type == 'rx':
        circuit2.rx(qubit, angle)
        circuit2.rx(qubit, -angle)
    elif gate_type == 'ry':
        circuit2.ry(qubit, angle)
        circuit2.ry(qubit, -angle)
    else:
        circuit2.rz(qubit, angle)
        circuit2.rz(qubit, -angle)
    
    result2 = circuit2.run()
    probs2 = result2.state_probabilities()
    
    # Should be back to |00...0⟩ state
    expected_state = '0' * n_qubits
    assert probs2.get(expected_state, 0) > 0.99


@create_quantum_property_test
@given(data=st.data())
def test_cnot_gate_properties(data):
    """Property: CNOT gates preserve normalization and create entanglement."""
    if not _NATIVE_AVAILABLE:
        return
    
    n_qubits, control, target = data.draw(qubit_pairs(max_qubits=4))
    
    circuit = _quantrs2.PyCircuit(n_qubits)
    circuit.h(control)  # Create superposition
    circuit.cnot(control, target)  # Entangle
    
    result = circuit.run()
    probs = result.state_probabilities()
    
    # Property: Valid probability distribution
    assert QuantumProperties.is_valid_probability_distribution(probs)
    
    # Property: Should create entanglement (non-zero correlation)
    # For 2-qubit case with H + CNOT, should have P(00) ≈ P(11) ≈ 0.5
    if n_qubits == 2 and control == 0 and target == 1:
        assert abs(probs.get('00', 0) - 0.5) < 0.1
        assert abs(probs.get('11', 0) - 0.5) < 0.1
        assert probs.get('01', 0) < 0.1
        assert probs.get('10', 0) < 0.1


@create_quantum_property_test
@given(data=st.data())
def test_gate_sequence_properties(data):
    """Property: Sequences of gates preserve quantum mechanics rules."""
    if not _NATIVE_AVAILABLE:
        return
    
    n_qubits, gates = data.draw(gate_sequences(max_qubits=3, max_depth=10))
    
    circuit = _quantrs2.PyCircuit(n_qubits)
    
    # Apply gate sequence
    for gate_type, params in gates:
        if gate_type == 'h':
            circuit.h(params[0])
        elif gate_type == 'x':
            circuit.x(params[0])
        elif gate_type == 'y':
            circuit.y(params[0])
        elif gate_type == 'z':
            circuit.z(params[0])
        elif gate_type == 's':
            circuit.s(params[0])
        elif gate_type == 't':
            circuit.t(params[0])
        elif gate_type == 'rx':
            circuit.rx(params[0], params[1])
        elif gate_type == 'ry':
            circuit.ry(params[0], params[1])
        elif gate_type == 'rz':
            circuit.rz(params[0], params[1])
        elif gate_type == 'cnot':
            circuit.cnot(params[0], params[1])
        elif gate_type == 'cz':
            circuit.cz(params[0], params[1])
    
    result = circuit.run()
    probs = result.state_probabilities()
    
    # Property: Always produces valid probability distribution
    assert QuantumProperties.is_valid_probability_distribution(probs)
    
    # Property: Circuit metrics should be reasonable
    assert circuit.gate_count() == len(gates)
    assert circuit.depth() <= len(gates)  # Depth should not exceed gate count


@create_quantum_property_test
@given(angle1=rotation_angles(), angle2=rotation_angles())
def test_rotation_composition_property(angle1, angle2):
    """Property: Composition of rotations follows expected rules."""
    if not _NATIVE_AVAILABLE:
        return
    
    # RZ(angle1) * RZ(angle2) = RZ(angle1 + angle2)
    circuit1 = _quantrs2.PyCircuit(1)
    circuit1.rz(0, angle1)
    circuit1.rz(0, angle2)
    
    circuit2 = _quantrs2.PyCircuit(1)
    circuit2.rz(0, angle1 + angle2)
    
    result1 = circuit1.run()
    result2 = circuit2.run()
    
    probs1 = result1.state_probabilities()
    probs2 = result2.state_probabilities()
    
    # Results should be equivalent (up to global phase)
    # For single qubit in |0⟩ state, RZ rotations don't change probabilities
    assert abs(probs1.get('0', 0) - probs2.get('0', 0)) < 1e-10
    assert abs(probs1.get('1', 0) - probs2.get('1', 0)) < 1e-10


@create_quantum_property_test
@given(n_qubits=st.integers(min_value=1, max_value=4))
def test_circuit_copy_property(n_qubits):
    """Property: Circuit copying preserves all properties."""
    if not _NATIVE_AVAILABLE:
        return
    
    # Create circuit with some gates
    original = _quantrs2.PyCircuit(n_qubits)
    original.h(0)
    if n_qubits >= 2:
        original.cnot(0, 1)
    
    # Copy circuit
    copied = original.copy()
    
    # Properties should be preserved
    assert copied.n_qubits == original.n_qubits
    assert copied.gate_count() == original.gate_count()
    assert copied.depth() == original.depth()
    
    # Results should be identical
    result1 = original.run()
    result2 = copied.run()
    
    probs1 = result1.state_probabilities()
    probs2 = result2.state_probabilities()
    
    for state in probs1:
        assert abs(probs1[state] - probs2.get(state, 0)) < 1e-10


# Test runner with custom settings
@settings(
    max_examples=100,
    deadline=10000,  # 10 seconds per example
    verbosity=Verbosity.normal,
    suppress_health_check=[HealthCheck.too_slow] if HYPOTHESIS_AVAILABLE else []
)
def run_property_tests():
    """Run all property tests with custom settings."""
    if not HYPOTHESIS_AVAILABLE:
        print("Hypothesis not available, skipping property tests")
        return
    
    test_functions = [
        test_circuit_creation_property,
        test_single_qubit_gate_properties,
        test_rotation_gate_properties,
        test_cnot_gate_properties,
        test_gate_sequence_properties,
        test_rotation_composition_property,
        test_circuit_copy_property,
    ]
    
    for test_func in test_functions:
        print(f"Running {test_func.__name__}...")
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")


if __name__ == "__main__":
    run_property_tests()