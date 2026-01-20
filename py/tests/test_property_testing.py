#!/usr/bin/env python3
"""
Test suite for property-based testing framework.
"""

import pytest
import numpy as np

try:
    from quantrs2.property_testing import (
        qubit_indices, qubit_pairs, rotation_angles, complex_amplitudes,
        quantum_states, unitary_matrices, hermitian_matrices, gate_sequences,
        QuantumProperties, QuantumCircuitStateMachine, create_quantum_property_test,
        test_circuit_creation_property, test_single_qubit_gate_properties,
        test_rotation_gate_properties, test_cnot_gate_properties,
        test_gate_sequence_properties, test_rotation_composition_property,
        test_circuit_copy_property, run_property_tests
    )
    HAS_PROPERTY_TESTING = True
except ImportError:
    HAS_PROPERTY_TESTING = False

try:
    from hypothesis import given, strategies as st, assume
    from hypothesis.stateful import run_state_machine_as_test
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
class TestQuantumStrategies:
    """Test quantum-specific Hypothesis strategies."""
    
    def test_qubit_indices_strategy(self):
        """Test qubit_indices strategy generates valid indices."""
        pytest.importorskip("hypothesis")
        
        @given(data=st.data())
        def test_qubit_indices_property(data):
            n_qubits, qubit = data.draw(qubit_indices(max_qubits=5))
            assert 1 <= n_qubits <= 5
            assert 0 <= qubit < n_qubits
        
        test_qubit_indices_property()
    
    def test_qubit_pairs_strategy(self):
        """Test qubit_pairs strategy generates valid pairs."""
        pytest.importorskip("hypothesis")
        
        @given(data=st.data())
        def test_qubit_pairs_property(data):
            n_qubits, control, target = data.draw(qubit_pairs(max_qubits=5))
            assert 2 <= n_qubits <= 5
            assert 0 <= control < n_qubits
            assert 0 <= target < n_qubits
            assert control != target
        
        test_qubit_pairs_property()
    
    def test_rotation_angles_strategy(self):
        """Test rotation_angles strategy generates valid angles."""
        pytest.importorskip("hypothesis")
        
        @given(angle=rotation_angles())
        def test_rotation_angles_property(angle):
            assert isinstance(angle, float)
            assert not np.isnan(angle)
            assert not np.isinf(angle)
            assert -4 * np.pi <= angle <= 4 * np.pi
        
        test_rotation_angles_property()
    
    def test_complex_amplitudes_strategy(self):
        """Test complex_amplitudes strategy generates valid amplitudes."""
        pytest.importorskip("hypothesis")
        
        @given(amplitude=complex_amplitudes())
        def test_complex_amplitudes_property(amplitude):
            assert isinstance(amplitude, complex)
            assert not np.isnan(amplitude.real)
            assert not np.isnan(amplitude.imag)
            assert not np.isinf(amplitude.real)
            assert not np.isinf(amplitude.imag)
            assert -1.0 <= amplitude.real <= 1.0
            assert -1.0 <= amplitude.imag <= 1.0
        
        test_complex_amplitudes_property()
    
    def test_quantum_states_strategy(self):
        """Test quantum_states strategy generates normalized states."""
        pytest.importorskip("hypothesis")
        
        @given(n_qubits=st.integers(min_value=1, max_value=3))
        def test_quantum_states_property(n_qubits):
            @given(state=quantum_states(n_qubits))
            def test_state_normalization(state):
                assert state.shape == (2**n_qubits,)
                assert QuantumProperties.is_normalized(state)
            
            test_state_normalization()
        
        test_quantum_states_property()
    
    def test_unitary_matrices_strategy(self):
        """Test unitary_matrices strategy generates unitary matrices."""
        pytest.importorskip("hypothesis")
        
        @given(n_qubits=st.integers(min_value=1, max_value=2))
        def test_unitary_matrices_property(n_qubits):
            @given(matrix=unitary_matrices(n_qubits))
            def test_matrix_unitarity(matrix):
                dim = 2**n_qubits
                assert matrix.shape == (dim, dim)
                assert QuantumProperties.is_unitary(matrix)
            
            test_matrix_unitarity()
        
        test_unitary_matrices_property()
    
    def test_hermitian_matrices_strategy(self):
        """Test hermitian_matrices strategy generates Hermitian matrices."""
        pytest.importorskip("hypothesis")
        
        @given(n_qubits=st.integers(min_value=1, max_value=2))
        def test_hermitian_matrices_property(n_qubits):
            @given(matrix=hermitian_matrices(n_qubits))
            def test_matrix_hermiticity(matrix):
                dim = 2**n_qubits
                assert matrix.shape == (dim, dim)
                assert QuantumProperties.is_hermitian(matrix)
            
            test_matrix_hermiticity()
        
        test_hermitian_matrices_property()
    
    def test_gate_sequences_strategy(self):
        """Test gate_sequences strategy generates valid sequences."""
        pytest.importorskip("hypothesis")
        
        @given(data=st.data())
        def test_gate_sequences_property(data):
            n_qubits, gates = data.draw(gate_sequences(max_qubits=3, max_depth=5))
            assert 1 <= n_qubits <= 3
            assert 1 <= len(gates) <= 5
            
            for gate_type, params in gates:
                assert gate_type in ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cnot', 'cz']
                
                if gate_type in ['h', 'x', 'y', 'z', 's', 't']:
                    assert len(params) == 1
                    assert 0 <= params[0] < n_qubits
                elif gate_type in ['rx', 'ry', 'rz']:
                    assert len(params) == 2
                    assert 0 <= params[0] < n_qubits
                    assert isinstance(params[1], float)
                elif gate_type in ['cnot', 'cz']:
                    assert len(params) == 2
                    assert 0 <= params[0] < n_qubits
                    assert 0 <= params[1] < n_qubits
                    assert params[0] != params[1]
        
        test_gate_sequences_property()


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
class TestQuantumProperties:
    """Test QuantumProperties utility functions."""
    
    def test_is_normalized(self):
        """Test normalization checking."""
        # Normalized state
        normalized = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        assert QuantumProperties.is_normalized(normalized)
        
        # Non-normalized state
        non_normalized = np.array([1.0, 1.0], dtype=complex)
        assert not QuantumProperties.is_normalized(non_normalized)
        
        # Zero state
        zero_state = np.array([0.0, 0.0], dtype=complex)
        assert not QuantumProperties.is_normalized(zero_state)
    
    def test_is_unitary(self):
        """Test unitarity checking."""
        # Identity matrix (unitary)
        identity = np.eye(2, dtype=complex)
        assert QuantumProperties.is_unitary(identity)
        
        # Pauli-X matrix (unitary)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        assert QuantumProperties.is_unitary(pauli_x)
        
        # Hadamard matrix (unitary)
        hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert QuantumProperties.is_unitary(hadamard)
        
        # Non-unitary matrix
        non_unitary = np.array([[1, 0], [0, 2]], dtype=complex)
        assert not QuantumProperties.is_unitary(non_unitary)
        
        # Non-square matrix
        non_square = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        assert not QuantumProperties.is_unitary(non_square)
    
    def test_is_hermitian(self):
        """Test Hermitian checking."""
        # Pauli-Z matrix (Hermitian)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert QuantumProperties.is_hermitian(pauli_z)
        
        # Pauli-Y matrix (Hermitian)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert QuantumProperties.is_hermitian(pauli_y)
        
        # Non-Hermitian matrix
        non_hermitian = np.array([[1, 1j], [0, 1]], dtype=complex)
        assert not QuantumProperties.is_hermitian(non_hermitian)
        
        # Non-square matrix
        non_square = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        assert not QuantumProperties.is_hermitian(non_square)
    
    def test_preserves_probabilities(self):
        """Test probability preservation checking."""
        state1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        state2 = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
        
        # Both normalized - should preserve probabilities
        assert QuantumProperties.preserves_probabilities(state1, state2)
        
        # Different norms - should not preserve probabilities
        state3 = np.array([1.0, 1.0], dtype=complex)
        assert not QuantumProperties.preserves_probabilities(state1, state3)
    
    def test_commutes(self):
        """Test commutativity checking."""
        # Pauli-X and Pauli-Z (don't commute)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert not QuantumProperties.commutes(pauli_x, pauli_z)
        
        # Identity commutes with everything
        identity = np.eye(2, dtype=complex)
        assert QuantumProperties.commutes(identity, pauli_x)
        assert QuantumProperties.commutes(identity, pauli_z)
        
        # Matrix commutes with itself
        assert QuantumProperties.commutes(pauli_x, pauli_x)
    
    def test_is_valid_probability_distribution(self):
        """Test probability distribution validation."""
        # Valid distribution
        valid_probs = {'00': 0.5, '01': 0.3, '10': 0.1, '11': 0.1}
        assert QuantumProperties.is_valid_probability_distribution(valid_probs)
        
        # Invalid: doesn't sum to 1
        invalid_sum = {'00': 0.5, '01': 0.3}
        assert not QuantumProperties.is_valid_probability_distribution(invalid_sum)
        
        # Invalid: negative probability
        negative_prob = {'00': 0.5, '01': -0.1, '10': 0.6}
        assert not QuantumProperties.is_valid_probability_distribution(negative_prob)
        
        # Edge case: empty distribution
        empty_probs = {}
        assert not QuantumProperties.is_valid_probability_distribution(empty_probs)
    
    def test_circuit_invariants(self):
        """Test circuit optimization invariants."""
        # Depth invariant
        assert QuantumProperties.circuit_depth_invariant(5, 4)  # Optimization improved
        assert QuantumProperties.circuit_depth_invariant(5, 5)  # No change
        assert QuantumProperties.circuit_depth_invariant(5, 10)  # Allow some increase
        assert not QuantumProperties.circuit_depth_invariant(5, 15)  # Too much increase
        
        # Gate count invariant
        assert QuantumProperties.gate_count_invariant(10, 8)  # Optimization improved
        assert QuantumProperties.gate_count_invariant(10, 10)  # No change
        assert not QuantumProperties.gate_count_invariant(10, 12)  # Increased count


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestQuantumStateMachine:
    """Test quantum circuit state machine."""
    
    def test_state_machine_basic(self):
        """Test basic state machine functionality."""
        # This test runs the state machine for a short time
        run_state_machine_as_test(QuantumCircuitStateMachine)
    
    def test_state_machine_properties(self):
        """Test that state machine maintains quantum properties."""
        # Create instance manually for testing
        sm = QuantumCircuitStateMachine()
        
        # Test initialization
        sm.init_circuit()
        
        # Test that we can access the circuit
        if hasattr(sm, 'current_circuit'):
            assert sm.current_circuit is not None
            assert sm.n_qubits > 0


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
class TestPropertyTestDecorator:
    """Test property test decorator functionality."""
    
    def test_create_quantum_property_test_with_hypothesis(self):
        """Test decorator when hypothesis is available."""
        if not HYPOTHESIS_AVAILABLE:
            pytest.skip("Hypothesis not available")
        
        @create_quantum_property_test
        def sample_property_test():
            assert True
        
        # Should be callable
        assert callable(sample_property_test)
        sample_property_test()
    
    def test_create_quantum_property_test_without_hypothesis(self):
        """Test decorator when hypothesis is not available."""
        # Mock hypothesis unavailability
        import quantrs2.property_testing as pt_module
        original_hypothesis = pt_module.HYPOTHESIS_AVAILABLE
        pt_module.HYPOTHESIS_AVAILABLE = False
        
        try:
            @create_quantum_property_test
            def sample_property_test():
                assert True
            
            # Should skip the test
            with pytest.raises(pytest.skip.Exception):
                sample_property_test()
        
        finally:
            # Restore original state
            pt_module.HYPOTHESIS_AVAILABLE = original_hypothesis


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestQuantumPropertyTests:
    """Test the actual quantum property tests."""
    
    def test_circuit_creation_property_test(self):
        """Test circuit creation property test."""
        # This should run without errors
        test_circuit_creation_property()
    
    def test_single_qubit_gate_properties_test(self):
        """Test single qubit gate properties test."""
        # This should run without errors
        test_single_qubit_gate_properties()
    
    def test_rotation_gate_properties_test(self):
        """Test rotation gate properties test."""
        # This should run without errors
        test_rotation_gate_properties()
    
    def test_cnot_gate_properties_test(self):
        """Test CNOT gate properties test."""
        # This should run without errors
        test_cnot_gate_properties()
    
    def test_gate_sequence_properties_test(self):
        """Test gate sequence properties test."""
        # This should run without errors
        test_gate_sequence_properties()
    
    def test_rotation_composition_property_test(self):
        """Test rotation composition property test."""
        # This should run without errors
        test_rotation_composition_property()
    
    def test_circuit_copy_property_test(self):
        """Test circuit copy property test."""
        # This should run without errors
        test_circuit_copy_property()


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
class TestPropertyTestRunner:
    """Test property test runner functionality."""
    
    def test_run_property_tests_with_hypothesis(self):
        """Test running property tests with hypothesis available."""
        if not HYPOTHESIS_AVAILABLE:
            pytest.skip("Hypothesis not available")
        
        # Should run without errors
        run_property_tests()
    
    def test_run_property_tests_without_hypothesis(self):
        """Test running property tests without hypothesis."""
        # Mock hypothesis unavailability
        import quantrs2.property_testing as pt_module
        original_hypothesis = pt_module.HYPOTHESIS_AVAILABLE
        pt_module.HYPOTHESIS_AVAILABLE = False
        
        try:
            # Should skip gracefully
            run_property_tests()
        finally:
            # Restore original state
            pt_module.HYPOTHESIS_AVAILABLE = original_hypothesis


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
class TestPropertyTestingIntegration:
    """Test integration of property testing with quantum operations."""
    
    def test_property_testing_with_mock_circuits(self):
        """Test property testing with mock circuit objects."""
        # Create a mock circuit class for testing
        class MockCircuit:
            def __init__(self, n_qubits):
                self.n_qubits = n_qubits
                self._gates = []
            
            def h(self, qubit):
                self._gates.append(('h', qubit))
            
            def cnot(self, control, target):
                self._gates.append(('cnot', control, target))
            
            def gate_count(self):
                return len(self._gates)
            
            def depth(self):
                return len(self._gates)  # Simplified
            
            def copy(self):
                new_circuit = MockCircuit(self.n_qubits)
                new_circuit._gates = self._gates.copy()
                return new_circuit
        
        # Test basic properties
        circuit = MockCircuit(2)
        assert circuit.n_qubits == 2
        assert circuit.gate_count() == 0
        
        circuit.h(0)
        circuit.cnot(0, 1)
        assert circuit.gate_count() == 2
        
        # Test copy property
        copied = circuit.copy()
        assert copied.n_qubits == circuit.n_qubits
        assert copied.gate_count() == circuit.gate_count()
    
    def test_quantum_properties_edge_cases(self):
        """Test quantum properties with edge cases."""
        # Very small tolerance
        tiny_tolerance = 1e-15
        
        # Perfect normalized state
        perfect_state = np.array([1.0, 0.0], dtype=complex)
        assert QuantumProperties.is_normalized(perfect_state, tiny_tolerance)
        
        # Nearly normalized state
        nearly_normalized = np.array([1.0 + 1e-16, 0.0], dtype=complex)
        assert QuantumProperties.is_normalized(nearly_normalized, 1e-10)
        assert not QuantumProperties.is_normalized(nearly_normalized, tiny_tolerance)
        
        # Perfect identity
        perfect_identity = np.eye(2, dtype=complex)
        assert QuantumProperties.is_unitary(perfect_identity, tiny_tolerance)
        
        # Nearly unitary
        nearly_unitary = np.eye(2, dtype=complex) * (1.0 + 1e-16)
        assert QuantumProperties.is_unitary(nearly_unitary, 1e-10)
        assert not QuantumProperties.is_unitary(nearly_unitary, tiny_tolerance)


@pytest.mark.skipif(not HAS_PROPERTY_TESTING, reason="property_testing module not available")
class TestPropertyTestingPerformance:
    """Test performance characteristics of property testing."""
    
    def test_strategy_performance(self):
        """Test that strategies generate data efficiently."""
        pytest.importorskip("hypothesis")
        
        import time
        
        # Test qubit_indices strategy
        start_time = time.time()
        for _ in range(100):
            example = qubit_indices(max_qubits=10).example()
            assert len(example) == 2
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 1.0
        
        # Test gate_sequences strategy
        start_time = time.time()
        for _ in range(50):
            example = gate_sequences(max_qubits=3, max_depth=5).example()
            n_qubits, gates = example
            assert 1 <= n_qubits <= 3
            assert 1 <= len(gates) <= 5
        end_time = time.time()
        
        # Should be reasonably fast
        assert end_time - start_time < 2.0
    
    def test_property_checking_performance(self):
        """Test that property checking is efficient."""
        # Test normalization checking
        import time
        
        states = [np.random.rand(4) + 1j * np.random.rand(4) for _ in range(100)]
        normalized_states = [s / np.linalg.norm(s) for s in states]
        
        start_time = time.time()
        for state in normalized_states:
            assert QuantumProperties.is_normalized(state)
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 0.5
        
        # Test unitary checking
        matrices = [np.random.rand(4, 4) + 1j * np.random.rand(4, 4) for _ in range(20)]
        unitary_matrices_list = [np.linalg.qr(m)[0] for m in matrices]
        
        start_time = time.time()
        for matrix in unitary_matrices_list:
            assert QuantumProperties.is_unitary(matrix)
        end_time = time.time()
        
        # Should be reasonably fast
        assert end_time - start_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])