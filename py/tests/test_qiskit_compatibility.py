"""
Tests for Qiskit compatibility layer.
"""

import unittest
import warnings
import numpy as np
from unittest.mock import Mock, patch

# Import the compatibility layer
from quantrs2.qiskit_compatibility import (
    CircuitConverter,
    QiskitBackendAdapter,
    QiskitAlgorithmLibrary,
    QiskitCompatibilityError,
    from_qiskit,
    to_qiskit,
    run_on_qiskit_backend,
    create_qiskit_compatible_vqe,
    test_conversion_fidelity,
    benchmark_conversion_performance,
    QISKIT_AVAILABLE,
    QUANTRS2_AVAILABLE
)

try:
    from quantrs2 import Circuit as QuantRS2Circuit
except ImportError:
    from quantrs2.qiskit_compatibility import QuantRS2Circuit

# Mock Qiskit components for testing when Qiskit is not available
class MockQiskitCircuit:
    def __init__(self, n_qubits):
        self.num_qubits = n_qubits
        self.data = []
        self._qubit_indices = {i: i for i in range(n_qubits)}
    
    def find_bit(self, qubit):
        class BitResult:
            def __init__(self, index):
                self.index = index
        return BitResult(self._qubit_indices[qubit])
    
    def h(self, qubit):
        self._add_gate('h', [qubit], [])
    
    def x(self, qubit):
        self._add_gate('x', [qubit], [])
    
    def cx(self, control, target):
        self._add_gate('cx', [control, target], [])
    
    def ry(self, angle, qubit):
        self._add_gate('ry', [qubit], [angle])
    
    def measure_all(self):
        for i in range(self.num_qubits):
            self._add_gate('measure', [i], [])
    
    def _add_gate(self, name, qubits, params):
        class MockInstruction:
            def __init__(self, name, qubits, params):
                self.operation = Mock()
                self.operation.name = name
                self.operation.params = params
                self.qubits = qubits
        
        self.data.append(MockInstruction(name, qubits, params))


class TestCircuitConverter(unittest.TestCase):
    """Test circuit conversion between QuantRS2 and Qiskit."""
    
    def setUp(self):
        self.converter = CircuitConverter()
    
    def test_converter_initialization(self):
        """Test converter initializes properly."""
        self.assertIsInstance(self.converter, CircuitConverter)
        self.assertIn('h', self.converter.gate_mapping_qiskit_to_quantrs2)
        self.assertIn('h', self.converter.gate_mapping_quantrs2_to_qiskit)
    
    def test_quantrs2_to_qiskit_basic_gates(self):
        """Test conversion of basic gates from QuantRS2 to Qiskit."""
        # Create simple QuantRS2 circuit
        circuit = QuantRS2Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Convert to Qiskit (will use mock when Qiskit not available)
        if not QISKIT_AVAILABLE:
            # Mock the conversion for testing
            result = self.converter.quantrs2_to_qiskit(circuit)
            self.assertIsNotNone(result)
        else:
            qiskit_circuit = self.converter.quantrs2_to_qiskit(circuit)
            self.assertEqual(qiskit_circuit.num_qubits, 2)
    
    def test_qiskit_to_quantrs2_basic_gates(self):
        """Test conversion from Qiskit to QuantRS2."""
        # Create mock Qiskit circuit
        qiskit_circuit = MockQiskitCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        
        # Convert to QuantRS2
        quantrs2_circuit = self.converter.qiskit_to_quantrs2(qiskit_circuit)
        self.assertEqual(quantrs2_circuit.n_qubits, 2)
        self.assertIsInstance(quantrs2_circuit, QuantRS2Circuit)
    
    def test_unsupported_gate_warning(self):
        """Test warning for unsupported gates."""
        qiskit_circuit = MockQiskitCircuit(1)
        qiskit_circuit._add_gate('unsupported_gate', [0], [])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.converter.qiskit_to_quantrs2(qiskit_circuit)
            self.assertTrue(any("Unsupported gate" in str(warning.message) for warning in w))
    
    def test_parametric_gates(self):
        """Test conversion of parametric gates."""
        qiskit_circuit = MockQiskitCircuit(1)
        qiskit_circuit.ry(np.pi/2, 0)
        
        quantrs2_circuit = self.converter.qiskit_to_quantrs2(qiskit_circuit)
        self.assertEqual(quantrs2_circuit.n_qubits, 1)


class TestQiskitBackendAdapter(unittest.TestCase):
    """Test Qiskit backend adapter."""
    
    def setUp(self):
        self.circuit = QuantRS2Circuit(2)
        self.circuit.h(0)
        self.circuit.cnot(0, 1)
    
    def test_adapter_initialization(self):
        """Test adapter initializes properly."""
        adapter = QiskitBackendAdapter()
        self.assertIsNotNone(adapter.backend)
        self.assertIsInstance(adapter.converter, CircuitConverter)
    
    def test_circuit_execution_mock(self):
        """Test circuit execution with mock backend."""
        adapter = QiskitBackendAdapter()
        result = adapter.execute(self.circuit, shots=1000)
        
        self.assertIsInstance(result, dict)
        self.assertIn('counts', result)
        self.assertIn('shots', result)
        self.assertEqual(result['shots'], 1000)
        self.assertTrue(result['success'])
    
    def test_execution_adds_measurements(self):
        """Test that measurements are added automatically."""
        adapter = QiskitBackendAdapter()
        result = adapter.execute(self.circuit)
        
        # Should succeed even without explicit measurements
        self.assertTrue(result['success'])
    
    @patch('quantrs2.qiskit_compatibility.QISKIT_AVAILABLE', False)
    def test_qiskit_unavailable_error(self):
        """Test error when Qiskit is not available."""
        with self.assertRaises(QiskitCompatibilityError):
            QiskitBackendAdapter()


class TestQiskitAlgorithmLibrary(unittest.TestCase):
    """Test Qiskit algorithm library."""
    
    def setUp(self):
        self.library = QiskitAlgorithmLibrary()
    
    def test_library_initialization(self):
        """Test library initializes properly."""
        self.assertIsInstance(self.library, QiskitAlgorithmLibrary)
        self.assertIsInstance(self.library.converter, CircuitConverter)
    
    def test_bell_state_creation(self):
        """Test Bell state creation."""
        bell_circuit = self.library.create_bell_state()
        self.assertEqual(bell_circuit.n_qubits, 2)
        self.assertIsInstance(bell_circuit, QuantRS2Circuit)
    
    def test_grover_oracle_creation(self):
        """Test Grover oracle creation."""
        oracle = self.library.create_grover_oracle(3, [5])
        self.assertEqual(oracle.n_qubits, 3)
        self.assertIsInstance(oracle, QuantRS2Circuit)
    
    def test_qft_creation(self):
        """Test QFT circuit creation."""
        qft_circuit = self.library.create_qft(3)
        self.assertEqual(qft_circuit.n_qubits, 3)
        self.assertIsInstance(qft_circuit, QuantRS2Circuit)
    
    def test_multiple_marked_items_grover(self):
        """Test Grover oracle with multiple marked items."""
        oracle = self.library.create_grover_oracle(2, [1, 3])
        self.assertEqual(oracle.n_qubits, 2)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_from_qiskit_function(self):
        """Test from_qiskit convenience function."""
        qiskit_circuit = MockQiskitCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        
        quantrs2_circuit = from_qiskit(qiskit_circuit)
        self.assertEqual(quantrs2_circuit.n_qubits, 2)
    
    def test_to_qiskit_function(self):
        """Test to_qiskit convenience function."""
        circuit = QuantRS2Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        qiskit_circuit = to_qiskit(circuit)
        self.assertIsNotNone(qiskit_circuit)
    
    def test_run_on_qiskit_backend_function(self):
        """Test run_on_qiskit_backend convenience function."""
        circuit = QuantRS2Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        result = run_on_qiskit_backend(circuit, shots=500)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['shots'], 500)


class TestQiskitCompatibleVQE(unittest.TestCase):
    """Test Qiskit-compatible VQE implementation."""
    
    def test_vqe_creation(self):
        """Test VQE algorithm creation."""
        vqe = create_qiskit_compatible_vqe("H2", ansatz_depth=2)
        self.assertIsNotNone(vqe)
        self.assertEqual(vqe.ansatz_depth, 2)
    
    def test_ansatz_creation(self):
        """Test VQE ansatz circuit creation."""
        vqe = create_qiskit_compatible_vqe("H2", ansatz_depth=2)
        parameters = [0.5, 1.0, 1.5, 2.0]
        
        ansatz = vqe.create_ansatz(parameters)
        self.assertIsInstance(ansatz, QuantRS2Circuit)
        self.assertEqual(ansatz.n_qubits, 2)
    
    def test_vqe_optimization(self):
        """Test VQE optimization process."""
        vqe = create_qiskit_compatible_vqe("H2", ansatz_depth=1)
        
        result = vqe.optimize()
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_parameters', result)
        self.assertIn('optimal_energy', result)
        self.assertIn('converged', result)
    
    def test_expectation_value_calculation(self):
        """Test expectation value calculation from counts."""
        vqe = create_qiskit_compatible_vqe("H2", ansatz_depth=1)
        
        counts = {'00': 500, '11': 500}
        expectation = vqe._compute_expectation_from_counts(counts)
        self.assertIsInstance(expectation, float)
        self.assertEqual(expectation, -1.0)  # All in computational basis states


class TestUtilityFunctions(unittest.TestCase):
    """Test utility and testing functions."""
    
    def test_conversion_fidelity_test(self):
        """Test conversion fidelity testing."""
        circuit = QuantRS2Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Should handle case when frameworks not available
        fidelity_result = test_conversion_fidelity(circuit)
        self.assertIsInstance(fidelity_result, bool)
    
    def test_benchmark_conversion_performance(self):
        """Test conversion performance benchmarking."""
        results = benchmark_conversion_performance()
        self.assertIsInstance(results, dict)
        
        for n_qubits, metrics in results.items():
            self.assertIn('conversion_time', metrics)
            self.assertIn('qubits', metrics)
            self.assertIsInstance(metrics['conversion_time'], float)
            self.assertEqual(metrics['qubits'], n_qubits)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_qiskit_compatibility_error(self):
        """Test QiskitCompatibilityError exception."""
        error = QiskitCompatibilityError("Test error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test error")
    
    def test_empty_circuit_conversion(self):
        """Test conversion of empty circuits."""
        empty_circuit = QuantRS2Circuit(1)
        qiskit_circuit = to_qiskit(empty_circuit)
        self.assertIsNotNone(qiskit_circuit)
    
    def test_large_circuit_conversion(self):
        """Test conversion of larger circuits."""
        large_circuit = QuantRS2Circuit(5)
        for i in range(5):
            large_circuit.h(i)
        for i in range(4):
            large_circuit.cnot(i, i+1)
        
        qiskit_circuit = to_qiskit(large_circuit)
        self.assertIsNotNone(qiskit_circuit)
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        vqe = create_qiskit_compatible_vqe("H2", ansatz_depth=2)
        
        # Test with insufficient parameters
        insufficient_params = [0.5]
        ansatz = vqe.create_ansatz(insufficient_params)
        self.assertIsInstance(ansatz, QuantRS2Circuit)


class TestMockComponents(unittest.TestCase):
    """Test mock components used for testing."""
    
    def test_mock_qiskit_circuit(self):
        """Test mock Qiskit circuit."""
        mock_circuit = MockQiskitCircuit(2)
        self.assertEqual(mock_circuit.num_qubits, 2)
        
        mock_circuit.h(0)
        mock_circuit.cx(0, 1)
        self.assertEqual(len(mock_circuit.data), 2)
    
    def test_mock_backend_execution(self):
        """Test mock backend execution."""
        from quantrs2.qiskit_compatibility import MockQiskitBackend
        
        backend = MockQiskitBackend()
        self.assertEqual(backend.name(), "mock_qiskit_backend")
        
        job = backend.run(None, shots=1000)
        result = job.result()
        counts = result.get_counts(None)
        
        self.assertIsInstance(counts, dict)
        self.assertEqual(sum(counts.values()), 1000)


class TestIntegration(unittest.TestCase):
    """Integration tests for the compatibility layer."""
    
    def test_full_workflow(self):
        """Test complete workflow with both frameworks."""
        # Create circuit
        circuit = QuantRS2Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Convert to Qiskit
        qiskit_circuit = to_qiskit(circuit)
        
        # Run on backend
        result = run_on_qiskit_backend(circuit, shots=1000)
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertEqual(result['shots'], 1000)
        self.assertTrue(result['success'])
    
    def test_algorithm_library_integration(self):
        """Test algorithm library integration."""
        library = QiskitAlgorithmLibrary()
        
        # Create different algorithm circuits
        bell_state = library.create_bell_state()
        grover_oracle = library.create_grover_oracle(2, [3])
        qft_circuit = library.create_qft(2)
        
        # Test they can all be converted
        for circuit in [bell_state, grover_oracle, qft_circuit]:
            qiskit_circuit = to_qiskit(circuit)
            self.assertIsNotNone(qiskit_circuit)
    
    def test_vqe_with_backend(self):
        """Test VQE with backend integration."""
        vqe = create_qiskit_compatible_vqe("H2", ansatz_depth=1)
        
        # Create mock backend
        from quantrs2.qiskit_compatibility import MockQiskitBackend
        backend = MockQiskitBackend()
        
        # Run optimization
        result = vqe.optimize(backend)
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimal_energy', result)


if __name__ == '__main__':
    # Configure test execution
    unittest.main(verbosity=2, buffer=True)