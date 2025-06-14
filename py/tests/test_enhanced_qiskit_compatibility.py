#!/usr/bin/env python3
"""Tests for enhanced Qiskit compatibility layer."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from quantrs2.enhanced_qiskit_compatibility import *


class MockQuantRS2Circuit:
    """Mock QuantRS2 circuit for testing."""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []
    
    def h(self, qubit): self.gates.append(('h', qubit))
    def x(self, qubit): self.gates.append(('x', qubit))
    def y(self, qubit): self.gates.append(('y', qubit))
    def z(self, qubit): self.gates.append(('z', qubit))
    def rx(self, qubit, angle): self.gates.append(('rx', qubit, angle))
    def ry(self, qubit, angle): self.gates.append(('ry', qubit, angle))
    def rz(self, qubit, angle): self.gates.append(('rz', qubit, angle))
    def cnot(self, control, target): self.gates.append(('cnot', control, target))
    def cz(self, control, target): self.gates.append(('cz', control, target))
    def s(self, qubit): self.gates.append(('s', qubit))
    def sdg(self, qubit): self.gates.append(('sdg', qubit))
    def t(self, qubit): self.gates.append(('t', qubit))
    def tdg(self, qubit): self.gates.append(('tdg', qubit))
    def swap(self, control, target): self.gates.append(('swap', control, target))
    def toffoli(self, c1, c2, target): self.gates.append(('toffoli', c1, c2, target))
    def crz(self, control, target, angle): self.gates.append(('crz', control, target, angle))


class MockQiskitCircuit:
    """Mock Qiskit circuit for testing."""
    
    class MockInstruction:
        def __init__(self, name, qubits, params=None):
            self.operation = Mock()
            self.operation.name = name
            self.operation.params = params or []
            self.qubits = qubits
    
    class MockQubit:
        def __init__(self, index):
            self.index = index
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.data = []
        self._qubits = [self.MockQubit(i) for i in range(num_qubits)]
    
    def find_bit(self, qubit):
        return Mock(index=qubit.index)
    
    def h(self, qubit): 
        self.data.append(self.MockInstruction('h', [self._qubits[qubit]]))
    
    def x(self, qubit):
        self.data.append(self.MockInstruction('x', [self._qubits[qubit]]))
    
    def cx(self, control, target):
        self.data.append(self.MockInstruction('cx', [self._qubits[control], self._qubits[target]]))
    
    def ry(self, angle, qubit):
        self.data.append(self.MockInstruction('ry', [self._qubits[qubit]], [angle]))
    
    def depth(self):
        return len(self.data)


@pytest.fixture
def mock_quantrs2_circuit():
    """Fixture providing mock QuantRS2 circuit."""
    with patch('quantrs2.enhanced_qiskit_compatibility.QuantRS2Circuit', MockQuantRS2Circuit):
        yield MockQuantRS2Circuit


class TestConversionOptions:
    """Test conversion options."""
    
    def test_default_options(self):
        """Test default conversion options."""
        options = ConversionOptions()
        assert options.mode == ConversionMode.EXACT
        assert options.compatibility_level == CompatibilityLevel.PERMISSIVE
        assert options.optimization_level == 1
        assert options.preserve_measurements is True
    
    def test_custom_options(self):
        """Test custom conversion options."""
        options = ConversionOptions(
            mode=ConversionMode.OPTIMIZED,
            compatibility_level=CompatibilityLevel.STRICT,
            optimization_level=3,
            basis_gates=['u3', 'cx']
        )
        assert options.mode == ConversionMode.OPTIMIZED
        assert options.compatibility_level == CompatibilityLevel.STRICT
        assert options.optimization_level == 3
        assert options.basis_gates == ['u3', 'cx']


class TestEnhancedCircuitConverter:
    """Test enhanced circuit converter."""
    
    @pytest.fixture
    def converter(self):
        """Converter fixture."""
        return EnhancedCircuitConverter()
    
    def test_initialization(self, converter):
        """Test converter initialization."""
        assert converter.options.mode == ConversionMode.EXACT
        assert len(converter.qiskit_to_quantrs2_gates) > 20
        assert len(converter.quantrs2_to_qiskit_gates) > 10
        assert converter.conversion_stats["successful_conversions"] == 0
    
    def test_gate_conversion_methods(self, converter, mock_quantrs2_circuit):
        """Test individual gate conversion methods."""
        circuit = mock_quantrs2_circuit(2)
        
        # Test single qubit gates
        converter._convert_h_gate(circuit, [0], [])
        converter._convert_x_gate(circuit, [0], [])
        converter._convert_ry_gate(circuit, [0], [np.pi/2])
        
        assert ('h', 0) in circuit.gates
        assert ('x', 0) in circuit.gates
        assert ('ry', 0, np.pi/2) in circuit.gates
        
        # Test two qubit gates
        converter._convert_cx_gate(circuit, [0, 1], [])
        converter._convert_cz_gate(circuit, [0, 1], [])
        
        assert ('cnot', 0, 1) in circuit.gates
        assert ('cz', 0, 1) in circuit.gates
    
    def test_u_gate_decomposition(self, converter, mock_quantrs2_circuit):
        """Test U gate decomposition."""
        circuit = mock_quantrs2_circuit(1)
        
        # Test U3 gate decomposition
        theta, phi, lam = np.pi/2, np.pi/4, np.pi/8
        converter._convert_u3_gate(circuit, [0], [theta, phi, lam])
        
        # Should have RZ, RY, RZ sequence
        rz_gates = [g for g in circuit.gates if g[0] == 'rz']
        ry_gates = [g for g in circuit.gates if g[0] == 'ry']
        
        assert len(rz_gates) == 2
        assert len(ry_gates) == 1
        assert ry_gates[0][2] == theta
    
    def test_controlled_gate_decomposition(self, converter, mock_quantrs2_circuit):
        """Test controlled gate decomposition."""
        circuit = mock_quantrs2_circuit(2)
        
        # Test controlled-RY decomposition
        theta = np.pi/3
        converter._convert_cry_gate(circuit, [0, 1], [theta])
        
        # Should have decomposed into RY and CNOT gates
        ry_gates = [g for g in circuit.gates if g[0] == 'ry']
        cnot_gates = [g for g in circuit.gates if g[0] == 'cnot']
        
        assert len(ry_gates) >= 2
        assert len(cnot_gates) >= 2
    
    def test_toffoli_decomposition(self, converter, mock_quantrs2_circuit):
        """Test Toffoli gate decomposition."""
        circuit = mock_quantrs2_circuit(3)
        
        # Mock toffoli method not available
        converter._decompose_toffoli(circuit, [0, 1, 2])
        
        # Should have many gates in decomposition
        assert len(circuit.gates) > 10
        
        # Should contain CNOTs, T, and H gates
        gate_types = set(g[0] for g in circuit.gates)
        assert 'cnot' in gate_types
        assert 't' in gate_types or 'tdg' in gate_types
        assert 'h' in gate_types
    
    def test_mcx_decomposition(self, converter, mock_quantrs2_circuit):
        """Test multi-controlled X decomposition."""
        circuit = mock_quantrs2_circuit(4)
        
        # Test 3-control X (should use Toffoli)
        converter._convert_mcx_gate(circuit, [0, 1, 2], [])
        assert len(circuit.gates) > 0
        
        # Test larger MCX
        circuit_large = mock_quantrs2_circuit(6)
        converter._decompose_mcx(circuit_large, [0, 1, 2, 3, 4])
        assert len(circuit_large.gates) > 0
    
    def test_qft_implementation(self, converter, mock_quantrs2_circuit):
        """Test QFT implementation."""
        circuit = mock_quantrs2_circuit(3)
        
        converter._convert_qft_gate(circuit, [0, 1, 2], [])
        
        # Should have Hadamards, controlled rotations, and swaps
        gate_types = set(g[0] for g in circuit.gates)
        assert 'h' in gate_types
        assert 'crz' in gate_types
        assert 'swap' in gate_types
    
    @patch('quantrs2.enhanced_qiskit_compatibility.QISKIT_AVAILABLE', True)
    @patch('quantrs2.enhanced_qiskit_compatibility.QUANTRS2_AVAILABLE', True)
    def test_qiskit_to_quantrs2_conversion(self, converter, mock_quantrs2_circuit):
        """Test Qiskit to QuantRS2 conversion."""
        # Create mock Qiskit circuit
        qiskit_circuit = MockQiskitCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        
        with patch('quantrs2.enhanced_qiskit_compatibility.QuantRS2Circuit', mock_quantrs2_circuit):
            quantrs2_circuit = converter.qiskit_to_quantrs2(qiskit_circuit)
            
            assert quantrs2_circuit.n_qubits == 2
            assert len(quantrs2_circuit.gates) == 2
            assert ('h', 0) in quantrs2_circuit.gates
            assert ('cnot', 0, 1) in quantrs2_circuit.gates
    
    def test_parse_quantrs2_gate_info(self, converter):
        """Test parsing QuantRS2 gate information."""
        # Single qubit gate
        qubits, params = converter._parse_quantrs2_gate_info(('h', 0))
        assert qubits == [0]
        assert params == []
        
        # Single qubit gate with parameter
        qubits, params = converter._parse_quantrs2_gate_info(('ry', 0, np.pi/2))
        assert qubits == [0]
        assert params == [np.pi/2]
        
        # Two qubit gate
        qubits, params = converter._parse_quantrs2_gate_info(('cnot', 0, 1))
        assert qubits == [0, 1]
        assert params == []
    
    def test_conversion_statistics(self, converter):
        """Test conversion statistics tracking."""
        initial_stats = converter.get_conversion_statistics()
        assert initial_stats["successful_conversions"] == 0
        
        # Simulate successful conversion
        converter.conversion_stats["successful_conversions"] += 1
        converter.conversion_stats["total_gates_converted"] += 5
        
        stats = converter.get_conversion_statistics()
        assert stats["successful_conversions"] == 1
        assert stats["total_gates_converted"] == 5
        
        # Reset statistics
        converter.reset_statistics()
        reset_stats = converter.get_conversion_statistics()
        assert reset_stats["successful_conversions"] == 0
    
    def test_strict_compatibility_mode(self, mock_quantrs2_circuit):
        """Test strict compatibility mode."""
        options = ConversionOptions(compatibility_level=CompatibilityLevel.STRICT)
        converter = EnhancedCircuitConverter(options)
        
        # This should raise an error for unsupported gates in strict mode
        qiskit_circuit = MockQiskitCircuit(1)
        qiskit_circuit.data.append(MockQiskitCircuit.MockInstruction('unsupported_gate', [qiskit_circuit._qubits[0]]))
        
        with patch('quantrs2.enhanced_qiskit_compatibility.QISKIT_AVAILABLE', True):
            with patch('quantrs2.enhanced_qiskit_compatibility.QUANTRS2_AVAILABLE', True):
                with patch('quantrs2.enhanced_qiskit_compatibility.QuantRS2Circuit', mock_quantrs2_circuit):
                    with pytest.raises(ValueError, match="Unsupported gate"):
                        converter.qiskit_to_quantrs2(qiskit_circuit)


class TestAdvancedQiskitIntegration:
    """Test advanced Qiskit integration features."""
    
    @pytest.fixture
    def integration(self):
        """Integration fixture."""
        return AdvancedQiskitIntegration()
    
    def test_initialization(self, integration):
        """Test integration initialization."""
        assert integration.converter is not None
        assert integration.logger is not None
    
    @patch('quantrs2.enhanced_qiskit_compatibility.QISKIT_AVAILABLE', True)
    def test_benchmark_circuit_size(self, integration):
        """Test circuit size benchmarking."""
        with patch('quantrs2.enhanced_qiskit_compatibility.QiskitCircuit', MockQiskitCircuit):
            result = integration._benchmark_circuit_size(2, 3, 2)
            
            assert "mean_conversion_time" in result
            assert "success_rate" in result
            assert 0 <= result["success_rate"] <= 1
    
    def test_create_hybrid_algorithm(self, integration):
        """Test hybrid algorithm creation."""
        qiskit_components = {"optimizer": "COBYLA"}
        quantrs2_components = {"ansatz_creator": lambda p: None}
        
        hybrid = integration.create_hybrid_algorithm(
            "hybrid_vqe", qiskit_components, quantrs2_components
        )
        
        assert isinstance(hybrid, HybridAlgorithm)
        assert hybrid.algorithm_type == "hybrid_vqe"
    
    @patch('quantrs2.enhanced_qiskit_compatibility.QISKIT_AVAILABLE', True)
    def test_optimize_for_hardware(self, integration, mock_quantrs2_circuit):
        """Test hardware optimization."""
        circuit = mock_quantrs2_circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        backend_properties = {
            "basis_gates": ["u3", "cx"],
            "coupling_map": [(0, 1)]
        }
        
        with patch('quantrs2.enhanced_qiskit_compatibility.QiskitCircuit', MockQiskitCircuit):
            with patch('quantrs2.enhanced_qiskit_compatibility.PassManager') as mock_pm:
                mock_pm.return_value.run.return_value = MockQiskitCircuit(2)
                
                optimized_circuit, stats = integration.optimize_for_hardware(
                    circuit, backend_properties
                )
                
                assert optimized_circuit is not None
                assert isinstance(stats, dict)
                assert "original_depth" in stats


class TestHybridAlgorithm:
    """Test hybrid algorithm functionality."""
    
    @pytest.fixture
    def hybrid_vqe(self):
        """Hybrid VQE fixture."""
        qiskit_components = {"optimizer": "COBYLA"}
        quantrs2_components = {
            "ansatz_creator": lambda params: MockQuantRS2Circuit(2)
        }
        converter = EnhancedCircuitConverter()
        
        return HybridAlgorithm("hybrid_vqe", qiskit_components, quantrs2_components, converter)
    
    def test_initialization(self, hybrid_vqe):
        """Test hybrid algorithm initialization."""
        assert hybrid_vqe.algorithm_type == "hybrid_vqe"
        assert "optimizer" in hybrid_vqe.qiskit_components
        assert "ansatz_creator" in hybrid_vqe.quantrs2_components
    
    def test_execute_hybrid_vqe(self, hybrid_vqe):
        """Test hybrid VQE execution."""
        result = hybrid_vqe.execute(
            hamiltonian=None,
            initial_params=[0.1, 0.2, 0.3, 0.4],
            max_iterations=10
        )
        
        assert "optimal_parameters" in result
        assert "optimal_energy" in result
        assert "algorithm_type" in result
        assert result["algorithm_type"] == "hybrid_vqe"
    
    def test_execute_unknown_algorithm(self):
        """Test execution of unknown algorithm type."""
        hybrid = HybridAlgorithm("unknown", {}, {}, EnhancedCircuitConverter())
        
        with pytest.raises(ValueError, match="Unknown algorithm type"):
            hybrid.execute()
    
    def test_mock_expectation_value(self, hybrid_vqe):
        """Test mock expectation value calculation."""
        params = [0.1, 0.2, 0.3]
        expectation = hybrid_vqe._mock_expectation_value(params)
        
        expected = sum(p**2 for p in params) - 1.0
        assert abs(expectation - expected) < 1e-10


class TestNoiseModelAdapter:
    """Test noise model adapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        mock_noise_model = Mock()
        adapter = NoiseModelAdapter(mock_noise_model)
        
        assert adapter.qiskit_noise_model == mock_noise_model
        assert adapter.logger is not None
    
    def test_convert_to_quantrs2_noise(self):
        """Test noise model conversion."""
        mock_noise_model = Mock()
        adapter = NoiseModelAdapter(mock_noise_model)
        
        quantrs2_noise = adapter.convert_to_quantrs2_noise()
        
        assert isinstance(quantrs2_noise, dict)
        assert "type" in quantrs2_noise
        assert quantrs2_noise["type"] == "qiskit_converted"
        assert "gate_errors" in quantrs2_noise
        assert "readout_errors" in quantrs2_noise
        assert "thermal_relaxation" in quantrs2_noise


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_enhanced_converter(self):
        """Test enhanced converter creation."""
        converter = create_enhanced_converter()
        assert isinstance(converter, EnhancedCircuitConverter)
        
        options = ConversionOptions(optimization_level=3)
        converter_with_options = create_enhanced_converter(options)
        assert converter_with_options.options.optimization_level == 3
    
    @patch('quantrs2.enhanced_qiskit_compatibility.AdvancedQiskitIntegration')
    def test_optimize_circuit_for_backend(self, mock_integration):
        """Test circuit optimization for backend."""
        mock_integration.return_value.optimize_for_hardware.return_value = (Mock(), {})
        
        circuit = Mock()
        optimized, stats = optimize_circuit_for_backend(circuit, "test_backend")
        
        assert optimized is not None
        assert isinstance(stats, dict)
    
    @patch('quantrs2.enhanced_qiskit_compatibility.AdvancedQiskitIntegration')
    def test_benchmark_conversion_performance(self, mock_integration):
        """Test conversion performance benchmarking."""
        mock_integration.return_value.benchmark_frameworks.return_value = {"2q_2d": {"mean_time": 0.001}}
        
        results = benchmark_conversion_performance(max_qubits=4, max_depth=5)
        
        assert isinstance(results, dict)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_converter_without_qiskit(self):
        """Test converter behavior when Qiskit is not available."""
        with patch('quantrs2.enhanced_qiskit_compatibility.QISKIT_AVAILABLE', False):
            converter = EnhancedCircuitConverter()
            
            with pytest.raises(ImportError, match="Qiskit not available"):
                converter.qiskit_to_quantrs2(Mock())
    
    def test_converter_without_quantrs2(self):
        """Test converter behavior when QuantRS2 is not available."""
        with patch('quantrs2.enhanced_qiskit_compatibility.QISKIT_AVAILABLE', True):
            with patch('quantrs2.enhanced_qiskit_compatibility.QUANTRS2_AVAILABLE', False):
                converter = EnhancedCircuitConverter()
                
                with pytest.raises(ImportError, match="QuantRS2 not available"):
                    converter.qiskit_to_quantrs2(MockQiskitCircuit(2))
    
    def test_conversion_failure_handling(self, mock_quantrs2_circuit):
        """Test handling of conversion failures."""
        converter = EnhancedCircuitConverter()
        
        # Mock a failing gate conversion
        def failing_gate_conversion(circuit, qubits, params):
            raise RuntimeError("Gate conversion failed")
        
        converter.qiskit_to_quantrs2_gates['failing_gate'] = failing_gate_conversion
        
        qiskit_circuit = MockQiskitCircuit(1)
        qiskit_circuit.data.append(MockQiskitCircuit.MockInstruction('failing_gate', [qiskit_circuit._qubits[0]]))
        
        with patch('quantrs2.enhanced_qiskit_compatibility.QISKIT_AVAILABLE', True):
            with patch('quantrs2.enhanced_qiskit_compatibility.QUANTRS2_AVAILABLE', True):
                with patch('quantrs2.enhanced_qiskit_compatibility.QuantRS2Circuit', mock_quantrs2_circuit):
                    # Should handle the failure gracefully in permissive mode
                    result = converter.qiskit_to_quantrs2(qiskit_circuit)
                    assert result is not None
    
    def test_invalid_gate_parameters(self, converter, mock_quantrs2_circuit):
        """Test handling of invalid gate parameters."""
        circuit = mock_quantrs2_circuit(1)
        
        # Test with missing parameters
        with pytest.raises((IndexError, TypeError)):
            converter._convert_ry_gate(circuit, [0], [])  # Missing angle parameter
        
        # Test with too many parameters for simple gates
        converter._convert_h_gate(circuit, [0], [1.0, 2.0])  # Extra parameters should be ignored


if __name__ == "__main__":
    pytest.main([__file__])