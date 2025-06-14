#!/usr/bin/env python3
"""Tests for enhanced PennyLane plugin."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from quantrs2.enhanced_pennylane_plugin import *


class TestDeviceConfig:
    """Test device configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DeviceConfig()
        assert config.mode == DeviceMode.STATEVECTOR
        assert config.gradient_method == GradientMethod.PARAMETER_SHIFT
        assert config.shots is None
        assert config.noise_model is None
        assert config.error_mitigation is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DeviceConfig(
            mode=DeviceMode.SAMPLING,
            gradient_method=GradientMethod.FINITE_DIFF,
            shots=1024,
            optimization_level=2
        )
        assert config.mode == DeviceMode.SAMPLING
        assert config.gradient_method == GradientMethod.FINITE_DIFF
        assert config.shots == 1024
        assert config.optimization_level == 2


class TestEnhancedQuantRS2Device:
    """Test enhanced QuantRS2 device."""
    
    @pytest.fixture
    def device(self):
        """Device fixture."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        config = DeviceConfig(shots=1024)
        return EnhancedQuantRS2Device(wires=2, config=config)
    
    def test_initialization(self, device):
        """Test device initialization."""
        assert device.num_wires == 2
        assert device.config.shots == 1024
        assert isinstance(device.state, np.ndarray)
    
    def test_capabilities(self, device):
        """Test device capabilities."""
        caps = device.capabilities()
        assert "returns_state" in caps
        assert "supports_finite_shots" in caps
        assert "supports_analytic_computation" in caps
    
    def test_operations(self, device):
        """Test supported operations."""
        ops = device.operations
        expected_ops = {"PauliX", "PauliY", "PauliZ", "Hadamard", "RX", "RY", "RZ", "CNOT"}
        assert expected_ops.issubset(set(ops))
    
    def test_observables(self, device):
        """Test supported observables."""
        obs = device.observables
        expected_obs = {"PauliX", "PauliY", "PauliZ", "Identity", "Hermitian"}
        assert expected_obs.issubset(set(obs))
    
    def test_state_vector_mode(self):
        """Test state vector execution mode."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        config = DeviceConfig(mode=DeviceMode.STATEVECTOR)
        device = EnhancedQuantRS2Device(wires=2, config=config)
        
        # Test initial state
        assert device.state.shape == (4,)
        assert np.allclose(device.state, [1, 0, 0, 0])
    
    def test_sampling_mode(self):
        """Test sampling execution mode."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        config = DeviceConfig(mode=DeviceMode.SAMPLING, shots=100)
        device = EnhancedQuantRS2Device(wires=2, config=config)
        
        assert device.config.shots == 100
        assert device.config.mode == DeviceMode.SAMPLING


class TestQuantRS2QMLModel:
    """Test quantum machine learning model."""
    
    @pytest.fixture
    def qml_model(self):
        """QML model fixture."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        config = DeviceConfig(mode=DeviceMode.STATEVECTOR)
        return QuantRS2QMLModel(n_qubits=2, n_layers=2, config=config)
    
    def test_initialization(self, qml_model):
        """Test model initialization."""
        assert qml_model.n_qubits == 2
        assert qml_model.n_layers == 2
        assert qml_model.n_params > 0
    
    def test_parameter_count(self, qml_model):
        """Test parameter count calculation."""
        # For 2 qubits, 2 layers: each layer has 2 rotations per qubit + entangling
        expected_params = 2 * 2 * 2  # n_qubits * n_layers * rotations_per_qubit
        assert qml_model.n_params == expected_params
    
    def test_forward_pass(self, qml_model):
        """Test forward pass."""
        params = np.random.random(qml_model.n_params)
        input_data = np.array([[0.5, 0.5], [0.1, 0.9]])
        
        output = qml_model.forward(params, input_data)
        assert output.shape == (2,)  # Two samples
        assert all(-1 <= x <= 1 for x in output)  # Valid expectation values
    
    def test_gradient_computation(self, qml_model):
        """Test gradient computation."""
        params = np.random.random(qml_model.n_params)
        input_data = np.array([[0.5, 0.5]])
        
        grad = qml_model.compute_gradient(params, input_data)
        assert grad.shape == (qml_model.n_params,)
        assert not np.allclose(grad, 0)  # Non-zero gradients


class TestQuantRS2VQC:
    """Test variational quantum classifier."""
    
    @pytest.fixture
    def vqc(self):
        """VQC fixture."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        config = DeviceConfig()
        return QuantRS2VQC(n_qubits=2, n_layers=1, config=config)
    
    def test_initialization(self, vqc):
        """Test VQC initialization."""
        assert vqc.n_qubits == 2
        assert vqc.n_layers == 1
        assert vqc.trained is False
    
    def test_fit_method(self, vqc):
        """Test fit method."""
        # Create simple training data
        X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        y = np.array([1, 1, -1, -1])  # XOR-like problem
        
        initial_params = vqc.get_initial_parameters()
        loss_history = vqc.fit(X, y, initial_params, max_iterations=5)
        
        assert len(loss_history) <= 5
        assert vqc.trained is True
        assert vqc.optimal_params is not None
    
    def test_predict_method(self, vqc):
        """Test predict method."""
        # Create dummy optimal parameters
        vqc.optimal_params = np.random.random(vqc.n_params)
        vqc.trained = True
        
        X_test = np.array([[0.5, 0.5], [0.2, 0.8]])
        predictions = vqc.predict(X_test)
        
        assert predictions.shape == (2,)
        assert all(p in [-1, 1] for p in predictions)
    
    def test_predict_untrained_error(self, vqc):
        """Test error when predicting with untrained model."""
        X_test = np.array([[0.5, 0.5]])
        
        with pytest.raises(ValueError, match="Model has not been trained"):
            vqc.predict(X_test)


class TestEnhancedPennyLaneIntegration:
    """Test enhanced PennyLane integration."""
    
    @pytest.fixture
    def integration(self):
        """Integration fixture."""
        return EnhancedPennyLaneIntegration()
    
    def test_initialization(self, integration):
        """Test integration initialization."""
        assert integration.logger is not None
        assert hasattr(integration, 'device_registry')
    
    def test_create_device(self, integration):
        """Test device creation."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        config = DeviceConfig(shots=512)
        device = integration.create_device(wires=3, config=config)
        
        assert device.num_wires == 3
        assert device.config.shots == 512
    
    def test_register_device(self, integration):
        """Test device registration."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        device = Mock()
        device.name = "test_device"
        
        integration.register_device("test_device", device)
        assert "test_device" in integration.device_registry
    
    def test_benchmark_device_performance(self, integration):
        """Test device performance benchmarking."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        results = integration.benchmark_device_performance(
            max_qubits=3, max_layers=2, num_trials=2
        )
        
        assert isinstance(results, dict)
        assert "timing_results" in results
        assert "memory_usage" in results
        assert "success_rate" in results
    
    def test_create_quantum_neural_network(self, integration):
        """Test quantum neural network creation."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        qnn = integration.create_quantum_neural_network(
            n_qubits=2, n_layers=1, learning_rate=0.01
        )
        
        assert hasattr(qnn, 'n_qubits')
        assert hasattr(qnn, 'n_layers')
        assert hasattr(qnn, 'learning_rate')
    
    def test_optimize_quantum_circuit(self, integration):
        """Test quantum circuit optimization."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        # Create a simple mock circuit
        mock_circuit = Mock()
        mock_circuit.n_qubits = 2
        mock_circuit.depth = 10
        
        optimized_circuit = integration.optimize_quantum_circuit(
            mock_circuit, optimization_level=2
        )
        
        assert optimized_circuit is not None


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_enhanced_pennylane_device(self):
        """Test enhanced device creation function."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        device = create_enhanced_pennylane_device(wires=2, shots=1024)
        assert device.num_wires == 2
        assert device.config.shots == 1024
    
    def test_create_qml_model(self):
        """Test QML model creation function."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        model = create_qml_model(n_qubits=3, n_layers=2)
        assert model.n_qubits == 3
        assert model.n_layers == 2
    
    def test_register_enhanced_device(self):
        """Test enhanced device registration."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        # This should not raise an error
        register_enhanced_device()
    
    def test_benchmark_pennylane_performance(self):
        """Test PennyLane performance benchmarking."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        results = benchmark_pennylane_performance(max_qubits=2, max_layers=1)
        assert isinstance(results, dict)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_device_without_pennylane(self):
        """Test device creation without PennyLane."""
        if PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane is available")
        
        with pytest.warns(UserWarning, match="PennyLane not available"):
            # This should use mock implementation
            device = EnhancedQuantRS2Device(wires=2)
            assert hasattr(device, 'capabilities')
    
    def test_invalid_device_config(self):
        """Test invalid device configuration."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        config = DeviceConfig(shots=-1)  # Invalid shots
        device = EnhancedQuantRS2Device(wires=2, config=config)
        
        # Should handle gracefully or raise appropriate error
        assert device is not None
    
    def test_gradient_computation_failure(self):
        """Test gradient computation failure handling."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        model = QuantRS2QMLModel(n_qubits=2, n_layers=1)
        
        # Test with invalid input
        with pytest.raises((ValueError, TypeError)):
            model.compute_gradient(None, np.array([[0.5, 0.5]]))
    
    def test_vqc_with_mismatched_data(self):
        """Test VQC with mismatched input dimensions."""
        if not PENNYLANE_AVAILABLE:
            pytest.skip("PennyLane not available")
        
        vqc = QuantRS2VQC(n_qubits=2, n_layers=1)
        
        # Wrong input dimension
        X = np.array([[0, 1, 2]])  # 3D input for 2-qubit system
        y = np.array([1])
        
        with pytest.raises((ValueError, IndexError)):
            vqc.fit(X, y, vqc.get_initial_parameters())


class TestMockImplementations:
    """Test mock implementations when dependencies are not available."""
    
    def test_mock_device_capabilities(self):
        """Test mock device capabilities."""
        if PENNYLANE_AVAILABLE:
            # Mock the import failure
            with patch('quantrs2.enhanced_pennylane_plugin.PENNYLANE_AVAILABLE', False):
                device = EnhancedQuantRS2Device(wires=2)
                caps = device.capabilities()
                assert isinstance(caps, dict)
    
    def test_mock_qml_model_forward(self):
        """Test mock QML model forward pass."""
        if PENNYLANE_AVAILABLE:
            # Mock the import failure
            with patch('quantrs2.enhanced_pennylane_plugin.PENNYLANE_AVAILABLE', False):
                model = QuantRS2QMLModel(n_qubits=2, n_layers=1)
                params = np.random.random(4)
                input_data = np.array([[0.5, 0.5]])
                
                output = model.forward(params, input_data)
                assert output.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])