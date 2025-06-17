"""
Tests for PennyLane plugin integration.
"""

import unittest
import warnings
import numpy as np
from unittest.mock import Mock, patch

# Safe import pattern for PennyLane plugin
HAS_PENNYLANE_PLUGIN = True
try:
    from quantrs2.pennylane_plugin import (
        QuantRS2Device,
        QuantRS2QMLModel,
        QuantRS2VQC,
        QuantRS2PennyLaneError,
        register_quantrs2_device,
        create_quantrs2_device,
        quantrs2_qnode,
        check_quantrs2_pennylane_integration,
        PENNYLANE_AVAILABLE,
        QUANTRS2_AVAILABLE
    )
except ImportError as e:
    HAS_PENNYLANE_PLUGIN = False
    
    # Create stub implementations
    class QuantRS2Device:
        def __init__(self, *args, **kwargs):
            pass
    
    class QuantRS2QMLModel:
        def __init__(self, *args, **kwargs):
            pass
    
    class QuantRS2VQC:
        def __init__(self, *args, **kwargs):
            pass
    
    class QuantRS2PennyLaneError(Exception):
        pass
    
    def register_quantrs2_device():
        pass
    
    def create_quantrs2_device(*args, **kwargs):
        return QuantRS2Device()
    
    def quantrs2_qnode(*args, **kwargs):
        pass
    
    def check_quantrs2_pennylane_integration():
        return False
    
    PENNYLANE_AVAILABLE = False
    QUANTRS2_AVAILABLE = False

# Mock PennyLane components when not available
if not PENNYLANE_AVAILABLE:
    class MockQNode:
        def __init__(self, func, device):
            self.func = func
            self.device = device
        def __call__(self, *args, **kwargs):
            return np.array([0.0, 0.0])
    
    class MockOperation:
        def __init__(self, name, wires, parameters=None):
            self.name = name
            self.wires = wires
            self.parameters = parameters or []
    
    # Patch qml if not available
    import sys
    qml_mock = Mock()
    qml_mock.QNode = MockQNode
    qml_mock.expval = lambda obs: 0.0
    qml_mock.PauliZ = lambda wires: MockOperation("PauliZ", [wires])
    qml_mock.PauliX = lambda wires: MockOperation("PauliX", [wires])
    qml_mock.PauliY = lambda wires: MockOperation("PauliY", [wires])
    qml_mock.Hadamard = lambda wires: MockOperation("Hadamard", [wires])
    qml_mock.RX = lambda angle, wires: MockOperation("RX", [wires], [angle])
    qml_mock.RY = lambda angle, wires: MockOperation("RY", [wires], [angle])
    qml_mock.RZ = lambda angle, wires: MockOperation("RZ", [wires], [angle])
    qml_mock.CNOT = lambda wires: MockOperation("CNOT", wires)
    sys.modules['pennylane'] = qml_mock


@unittest.skipIf(not HAS_PENNYLANE_PLUGIN, "quantrs2.pennylane_plugin not available")
class TestQuantRS2Device(unittest.TestCase):
    """Test QuantRS2Device class."""
    
    def setUp(self):
        """Set up test device."""
        if not PENNYLANE_AVAILABLE:
            self.skipTest("PennyLane not available")
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_device_initialization(self):
        """Test device initialization."""
        device = QuantRS2Device(wires=2)
        
        self.assertEqual(device.n_qubits, 2)
        self.assertEqual(len(device.wires), 2)
        self.assertIsNone(device.shots)
        self.assertIsNone(device._circuit)
        self.assertIsNone(device._state)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_device_with_shots(self):
        """Test device initialization with shots."""
        device = QuantRS2Device(wires=3, shots=1000)
        
        self.assertEqual(device.n_qubits, 3)
        self.assertEqual(device.shots, 1000)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', False)
    def test_device_quantrs2_unavailable(self):
        """Test device initialization when QuantRS2 unavailable."""
        with self.assertRaises(QuantRS2PennyLaneError):
            QuantRS2Device(wires=2)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', False)
    def test_device_pennylane_unavailable(self):
        """Test device initialization when PennyLane unavailable."""
        with self.assertRaises(QuantRS2PennyLaneError):
            QuantRS2Device(wires=2)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_device_reset(self):
        """Test device reset functionality."""
        device = QuantRS2Device(wires=2)
        
        # Set some state
        device._circuit = Mock()
        device._state = np.array([1, 0, 0, 0])
        
        # Reset
        device.reset()
        
        self.assertIsNone(device._circuit)
        self.assertIsNone(device._state)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_apply_operations(self):
        """Test applying operations to device."""
        device = QuantRS2Device(wires=2)
        
        # Mock operations
        h_op = MockOperation("Hadamard", [0])
        cnot_op = MockOperation("CNOT", [0, 1])
        
        operations = [h_op, cnot_op]
        
        # This should not raise an error
        device.apply(operations)
        
        # Circuit should be created
        self.assertIsNotNone(device._circuit)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_single_qubit_operations(self):
        """Test single-qubit operation conversion."""
        device = QuantRS2Device(wires=2)
        
        # Test various single-qubit operations
        operations = [
            MockOperation("PauliX", [0]),
            MockOperation("PauliY", [1]),
            MockOperation("PauliZ", [0]),
            MockOperation("Hadamard", [1]),
            MockOperation("S", [0]),
            MockOperation("T", [1])
        ]
        
        device.apply(operations)
        
        # Should not raise errors
        self.assertIsNotNone(device._circuit)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_parametric_operations(self):
        """Test parametric operation conversion."""
        device = QuantRS2Device(wires=2)
        
        operations = [
            MockOperation("RX", [0], [np.pi/2]),
            MockOperation("RY", [1], [np.pi/4]),
            MockOperation("RZ", [0], [np.pi]),
            MockOperation("PhaseShift", [1], [np.pi/3])
        ]
        
        device.apply(operations)
        self.assertIsNotNone(device._circuit)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_two_qubit_operations(self):
        """Test two-qubit operation conversion."""
        device = QuantRS2Device(wires=2)
        
        operations = [
            MockOperation("CNOT", [0, 1]),
            MockOperation("CZ", [1, 0]),
            MockOperation("SWAP", [0, 1]),
            MockOperation("ControlledPhaseShift", [0, 1], [np.pi/2])
        ]
        
        device.apply(operations)
        self.assertIsNotNone(device._circuit)
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_unsupported_operation(self):
        """Test handling of unsupported operations."""
        device = QuantRS2Device(wires=2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            operations = [MockOperation("UnsupportedGate", [0])]
            device.apply(operations)
            
            # Should issue a warning
            self.assertTrue(len(w) > 0)
            self.assertIn("not supported", str(w[0].message))


@unittest.skipIf(not HAS_PENNYLANE_PLUGIN, "quantrs2.pennylane_plugin not available")
class TestQuantRS2QMLModel(unittest.TestCase):
    """Test QuantRS2QMLModel class."""
    
    def setUp(self):
        """Set up test model."""
        if not PENNYLANE_AVAILABLE:
            self.skipTest("PennyLane not available")
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_model_initialization(self):
        """Test QML model initialization."""
        model = QuantRS2QMLModel(n_qubits=2, n_layers=2)
        
        self.assertEqual(model.n_qubits, 2)
        self.assertEqual(model.n_layers, 2)
        self.assertIsNotNone(model.device)
        self.assertIsNotNone(model.qnode)
        self.assertIsNone(model.params)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        model = QuantRS2QMLModel(n_qubits=2, n_layers=1)
        
        params = model.initialize_params(seed=42)
        
        # Should have 3 parameters per qubit per layer (RX, RY, RZ)
        expected_params = 2 * 1 * 3  # 2 qubits, 1 layer, 3 params each
        self.assertEqual(len(params), expected_params)
        self.assertIsNotNone(model.params)
        
        # Test reproducibility
        params2 = model.initialize_params(seed=42)
        np.testing.assert_array_equal(params, params2)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_forward_pass(self):
        """Test forward pass through model."""
        model = QuantRS2QMLModel(n_qubits=2, n_layers=1)
        model.initialize_params(seed=42)
        
        # Test input
        x = np.array([0.5, 1.0])
        
        # Forward pass
        output = model.forward(x)
        
        # Should return output from 2 qubits
        self.assertEqual(len(output), 2)
        self.assertTrue(all(isinstance(val, (int, float, complex, np.number)) for val in output))
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_forward_without_params(self):
        """Test forward pass without initialized parameters."""
        model = QuantRS2QMLModel(n_qubits=2, n_layers=1)
        
        x = np.array([0.5, 1.0])
        
        with self.assertRaises(QuantRS2PennyLaneError):
            model.forward(x)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_prediction(self):
        """Test model prediction."""
        model = QuantRS2QMLModel(n_qubits=2, n_layers=1)
        model.initialize_params(seed=42)
        
        # Test data
        X = np.array([[0.5, 1.0], [1.5, 0.5], [0.0, 2.0]])
        
        predictions = model.predict(X)
        
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_training(self):
        """Test model training."""
        model = QuantRS2QMLModel(n_qubits=2, n_layers=1)
        
        # Generate simple training data
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        y = np.array([1, -1, 0])
        
        # Train for a few epochs
        history = model.train(X, y, n_epochs=3, learning_rate=0.1)
        
        self.assertEqual(len(history), 3)
        self.assertTrue(all(isinstance(cost, (int, float)) for cost in history))
        self.assertIsNotNone(model.params)


@unittest.skipIf(not HAS_PENNYLANE_PLUGIN, "quantrs2.pennylane_plugin not available")
class TestQuantRS2VQC(unittest.TestCase):
    """Test QuantRS2VQC class."""
    
    def setUp(self):
        """Set up test VQC."""
        if not PENNYLANE_AVAILABLE:
            self.skipTest("PennyLane not available")
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_vqc_initialization(self):
        """Test VQC initialization."""
        vqc = QuantRS2VQC(n_features=3, n_qubits=3, n_layers=2)
        
        self.assertEqual(vqc.n_features, 3)
        self.assertEqual(vqc.n_qubits, 3)
        self.assertEqual(vqc.n_layers, 2)
        self.assertIsNotNone(vqc.model)
        self.assertFalse(vqc.is_trained)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_vqc_default_qubits(self):
        """Test VQC with default number of qubits."""
        vqc = QuantRS2VQC(n_features=4)
        
        self.assertEqual(vqc.n_features, 4)
        self.assertEqual(vqc.n_qubits, 4)  # Should default to n_features
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_feature_normalization(self):
        """Test feature normalization."""
        vqc = QuantRS2VQC(n_features=2)
        
        # Test data with different ranges
        X = np.array([[0, 10], [5, 20], [10, 30]])
        
        X_normalized = vqc._normalize_features(X)
        
        # Should be normalized to [0, π]
        self.assertTrue(np.all(X_normalized >= 0))
        self.assertTrue(np.all(X_normalized <= np.pi))
        
        # Check that min and max are mapped correctly
        self.assertAlmostEqual(np.min(X_normalized, axis=0)[0], 0)
        self.assertAlmostEqual(np.max(X_normalized, axis=0)[0], np.pi)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_vqc_training(self):
        """Test VQC training."""
        vqc = QuantRS2VQC(n_features=2, n_qubits=2, n_layers=1)
        
        # Simple binary classification data
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([1, 1, -1, -1])
        
        # Train for a few epochs
        history = vqc.fit(X, y, n_epochs=3, learning_rate=0.1)
        
        self.assertTrue(vqc.is_trained)
        self.assertEqual(len(history), 3)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_vqc_prediction(self):
        """Test VQC prediction."""
        vqc = QuantRS2VQC(n_features=2, n_qubits=2, n_layers=1)
        
        # Train first
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([1, -1])
        vqc.fit(X_train, y_train, n_epochs=2)
        
        # Test prediction
        X_test = np.array([[0.5, 0.5], [0.1, 0.1]])
        predictions = vqc.predict(X_test)
        
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_vqc_score(self):
        """Test VQC scoring."""
        vqc = QuantRS2VQC(n_features=2, n_qubits=2, n_layers=1)
        
        # Train first
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([1, 1, -1, -1])
        vqc.fit(X, y, n_epochs=2)
        
        # Test scoring
        score = vqc.score(X, y)
        
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_vqc_untrained_prediction(self):
        """Test prediction on untrained VQC."""
        vqc = QuantRS2VQC(n_features=2)
        
        X = np.array([[0, 0], [1, 1]])
        
        with self.assertRaises(QuantRS2PennyLaneError):
            vqc.predict(X)


@unittest.skipIf(not HAS_PENNYLANE_PLUGIN, "quantrs2.pennylane_plugin not available")
class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_register_device(self):
        """Test device registration."""
        # This should not raise an error
        register_quantrs2_device()
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_create_device(self):
        """Test device creation utility."""
        device = create_quantrs2_device(wires=3, shots=1000)
        
        self.assertIsInstance(device, QuantRS2Device)
        self.assertEqual(device.n_qubits, 3)
        self.assertEqual(device.shots, 1000)
    
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', False)
    def test_qnode_pennylane_unavailable(self):
        """Test QNode creation when PennyLane unavailable."""
        def dummy_circuit():
            pass
        
        with self.assertRaises(QuantRS2PennyLaneError):
            quantrs2_qnode(dummy_circuit, wires=2)


@unittest.skipIf(not HAS_PENNYLANE_PLUGIN, "quantrs2.pennylane_plugin not available")
class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_integration_test_function(self):
        """Test the integration test function."""
        # This should handle missing dependencies gracefully
        result = check_quantrs2_pennylane_integration()
        
        # Should return boolean
        self.assertIsInstance(result, bool)
        
        # If dependencies are missing, should print message and return False
        if not (PENNYLANE_AVAILABLE and QUANTRS2_AVAILABLE):
            self.assertFalse(result)


@unittest.skipIf(not HAS_PENNYLANE_PLUGIN, "quantrs2.pennylane_plugin not available")
class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_quantrs2_pennylane_error(self):
        """Test custom exception."""
        error = QuantRS2PennyLaneError("Test error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test error")
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True)
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_device_operations_without_circuit(self):
        """Test device operations without applied circuit."""
        device = QuantRS2Device(wires=2)
        
        # Should raise error when trying to generate samples without circuit
        with self.assertRaises(QuantRS2PennyLaneError):
            device.generate_samples()
        
        # Should raise error when trying to get expectation without state
        with self.assertRaises(QuantRS2PennyLaneError):
            mock_observable = Mock()
            device.expval(mock_observable, [0], [])
    
    @patch('quantrs2.pennylane_plugin.QUANTRS2_AVAILABLE', True) 
    @patch('quantrs2.pennylane_plugin.PENNYLANE_AVAILABLE', True)
    def test_empty_feature_normalization(self):
        """Test feature normalization with edge cases."""
        vqc = QuantRS2VQC(n_features=2)
        
        # Test with identical features (zero range)
        X = np.array([[1, 1], [1, 1], [1, 1]])
        X_normalized = vqc._normalize_features(X)
        
        # Should handle zero range gracefully
        self.assertEqual(X_normalized.shape, X.shape)
        self.assertTrue(np.all(np.isfinite(X_normalized)))


@unittest.skipIf(not HAS_PENNYLANE_PLUGIN, "quantrs2.pennylane_plugin not available")
class TestMockComponents(unittest.TestCase):
    """Test mock components used when dependencies unavailable."""
    
    def test_mock_operation(self):
        """Test mock operation creation."""
        if not PENNYLANE_AVAILABLE:
            from quantrs2.pennylane_plugin import MockOperation
            
            op = MockOperation("TestGate", [0, 1], [np.pi/2])
            
            self.assertEqual(op.name, "TestGate")
            self.assertEqual(op.wires, [0, 1])
            self.assertEqual(op.parameters, [np.pi/2])
    
    def test_mock_qnode(self):
        """Test mock QNode functionality."""
        if not PENNYLANE_AVAILABLE:
            from quantrs2.pennylane_plugin import MockQNode
            
            def dummy_func():
                return "test"
            
            qnode = MockQNode(dummy_func, Mock())
            result = qnode()
            
            self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main(verbosity=2)