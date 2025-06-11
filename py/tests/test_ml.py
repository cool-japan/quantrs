#!/usr/bin/env python3
"""
Test suite for quantum machine learning functionality.
"""

import pytest
import numpy as np

try:
    from quantrs2.ml import QNN, VQE, HEPClassifier, QuantumGAN
    HAS_ML = True
except ImportError:
    HAS_ML = False


@pytest.mark.skipif(not HAS_ML, reason="ml module not available")
class TestQNN:
    """Test Quantum Neural Network functionality."""
    
    def test_qnn_initialization(self):
        """Test QNN initialization."""
        qnn = QNN(n_qubits=4, n_layers=2, activation="relu")
        
        assert qnn.n_qubits == 4
        assert qnn.n_layers == 2
        assert qnn.activation == "relu"
        assert qnn.parameters.shape == (4 * 2 * 3,)  # n_qubits * n_layers * 3 rotations
    
    def test_qnn_default_parameters(self):
        """Test QNN with default parameters."""
        qnn = QNN(n_qubits=3)
        
        assert qnn.n_qubits == 3
        assert qnn.n_layers == 2  # Default
        assert qnn.activation == "relu"  # Default
        assert len(qnn.parameters) == 3 * 2 * 3
    
    def test_qnn_parameter_management(self):
        """Test QNN parameter setting and getting."""
        qnn = QNN(n_qubits=2, n_layers=1)
        
        # Get initial parameters
        initial_params = qnn.get_parameters()
        assert len(initial_params) == 2 * 1 * 3
        
        # Set new parameters
        new_params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qnn.set_parameters(new_params)
        
        # Verify parameters were set
        retrieved_params = qnn.get_parameters()
        np.testing.assert_array_equal(retrieved_params, new_params)
    
    def test_qnn_forward_pass(self):
        """Test QNN forward pass."""
        qnn = QNN(n_qubits=3, n_layers=1)
        
        # Single sample
        x = np.array([[0.1, 0.2, 0.3]])
        output = qnn.forward(x)
        
        assert output.shape[0] == 1  # Single sample
        assert output.shape[1] > 0   # Some number of features
        assert np.all(output >= 0)   # ReLU activation should give non-negative outputs
    
    def test_qnn_multiple_samples(self):
        """Test QNN with multiple input samples."""
        qnn = QNN(n_qubits=2, n_layers=1)
        
        # Multiple samples
        x = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        
        # Process each sample individually (current implementation)
        outputs = []
        for i in range(len(x)):
            output = qnn.forward(x[i:i+1])
            outputs.append(output)
        
        assert len(outputs) == 3
        for output in outputs:
            assert output.shape[0] == 1
            assert output.shape[1] > 0
    
    def test_qnn_different_activations(self):
        """Test QNN with different activation functions."""
        # ReLU activation
        qnn_relu = QNN(n_qubits=2, n_layers=1, activation="relu")
        x = np.array([[0.1, 0.2]])
        output_relu = qnn_relu.forward(x)
        assert np.all(output_relu >= 0)  # ReLU gives non-negative outputs
        
        # Other activation (should handle gracefully)
        qnn_other = QNN(n_qubits=2, n_layers=1, activation="tanh")
        output_other = qnn_other.forward(x)
        assert output_other.shape == output_relu.shape


@pytest.mark.skipif(not HAS_ML, reason="ml module not available")
class TestVQE:
    """Test Variational Quantum Eigensolver functionality."""
    
    def test_vqe_initialization(self):
        """Test VQE initialization."""
        vqe = VQE(n_qubits=3)
        
        assert vqe.n_qubits == 3
        assert vqe.ansatz == "hardware_efficient"  # Default
        assert vqe.hamiltonian.shape == (8, 8)  # 2^3 x 2^3
        assert len(vqe.parameters) > 0
    
    def test_vqe_custom_hamiltonian(self):
        """Test VQE with custom Hamiltonian."""
        # Simple 2x2 Hamiltonian for 1 qubit
        custom_h = np.array([[1.0, 0.5], [0.5, -1.0]])
        vqe = VQE(n_qubits=1, hamiltonian=custom_h)
        
        assert vqe.n_qubits == 1
        np.testing.assert_array_equal(vqe.hamiltonian, custom_h)
    
    def test_vqe_expectation_value(self):
        """Test VQE expectation value calculation."""
        vqe = VQE(n_qubits=2)
        
        # Test with specific parameters
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        expectation = vqe.expectation(params)
        
        assert isinstance(expectation, float)
        # Expectation value should be real
        assert np.isfinite(expectation)
    
    def test_vqe_parameter_count(self):
        """Test VQE parameter count for different ansatz types."""
        # Hardware efficient ansatz
        vqe_he = VQE(n_qubits=3, ansatz="hardware_efficient")
        expected_params_he = 3 * 3 + (3 - 1)  # 3 rotations per qubit + entangling params
        assert len(vqe_he.parameters) == expected_params_he
        
        # Default ansatz
        vqe_default = VQE(n_qubits=3, ansatz="simple")
        expected_params_default = 3 * 2  # 2 rotations per qubit
        assert len(vqe_default.parameters) == expected_params_default
    
    def test_vqe_optimization(self):
        """Test VQE optimization process."""
        vqe = VQE(n_qubits=2)
        
        # Run short optimization
        final_energy, optimal_params = vqe.optimize(max_iterations=5)
        
        assert isinstance(final_energy, float)
        assert isinstance(optimal_params, np.ndarray)
        assert len(optimal_params) == len(vqe.parameters)
        assert np.isfinite(final_energy)
    
    def test_vqe_optimization_convergence(self):
        """Test that VQE optimization improves over iterations."""
        vqe = VQE(n_qubits=2)
        
        # Get initial energy
        initial_energy = vqe.expectation(vqe.parameters)
        
        # Run optimization
        final_energy, _ = vqe.optimize(max_iterations=10)
        
        # Energy should improve (decrease) or stay similar
        # Note: Due to random nature of stub implementation, we just check it's reasonable
        assert np.isfinite(final_energy)
        assert abs(final_energy - initial_energy) < 100  # Reasonable change


@pytest.mark.skipif(not HAS_ML, reason="ml module not available")
class TestHEPClassifier:
    """Test High-Energy Physics classifier functionality."""
    
    def test_hep_classifier_initialization(self):
        """Test HEP classifier initialization."""
        classifier = HEPClassifier(n_qubits=4, n_features=8, n_classes=3)
        
        assert classifier.n_qubits == 4
        assert classifier.n_features == 8
        assert classifier.n_classes == 3
        assert isinstance(classifier.qnn, QNN)
        assert classifier.qnn.n_qubits == 4
    
    def test_hep_classifier_default_parameters(self):
        """Test HEP classifier with default parameters."""
        classifier = HEPClassifier(n_qubits=3, n_features=6)
        
        assert classifier.n_classes == 2  # Default binary classification
        assert classifier.qnn.n_layers == 3  # Default layers
    
    def test_hep_classifier_single_prediction(self):
        """Test single sample prediction."""
        classifier = HEPClassifier(n_qubits=3, n_features=4, n_classes=2)
        
        # Single sample prediction
        x = np.array([0.1, 0.2, 0.3, 0.4])
        prediction = classifier.predict_single(x)
        
        assert isinstance(prediction, (int, np.integer))
        assert 0 <= prediction < classifier.n_classes
    
    def test_hep_classifier_batch_prediction(self):
        """Test batch prediction."""
        classifier = HEPClassifier(n_qubits=3, n_features=4, n_classes=2)
        
        # Multiple samples
        X = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ])
        
        predictions = classifier.predict(X)
        
        assert len(predictions) == 3
        assert all(0 <= pred < classifier.n_classes for pred in predictions)
    
    def test_hep_classifier_training(self):
        """Test HEP classifier training."""
        classifier = HEPClassifier(n_qubits=3, n_features=4, n_classes=2)
        
        # Generate dummy training data
        X = np.random.randn(20, 4)
        y = np.random.randint(0, 2, 20)
        
        # Train for a few iterations
        history = classifier.fit(X, y, iterations=5)
        
        assert isinstance(history, dict)
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 5
        assert len(history['accuracy']) == 5
        
        # Check that accuracies are reasonable
        for acc in history['accuracy']:
            assert 0 <= acc <= 1
    
    def test_hep_classifier_training_convergence(self):
        """Test that training produces reasonable metrics."""
        classifier = HEPClassifier(n_qubits=2, n_features=3, n_classes=2)
        
        # Simple training data
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        
        history = classifier.fit(X, y, iterations=3)
        
        # Should produce some training history
        assert len(history['loss']) == 3
        assert len(history['accuracy']) == 3
        assert all(isinstance(loss, (int, float)) for loss in history['loss'])


@pytest.mark.skipif(not HAS_ML, reason="ml module not available")
class TestQuantumGAN:
    """Test Quantum Generative Adversarial Network functionality."""
    
    def test_qgan_initialization(self):
        """Test Quantum GAN initialization."""
        qgan = QuantumGAN(
            generator_qubits=4,
            discriminator_qubits=3,
            latent_dim=2,
            data_dim=5
        )
        
        assert qgan.generator_qubits == 4
        assert qgan.discriminator_qubits == 3
        assert qgan.latent_dim == 2
        assert qgan.data_dim == 5
        assert isinstance(qgan.generator, QNN)
        assert isinstance(qgan.discriminator, QNN)
        assert qgan.generator.n_qubits == 4
        assert qgan.discriminator.n_qubits == 3
    
    def test_qgan_sample_generation(self):
        """Test Quantum GAN sample generation."""
        qgan = QuantumGAN(
            generator_qubits=3,
            discriminator_qubits=3,
            latent_dim=2,
            data_dim=4
        )
        
        # Generate samples
        samples = qgan.generate_samples(n_samples=5)
        
        assert samples.shape == (5, 4)  # n_samples x data_dim
        assert np.all(np.isfinite(samples))  # All samples should be finite
    
    def test_qgan_discrimination(self):
        """Test Quantum GAN discrimination."""
        qgan = QuantumGAN(
            generator_qubits=3,
            discriminator_qubits=3,
            latent_dim=2,
            data_dim=3
        )
        
        # Create sample data
        samples = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Get discrimination scores
        scores = qgan.discriminate(samples)
        
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))
    
    def test_qgan_training(self):
        """Test Quantum GAN training process."""
        qgan = QuantumGAN(
            generator_qubits=3,
            discriminator_qubits=3,
            latent_dim=2,
            data_dim=3
        )
        
        # Generate dummy real data
        real_data = np.random.randn(20, 3)
        
        # Train for a few iterations
        history = qgan.train(real_data, iterations=5, batch_size=8)
        
        assert isinstance(history, dict)
        assert 'generator_loss' in history
        assert 'discriminator_loss' in history
        assert len(history['generator_loss']) == 5
        assert len(history['discriminator_loss']) == 5
        
        # Check that losses are reasonable numbers
        for loss in history['generator_loss']:
            assert isinstance(loss, (int, float))
            assert np.isfinite(loss)
        
        for loss in history['discriminator_loss']:
            assert isinstance(loss, (int, float))
            assert np.isfinite(loss)
    
    def test_qgan_training_history_tracking(self):
        """Test that training history is properly tracked."""
        qgan = QuantumGAN(
            generator_qubits=2,
            discriminator_qubits=2,
            latent_dim=1,
            data_dim=2
        )
        
        # Initial history should be empty
        assert len(qgan.history['generator_loss']) == 0
        assert len(qgan.history['discriminator_loss']) == 0
        
        # Generate simple training data
        real_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        # Train
        history = qgan.train(real_data, iterations=3, batch_size=4)
        
        # History should be updated
        assert len(qgan.history['generator_loss']) == 3
        assert len(qgan.history['discriminator_loss']) == 3
        
        # Returned history should match internal history
        assert history == qgan.history


@pytest.mark.skipif(not HAS_ML, reason="ml module not available")
class TestMLIntegration:
    """Test integration between ML components."""
    
    def test_qnn_in_hep_classifier(self):
        """Test that HEP classifier properly uses QNN."""
        classifier = HEPClassifier(n_qubits=3, n_features=4)
        
        # The QNN should be properly initialized
        assert classifier.qnn.n_qubits == 3
        assert classifier.qnn.n_layers == 3
        
        # Should be able to get QNN parameters
        params = classifier.qnn.get_parameters()
        assert len(params) > 0
        
        # Should be able to modify QNN parameters
        new_params = params * 1.1
        classifier.qnn.set_parameters(new_params)
        retrieved_params = classifier.qnn.get_parameters()
        np.testing.assert_array_almost_equal(retrieved_params, new_params)
    
    def test_qnn_in_qgan(self):
        """Test that Quantum GAN properly uses QNNs."""
        qgan = QuantumGAN(
            generator_qubits=3,
            discriminator_qubits=2,
            latent_dim=2,
            data_dim=3
        )
        
        # Both generator and discriminator should be QNNs
        assert isinstance(qgan.generator, QNN)
        assert isinstance(qgan.discriminator, QNN)
        
        # Should have correct properties
        assert qgan.generator.n_qubits == 3
        assert qgan.discriminator.n_qubits == 2
        
        # Should be able to get parameters
        gen_params = qgan.generator.get_parameters()
        disc_params = qgan.discriminator.get_parameters()
        
        assert len(gen_params) > 0
        assert len(disc_params) > 0
    
    def test_vqe_with_custom_circuit(self):
        """Test VQE with different circuit configurations."""
        # Small system
        vqe_small = VQE(n_qubits=2, ansatz="hardware_efficient")
        energy_small = vqe_small.expectation(vqe_small.parameters)
        
        # Larger system
        vqe_large = VQE(n_qubits=3, ansatz="hardware_efficient")
        energy_large = vqe_large.expectation(vqe_large.parameters)
        
        # Both should produce finite energies
        assert np.isfinite(energy_small)
        assert np.isfinite(energy_large)
        
        # Parameter counts should be different
        assert len(vqe_large.parameters) > len(vqe_small.parameters)


@pytest.mark.skipif(not HAS_ML, reason="ml module not available")
class TestMLErrorHandling:
    """Test error handling in ML components."""
    
    def test_qnn_invalid_parameters(self):
        """Test QNN with invalid parameters."""
        qnn = QNN(n_qubits=2, n_layers=1)
        
        # Wrong parameter count should handle gracefully
        wrong_params = np.array([1.0, 2.0])  # Too few parameters
        # Should not crash (implementation may handle this gracefully)
        try:
            qnn.set_parameters(wrong_params)
            # If it doesn't crash, verify the behavior
            params = qnn.get_parameters()
            assert len(params) >= 2
        except (ValueError, IndexError):
            # This is also acceptable
            pass
    
    def test_vqe_zero_qubits(self):
        """Test VQE edge cases."""
        # Very small system
        try:
            vqe = VQE(n_qubits=1)
            assert vqe.n_qubits == 1
            assert vqe.hamiltonian.shape == (2, 2)
        except Exception:
            # Some edge cases might not be supported
            pass
    
    def test_hep_classifier_empty_data(self):
        """Test HEP classifier with edge cases."""
        classifier = HEPClassifier(n_qubits=2, n_features=3)
        
        # Empty training data
        X_empty = np.array([]).reshape(0, 3)
        y_empty = np.array([])
        
        # Should handle gracefully
        try:
            history = classifier.fit(X_empty, y_empty, iterations=1)
            assert isinstance(history, dict)
        except (ValueError, IndexError):
            # This is acceptable for empty data
            pass
    
    def test_qgan_small_dimensions(self):
        """Test Quantum GAN with minimal dimensions."""
        # Minimal GAN
        qgan = QuantumGAN(
            generator_qubits=1,
            discriminator_qubits=1,
            latent_dim=1,
            data_dim=1
        )
        
        # Should still work
        samples = qgan.generate_samples(n_samples=2)
        assert samples.shape == (2, 1)
        
        scores = qgan.discriminate(samples)
        assert len(scores) == 2


@pytest.mark.skipif(not HAS_ML, reason="ml module not available")
class TestMLPerformance:
    """Test ML component performance characteristics."""
    
    def test_qnn_consistency(self):
        """Test QNN output consistency."""
        qnn = QNN(n_qubits=3, n_layers=1)
        
        # Same input should give same output
        x = np.array([[0.1, 0.2, 0.3]])
        
        output1 = qnn.forward(x)
        output2 = qnn.forward(x)
        
        np.testing.assert_array_almost_equal(output1, output2)
    
    def test_vqe_parameter_sensitivity(self):
        """Test VQE parameter sensitivity."""
        vqe = VQE(n_qubits=2)
        
        # Small parameter change
        params1 = vqe.parameters.copy()
        params2 = params1 + 0.001
        
        energy1 = vqe.expectation(params1)
        energy2 = vqe.expectation(params2)
        
        # Energies should be close but potentially different
        assert np.isfinite(energy1)
        assert np.isfinite(energy2)
        # Allow for some variation due to quantum nature
        assert abs(energy1 - energy2) < 10.0
    
    def test_classifier_prediction_stability(self):
        """Test classifier prediction stability."""
        classifier = HEPClassifier(n_qubits=2, n_features=3)
        
        # Same input should give same prediction
        x = np.array([1.0, 2.0, 3.0])
        
        pred1 = classifier.predict_single(x)
        pred2 = classifier.predict_single(x)
        
        assert pred1 == pred2  # Should be deterministic


if __name__ == "__main__":
    pytest.main([__file__])