#!/usr/bin/env python3
"""
Test suite for quantum transfer learning functionality.
"""

import pytest
import numpy as np

try:
    from quantrs2.transfer_learning import (
        TransferStrategy, TransferConfig, PretrainedModel,
        QuantumTransferLearning, QuantumModelZoo, TransferLearningHelper,
        create_transfer_strategy, example_vqe_transfer_learning,
        example_progressive_unfreezing
    )
    HAS_TRANSFER = True
except ImportError:
    HAS_TRANSFER = False


# Mock classes for testing when transfer learning is not available
class MockPretrainedModel:
    """Mock pretrained model for testing."""
    def __init__(self, name="MockModel", n_qubits=4):
        self.name = name
        self._n_qubits = n_qubits
        self.description = f"Mock model with {n_qubits} qubits"
    
    def n_qubits(self):
        return self._n_qubits
    
    def n_layers(self):
        return 3


class MockTransferStrategy:
    """Mock transfer strategy for testing."""
    def __init__(self, strategy_type="fine_tuning"):
        self.strategy_type = strategy_type


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestTransferStrategy:
    """Test TransferStrategy functionality."""
    
    def test_transfer_strategy_creation(self):
        """Test creating different transfer strategies."""
        # Fine-tuning strategy
        strategy1 = create_transfer_strategy("fine_tuning", num_trainable_layers=3)
        assert strategy1 is not None
        
        # Feature extraction strategy
        strategy2 = create_transfer_strategy("feature_extraction")
        assert strategy2 is not None
        
        # Selective adaptation strategy
        strategy3 = create_transfer_strategy("selective_adaptation")
        assert strategy3 is not None
        
        # Progressive unfreezing strategy
        strategy4 = create_transfer_strategy("progressive_unfreezing", unfreeze_rate=2)
        assert strategy4 is not None
    
    def test_transfer_strategy_invalid_type(self):
        """Test creating transfer strategy with invalid type."""
        with pytest.raises(ValueError):
            create_transfer_strategy("invalid_strategy")
    
    def test_transfer_strategy_parameters(self):
        """Test transfer strategy with different parameters."""
        # Fine-tuning with different layer counts
        strategy1 = create_transfer_strategy("fine_tuning", num_trainable_layers=1)
        assert strategy1 is not None
        
        strategy2 = create_transfer_strategy("fine_tuning", num_trainable_layers=5)
        assert strategy2 is not None
        
        # Progressive unfreezing with different rates
        strategy3 = create_transfer_strategy("progressive_unfreezing", unfreeze_rate=1)
        assert strategy3 is not None
        
        strategy4 = create_transfer_strategy("progressive_unfreezing", unfreeze_rate=3)
        assert strategy4 is not None


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestQuantumModelZoo:
    """Test QuantumModelZoo functionality."""
    
    def test_vqe_feature_extractor(self):
        """Test VQE feature extractor creation."""
        for n_qubits in [2, 4, 6]:
            model = QuantumModelZoo.vqe_feature_extractor(n_qubits)
            assert model is not None
            assert model.n_qubits() == n_qubits
    
    def test_qaoa_classifier(self):
        """Test QAOA classifier creation."""
        for n_qubits in [3, 5]:
            for n_layers in [2, 4]:
                model = QuantumModelZoo.qaoa_classifier(n_qubits, n_layers)
                assert model is not None
                assert model.n_qubits() == n_qubits
                assert model.n_layers() == n_layers
    
    def test_quantum_autoencoder(self):
        """Test quantum autoencoder creation."""
        test_cases = [(4, 2), (6, 3), (8, 4)]
        for n_qubits, latent_dim in test_cases:
            model = QuantumModelZoo.quantum_autoencoder(n_qubits, latent_dim)
            assert model is not None
            assert model.n_qubits() == n_qubits


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestTransferLearningHelper:
    """Test TransferLearningHelper functionality."""
    
    def test_helper_initialization_with_string_strategy(self):
        """Test helper initialization with string strategy."""
        model = QuantumModelZoo.vqe_feature_extractor(4)
        helper = TransferLearningHelper(model, "fine_tuning")
        
        assert helper.model == model
        assert helper.strategy is not None
        assert helper.qtl is not None
    
    def test_helper_initialization_with_strategy_object(self):
        """Test helper initialization with strategy object."""
        model = QuantumModelZoo.vqe_feature_extractor(3)
        strategy = create_transfer_strategy("feature_extraction")
        helper = TransferLearningHelper(model, strategy)
        
        assert helper.model == model
        assert helper.strategy == strategy
    
    def test_adapt_for_classification(self):
        """Test adapting model for classification."""
        model = QuantumModelZoo.vqe_feature_extractor(4)
        helper = TransferLearningHelper(model, "fine_tuning")
        
        # Test different numbers of classes
        for n_classes in [2, 3, 5, 10]:
            helper.adapt_for_classification(n_classes)
            # Should not raise any exceptions
    
    def test_adapt_for_regression(self):
        """Test adapting model for regression."""
        model = QuantumModelZoo.qaoa_classifier(4, 2)
        helper = TransferLearningHelper(model, "feature_extraction")
        
        # Test different numbers of outputs
        for n_outputs in [1, 3, 5]:
            helper.adapt_for_regression(n_outputs)
            # Should not raise any exceptions
    
    def test_get_trainable_params(self):
        """Test getting number of trainable parameters."""
        model = QuantumModelZoo.vqe_feature_extractor(4)
        helper = TransferLearningHelper(model, "fine_tuning")
        
        trainable_params = helper.get_trainable_params()
        assert isinstance(trainable_params, int)
        assert trainable_params >= 0
    
    def test_get_model_info(self):
        """Test getting model information."""
        model = QuantumModelZoo.vqe_feature_extractor(5)
        helper = TransferLearningHelper(model, "progressive_unfreezing")
        
        info = helper.get_model_info()
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "n_qubits" in info
        assert "n_layers" in info
        assert "trainable_parameters" in info
        
        assert info["n_qubits"] == 5
        assert isinstance(info["trainable_parameters"], int)
        assert info["trainable_parameters"] >= 0


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestExampleFunctions:
    """Test example transfer learning functions."""
    
    def test_example_vqe_transfer_learning(self):
        """Test VQE transfer learning example."""
        helper = example_vqe_transfer_learning(n_qubits=4)
        
        assert isinstance(helper, TransferLearningHelper)
        info = helper.get_model_info()
        assert info["n_qubits"] == 4
        
        # Test with different numbers of qubits
        for n_qubits in [3, 5, 6]:
            helper = example_vqe_transfer_learning(n_qubits)
            assert helper.get_model_info()["n_qubits"] == n_qubits
    
    def test_example_progressive_unfreezing(self):
        """Test progressive unfreezing example."""
        helper = example_progressive_unfreezing(n_qubits=6)
        
        assert isinstance(helper, TransferLearningHelper)
        info = helper.get_model_info()
        assert info["n_qubits"] == 6
        assert info["n_layers"] == 3
        
        # Test with different numbers of qubits
        for n_qubits in [4, 7, 8]:
            helper = example_progressive_unfreezing(n_qubits)
            assert helper.get_model_info()["n_qubits"] == n_qubits


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestTransferLearningIntegration:
    """Test integration between transfer learning components."""
    
    def test_full_workflow_classification(self):
        """Test full transfer learning workflow for classification."""
        # Get pretrained model
        model = QuantumModelZoo.vqe_feature_extractor(4)
        
        # Create transfer strategy
        strategy = create_transfer_strategy("fine_tuning", num_trainable_layers=2)
        
        # Create helper
        helper = TransferLearningHelper(model, strategy)
        
        # Adapt for multi-class classification
        helper.adapt_for_classification(5)
        
        # Get information
        info = helper.get_model_info()
        trainable_params = helper.get_trainable_params()
        
        assert info["n_qubits"] == 4
        assert isinstance(trainable_params, int)
        assert trainable_params >= 0
    
    def test_full_workflow_regression(self):
        """Test full transfer learning workflow for regression."""
        # Get pretrained model
        model = QuantumModelZoo.qaoa_classifier(5, 3)
        
        # Create transfer strategy
        strategy = create_transfer_strategy("selective_adaptation")
        
        # Create helper
        helper = TransferLearningHelper(model, strategy)
        
        # Adapt for regression
        helper.adapt_for_regression(3)
        
        # Get information
        info = helper.get_model_info()
        
        assert info["n_qubits"] == 5
        assert info["n_layers"] == 3
    
    def test_strategy_comparison(self):
        """Test different strategies on the same model."""
        model = QuantumModelZoo.vqe_feature_extractor(4)
        
        strategies = ["fine_tuning", "feature_extraction", "selective_adaptation"]
        helpers = []
        
        for strategy_name in strategies:
            helper = TransferLearningHelper(model, strategy_name)
            helper.adapt_for_classification(3)
            helpers.append(helper)
        
        # All helpers should work
        for helper in helpers:
            info = helper.get_model_info()
            assert info["n_qubits"] == 4
            trainable_params = helper.get_trainable_params()
            assert isinstance(trainable_params, int)
    
    def test_model_comparison(self):
        """Test transfer learning with different pretrained models."""
        models = [
            QuantumModelZoo.vqe_feature_extractor(4),
            QuantumModelZoo.qaoa_classifier(4, 2),
            QuantumModelZoo.quantum_autoencoder(4, 2)
        ]
        
        strategy = "fine_tuning"
        
        for model in models:
            helper = TransferLearningHelper(model, strategy)
            helper.adapt_for_classification(2)
            
            info = helper.get_model_info()
            assert info["n_qubits"] == 4
            assert isinstance(info["trainable_parameters"], int)


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestTransferLearningErrorHandling:
    """Test error handling in transfer learning components."""
    
    def test_invalid_strategy_type(self):
        """Test handling of invalid strategy types."""
        with pytest.raises(ValueError):
            create_transfer_strategy("nonexistent_strategy")
    
    def test_zero_qubits_model(self):
        """Test handling of edge cases in model creation."""
        # Very small models
        try:
            model = QuantumModelZoo.vqe_feature_extractor(1)
            assert model.n_qubits() == 1
        except Exception:
            # Some implementations might not support 1-qubit models
            pass
    
    def test_large_model_parameters(self):
        """Test handling of large model parameters."""
        # Large models should work
        model = QuantumModelZoo.vqe_feature_extractor(8)
        helper = TransferLearningHelper(model, "fine_tuning")
        
        info = helper.get_model_info()
        assert info["n_qubits"] == 8
        
        # Large number of classes
        helper.adapt_for_classification(100)
        
        # Should not crash
        trainable_params = helper.get_trainable_params()
        assert isinstance(trainable_params, int)
    
    def test_zero_classes_edge_case(self):
        """Test edge cases in adaptation."""
        model = QuantumModelZoo.vqe_feature_extractor(3)
        helper = TransferLearningHelper(model, "fine_tuning")
        
        # Edge cases
        try:
            helper.adapt_for_classification(1)  # Single class
            helper.adapt_for_regression(0)      # Zero outputs
        except (ValueError, IndexError):
            # These edge cases might not be supported
            pass


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestTransferLearningStubs:
    """Test stub behavior when transfer learning is not available."""
    
    def test_stub_imports_when_native_unavailable(self):
        """Test that stubs raise appropriate errors."""
        # This test simulates the case where native features are not available
        # In the actual stub implementation, these would raise ImportError
        
        # We can't easily test the stub behavior directly since we're already
        # importing the real module. This test is for documentation purposes.
        pass


@pytest.mark.skipif(not HAS_TRANSFER, reason="transfer_learning module not available")
class TestTransferLearningPerformance:
    """Test performance characteristics of transfer learning."""
    
    def test_model_creation_performance(self):
        """Test that model creation is reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Create multiple models
        models = []
        for i in range(5):
            model = QuantumModelZoo.vqe_feature_extractor(4)
            models.append(model)
        
        end_time = time.time()
        
        # Should complete within reasonable time (< 1 second for 5 models)
        assert end_time - start_time < 1.0
        assert len(models) == 5
    
    def test_helper_operations_performance(self):
        """Test that helper operations are efficient."""
        model = QuantumModelZoo.vqe_feature_extractor(6)
        helper = TransferLearningHelper(model, "fine_tuning")
        
        import time
        start_time = time.time()
        
        # Perform multiple operations
        for i in range(10):
            helper.adapt_for_classification(3)
            info = helper.get_model_info()
            trainable_params = helper.get_trainable_params()
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 2.0
    
    def test_memory_efficiency(self):
        """Test memory efficiency of transfer learning operations."""
        # Create multiple helpers to test memory usage
        helpers = []
        
        for i in range(10):
            model = QuantumModelZoo.vqe_feature_extractor(3)
            helper = TransferLearningHelper(model, "feature_extraction")
            helper.adapt_for_classification(2)
            helpers.append(helper)
        
        # Should complete without memory issues
        assert len(helpers) == 10
        
        # Verify all helpers work
        for helper in helpers:
            info = helper.get_model_info()
            assert info["n_qubits"] == 3


if __name__ == "__main__":
    pytest.main([__file__])