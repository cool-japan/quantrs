"""
Quantum Transfer Learning Module

This module provides quantum transfer learning capabilities,
allowing you to leverage pretrained quantum models for new tasks.
"""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = hasattr(_quantrs2, 'transfer')
except ImportError:
    _NATIVE_AVAILABLE = False

if _NATIVE_AVAILABLE:
    # Import native implementations
    TransferStrategy = _quantrs2.transfer.PyTransferStrategy
    TransferConfig = _quantrs2.transfer.PyTransferConfig
    PretrainedModel = _quantrs2.transfer.PyPretrainedModel
    QuantumTransferLearning = _quantrs2.transfer.PyQuantumTransferLearning
    QuantumModelZoo = _quantrs2.transfer.PyQuantumModelZoo
else:
    # Provide stubs for when ML features are not available
    class TransferStrategy:
        """Transfer learning strategies (stub)"""
        def __init__(self):
            raise ImportError("ML features not available. Install with: pip install quantrs2[ml]")
    
    class TransferConfig:
        """Transfer learning configuration (stub)"""
        def __init__(self):
            raise ImportError("ML features not available")
    
    class PretrainedModel:
        """Pretrained quantum model (stub)"""
        def __init__(self):
            raise ImportError("ML features not available")
    
    class QuantumTransferLearning:
        """Quantum transfer learning (stub)"""
        def __init__(self, *args, **kwargs):
            raise ImportError("ML features not available")
    
    class QuantumModelZoo:
        """Quantum model zoo (stub)"""
        @staticmethod
        def vqe_feature_extractor(n_qubits: int):
            raise ImportError("ML features not available")
        
        @staticmethod
        def qaoa_classifier(n_qubits: int, n_layers: int):
            raise ImportError("ML features not available")
        
        @staticmethod
        def quantum_autoencoder(n_qubits: int, latent_dim: int):
            raise ImportError("ML features not available")


def create_transfer_strategy(strategy_type: str, **kwargs) -> TransferStrategy:
    """
    Create a transfer learning strategy.
    
    Args:
        strategy_type: Type of strategy ('fine_tuning', 'feature_extraction', 
                      'selective_adaptation', 'progressive_unfreezing')
        **kwargs: Strategy-specific parameters
        
    Returns:
        TransferStrategy object
    """
    if not _NATIVE_AVAILABLE:
        raise ImportError("ML features not available")
    
    if strategy_type == "fine_tuning":
        num_trainable_layers = kwargs.get("num_trainable_layers", 2)
        return TransferStrategy.FineTuning(num_trainable_layers=num_trainable_layers)
    elif strategy_type == "feature_extraction":
        return TransferStrategy.FeatureExtraction()
    elif strategy_type == "selective_adaptation":
        return TransferStrategy.SelectiveAdaptation()
    elif strategy_type == "progressive_unfreezing":
        unfreeze_rate = kwargs.get("unfreeze_rate", 1)
        return TransferStrategy.ProgressiveUnfreezing(unfreeze_rate=unfreeze_rate)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


class TransferLearningHelper:
    """
    Helper class for quantum transfer learning workflows.
    """
    
    def __init__(self, pretrained_model: PretrainedModel, strategy: Union[str, TransferStrategy]):
        """
        Initialize transfer learning helper.
        
        Args:
            pretrained_model: Pretrained quantum model
            strategy: Transfer learning strategy (string or TransferStrategy object)
        """
        if isinstance(strategy, str):
            strategy = create_transfer_strategy(strategy)
        
        self.model = pretrained_model
        self.strategy = strategy
        self.qtl = QuantumTransferLearning(pretrained_model, strategy)
    
    def adapt_for_classification(self, n_classes: int) -> None:
        """Adapt model for classification task."""
        self.qtl.adapt_for_task(n_classes)
    
    def adapt_for_regression(self, n_outputs: int = 1) -> None:
        """Adapt model for regression task."""
        self.qtl.adapt_for_task(n_outputs)
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return self.qtl.trainable_parameters()
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get model information."""
        return {
            "name": self.model.name,
            "description": self.model.description,
            "n_qubits": self.model.n_qubits(),
            "n_layers": self.model.n_layers(),
            "trainable_parameters": self.get_trainable_params(),
        }


# Example usage functions
def example_vqe_transfer_learning(n_qubits: int = 4) -> TransferLearningHelper:
    """
    Example: Transfer learning with VQE feature extractor.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Configured TransferLearningHelper
    """
    # Get pretrained model
    model = QuantumModelZoo.vqe_feature_extractor(n_qubits)
    
    # Use fine-tuning strategy
    helper = TransferLearningHelper(model, "fine_tuning")
    
    # Adapt for binary classification
    helper.adapt_for_classification(2)
    
    return helper


def example_progressive_unfreezing(n_qubits: int = 6) -> TransferLearningHelper:
    """
    Example: Progressive unfreezing with QAOA classifier.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Configured TransferLearningHelper
    """
    # Get pretrained QAOA model
    model = QuantumModelZoo.qaoa_classifier(n_qubits, n_layers=3)
    
    # Use progressive unfreezing
    strategy = create_transfer_strategy("progressive_unfreezing", unfreeze_rate=1)
    helper = TransferLearningHelper(model, strategy)
    
    # Adapt for multi-class classification
    helper.adapt_for_classification(4)
    
    return helper


__all__ = [
    'TransferStrategy',
    'TransferConfig', 
    'PretrainedModel',
    'QuantumTransferLearning',
    'QuantumModelZoo',
    'TransferLearningHelper',
    'create_transfer_strategy',
    'example_vqe_transfer_learning',
    'example_progressive_unfreezing',
]