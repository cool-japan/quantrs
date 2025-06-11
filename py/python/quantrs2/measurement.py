"""
Quantum measurement functionality.

This module provides measurement-related classes and utilities.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class MeasurementResult:
    """Result of a quantum measurement."""
    counts: Dict[str, int]
    shots: int
    
    def get_counts(self) -> Dict[str, int]:
        """Get measurement counts."""
        return self.counts
        
    def get_probability(self, outcome: str) -> float:
        """Get probability of a specific outcome."""
        return self.counts.get(outcome, 0) / self.shots

class MeasurementSampler:
    """Samples measurements from quantum states."""
    
    def __init__(self, shots: int = 1024):
        self.shots = shots
        
    def sample(self, state_vector: np.ndarray) -> MeasurementResult:
        """Sample measurements from a state vector."""
        # Simple implementation for testing
        n_qubits = int(np.log2(len(state_vector)))
        probabilities = np.abs(state_vector) ** 2
        
        # Sample according to probabilities
        outcomes = np.random.choice(len(state_vector), size=self.shots, p=probabilities)
        
        # Convert to binary strings
        counts = {}
        for outcome in outcomes:
            binary = format(outcome, f'0{n_qubits}b')
            counts[binary] = counts.get(binary, 0) + 1
            
        return MeasurementResult(counts=counts, shots=self.shots)

__all__ = ["MeasurementResult", "MeasurementSampler"]