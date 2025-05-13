"""
QuantRS2 Python bindings.

This module provides Python access to the QuantRS2 quantum computing framework.
"""

# Version information
__version__ = "0.1.0a1"

# This allows the package to be importable even when the native module is not built
try:
    # Try to import the actual native module
    from . import _quantrs2
    from ._quantrs2 import *
except ImportError:
    # Stub for verification purposes
    class PyCircuit:
        """Stub for Python bindings verification."""
        def __init__(self, n_qubits):
            pass
        
        def h(self, qubit):
            pass
        
        def x(self, qubit):
            pass
        
        def cnot(self, control, target):
            pass
        
        def run(self, use_gpu=False):
            pass
    
    class PySimulationResult:
        """Stub for Python bindings verification."""
        def __init__(self):
            pass
        
        def amplitudes(self):
            pass
        
        def probabilities(self):
            pass
        
        def state_probabilities(self):
            pass