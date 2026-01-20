"""
Bell state implementation for QuantRS2.

This module provides direct implementation of Bell state and related utility
functions for the QuantRS2 framework.
"""

from math import sqrt
import numpy as np

class BellState:
    """Represents a Bell state simulation with helper methods."""
    
    @staticmethod
    def phi_plus():
        """Create a Φ⁺ = (|00⟩ + |11⟩)/√2 Bell state simulation result."""
        # Import needed modules
        from . import _quantrs2
        
        # Create a circuit that generates a Bell state
        circuit = _quantrs2.PyCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Run the circuit to get a Bell state
        result = circuit.run()
        return result
    
    @staticmethod
    def phi_minus():
        """Create a Φ⁻ = (|00⟩ - |11⟩)/√2 Bell state simulation result."""
        # Import needed modules
        from . import _quantrs2
        
        # Create a circuit that generates a Bell state
        circuit = _quantrs2.PyCircuit(2)
        circuit.h(0)
        circuit.z(0)  # Add Z gate to flip the phase
        circuit.cnot(0, 1)
        
        # Run the circuit to get a Bell state
        result = circuit.run()
        return result
    
    @staticmethod
    def psi_plus():
        """Create a Ψ⁺ = (|01⟩ + |10⟩)/√2 Bell state simulation result."""
        # Import needed modules
        from . import _quantrs2
        
        # Create a circuit that generates a Bell state
        circuit = _quantrs2.PyCircuit(2)
        circuit.h(0)
        circuit.x(1)  # Flip the second qubit
        circuit.cnot(0, 1)
        
        # Run the circuit to get a Bell state
        result = circuit.run()
        return result
    
    @staticmethod
    def psi_minus():
        """Create a Ψ⁻ = (|01⟩ - |10⟩)/√2 Bell state simulation result."""
        # Import needed modules
        from . import _quantrs2
        
        # Create a circuit that generates a Bell state
        circuit = _quantrs2.PyCircuit(2)
        circuit.h(0)
        circuit.z(0)  # Add Z gate to flip the phase
        circuit.x(1)  # Flip the second qubit
        circuit.cnot(0, 1)
        
        # Run the circuit to get a Bell state
        result = circuit.run()
        return result

# Legacy functions for backward compatibility
def create_bell_state():
    """
    Create a Bell state simulation result (Φ⁺ = (|00⟩ + |11⟩)/√2).

    Returns:
        result: A PySimulationResult object representing a Bell state.
    """
    return BellState.phi_plus()

def bell_state_probabilities():
    """
    Get the probability distribution for a Φ⁺ Bell state.

    Returns:
        dict: A dictionary mapping basis states to probabilities.
    """
    # Get probabilities from a Bell state
    result = create_bell_state()
    return result.state_probabilities()

def simulate_bell_circuit():
    """
    Simulate a Bell circuit using direct implementation.
    Creates a Φ⁺ = (|00⟩ + |11⟩)/√2 Bell state.

    Returns:
        result: A PySimulationResult object representing a Bell state.
    """
    # Import directly from _quantrs2
    from . import _quantrs2
    
    # Create a circuit
    circuit = _quantrs2.PyCircuit(2)
    
    # Apply Hadamard to qubit 0
    circuit.h(0)
    
    # Apply CNOT with control=0, target=1
    circuit.cnot(0, 1)
    
    # Run the circuit
    result = circuit.run()
    return result