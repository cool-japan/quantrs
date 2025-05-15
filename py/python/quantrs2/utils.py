"""
Utility functions for QuantRS2.

This module provides various utility functions for working with
quantum circuits, state vectors, and other quantum objects.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import math

def binary_to_int(binary: str) -> int:
    """
    Convert a binary string to an integer.
    
    Args:
        binary: Binary string (e.g., '1101')
        
    Returns:
        Corresponding integer
    """
    return int(binary, 2)

def int_to_binary(value: int, width: int) -> str:
    """
    Convert an integer to a binary string with fixed width.
    
    Args:
        value: Integer value
        width: Number of bits in the output
        
    Returns:
        Binary string of specified width
    """
    return format(value, f'0{width}b')

def state_index(bit_string: str) -> int:
    """
    Convert a bit string to a state vector index.
    
    Args:
        bit_string: Binary string representing a basis state
        
    Returns:
        Index in the state vector
    """
    return binary_to_int(bit_string)

def get_basis_states(n_qubits: int) -> List[str]:
    """
    Get all basis states for a given number of qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        List of all basis states as binary strings
    """
    return [int_to_binary(i, n_qubits) for i in range(2**n_qubits)]

def state_to_vector(state: Dict[str, complex], n_qubits: int) -> np.ndarray:
    """
    Convert a state dictionary to a state vector.
    
    Args:
        state: Dictionary mapping basis states to amplitudes
        n_qubits: Number of qubits
        
    Returns:
        State vector as a numpy array
    """
    vec = np.zeros(2**n_qubits, dtype=np.complex128)
    for basis, amplitude in state.items():
        idx = state_index(basis)
        vec[idx] = amplitude
    return vec

def vector_to_state(vector: np.ndarray, n_qubits: int) -> Dict[str, complex]:
    """
    Convert a state vector to a state dictionary.
    
    Args:
        vector: State vector as a numpy array
        n_qubits: Number of qubits
        
    Returns:
        Dictionary mapping basis states to amplitudes
    """
    state = {}
    for i, amplitude in enumerate(vector):
        if abs(amplitude) > 1e-10:  # Ignore very small amplitudes
            basis = int_to_binary(i, n_qubits)
            state[basis] = amplitude
    return state

def fidelity(state1: Union[np.ndarray, Dict[str, complex]], 
             state2: Union[np.ndarray, Dict[str, complex]],
             n_qubits: Optional[int] = None) -> float:
    """
    Calculate the fidelity between two quantum states.
    
    Args:
        state1: First quantum state (vector or dictionary)
        state2: Second quantum state (vector or dictionary)
        n_qubits: Number of qubits (required if states are dictionaries)
        
    Returns:
        Fidelity between the states (0 to 1)
    """
    # Convert dictionaries to vectors if necessary
    if isinstance(state1, dict):
        if n_qubits is None:
            n_qubits = max(len(key) for key in state1.keys())
        state1 = state_to_vector(state1, n_qubits)
    
    if isinstance(state2, dict):
        if n_qubits is None:
            n_qubits = max(len(key) for key in state2.keys())
        state2 = state_to_vector(state2, n_qubits)
    
    # Ensure vectors are the same length
    if len(state1) != len(state2):
        raise ValueError("State vectors must have the same dimension")
    
    # Calculate fidelity
    overlap = np.abs(np.vdot(state1, state2))**2
    return float(overlap)

def entropy(state: Union[np.ndarray, Dict[str, complex]]) -> float:
    """
    Calculate the von Neumann entropy of a quantum state.
    
    Args:
        state: Quantum state (vector or dictionary)
        
    Returns:
        Entropy value
    """
    # Get probabilities
    if isinstance(state, dict):
        probs = [abs(amp)**2 for amp in state.values()]
    else:
        probs = [abs(amp)**2 for amp in state]
    
    # Calculate entropy
    entropy = 0.0
    for p in probs:
        if p > 1e-10:  # Avoid log(0)
            entropy -= p * math.log2(p)
    
    return entropy

def measure_qubit(state: Dict[str, complex], qubit: int) -> Tuple[Dict[str, complex], Dict[str, complex], float]:
    """
    Simulate measurement of a single qubit.
    
    Args:
        state: Quantum state as a dictionary
        qubit: Index of qubit to measure
        
    Returns:
        Tuple of (state_0, state_1, prob_1), where:
          - state_0: Post-measurement state if result is 0
          - state_1: Post-measurement state if result is 1
          - prob_1: Probability of measuring 1
    """
    state_0 = {}
    state_1 = {}
    prob_0 = 0.0
    prob_1 = 0.0
    
    # Calculate probabilities and post-measurement states
    for basis, amplitude in state.items():
        prob = abs(amplitude)**2
        # Check if the qubit is 0 or 1
        if basis[-(qubit+1)] == '0':
            state_0[basis] = amplitude
            prob_0 += prob
        else:
            state_1[basis] = amplitude
            prob_1 += prob
    
    # Normalize the post-measurement states
    if prob_0 > 0:
        norm_factor = 1.0 / math.sqrt(prob_0)
        state_0 = {basis: amp * norm_factor for basis, amp in state_0.items()}
    
    if prob_1 > 0:
        norm_factor = 1.0 / math.sqrt(prob_1)
        state_1 = {basis: amp * norm_factor for basis, amp in state_1.items()}
    
    return state_0, state_1, prob_1

def bell_state(variant: str = 'phi_plus') -> Dict[str, complex]:
    """
    Create a Bell state.
    
    Args:
        variant: Type of Bell state ('phi_plus', 'phi_minus', 'psi_plus', 'psi_minus')
        
    Returns:
        Bell state as a dictionary
    """
    if variant == 'phi_plus':
        # (|00⟩ + |11⟩)/√2
        return {'00': 1/math.sqrt(2), '11': 1/math.sqrt(2)}
    elif variant == 'phi_minus':
        # (|00⟩ - |11⟩)/√2
        return {'00': 1/math.sqrt(2), '11': -1/math.sqrt(2)}
    elif variant == 'psi_plus':
        # (|01⟩ + |10⟩)/√2
        return {'01': 1/math.sqrt(2), '10': 1/math.sqrt(2)}
    elif variant == 'psi_minus':
        # (|01⟩ - |10⟩)/√2
        return {'01': 1/math.sqrt(2), '10': -1/math.sqrt(2)}
    else:
        raise ValueError(f"Unknown Bell state variant: {variant}")

def ghz_state(n_qubits: int) -> Dict[str, complex]:
    """
    Create a GHZ state for n qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        GHZ state as a dictionary
    """
    # (|00...0⟩ + |11...1⟩)/√2
    zeros = '0' * n_qubits
    ones = '1' * n_qubits
    return {zeros: 1/math.sqrt(2), ones: 1/math.sqrt(2)}

def w_state(n_qubits: int) -> Dict[str, complex]:
    """
    Create a W state for n qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        W state as a dictionary
    """
    # (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    norm_factor = 1.0 / math.sqrt(n_qubits)
    state = {}
    
    for i in range(n_qubits):
        # Create a basis state with a single 1 at position i
        basis = ['0'] * n_qubits
        basis[i] = '1'
        basis_str = ''.join(basis)
        state[basis_str] = norm_factor
    
    return state

def uniform_superposition(n_qubits: int) -> Dict[str, complex]:
    """
    Create a uniform superposition state for n qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Uniform superposition state as a dictionary
    """
    # (|00...0⟩ + |00...1⟩ + ... + |11...1⟩)/√(2^n)
    norm_factor = 1.0 / math.sqrt(2**n_qubits)
    state = {}
    
    for i in range(2**n_qubits):
        basis = int_to_binary(i, n_qubits)
        state[basis] = norm_factor
    
    return state