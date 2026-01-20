#!/usr/bin/env python3
"""
Dynamic Qubit Count Example for QuantRS2

This example demonstrates the usage of dynamic qubit count support in QuantRS2,
showing how to create circuits with different qubit counts.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantrs2 import Circuit

def create_bell_state(n_qubits=2):
    """Create a Bell state using the first two qubits in a circuit."""
    circuit = Circuit(n_qubits)
    circuit.h(0)
    circuit.cnot(0, 1)
    return circuit

def create_ghz_state(n_qubits):
    """Create a GHZ state with the specified number of qubits."""
    if n_qubits < 3:
        raise ValueError("GHZ state requires at least 3 qubits")
    
    circuit = Circuit(n_qubits)
    circuit.h(0)
    
    # Apply CNOT gates to entangle all qubits
    for i in range(n_qubits - 1):
        circuit.cnot(i, i + 1)
    
    return circuit

def create_w_state(n_qubits):
    """Create an approximation of a W state with the specified number of qubits."""
    if n_qubits < 3:
        raise ValueError("W state requires at least 3 qubits")
    
    circuit = Circuit(n_qubits)
    
    # Apply Ry rotation to the first qubit
    theta = 2 * np.arccos(1.0 / np.sqrt(n_qubits))
    circuit.ry(0, theta)
    
    # Apply sequence of controlled rotations and CNOTs
    for i in range(1, n_qubits):
        remained_qubits = n_qubits - i
        theta_i = 2 * np.arccos(1.0 / np.sqrt(remained_qubits + 1))
        
        circuit.cnot(i-1, i)
        circuit.ry(i, theta_i)
    
    return circuit

def plot_probabilities(result, title):
    """Plot the probabilities of each basis state."""
    probs = result.probabilities()
    
    n_qubits = result.n_qubits
    basis_states = [format(i, f'0{n_qubits}b') for i in range(len(probs))]
    
    # Only show states with probability > 0.01
    significant_indices = [i for i, p in enumerate(probs) if p > 0.01]
    significant_states = [basis_states[i] for i in significant_indices]
    significant_probs = [probs[i] for i in significant_indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(significant_states, significant_probs)
    plt.xlabel('Basis State')
    plt.ylabel('Probability')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    
    # Print the probabilities for significant states
    print(f"\n{title} - Probabilities:")
    state_probs = result.state_probabilities()
    for state, prob in sorted(state_probs.items()):
        if prob > 0.01:
            print(f"|{state}⟩: {prob:.4f}")
    print()

def demonstrate_different_qubit_counts():
    """Demonstrate dynamic qubit count support by creating circuits with different sizes."""
    
    # 1. Bell state with 2 qubits
    bell_circuit = create_bell_state(2)
    bell_result = bell_circuit.run()
    plot_probabilities(bell_result, "Bell State (2 qubits)")
    
    # 2. GHZ state with 5 qubits
    ghz5_circuit = create_ghz_state(5)
    ghz5_result = ghz5_circuit.run()
    plot_probabilities(ghz5_result, "GHZ State (5 qubits)")
    
    # 3. GHZ state with 8 qubits
    ghz8_circuit = create_ghz_state(8)
    ghz8_result = ghz8_circuit.run()
    plot_probabilities(ghz8_result, "GHZ State (8 qubits)")
    
    # 4. W state with 3 qubits
    w3_circuit = create_w_state(3)
    w3_result = w3_circuit.run()
    plot_probabilities(w3_result, "W State (3 qubits)")
    
    # 5. W state with 4 qubits
    w4_circuit = create_w_state(4)
    w4_result = w4_circuit.run()
    plot_probabilities(w4_result, "W State (4 qubits)")
    
    # Print supported qubit counts
    from quantrs2 import SUPPORTED_QUBITS, MAX_QUBITS
    print(f"Supported qubit counts: {SUPPORTED_QUBITS}")
    print(f"Maximum supported qubits: {MAX_QUBITS}")

if __name__ == "__main__":
    demonstrate_different_qubit_counts()