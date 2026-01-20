"""
Gate Decomposition and Optimization Example

This example demonstrates how to decompose complex gates into simpler ones 
and optimize circuits by removing redundant gates.
"""

import quantrs2 as qrs
import numpy as np
import math

def main():
    print("Gate Decomposition and Optimization Example")
    print("==========================================")
    
    # Create a circuit with complex gates
    print("\nCreating circuit with complex gates...")
    circuit = qrs.PyCircuit(3)
    
    # Add some gates including complex ones
    circuit.h(0)
    circuit.h(0)  # Two Hadamards in a row cancel out
    circuit.x(1)
    circuit.toffoli(0, 1, 2)  # Complex gate that can be decomposed
    circuit.swap(1, 2)  # Can be decomposed into 3 CNOTs
    
    # Visualize the original circuit
    print("\nOriginal Circuit:")
    print(circuit.draw())
    
    # Simulate the original circuit
    print("\nSimulating original circuit...")
    result = circuit.run_auto()
    state_probs = result.state_probabilities()
    print("State probabilities:")
    for state, prob in state_probs.items():
        print(f"|{state}⟩: {prob:.6f}")
    
    # Decompose the circuit
    print("\nDecomposing the circuit...")
    decomposed = circuit.decompose()
    
    # Visualize the decomposed circuit
    print("\nDecomposed Circuit:")
    print(decomposed.draw())
    
    # Simulate the decomposed circuit
    print("\nSimulating decomposed circuit...")
    decomposed_result = decomposed.run_auto()
    decomposed_probs = decomposed_result.state_probabilities()
    print("State probabilities:")
    for state, prob in decomposed_probs.items():
        print(f"|{state}⟩: {prob:.6f}")
    
    # Compare results to verify equivalence
    print("\nComparing results...")
    original_probs = result.probabilities()
    decomp_probs = decomposed_result.probabilities()
    
    # Calculate maximum difference
    max_diff = max(abs(o - d) for o, d in zip(original_probs, decomp_probs))
    print(f"Maximum probability difference: {max_diff:.10f}")
    
    if max_diff < 1e-10:
        print("Results match! The decomposition preserves the circuit's behavior.")
    else:
        print("Results differ! The decomposition might have issues.")
    
    # Optimization Example
    print("\n\nOptimization Example")
    print("===================")
    
    # Create a circuit with redundant gates
    print("\nCreating circuit with redundant gates...")
    redundant = qrs.PyCircuit(2)
    
    # Add gates that should cancel out or be combined
    redundant.h(0)
    redundant.h(0)  # These two H gates cancel out
    
    redundant.x(1)
    redundant.x(1)  # These two X gates cancel out
    
    # Add gates that can be combined
    redundant.rx(0, np.pi/4)
    redundant.rx(0, np.pi/4)  # These combine to rx(0, pi/2)
    
    # Visualize the redundant circuit
    print("\nRedundant Circuit:")
    print(redundant.draw())
    
    # Optimize the circuit
    print("\nOptimizing the circuit...")
    optimized = redundant.optimize()
    
    # Visualize the optimized circuit
    print("\nOptimized Circuit:")
    print(optimized.draw())
    
    # Simulate both to verify they're equivalent
    print("\nSimulating redundant circuit...")
    redundant_result = redundant.run_auto()
    
    print("\nSimulating optimized circuit...")
    optimized_result = optimized.run_auto()
    
    # Compare the results
    redundant_probs = redundant_result.probabilities()
    optimized_probs = optimized_result.probabilities()
    
    # Calculate maximum difference
    max_diff = max(abs(r - o) for r, o in zip(redundant_probs, optimized_probs))
    print(f"\nMaximum probability difference: {max_diff:.10f}")
    
    if max_diff < 1e-10:
        print("Results match! The optimization preserves the circuit's behavior.")
    else:
        print("Results differ! The optimization might have issues.")

if __name__ == "__main__":
    main()