#!/usr/bin/env python3
"""
QuantRS2 Python Realistic Noise Example

This example demonstrates how to use realistic device noise models
with the QuantRS2 Python bindings.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantrs2 import Circuit, DynamicCircuit, RealisticNoiseModel

def compare_noise_models():
    """Compare different noise models for a Bell state circuit."""
    print("Comparing noise models for Bell state")
    print("====================================")
    
    # Create a Bell state circuit
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Run without noise
    result_ideal = circuit.run()
    print("\nIdeal simulation (no noise):")
    print_state(result_ideal)
    
    # Create and run with IBM device noise
    ibm_noise = RealisticNoiseModel.ibm_device("ibmq_lima")
    result_ibm = circuit.simulate_with_noise(ibm_noise)
    print("\nSimulation with IBM Lima device noise:")
    print_state(result_ibm)
    
    # Create and run with Rigetti device noise
    rigetti_noise = RealisticNoiseModel.rigetti_device("Aspen-M-2")
    result_rigetti = circuit.simulate_with_noise(rigetti_noise)
    print("\nSimulation with Rigetti Aspen-M-2 device noise:")
    print_state(result_rigetti)
    
    # Create and run with custom noise parameters
    custom_noise = RealisticNoiseModel.custom(
        t1_us=80,            # T1 relaxation time in microseconds
        t2_us=40,            # T2 dephasing time in microseconds
        gate_time_ns=50,     # Gate time in nanoseconds
        gate_error_1q=0.001, # Single-qubit gate error rate
        gate_error_2q=0.01,  # Two-qubit gate error rate
        readout_error=0.02   # Readout error rate
    )
    result_custom = circuit.simulate_with_noise(custom_noise)
    print("\nSimulation with custom noise parameters:")
    print_state(result_custom)
    
    # Calculate state fidelities to ideal state
    print("\nState fidelities:")
    print(f"IBM noise fidelity:      {calculate_fidelity(result_ideal, result_ibm):.6f}")
    print(f"Rigetti noise fidelity:  {calculate_fidelity(result_ideal, result_rigetti):.6f}")
    print(f"Custom noise fidelity:   {calculate_fidelity(result_ideal, result_custom):.6f}")
    
    # Visualize the results
    plot_state_comparison([
        ("Ideal", result_ideal),
        ("IBM", result_ibm),
        ("Rigetti", result_rigetti),
        ("Custom", result_custom)
    ])

def simulate_ghz_state():
    """Simulate a GHZ state with different noise levels."""
    print("\nSimulating GHZ state with different noise levels")
    print("===============================================")
    
    # Create a 5-qubit GHZ state circuit using dynamic circuit
    circuit = Circuit(5)
    circuit.h(0)
    for i in range(4):
        circuit.cnot(i, i+1)
    
    # Set different error levels for custom noise
    error_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    results = []
    
    for error in error_levels:
        # Create custom noise with specified error rate
        noise = RealisticNoiseModel.custom(
            t1_us=100,
            t2_us=50,
            gate_time_ns=40,
            gate_error_1q=error/2,  # Half the error for 1-qubit gates
            gate_error_2q=error,    # Full error for 2-qubit gates
            readout_error=error
        )
        
        # Simulate with noise
        if error == 0.0:
            result = circuit.run()  # No noise for the ideal case
        else:
            result = circuit.simulate_with_noise(noise)
        results.append(result)
        
        # Print top 3 states
        print(f"\nResults with {error*100:.1f}% error rate:")
        probs = [(i, p) for i, p in enumerate(result)]
        probs.sort(key=lambda x: x[1], reverse=True)
        for i, (idx, prob) in enumerate(probs[:3]):
            state = format(idx, f"0{circuit.num_qubits()}b")
            print(f"  {i+1}. |{state}⟩: {prob:.6f} ({prob*100:.2f}%)")
    
    # Plot fidelity vs error rate
    ideal_states = [0, 2**(circuit.num_qubits())-1]  # |00000⟩ and |11111⟩
    fidelities = [calculate_fidelity_to_states(result, ideal_states) for result in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(error_levels)*100, fidelities, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Error Rate (%)')
    plt.ylabel('Fidelity to Ideal GHZ State')
    plt.title('GHZ State Fidelity vs Noise Level')
    plt.grid(True)
    plt.savefig('ghz_fidelity.png')
    print("\nGHZ fidelity plot saved as 'ghz_fidelity.png'")

def simulate_quantum_fourier_transform():
    """Simulate a Quantum Fourier Transform with realistic noise."""
    print("\nSimulating Quantum Fourier Transform with realistic noise")
    print("======================================================")
    
    # Create a 3-qubit QFT circuit
    circuit = Circuit(3)
    
    # Apply Hadamard to all qubits
    for i in range(3):
        circuit.h(i)
    
    # Apply controlled phase rotations
    circuit.cz(0, 1)
    circuit.cz(0, 2)
    circuit.cz(1, 2)
    
    # Apply more Hadamards
    for i in range(3):
        circuit.h(i)
    
    # Run without noise
    result_ideal = circuit.run()
    print("\nIdeal QFT simulation (no noise):")
    print_state(result_ideal)
    
    # Simulate with IBM noise
    ibm_noise = RealisticNoiseModel.ibm_device("ibm_cairo")
    result_ibm = circuit.simulate_with_noise(ibm_noise)
    print("\nQFT simulation with IBM Cairo device noise:")
    print_state(result_ibm)
    
    # Calculate fidelity
    fidelity = calculate_fidelity(result_ideal, result_ibm)
    print(f"\nFidelity to ideal QFT state: {fidelity:.6f}")
    
    # Plot state comparison
    plot_state_comparison([
        ("Ideal QFT", result_ideal),
        ("Noisy QFT", result_ibm)
    ])

def print_state(state_vector):
    """Print state vector probabilities in a readable format."""
    probs = [(i, abs(amp)**2) for i, amp in enumerate(state_vector)]
    significant = [(i, p) for i, p in probs if p > 0.001]  # Only states with p > 0.1%
    
    # Print in descending probability order
    significant.sort(key=lambda x: x[1], reverse=True)
    for idx, prob in significant:
        num_qubits = int(np.log2(len(state_vector)))
        state = format(idx, f"0{num_qubits}b")
        print(f"  |{state}⟩: {prob:.6f} ({prob*100:.2f}%)")

def calculate_fidelity(state1, state2):
    """Calculate fidelity between two quantum states."""
    fidelity = abs(np.vdot(state1, state2))**2
    return fidelity

def calculate_fidelity_to_states(state, target_indices):
    """Calculate fidelity to specific target states."""
    # For states like (|00000⟩ + |11111⟩)/√2
    target_prob = 1.0 / len(target_indices)
    fidelity = sum(abs(state[idx])**2 for idx in target_indices)
    return fidelity / len(target_indices)

def plot_state_comparison(state_list):
    """Plot comparison of multiple quantum states."""
    num_states = len(state_list)
    num_qubits = int(np.log2(len(state_list[0][1])))
    dim = 2**num_qubits
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(dim)
    width = 0.8 / num_states
    
    for i, (label, state) in enumerate(state_list):
        probs = [abs(amp)**2 for amp in state]
        plt.bar(x + i*width - 0.4 + width/2, probs, width, label=label)
    
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.title('Quantum State Comparison')
    plt.xticks(range(dim), [format(i, f"0{num_qubits}b") for i in range(dim)], rotation=70)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('state_comparison.png')
    print("State comparison plot saved as 'state_comparison.png'")

if __name__ == "__main__":
    print("QuantRS2 Python Realistic Noise Example")
    print("======================================")
    
    compare_noise_models()
    simulate_ghz_state()
    simulate_quantum_fourier_transform()