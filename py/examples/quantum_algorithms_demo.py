#!/usr/bin/env python3
"""
Demonstration of quantum algorithm templates.

This example shows how to use:
1. Variational Quantum Eigensolver (VQE)
2. Quantum Approximate Optimization Algorithm (QAOA)
3. Quantum Fourier Transform (QFT)
4. Grover's Search Algorithm
5. Quantum Phase Estimation (QPE)
"""

import numpy as np
from quantrs2.algorithms import VQE, QAOA, QFT, Grover, QPE, create_ising_hamiltonian
from quantrs2.measurement import MeasurementSampler

def vqe_example():
    """Example of using VQE for finding ground states."""
    print("=== Variational Quantum Eigensolver (VQE) ===")
    
    # Create VQE instance
    vqe = VQE(n_qubits=4, ansatz_type="hardware_efficient")
    
    # Create ansatz circuit
    ansatz = vqe.create_ansatz(num_layers=2)
    print(f"Created ansatz circuit with {ansatz.n_qubits} qubits")
    
    # Create a simple Hamiltonian (Ising model)
    H = create_ising_hamiltonian(n_qubits=4, j_coupling=1.0, h_field=0.5)
    print("Created Ising Hamiltonian")
    
    # In practice, you would optimize the ansatz parameters here
    # For demonstration, we just show the structure
    print("VQE optimization would find optimal parameters for the ansatz")


def qaoa_example():
    """Example of QAOA for combinatorial optimization."""
    print("\n=== Quantum Approximate Optimization Algorithm (QAOA) ===")
    
    # Define a MaxCut problem on a triangle graph
    n_qubits = 3
    edges = [(0, 1), (1, 2), (0, 2)]  # Triangle graph
    
    # Create QAOA instance with p=2 layers
    qaoa = QAOA(n_qubits=n_qubits, p=2)
    
    # Initialize parameters (gamma and beta for each layer)
    params = [0.5, 0.8, 0.3, 0.6]  # [gamma_0, beta_0, gamma_1, beta_1]
    
    # Create QAOA circuit
    circuit = qaoa.maxcut_circuit(edges=edges, params=params)
    
    # Run the circuit
    result = circuit.run()
    
    # Sample measurements
    sampler = MeasurementSampler()
    measurements = sampler.sample_counts(result, shots=1000)
    
    print(f"QAOA circuit created for MaxCut on {n_qubits} qubits")
    print(f"Graph edges: {edges}")
    print(f"Parameters: γ={params[::2]}, β={params[1::2]}")
    print(f"Measurement outcomes: {measurements.get_counts()}")
    
    # The most probable outcome should correspond to a good cut
    print(f"Most probable cut: {measurements.most_probable()}")


def qft_example():
    """Example of Quantum Fourier Transform."""
    print("\n=== Quantum Fourier Transform (QFT) ===")
    
    # Create QFT circuit for 3 qubits
    n_qubits = 3
    qft_circuit = QFT.circuit(n_qubits=n_qubits)
    
    print(f"Created QFT circuit for {n_qubits} qubits")
    
    # Apply QFT to a basis state |5⟩ = |101⟩
    prep_circuit = QFT.circuit(n_qubits=n_qubits)  # Create new circuit
    prep_circuit.x(0)  # Set bit 0
    prep_circuit.x(2)  # Set bit 2
    # Now state is |101⟩ = |5⟩
    
    # Apply QFT
    QFT.apply_to_circuit(prep_circuit, qubits=[0, 1, 2])
    
    # Run and get the result
    result = prep_circuit.run()
    print("Applied QFT to state |101⟩")
    
    # The result should show phase information
    amplitudes = result.amplitudes()
    print("QFT output amplitudes (first 4):")
    for i in range(min(4, len(amplitudes))):
        amp = amplitudes[i]
        print(f"  |{i:03b}⟩: {amp:.3f}")
    
    # Inverse QFT
    inverse_qft = QFT.circuit(n_qubits=n_qubits, inverse=True)
    print("\nCreated inverse QFT circuit")


def grover_example():
    """Example of Grover's search algorithm."""
    print("\n=== Grover's Search Algorithm ===")
    
    # Search in 3-qubit space (8 items)
    grover = Grover(n_qubits=3)
    
    # Mark items 3 (011) and 5 (101)
    marked_items = [3, 5]
    
    # Create Grover circuit
    circuit = grover.create_circuit(marked_items=marked_items)
    
    print(f"Searching in {2**3} items")
    print(f"Marked items: {marked_items} (binary: {[f'{x:03b}' for x in marked_items]})")
    print(f"Optimal iterations: {grover.n_iterations}")
    
    # Run the circuit
    result = circuit.run()
    
    # Sample measurements
    sampler = MeasurementSampler()
    measurements = sampler.sample_counts(result, shots=1000)
    
    print("\nMeasurement outcomes:")
    for outcome, count in sorted(measurements.get_counts().items(), key=lambda x: x[1], reverse=True):
        marked = " (marked)" if int(outcome, 2) in marked_items else ""
        print(f"  |{outcome}⟩: {count}{marked}")
    
    print(f"\nMost probable outcome: |{measurements.most_probable()}⟩")


def qpe_example():
    """Example of Quantum Phase Estimation."""
    print("\n=== Quantum Phase Estimation (QPE) ===")
    
    # Estimate the phase of a Z rotation
    phase = 0.375  # 3/8 in binary: 0.011
    
    # Create QPE circuit
    # 3 counting qubits to estimate phase with 3-bit precision
    # 1 state qubit
    circuit = QPE.circuit(
        n_counting_qubits=3,
        n_state_qubits=1,
        unitary_gate="RZ",
        phase=2 * np.pi * phase  # Convert to radians
    )
    
    print(f"Estimating phase: {phase} = {phase:.3f}")
    print(f"Using 3 counting qubits for 3-bit precision")
    
    # Run the circuit
    result = circuit.run()
    
    # Sample measurements
    sampler = MeasurementSampler()
    measurements = sampler.sample_counts(result, shots=1000)
    
    print("\nMeasurement outcomes (counting qubits only):")
    # Extract only the counting qubit measurements
    counting_measurements = {}
    for outcome, count in measurements.get_counts().items():
        counting_bits = outcome[:3]  # First 3 bits are counting qubits
        counting_measurements[counting_bits] = counting_measurements.get(counting_bits, 0) + count
    
    for outcome, count in sorted(counting_measurements.items(), key=lambda x: x[1], reverse=True)[:4]:
        measured_phase = int(outcome, 2) / 8  # Convert to decimal fraction
        print(f"  |{outcome}⟩: {count} (phase ≈ {measured_phase:.3f})")
    
    # Most probable should be |011⟩ = 3/8 = 0.375
    most_probable = max(counting_measurements.items(), key=lambda x: x[1])[0]
    estimated_phase = int(most_probable, 2) / 8
    print(f"\nEstimated phase: {estimated_phase:.3f}")
    print(f"Error: {abs(estimated_phase - phase):.3f}")


def algorithm_comparison():
    """Compare different algorithms for the same problem."""
    print("\n=== Algorithm Comparison ===")
    
    print("Different quantum algorithms serve different purposes:")
    print("- VQE: Finding ground states of quantum systems")
    print("- QAOA: Solving combinatorial optimization problems")
    print("- QFT: Basis transformation for phase estimation")
    print("- Grover: Searching unsorted databases")
    print("- QPE: Estimating eigenvalues of unitary operators")
    
    print("\nCircuit depths (approximate):")
    print("- VQE: O(p) where p is number of ansatz layers")
    print("- QAOA: O(p) where p is number of QAOA rounds")
    print("- QFT: O(n²) where n is number of qubits")
    print("- Grover: O(√N) where N is search space size")
    print("- QPE: O(n) + cost of controlled unitaries")


if __name__ == "__main__":
    vqe_example()
    qaoa_example()
    qft_example()
    grover_example()
    qpe_example()
    algorithm_comparison()
    
    print("\n=== Quantum Algorithms Demo Complete ===")