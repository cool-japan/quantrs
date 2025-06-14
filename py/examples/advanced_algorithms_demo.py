#!/usr/bin/env python3
"""
Advanced Quantum Algorithms Demo for QuantRS2

This example demonstrates the advanced quantum algorithm implementations,
including VQE, QAOA, quantum walks, error correction, and other algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any

try:
    from quantrs2.advanced_algorithms import (
        AnsatzType,
        OptimizationMethod,
        AdvancedVQE,
        AdvancedQAOA,
        QuantumWalkAlgorithms,
        QuantumErrorCorrection,
        QuantumTeleportation,
        ShorsAlgorithm,
        QuantumSimulatedAnnealing,
        create_advanced_vqe,
        create_advanced_qaoa,
        create_quantum_walk,
        create_error_correction_circuit,
        create_shors_circuit,
        create_teleportation_circuit
    )
    from quantrs2 import Circuit
except ImportError as e:
    print(f"Import error: {e}")
    print("Advanced algorithms module not available. Please check installation.")
    exit(1)


def demo_advanced_vqe():
    """Demonstrate advanced VQE capabilities."""
    print("=== Advanced VQE Demo ===")
    
    # Test different ansätze
    ansatz_types = [
        AnsatzType.HARDWARE_EFFICIENT,
        AnsatzType.REAL_AMPLITUDES,
        AnsatzType.EFFICIENT_SU2,
        AnsatzType.TWO_LOCAL
    ]
    
    n_qubits = 4
    
    for ansatz_type in ansatz_types:
        print(f"\nTesting {ansatz_type.value} ansatz:")
        
        try:
            vqe = AdvancedVQE(
                n_qubits=n_qubits,
                ansatz=ansatz_type,
                optimizer=OptimizationMethod.COBYLA,
                max_iterations=100
            )
            
            # Get parameter count
            param_count = vqe.get_parameter_count(reps=2)
            print(f"  Parameter count (2 reps): {param_count}")
            
            # Create test parameters
            parameters = np.random.uniform(0, 2*np.pi, param_count)
            
            # Create ansatz circuit
            circuit = vqe.create_ansatz_circuit(parameters.tolist(), reps=2)
            print(f"  Circuit created successfully with {circuit.n_qubits} qubits")
            
            # Run simulation (if possible)
            try:
                result = circuit.run()
                probabilities = result.probabilities()
                print(f"  Simulation completed. Max probability: {max(probabilities):.4f}")
            except Exception as e:
                print(f"  Simulation not available: {e}")
                
        except Exception as e:
            print(f"  Error with {ansatz_type.value}: {e}")
    
    # Demonstrate UCCSD for quantum chemistry
    print(f"\nTesting UCCSD ansatz for quantum chemistry:")
    try:
        vqe_uccsd = AdvancedVQE(4, ansatz=AnsatzType.UCCSD)
        param_count = vqe_uccsd.get_parameter_count()
        parameters = np.random.uniform(-np.pi, np.pi, param_count)
        
        circuit = vqe_uccsd.create_ansatz_circuit(parameters.tolist())
        print(f"  UCCSD circuit created with {param_count} parameters")
        
    except Exception as e:
        print(f"  UCCSD error: {e}")


def demo_advanced_qaoa():
    """Demonstrate advanced QAOA capabilities."""
    print("\n=== Advanced QAOA Demo ===")
    
    # MaxCut problem
    print("\n1. MaxCut Problem:")
    n_qubits = 4
    qaoa_maxcut = AdvancedQAOA(
        n_qubits=n_qubits,
        p_layers=2,
        problem_type="maxcut",
        mixer_type="x_mixer"
    )
    
    # Define a graph
    problem_instance = {
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        "weights": [1.0, 1.0, 1.0, 1.0, 0.5]
    }
    
    # QAOA parameters
    parameters = [0.8, 0.4, 0.6, 0.3]  # [γ₁, β₁, γ₂, β₂]
    
    try:
        circuit = qaoa_maxcut.create_qaoa_circuit(problem_instance, parameters)
        print(f"  MaxCut circuit created for {len(problem_instance['edges'])} edges")
        
        result = circuit.run()
        probabilities = result.probabilities()
        
        # Find most probable states
        max_prob_idx = np.argmax(probabilities)
        max_prob_state = format(max_prob_idx, f'0{n_qubits}b')
        print(f"  Most probable state: |{max_prob_state}⟩ with probability {probabilities[max_prob_idx]:.4f}")
        
    except Exception as e:
        print(f"  MaxCut QAOA error: {e}")
    
    # MAX-k-SAT problem
    print("\n2. MAX-3-SAT Problem:")
    qaoa_sat = AdvancedQAOA(
        n_qubits=3,
        p_layers=1,
        problem_type="max_k_sat"
    )
    
    # Define 3-SAT clauses: (x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂ ∨ ¬x₃)
    sat_instance = {
        "clauses": [[1, -2, 3], [-1, 2, -3]]
    }
    
    try:
        circuit = qaoa_sat.create_qaoa_circuit(sat_instance, [0.5, 0.3])
        print(f"  MAX-3-SAT circuit created for {len(sat_instance['clauses'])} clauses")
        
    except Exception as e:
        print(f"  MAX-3-SAT QAOA error: {e}")
    
    # Number partitioning
    print("\n3. Number Partitioning Problem:")
    qaoa_partition = AdvancedQAOA(
        n_qubits=4,
        p_layers=1,
        problem_type="number_partitioning"
    )
    
    partition_instance = {
        "numbers": [3, 1, 1, 2]  # Can we partition into two equal-sum sets?
    }
    
    try:
        circuit = qaoa_partition.create_qaoa_circuit(partition_instance, [0.4, 0.2])
        print(f"  Number partitioning circuit created for numbers {partition_instance['numbers']}")
        
    except Exception as e:
        print(f"  Number partitioning QAOA error: {e}")


def demo_quantum_walks():
    """Demonstrate quantum walk algorithms."""
    print("\n=== Quantum Walks Demo ===")
    
    # Continuous-time quantum walk
    print("\n1. Continuous-Time Quantum Walk:")
    n_vertices = 4
    
    # Create a cycle graph
    adjacency_matrix = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    
    try:
        circuit = create_quantum_walk(
            "continuous",
            n_qubits=n_vertices,
            adjacency_matrix=adjacency_matrix,
            time=1.0,
            initial_state=0
        )
        
        print(f"  Continuous-time walk circuit created for {n_vertices} vertices")
        
        result = circuit.run()
        probabilities = result.probabilities()
        
        print("  Final probability distribution:")
        for i, prob in enumerate(probabilities):
            if prob > 0.01:
                state = format(i, f'0{n_vertices}b')
                print(f"    |{state}⟩: {prob:.4f}")
                
    except Exception as e:
        print(f"  Continuous-time walk error: {e}")
    
    # Discrete-time quantum walk
    print("\n2. Discrete-Time Quantum Walk:")
    try:
        circuit = create_quantum_walk(
            "discrete",
            n_position_qubits=3,
            n_coin_qubits=1,
            steps=4,
            coin_operator="hadamard"
        )
        
        print(f"  Discrete-time walk circuit created (3 position + 1 coin qubits, 4 steps)")
        
    except Exception as e:
        print(f"  Discrete-time walk error: {e}")


def demo_error_correction():
    """Demonstrate quantum error correction codes."""
    print("\n=== Quantum Error Correction Demo ===")
    
    # 3-qubit repetition code
    print("\n1. Three-Qubit Repetition Code:")
    try:
        data_state = [np.sqrt(0.7), np.sqrt(0.3)]  # |ψ⟩ = √0.7|0⟩ + √0.3|1⟩
        circuit = create_error_correction_circuit("repetition", data_qubit_state=data_state)
        
        print(f"  Repetition code circuit created (5 qubits total)")
        
        result = circuit.run()
        probabilities = result.probabilities()
        
        # Show encoded states
        print("  Encoded state probabilities (showing |000⟩ and |111⟩ blocks):")
        for i, prob in enumerate(probabilities):
            if prob > 0.01:
                state = format(i, '05b')
                if state[:3] in ['000', '111']:  # Logical codewords
                    print(f"    |{state}⟩: {prob:.4f}")
                
    except Exception as e:
        print(f"  Repetition code error: {e}")
    
    # Steane code
    print("\n2. Steane [[7,1,3]] Code:")
    try:
        circuit = create_error_correction_circuit("steane")
        print(f"  Steane code circuit created (7 qubits)")
        
    except Exception as e:
        print(f"  Steane code error: {e}")
    
    # Surface code
    print("\n3. Surface Code Patch:")
    try:
        circuit = create_error_correction_circuit("surface", distance=3)
        print(f"  Surface code patch created (distance 3, {3*3} qubits)")
        
    except Exception as e:
        print(f"  Surface code error: {e}")


def demo_teleportation():
    """Demonstrate quantum teleportation."""
    print("\n=== Quantum Teleportation Demo ===")
    
    try:
        circuit = create_teleportation_circuit()
        print(f"  Teleportation circuit created (3 qubits)")
        
        result = circuit.run()
        probabilities = result.probabilities()
        
        print("  Teleportation outcome probabilities:")
        for i, prob in enumerate(probabilities):
            if prob > 0.01:
                state = format(i, '03b')
                print(f"    |{state}⟩: {prob:.4f}")
                
    except Exception as e:
        print(f"  Teleportation error: {e}")


def demo_shors_algorithm():
    """Demonstrate Shor's factorization algorithm."""
    print("\n=== Shor's Algorithm Demo ===")
    
    # Factor N = 15
    N = 15
    a = 7  # Coprime to N
    
    try:
        circuit = create_shors_circuit(N, a)
        print(f"  Shor's circuit created for N={N}, a={a}")
        print(f"  Circuit uses {circuit.n_qubits} qubits")
        
        # Note: Actual factorization would require classical post-processing
        # of the measurement results to find the period
        
    except Exception as e:
        print(f"  Shor's algorithm error: {e}")


def demo_quantum_annealing():
    """Demonstrate quantum simulated annealing."""
    print("\n=== Quantum Simulated Annealing Demo ===")
    
    n_qubits = 4
    
    try:
        qsa = QuantumSimulatedAnnealing(
            n_qubits=n_qubits,
            initial_temp=1.0,
            final_temp=0.1,
            n_steps=20
        )
        
        # Define an Ising problem
        problem_hamiltonian = {
            "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],  # Ring topology
            "fields": [0.1, -0.2, 0.3, -0.1]  # Random fields
        }
        
        circuit = qsa.create_annealing_circuit(problem_hamiltonian)
        print(f"  Quantum annealing circuit created ({n_qubits} qubits, 20 steps)")
        
        result = circuit.run()
        probabilities = result.probabilities()
        
        # Find ground state candidate
        max_prob_idx = np.argmax(probabilities)
        ground_state = format(max_prob_idx, f'0{n_qubits}b')
        print(f"  Candidate ground state: |{ground_state}⟩ with probability {probabilities[max_prob_idx]:.4f}")
        
    except Exception as e:
        print(f"  Quantum annealing error: {e}")


def demo_algorithm_comparison():
    """Compare different algorithms for optimization problems."""
    print("\n=== Algorithm Comparison Demo ===")
    
    # Compare VQE ansätze performance
    print("\n1. VQE Ansatz Comparison:")
    n_qubits = 3
    ansätze = [AnsatzType.HARDWARE_EFFICIENT, AnsatzType.REAL_AMPLITUDES, AnsatzType.EFFICIENT_SU2]
    
    for ansatz in ansätze:
        try:
            vqe = create_advanced_vqe(n_qubits, ansatz=ansatz)
            param_count = vqe.get_parameter_count(reps=1)
            
            # Create random parameters
            parameters = np.random.uniform(0, 2*np.pi, param_count)
            
            start_time = time.time()
            circuit = vqe.create_ansatz_circuit(parameters.tolist(), reps=1)
            result = circuit.run()
            execution_time = time.time() - start_time
            
            print(f"  {ansatz.value}:")
            print(f"    Parameters: {param_count}")
            print(f"    Execution time: {execution_time:.4f}s")
            print(f"    Quantum state entropy: {calculate_entropy(result.probabilities()):.4f}")
            
        except Exception as e:
            print(f"  {ansatz.value} error: {e}")
    
    # Compare QAOA vs Quantum Annealing for optimization
    print("\n2. QAOA vs Quantum Annealing:")
    
    # Define common optimization problem
    problem = {
        "edges": [(0, 1), (1, 2), (2, 0)],  # Triangle graph
        "weights": [1.0, 1.0, 1.0]
    }
    
    try:
        # QAOA approach
        qaoa = create_advanced_qaoa(3, p_layers=1, problem_type="maxcut")
        start_time = time.time()
        qaoa_circuit = qaoa.create_qaoa_circuit(problem, [0.6, 0.4])
        qaoa_result = qaoa_circuit.run()
        qaoa_time = time.time() - start_time
        
        print(f"  QAOA:")
        print(f"    Execution time: {qaoa_time:.4f}s")
        print(f"    Best probability: {max(qaoa_result.probabilities()):.4f}")
        
    except Exception as e:
        print(f"  QAOA error: {e}")
    
    try:
        # Quantum Annealing approach
        qsa = QuantumSimulatedAnnealing(3, n_steps=10)
        annealing_problem = {"edges": problem["edges"], "fields": [0.0, 0.0, 0.0]}
        
        start_time = time.time()
        annealing_circuit = qsa.create_annealing_circuit(annealing_problem)
        annealing_result = annealing_circuit.run()
        annealing_time = time.time() - start_time
        
        print(f"  Quantum Annealing:")
        print(f"    Execution time: {annealing_time:.4f}s")
        print(f"    Best probability: {max(annealing_result.probabilities()):.4f}")
        
    except Exception as e:
        print(f"  Quantum Annealing error: {e}")


def calculate_entropy(probabilities):
    """Calculate Shannon entropy of probability distribution."""
    probs = np.array(probabilities)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs))


def demo_scalability_analysis():
    """Analyze algorithm scalability with qubit count."""
    print("\n=== Scalability Analysis Demo ===")
    
    qubit_counts = [2, 3, 4, 5]
    
    print("\nParameter scaling analysis:")
    print("Qubits | HE-VQE | Real-Amp | UCCSD | QAOA-p1")
    print("-------|--------|----------|-------|--------")
    
    for n_qubits in qubit_counts:
        try:
            # Hardware-efficient VQE
            he_vqe = create_advanced_vqe(n_qubits, ansatz=AnsatzType.HARDWARE_EFFICIENT)
            he_params = he_vqe.get_parameter_count(reps=1)
            
            # Real amplitudes VQE
            ra_vqe = create_advanced_vqe(n_qubits, ansatz=AnsatzType.REAL_AMPLITUDES)
            ra_params = ra_vqe.get_parameter_count(reps=1)
            
            # UCCSD VQE
            uccsd_vqe = create_advanced_vqe(n_qubits, ansatz=AnsatzType.UCCSD)
            uccsd_params = uccsd_vqe.get_parameter_count()
            
            # QAOA parameters (p=1)
            qaoa_params = 2  # Always 2 parameters per layer
            
            print(f"   {n_qubits}   |   {he_params:2d}   |    {ra_params:2d}    |  {uccsd_params:2d}   |   {qaoa_params:2d}")
            
        except Exception as e:
            print(f"   {n_qubits}   | Error: {e}")
    
    print("\nCircuit depth analysis (approximate gate counts):")
    for n_qubits in qubit_counts:
        try:
            # Create circuits and count operations
            vqe = create_advanced_vqe(n_qubits, ansatz=AnsatzType.HARDWARE_EFFICIENT)
            param_count = vqe.get_parameter_count(reps=1)
            parameters = [0.1] * param_count
            
            circuit = vqe.create_ansatz_circuit(parameters, reps=1)
            
            print(f"  {n_qubits} qubits: ~{len(circuit.operations) if hasattr(circuit, 'operations') else 'N/A'} operations")
            
        except Exception as e:
            print(f"  {n_qubits} qubits: Error - {e}")


def main():
    """Run all demonstrations."""
    print("QuantRS2 Advanced Quantum Algorithms Demo")
    print("=" * 50)
    
    try:
        demo_advanced_vqe()
        demo_advanced_qaoa()
        demo_quantum_walks()
        demo_error_correction()
        demo_teleportation()
        demo_shors_algorithm()
        demo_quantum_annealing()
        demo_algorithm_comparison()
        demo_scalability_analysis()
        
        print("\n" + "=" * 50)
        print("All advanced algorithm demonstrations completed!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()