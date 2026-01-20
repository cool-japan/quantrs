#!/usr/bin/env python3
"""
GPU Benchmark Example for QuantRS2

This example benchmarks GPU acceleration in QuantRS2 by comparing CPU and GPU
simulation times for different qubit counts.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from quantrs2 import Circuit

def create_random_circuit(n_qubits, depth):
    """Create a random quantum circuit with the specified number of qubits and depth."""
    circuit = Circuit(n_qubits)
    
    # Add random gates
    for _ in range(depth):
        # Randomly select gate type
        gate_type = np.random.choice(['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cnot'])
        
        if gate_type in ['h', 'x', 'y', 'z']:
            # Single-qubit gate
            qubit = np.random.randint(0, n_qubits)
            getattr(circuit, gate_type)(qubit)
        elif gate_type in ['rx', 'ry', 'rz']:
            # Rotation gate
            qubit = np.random.randint(0, n_qubits)
            angle = np.random.random() * 2 * np.pi
            getattr(circuit, gate_type)(qubit, angle)
        elif gate_type == 'cnot':
            # Two-qubit gate
            qubits = np.random.choice(range(n_qubits), size=2, replace=False)
            circuit.cnot(qubits[0], qubits[1])
    
    return circuit

def benchmark_circuit(circuit, use_gpu=False, use_auto=False):
    """Benchmark the circuit simulation."""
    start_time = time.time()
    
    if use_auto:
        result = circuit.run_auto()
    else:
        result = circuit.run(use_gpu=use_gpu)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time, result

def run_benchmark():
    """Run the benchmark for different qubit counts."""
    # Check if GPU is available
    gpu_available = Circuit.is_gpu_available()
    if not gpu_available:
        print("GPU acceleration is not available. Running CPU-only benchmark.")
    
    # Qubit counts to benchmark
    qubit_counts = [4, 8, 12, 16, 20]
    circuit_depth = 20
    
    cpu_times = []
    gpu_times = []
    auto_times = []
    
    print(f"Benchmarking circuits with depth {circuit_depth}")
    print("=" * 50)
    
    for n_qubits in qubit_counts:
        print(f"Testing with {n_qubits} qubits...")
        
        # Create the circuit
        circuit = create_random_circuit(n_qubits, circuit_depth)
        
        # Benchmark CPU
        cpu_time, _ = benchmark_circuit(circuit, use_gpu=False)
        cpu_times.append(cpu_time)
        print(f"  CPU time: {cpu_time:.4f} seconds")
        
        # Benchmark GPU (if available)
        if gpu_available:
            gpu_time, _ = benchmark_circuit(circuit, use_gpu=True)
            gpu_times.append(gpu_time)
            print(f"  GPU time: {gpu_time:.4f} seconds")
            
            # Benchmark auto mode
            auto_time, _ = benchmark_circuit(circuit, use_auto=True)
            auto_times.append(auto_time)
            print(f"  Auto time: {auto_time:.4f} seconds")
            
            # Calculate speedup
            speedup = cpu_time / gpu_time
            print(f"  GPU speedup: {speedup:.2f}x")
        
        print("-" * 30)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(qubit_counts, cpu_times, 'o-', label='CPU')
    
    if gpu_available:
        plt.plot(qubit_counts, gpu_times, 'o-', label='GPU')
        plt.plot(qubit_counts, auto_times, 'o-', label='Auto')
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Simulation Time (seconds)')
    plt.title(f'Quantum Circuit Simulation Performance (Depth {circuit_depth})')
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark_results.png')
    
    print("\nBenchmark results:")
    print("Qubit Counts:", qubit_counts)
    print("CPU Times:", [f"{t:.4f}" for t in cpu_times])
    
    if gpu_available:
        print("GPU Times:", [f"{t:.4f}" for t in gpu_times])
        print("Auto Times:", [f"{t:.4f}" for t in auto_times])
        print("Speedups:", [f"{cpu/gpu:.2f}x" for cpu, gpu in zip(cpu_times, gpu_times)])
    
    print("\nBenchmark complete! Results saved to benchmark_results.png")

if __name__ == "__main__":
    run_benchmark()