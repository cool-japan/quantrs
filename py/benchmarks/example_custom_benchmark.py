#!/usr/bin/env python3
"""
Example: Creating Custom Benchmarks for QuantRS2

This example shows how to create and integrate custom benchmarks
into the QuantRS2 benchmarking framework.
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the benchmark suite
from benchmark_suite import BenchmarkSuite

# Try to import quantrs2
try:
    import quantrs2
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False
    print("Warning: quantrs2 not available, using mock implementations")


# Example 1: Custom gate benchmark
def benchmark_custom_gate_sequence(**kwargs):
    """Benchmark a specific gate sequence pattern."""
    n_qubits = kwargs.get('n_qubits', 5)
    pattern = kwargs.get('pattern', 'ladder')
    repetitions = kwargs.get('repetitions', 10)
    
    if not HAS_QUANTRS2:
        # Mock implementation
        time.sleep(0.1)
        return {'metrics': {'pattern_time': 0.1}}
    
    circuit = quantrs2.Circuit(n_qubits)
    
    start = time.perf_counter()
    
    for rep in range(repetitions):
        if pattern == 'ladder':
            # Ladder pattern: sequential CNOTs
            for i in range(n_qubits - 1):
                circuit.h(i)
                circuit.cx(i, i + 1)
        
        elif pattern == 'star':
            # Star pattern: all qubits connected to first
            circuit.h(0)
            for i in range(1, n_qubits):
                circuit.cx(0, i)
        
        elif pattern == 'grid':
            # Grid pattern: nearest neighbor on a line
            for i in range(0, n_qubits - 1, 2):
                circuit.h(i)
                if i + 1 < n_qubits:
                    circuit.cx(i, i + 1)
            for i in range(1, n_qubits - 1, 2):
                if i + 1 < n_qubits:
                    circuit.cx(i, i + 1)
    
    # Run the circuit
    result = circuit.run()
    
    total_time = time.perf_counter() - start
    gates_per_second = (repetitions * n_qubits * 2) / total_time  # Approximate
    
    return {
        'metrics': {
            'pattern_time': total_time,
            'gates_per_second': gates_per_second,
            'final_entropy': calculate_entropy(result.probabilities())
        }
    }


# Example 2: Custom optimization benchmark
def benchmark_custom_vqe_ansatz(**kwargs):
    """Benchmark custom VQE ansatz performance."""
    n_qubits = kwargs.get('n_qubits', 4)
    ansatz_type = kwargs.get('ansatz_type', 'hardware_efficient')
    optimization_steps = kwargs.get('optimization_steps', 20)
    
    if not HAS_QUANTRS2 or not hasattr(quantrs2, 'ml'):
        return {'metrics': {'convergence_time': 0}}
    
    from quantrs2.ml import VQE
    
    # Define Hamiltonian (simple Ising chain)
    hamiltonian = []
    for i in range(n_qubits - 1):
        hamiltonian.append(('ZZ', [i, i+1], 1.0))
    for i in range(n_qubits):
        hamiltonian.append(('X', [i], 0.5))
    
    # Create custom ansatz based on type
    if ansatz_type == 'hardware_efficient':
        # Hardware efficient ansatz
        layers = []
        for _ in range(2):  # 2 layers
            # Single qubit rotations
            layers.append({'type': 'ry', 'qubits': list(range(n_qubits))})
            layers.append({'type': 'rz', 'qubits': list(range(n_qubits))})
            # Entangling layer
            layers.append({
                'type': 'entangle',
                'pairs': [(i, i+1) for i in range(n_qubits-1)]
            })
    else:  # 'chemistry_inspired'
        # Chemistry-inspired ansatz
        layers = []
        # Pair excitations
        for i in range(0, n_qubits, 2):
            if i + 1 < n_qubits:
                layers.append({
                    'type': 'givens_rotation',
                    'qubits': [i, i+1]
                })
    
    # Run VQE
    start = time.perf_counter()
    
    # Note: This is a simplified example
    # In practice, you'd create the ansatz properly
    convergence_history = []
    for step in range(optimization_steps):
        # Simulate optimization step
        energy = -step / optimization_steps + np.random.normal(0, 0.1)
        convergence_history.append(energy)
    
    optimization_time = time.perf_counter() - start
    
    return {
        'metrics': {
            'convergence_time': optimization_time,
            'final_energy': convergence_history[-1],
            'convergence_rate': abs(convergence_history[-1] - convergence_history[0]) / optimization_steps,
            'energy_variance': np.var(convergence_history)
        }
    }


# Helper function
def calculate_entropy(probabilities):
    """Calculate Shannon entropy of probability distribution."""
    probs = np.array(probabilities)
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))


# Example 3: Custom memory benchmark
def benchmark_circuit_memory_patterns(**kwargs):
    """Analyze memory allocation patterns for different circuit types."""
    circuit_type = kwargs.get('circuit_type', 'random')
    n_qubits = kwargs.get('n_qubits', 10)
    depth = kwargs.get('depth', 100)
    
    if not HAS_QUANTRS2:
        return {'metrics': {'memory_efficiency': 0}}
    
    import tracemalloc
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create circuit based on type
    circuit = quantrs2.Circuit(n_qubits)
    
    if circuit_type == 'random':
        # Random circuit with mixed gates
        gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
        for _ in range(depth):
            for q in range(n_qubits):
                gate = np.random.choice(gates)
                if gate in ['rx', 'ry', 'rz']:
                    getattr(circuit, gate)(q, np.random.random() * 2 * np.pi)
                else:
                    getattr(circuit, gate)(q)
    
    elif circuit_type == 'qft':
        # Quantum Fourier Transform pattern
        for j in range(n_qubits):
            circuit.h(j)
            for k in range(j + 1, n_qubits):
                circuit.cp(j, k, np.pi / (2 ** (k - j)))
    
    elif circuit_type == 'variational':
        # Variational circuit pattern
        for layer in range(depth // (n_qubits * 3)):
            # Rotation layer
            for q in range(n_qubits):
                circuit.rx(q, np.random.random() * 2 * np.pi)
                circuit.rz(q, np.random.random() * 2 * np.pi)
            # Entangling layer
            for q in range(0, n_qubits - 1, 2):
                circuit.cx(q, q + 1)
            for q in range(1, n_qubits - 1, 2):
                circuit.cx(q, q + 1)
    
    # Get memory usage before execution
    current, peak = tracemalloc.get_traced_memory()
    pre_execution_memory = peak
    
    # Execute circuit
    result = circuit.run()
    
    # Get memory usage after execution
    current, peak = tracemalloc.get_traced_memory()
    post_execution_memory = peak
    
    tracemalloc.stop()
    
    # Calculate metrics
    memory_efficiency = (depth * n_qubits) / (post_execution_memory / 1e6)  # gates per MB
    
    return {
        'metrics': {
            'pre_execution_mb': pre_execution_memory / 1e6,
            'post_execution_mb': post_execution_memory / 1e6,
            'memory_efficiency': memory_efficiency,
            'memory_per_gate': (post_execution_memory - pre_execution_memory) / (depth * n_qubits)
        }
    }


def run_custom_benchmarks():
    """Run all custom benchmarks."""
    # Create benchmark suite instance
    suite = BenchmarkSuite(output_dir="custom_benchmark_results")
    
    print("Running Custom Benchmarks")
    print("=" * 60)
    
    # Define custom benchmark configurations
    custom_benchmarks = [
        # Gate pattern benchmarks
        ("Gate Pattern Performance", benchmark_custom_gate_sequence, [
            {'n_qubits': 5, 'pattern': 'ladder', 'repetitions': 10},
            {'n_qubits': 5, 'pattern': 'star', 'repetitions': 10},
            {'n_qubits': 5, 'pattern': 'grid', 'repetitions': 10},
            {'n_qubits': 10, 'pattern': 'ladder', 'repetitions': 5},
            {'n_qubits': 10, 'pattern': 'star', 'repetitions': 5},
        ]),
        
        # VQE ansatz benchmarks
        ("VQE Ansatz Comparison", benchmark_custom_vqe_ansatz, [
            {'n_qubits': 4, 'ansatz_type': 'hardware_efficient', 'optimization_steps': 20},
            {'n_qubits': 4, 'ansatz_type': 'chemistry_inspired', 'optimization_steps': 20},
            {'n_qubits': 6, 'ansatz_type': 'hardware_efficient', 'optimization_steps': 15},
            {'n_qubits': 6, 'ansatz_type': 'chemistry_inspired', 'optimization_steps': 15},
        ]),
        
        # Memory pattern benchmarks
        ("Circuit Memory Patterns", benchmark_circuit_memory_patterns, [
            {'circuit_type': 'random', 'n_qubits': 10, 'depth': 100},
            {'circuit_type': 'qft', 'n_qubits': 10, 'depth': 100},
            {'circuit_type': 'variational', 'n_qubits': 10, 'depth': 100},
            {'circuit_type': 'random', 'n_qubits': 15, 'depth': 50},
            {'circuit_type': 'qft', 'n_qubits': 8, 'depth': 200},
        ])
    ]
    
    # Run benchmarks
    suite.run_category("Custom Benchmarks", custom_benchmarks)
    
    # Save results
    df = suite.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("Custom Benchmark Summary")
    print("="*60)
    
    if not df.empty:
        # Best gate pattern
        gate_results = df[df['name'] == 'Gate Pattern Performance']
        if not gate_results.empty:
            best_pattern = gate_results.nsmallest(1, 'execution_time_ms').iloc[0]
            print(f"Fastest gate pattern: {best_pattern['pattern']} ({best_pattern['execution_time_ms']:.2f} ms)")
        
        # VQE convergence
        vqe_results = df[df['name'] == 'VQE Ansatz Comparison']
        if not vqe_results.empty:
            best_ansatz = vqe_results.nsmallest(1, 'execution_time_ms').iloc[0]
            print(f"Fastest VQE ansatz: {best_ansatz['ansatz_type']}")
        
        # Memory efficiency
        memory_results = df[df['name'] == 'Circuit Memory Patterns']
        if not memory_results.empty and 'memory_efficiency' in memory_results.columns:
            best_memory = memory_results.nlargest(1, 'memory_efficiency').iloc[0]
            print(f"Most memory efficient: {best_memory['circuit_type']} circuit")
    
    print(f"\nResults saved to: custom_benchmark_results/")


if __name__ == "__main__":
    run_custom_benchmarks()