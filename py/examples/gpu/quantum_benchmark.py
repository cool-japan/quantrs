#!/usr/bin/env python3
"""
QuantRS2 CPU vs GPU Comparison

This script compares CPU and GPU execution paths for quantum circuits of increasing size.
Even though our current implementation is a GPU stub, this shows the different code paths
and prepares your code for when real GPU acceleration is available.
"""

import time
import math
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark(max_qubits=16, mode="both", verbose=True, plot=False):
    """
    Run quantum circuit benchmarks comparing CPU and GPU execution paths.
    
    Args:
        max_qubits: Maximum number of qubits to test (default: 16)
        mode: "cpu", "gpu", or "both" (default: "both")
        verbose: Print detailed results (default: True)
        plot: Generate performance plot (default: False)
    
    Returns:
        Dictionary with benchmark results
    """
    try:
        import _quantrs2 as qr
    except ImportError:
        print("❌ Could not import _quantrs2 module")
        print("Make sure you've built with: ./build_with_gpu_stub.sh")
        print("And activated the virtual environment: source .venv/bin/activate")
        return None
    
    # Supported qubit counts
    qubit_counts = [1, 2, 3, 4, 5, 8, 10, 16]
    
    # Filter to max_qubits
    qubit_counts = [q for q in qubit_counts if q <= max_qubits]
    
    results = {
        "qubits": qubit_counts,
        "state_size": [2**n for n in qubit_counts],
        "cpu_time": [],
        "gpu_time": [],
        "cpu_states": [],
        "gpu_states": []
    }
    
    print(f"QuantRS2 Benchmark: Testing with {qubit_counts} qubits")
    print("=" * 60)
    print()
    
    # Run benchmarks for each qubit count
    for n_qubits in qubit_counts:
        print(f"Creating circuit with {n_qubits} qubits (state size: {2**n_qubits})...")
        
        # Create a circuit that generates an interesting state
        # This will be a superposition of all states
        circuit = qr.PyCircuit(n_qubits)
        
        # Apply Hadamard to all qubits to create superposition
        for i in range(n_qubits):
            circuit.h(i)
        
        # Apply some CNOT gates to create entanglement
        for i in range(n_qubits-1):
            circuit.cnot(i, i+1)
        
        # CPU benchmark
        if mode in ["cpu", "both"]:
            print("Running on CPU...")
            start_time = time.time()
            cpu_result = circuit.run(use_gpu=False)
            cpu_time = time.time() - start_time
            results["cpu_time"].append(cpu_time)
            
            # Get top 5 states with highest probability
            probs = cpu_result.state_probabilities()
            top_states = sorted([(state, prob) for state, prob in probs.items()], 
                                key=lambda x: x[1], reverse=True)[:5]
            results["cpu_states"].append(top_states)
            
            print(f"CPU time: {cpu_time:.4f} seconds")
            
            if verbose:
                print("Top 5 states (CPU):")
                for state, prob in top_states:
                    print(f"  |{state}⟩: {prob:.4f}")
        
        # GPU benchmark
        if mode in ["gpu", "both"]:
            print("Running with GPU code path...")
            start_time = time.time()
            try:
                gpu_result = circuit.run(use_gpu=True)
                gpu_time = time.time() - start_time
                results["gpu_time"].append(gpu_time)
                
                # Get top 5 states with highest probability
                probs = gpu_result.state_probabilities()
                top_states = sorted([(state, prob) for state, prob in probs.items()], 
                                   key=lambda x: x[1], reverse=True)[:5]
                results["gpu_states"].append(top_states)
                
                print(f"GPU time: {gpu_time:.4f} seconds")
                
                if verbose:
                    print("Top 5 states (GPU):")
                    for state, prob in top_states:
                        print(f"  |{state}⟩: {prob:.4f}")
                    
                # Compare results
                if mode == "both":
                    if set([s[0] for s in results["cpu_states"][-1]]) == set([s[0] for s in results["gpu_states"][-1]]):
                        print("✓ CPU and GPU results match")
                    else:
                        print("⚠️ CPU and GPU results differ")
            
            except Exception as e:
                print(f"❌ GPU execution failed: {e}")
                results["gpu_time"].append(None)
                results["gpu_states"].append(None)
        
        print()
    
    # Print summary table
    headers = ["Qubits", "State Size", "CPU Time (s)"]
    table_data = []
    
    for i, n_qubits in enumerate(qubit_counts):
        row = [n_qubits, f"{2**n_qubits:,}", f"{results['cpu_time'][i]:.4f}"]
        if mode == "both":
            headers = ["Qubits", "State Size", "CPU Time (s)", "GPU Time (s)", "Speedup"]
            if results["gpu_time"][i] is not None:
                speedup = results["cpu_time"][i] / results["gpu_time"][i]
                row.extend([f"{results['gpu_time'][i]:.4f}", f"{speedup:.2f}x"])
            else:
                row.extend(["Failed", "N/A"])
        table_data.append(row)
    
    print(tabulate(table_data, headers, tablefmt="grid"))
    
    # Generate plot if requested
    if plot and mode == "both":
        plt.figure(figsize=(10, 6))
        
        # Filter out None values for plotting
        valid_indices = [i for i, t in enumerate(results["gpu_time"]) if t is not None]
        valid_qubits = [results["qubits"][i] for i in valid_indices]
        valid_cpu_times = [results["cpu_time"][i] for i in valid_indices]
        valid_gpu_times = [results["gpu_time"][i] for i in valid_indices]
        
        if valid_indices:
            # For log scale, add small epsilon to avoid log(0)
            epsilon = 1e-6
            
            plt.yscale('log')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            # Plot times
            plt.plot(valid_qubits, [t + epsilon for t in valid_cpu_times], 'o-', label='CPU Time')
            plt.plot(valid_qubits, [t + epsilon for t in valid_gpu_times], 'o-', label='GPU Time')
            
            plt.xlabel('Number of Qubits')
            plt.ylabel('Execution Time (seconds, log scale)')
            plt.title('QuantRS2 Performance: CPU vs GPU')
            plt.legend()
            
            # Show state sizes on top x-axis
            ax1 = plt.gca()
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(valid_qubits)
            ax2.set_xticklabels([f'{2**n}' for n in valid_qubits])
            ax2.set_xlabel('State Vector Size (Amplitudes)')
            
            plt.tight_layout()
            plt.savefig('quantrs2_benchmark.png', dpi=300)
            plt.show()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantRS2 CPU vs GPU Benchmark")
    parser.add_argument("--max-qubits", type=int, default=16, help="Maximum number of qubits to test")
    parser.add_argument("--mode", choices=["cpu", "gpu", "both"], default="both", help="Benchmark mode")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose output")
    parser.add_argument("--plot", action="store_true", help="Generate performance plot")
    
    args = parser.parse_args()
    
    try:
        run_benchmark(
            max_qubits=args.max_qubits,
            mode=args.mode,
            verbose=not args.no_verbose,
            plot=args.plot
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark: {e}")