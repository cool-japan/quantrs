#!/usr/bin/env python3
"""
Memory Efficiency Benchmarks for QuantRS2

This module specifically tests memory usage patterns and efficiency
of various QuantRS2 components.
"""

import gc
import tracemalloc
import numpy as np
from typing import Dict, List, Tuple, Callable
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

try:
    import quantrs2
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False
    print("Warning: quantrs2 not available")


class MemoryBenchmark:
    """Memory usage profiling for QuantRS2 operations."""
    
    def __init__(self, output_dir: str = "benchmark_results/memory"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        
    def profile_memory(self, func: Callable, name: str, **kwargs) -> Dict:
        """Profile memory usage of a function."""
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1e6  # MB
        
        # Start tracing
        tracemalloc.start()
        
        try:
            # Run function
            result = func(**kwargs)
            
            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get process memory
            final_memory = process.memory_info().rss / 1e6  # MB
            
            memory_stats = {
                'name': name,
                'parameters': kwargs,
                'peak_traced_mb': peak / 1e6,
                'current_traced_mb': current / 1e6,
                'process_delta_mb': final_memory - baseline_memory,
                'baseline_mb': baseline_memory,
                'final_mb': final_memory,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            tracemalloc.stop()
            memory_stats = {
                'name': name,
                'parameters': kwargs,
                'peak_traced_mb': 0,
                'current_traced_mb': 0,
                'process_delta_mb': 0,
                'baseline_mb': baseline_memory,
                'final_mb': baseline_memory,
                'success': False,
                'error': str(e)
            }
        
        self.results.append(memory_stats)
        return memory_stats
    
    def run_scaling_analysis(self, func: Callable, name: str, 
                           size_range: List[int], 
                           param_name: str = 'n_qubits'):
        """Run memory scaling analysis."""
        print(f"\nMemory Scaling Analysis: {name}")
        print("-" * 50)
        
        scaling_results = []
        
        for size in size_range:
            print(f"{param_name}={size}: ", end='', flush=True)
            
            stats = self.profile_memory(func, name, **{param_name: size})
            
            if stats['success']:
                print(f"{stats['peak_traced_mb']:.2f} MB peak")
            else:
                print(f"ERROR: {stats['error']}")
            
            scaling_results.append({
                param_name: size,
                'peak_mb': stats['peak_traced_mb'],
                'delta_mb': stats['process_delta_mb']
            })
        
        # Analyze scaling
        df = pd.DataFrame(scaling_results)
        if len(df) > 1:
            # Fit exponential for state vector simulators
            sizes = df[param_name].values
            memory = df['peak_mb'].values
            
            if param_name == 'n_qubits' and all(memory > 0):
                # Expected: O(2^n) for state vector
                theoretical = [2**(n-10) * 8 / 1e6 for n in sizes]  # 8 bytes per complex
                
                plt.figure(figsize=(10, 6))
                plt.plot(sizes, memory, 'o-', label='Measured')
                plt.plot(sizes, theoretical, '--', label='Theoretical O(2^n)')
                plt.xlabel(param_name)
                plt.ylabel('Memory (MB)')
                plt.title(f'Memory Scaling: {name}')
                plt.yscale('log')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{name.replace(" ", "_")}_scaling.png')
                plt.close()
        
        return scaling_results
    
    def compare_implementations(self, implementations: List[Tuple[str, Callable]], 
                              test_params: Dict):
        """Compare memory usage across different implementations."""
        print(f"\nComparing Implementations")
        print("-" * 50)
        
        comparison_results = []
        
        for impl_name, impl_func in implementations:
            stats = self.profile_memory(impl_func, impl_name, **test_params)
            comparison_results.append({
                'implementation': impl_name,
                'peak_mb': stats['peak_traced_mb'],
                'delta_mb': stats['process_delta_mb'],
                'success': stats['success']
            })
            
            print(f"{impl_name}: {stats['peak_traced_mb']:.2f} MB peak")
        
        # Create comparison plot
        df = pd.DataFrame(comparison_results)
        df = df[df['success']]
        
        if not df.empty:
            plt.figure(figsize=(10, 6))
            x = range(len(df))
            plt.bar(x, df['peak_mb'])
            plt.xticks(x, df['implementation'], rotation=45)
            plt.ylabel('Peak Memory (MB)')
            plt.title('Memory Usage Comparison')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'implementation_comparison.png')
            plt.close()
        
        return comparison_results
    
    def save_results(self):
        """Save all results to files."""
        # Save raw results
        with open(self.output_dir / 'memory_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary DataFrame
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / 'memory_summary.csv', index=False)
        
        # Generate summary statistics
        summary = {
            'total_tests': len(self.results),
            'successful_tests': sum(1 for r in self.results if r['success']),
            'average_peak_mb': df[df['success']]['peak_traced_mb'].mean() if any(df['success']) else 0,
            'max_peak_mb': df[df['success']]['peak_traced_mb'].max() if any(df['success']) else 0
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")
        return df


# Memory benchmark functions
def memory_statevector_simulation(n_qubits: int = 10):
    """Memory usage for state vector simulation."""
    if not HAS_QUANTRS2:
        raise ImportError("quantrs2 not available")
    
    circuit = quantrs2.Circuit(n_qubits)
    
    # Add gates to ensure state vector is fully utilized
    for i in range(n_qubits):
        circuit.h(i)
    
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    
    # Run simulation
    result = circuit.run()
    
    # Access state vector to ensure it's allocated
    state = result.state_vector()
    
    return {'state_size': len(state)}


def memory_tensor_network_simulation(n_qubits: int = 20):
    """Memory usage for tensor network simulation."""
    if not HAS_QUANTRS2:
        raise ImportError("quantrs2 not available")
    
    # Create circuit with limited entanglement for TN efficiency
    circuit = quantrs2.Circuit(n_qubits)
    
    # Linear chain of gates
    for i in range(n_qubits):
        circuit.h(i)
    
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    
    # Use tensor network backend if available
    try:
        result = circuit.run(backend='tensor_network')
    except:
        result = circuit.run()  # Fallback
    
    return {'success': True}


def memory_batch_circuits(n_circuits: int = 100, n_qubits: int = 5):
    """Memory usage for batch circuit execution."""
    if not HAS_QUANTRS2:
        raise ImportError("quantrs2 not available")
    
    circuits = []
    
    # Create multiple circuits
    for _ in range(n_circuits):
        circuit = quantrs2.Circuit(n_qubits)
        
        # Random circuit
        for _ in range(10):  # depth
            for q in range(n_qubits):
                gate = np.random.choice(['h', 'x', 'y', 'z'])
                getattr(circuit, gate)(q)
        
        circuits.append(circuit)
    
    # Execute all circuits
    results = []
    for circuit in circuits:
        results.append(circuit.run())
    
    return {'n_results': len(results)}


def memory_ml_model(n_qubits: int = 6, n_layers: int = 3):
    """Memory usage for quantum ML models."""
    if not HAS_QUANTRS2 or not hasattr(quantrs2, 'ml'):
        raise ImportError("ML features not available")
    
    from quantrs2.ml import QNN, QNNConfig
    
    # Create QNN
    config = QNNConfig(
        n_qubits=n_qubits,
        n_features=2**n_qubits,
        n_outputs=2,
        layers=[
            {'type': 'rx', 'qubits': list(range(n_qubits))},
            {'type': 'ry', 'qubits': list(range(n_qubits))},
            {'type': 'entangle', 'pairs': [(i, i+1) for i in range(n_qubits-1)]}
        ] * n_layers
    )
    
    qnn = QNN(config)
    
    # Create dummy data
    x = np.random.randn(10, 2**n_qubits)
    
    # Forward pass
    output = qnn.forward(x)
    
    return {'output_shape': output.shape}


def memory_sparse_operations(n_qubits: int = 20):
    """Memory usage for sparse gate operations."""
    if not HAS_QUANTRS2:
        raise ImportError("quantrs2 not available")
    
    circuit = quantrs2.Circuit(n_qubits)
    
    # Sparse operations (only few qubits interact)
    for i in range(0, n_qubits, 3):
        circuit.h(i)
        if i + 1 < n_qubits:
            circuit.cx(i, i + 1)
    
    # Use optimized backend
    result = circuit.run(backend='optimized')
    
    return {'success': True}


def run_memory_benchmarks():
    """Run complete memory benchmark suite."""
    benchmark = MemoryBenchmark()
    
    # Test 1: State vector scaling
    benchmark.run_scaling_analysis(
        memory_statevector_simulation,
        "State Vector Simulation",
        size_range=[5, 10, 15, 20, 23],
        param_name='n_qubits'
    )
    
    # Test 2: Tensor network scaling
    benchmark.run_scaling_analysis(
        memory_tensor_network_simulation,
        "Tensor Network Simulation",
        size_range=[10, 20, 30, 40, 50],
        param_name='n_qubits'
    )
    
    # Test 3: Batch processing
    benchmark.run_scaling_analysis(
        memory_batch_circuits,
        "Batch Circuit Processing",
        size_range=[10, 50, 100, 200, 500],
        param_name='n_circuits'
    )
    
    # Test 4: ML model scaling
    if HAS_QUANTRS2 and hasattr(quantrs2, 'ml'):
        benchmark.run_scaling_analysis(
            memory_ml_model,
            "Quantum ML Model",
            size_range=[4, 6, 8, 10],
            param_name='n_qubits'
        )
    
    # Test 5: Implementation comparison
    implementations = [
        ("Dense State Vector", lambda: memory_statevector_simulation(15)),
        ("Sparse Operations", lambda: memory_sparse_operations(15)),
    ]
    
    if HAS_QUANTRS2:
        # Add tensor network if available
        try:
            implementations.append(
                ("Tensor Network", lambda: memory_tensor_network_simulation(15))
            )
        except:
            pass
    
    benchmark.compare_implementations(
        implementations,
        test_params={}  # Parameters are embedded in lambdas
    )
    
    # Save all results
    df = benchmark.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("Memory Benchmark Summary")
    print("="*60)
    print(f"Total tests: {len(benchmark.results)}")
    print(f"Successful: {sum(1 for r in benchmark.results if r['success'])}")
    
    if any(r['success'] for r in benchmark.results):
        successful_results = [r for r in benchmark.results if r['success']]
        avg_peak = np.mean([r['peak_traced_mb'] for r in successful_results])
        max_peak = max(r['peak_traced_mb'] for r in successful_results)
        
        print(f"Average peak memory: {avg_peak:.2f} MB")
        print(f"Maximum peak memory: {max_peak:.2f} MB")


if __name__ == "__main__":
    run_memory_benchmarks()