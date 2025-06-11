#!/usr/bin/env python3
"""
Parallel Performance Benchmarks for QuantRS2

This module tests parallel execution efficiency, thread scaling,
and GPU acceleration performance.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Callable, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import threading
import queue
import psutil

try:
    import quantrs2
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False
    print("Warning: quantrs2 not available")


class ParallelBenchmark:
    """Benchmark parallel and GPU performance."""
    
    def __init__(self, output_dir: str = "benchmark_results/parallel"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.cpu_count = psutil.cpu_count(logical=False)
        self.thread_count = psutil.cpu_count(logical=True)
        
    def benchmark_thread_scaling(self, func: Callable, name: str,
                               thread_counts: List[int],
                               workload_size: int = 100):
        """Test performance with different thread counts."""
        print(f"\nThread Scaling Benchmark: {name}")
        print("-" * 50)
        
        scaling_results = []
        
        # Single-threaded baseline
        print("Baseline (1 thread): ", end='', flush=True)
        start = time.perf_counter()
        baseline_results = []
        for i in range(workload_size):
            baseline_results.append(func(work_id=i))
        baseline_time = time.perf_counter() - start
        print(f"{baseline_time:.3f}s")
        
        scaling_results.append({
            'threads': 1,
            'time': baseline_time,
            'speedup': 1.0,
            'efficiency': 1.0
        })
        
        # Multi-threaded tests
        for n_threads in thread_counts:
            if n_threads == 1:
                continue
                
            print(f"{n_threads} threads: ", end='', flush=True)
            
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                start = time.perf_counter()
                futures = []
                for i in range(workload_size):
                    futures.append(executor.submit(func, work_id=i))
                
                results = [f.result() for f in futures]
                elapsed = time.perf_counter() - start
            
            speedup = baseline_time / elapsed
            efficiency = speedup / n_threads
            
            print(f"{elapsed:.3f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.2%})")
            
            scaling_results.append({
                'threads': n_threads,
                'time': elapsed,
                'speedup': speedup,
                'efficiency': efficiency
            })
        
        # Plot scaling
        df = pd.DataFrame(scaling_results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Speedup plot
        ax1.plot(df['threads'], df['speedup'], 'o-', label='Actual')
        ax1.plot(df['threads'], df['threads'], '--', label='Ideal', alpha=0.5)
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Speedup')
        ax1.set_title(f'Thread Scaling: {name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax2.plot(df['threads'], df['efficiency'] * 100, 'o-')
        ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Parallel Efficiency')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name.replace(" ", "_")}_scaling.png')
        plt.close()
        
        self.results.append({
            'name': name,
            'type': 'thread_scaling',
            'data': scaling_results
        })
        
        return scaling_results
    
    def benchmark_gpu_vs_cpu(self, circuit_func: Callable, name: str,
                           problem_sizes: List[int]):
        """Compare GPU vs CPU performance."""
        if not HAS_QUANTRS2:
            print("QuantRS2 not available, skipping GPU benchmarks")
            return []
        
        print(f"\nGPU vs CPU Benchmark: {name}")
        print("-" * 50)
        
        comparison_results = []
        
        for size in problem_sizes:
            print(f"Size {size}: ", end='', flush=True)
            
            # Create circuit
            circuit = circuit_func(size)
            
            # CPU timing
            cpu_times = []
            for _ in range(3):
                start = time.perf_counter()
                try:
                    result_cpu = circuit.run(backend='cpu')
                    cpu_times.append(time.perf_counter() - start)
                except Exception as e:
                    print(f"CPU error: {e}")
                    cpu_times.append(float('inf'))
            
            cpu_time = min(cpu_times) if cpu_times else float('inf')
            
            # GPU timing
            gpu_time = float('inf')
            gpu_available = False
            
            try:
                gpu_times = []
                for _ in range(3):
                    start = time.perf_counter()
                    result_gpu = circuit.run(backend='gpu')
                    gpu_times.append(time.perf_counter() - start)
                
                gpu_time = min(gpu_times)
                gpu_available = True
            except Exception as e:
                print(f"GPU not available: {e}")
            
            if gpu_available:
                speedup = cpu_time / gpu_time
                print(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {speedup:.2f}x")
            else:
                speedup = 0
                print(f"CPU: {cpu_time:.3f}s, GPU: N/A")
            
            comparison_results.append({
                'size': size,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'gpu_available': gpu_available
            })
        
        # Plot comparison
        df = pd.DataFrame(comparison_results)
        df_gpu = df[df['gpu_available']]
        
        if not df_gpu.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time comparison
            x = range(len(df_gpu))
            width = 0.35
            ax1.bar([i - width/2 for i in x], df_gpu['cpu_time'], width, label='CPU')
            ax1.bar([i + width/2 for i in x], df_gpu['gpu_time'], width, label='GPU')
            ax1.set_xlabel('Problem Size')
            ax1.set_ylabel('Time (s)')
            ax1.set_title(f'Execution Time: {name}')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_gpu['size'])
            ax1.legend()
            ax1.set_yscale('log')
            
            # Speedup plot
            ax2.plot(df_gpu['size'], df_gpu['speedup'], 'o-')
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Problem Size')
            ax2.set_ylabel('GPU Speedup')
            ax2.set_title('GPU Acceleration Factor')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{name.replace(" ", "_")}_gpu_comparison.png')
            plt.close()
        
        self.results.append({
            'name': name,
            'type': 'gpu_comparison',
            'data': comparison_results
        })
        
        return comparison_results
    
    def benchmark_batch_parallelism(self, operation: Callable, name: str,
                                  batch_sizes: List[int],
                                  n_items: int = 1000):
        """Test batch processing efficiency."""
        print(f"\nBatch Parallelism Benchmark: {name}")
        print("-" * 50)
        
        batch_results = []
        
        for batch_size in batch_sizes:
            print(f"Batch size {batch_size}: ", end='', flush=True)
            
            n_batches = (n_items + batch_size - 1) // batch_size
            
            start = time.perf_counter()
            results = []
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_items)
                batch_items = list(range(batch_start, batch_end))
                
                # Process batch
                batch_results_temp = operation(batch_items)
                results.extend(batch_results_temp)
            
            elapsed = time.perf_counter() - start
            throughput = n_items / elapsed
            
            print(f"{elapsed:.3f}s ({throughput:.0f} items/s)")
            
            batch_results.append({
                'batch_size': batch_size,
                'time': elapsed,
                'throughput': throughput,
                'n_batches': n_batches
            })
        
        # Plot results
        df = pd.DataFrame(batch_results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Throughput vs batch size
        ax1.semilogx(df['batch_size'], df['throughput'], 'o-')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (items/s)')
        ax1.set_title(f'Batch Processing: {name}')
        ax1.grid(True, alpha=0.3)
        
        # Time vs batch size
        ax2.loglog(df['batch_size'], df['time'], 'o-')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Total Time (s)')
        ax2.set_title('Processing Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name.replace(" ", "_")}_batch.png')
        plt.close()
        
        self.results.append({
            'name': name,
            'type': 'batch_parallelism',
            'data': batch_results
        })
        
        return batch_results
    
    def save_results(self):
        """Save all benchmark results."""
        # Save raw results
        with open(self.output_dir / 'parallel_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary
        summary = {
            'system_info': {
                'physical_cores': self.cpu_count,
                'logical_cores': self.thread_count,
                'cpu_name': platform.processor()
            },
            'benchmarks_run': len(self.results),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")


# Benchmark implementations
def parallel_circuit_simulation(work_id: int) -> Dict:
    """Simulate a quantum circuit (for thread scaling)."""
    if not HAS_QUANTRS2:
        # Mock implementation
        time.sleep(0.01)
        return {'work_id': work_id, 'result': np.random.random()}
    
    # Create random circuit
    n_qubits = 10
    circuit = quantrs2.Circuit(n_qubits)
    
    np.random.seed(work_id)
    for _ in range(20):  # depth
        q = np.random.randint(n_qubits)
        gate = np.random.choice(['h', 'x', 'rx'])
        
        if gate in ['h', 'x']:
            getattr(circuit, gate)(q)
        else:
            angle = np.random.uniform(0, 2*np.pi)
            circuit.rx(q, angle)
        
        # Add entanglement
        if q < n_qubits - 1:
            circuit.cx(q, q + 1)
    
    result = circuit.run()
    probs = result.probabilities()
    
    return {
        'work_id': work_id,
        'max_prob': max(probs),
        'entropy': -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
    }


def create_benchmark_circuit(n_qubits: int):
    """Create a circuit for GPU benchmarking."""
    if not HAS_QUANTRS2:
        raise ImportError("quantrs2 not available")
    
    circuit = quantrs2.Circuit(n_qubits)
    
    # Layer of Hadamards
    for i in range(n_qubits):
        circuit.h(i)
    
    # Entangling layers
    for layer in range(3):
        for i in range(0, n_qubits - 1, 2):
            if i + 1 < n_qubits:
                circuit.cx(i, i + 1)
        
        for i in range(1, n_qubits - 1, 2):
            if i + 1 < n_qubits:
                circuit.cx(i, i + 1)
    
    # Rotation layer
    for i in range(n_qubits):
        circuit.rx(i, np.pi / 4)
        circuit.ry(i, np.pi / 3)
    
    return circuit


def batch_vqe_optimization(circuit_indices: List[int]) -> List[Dict]:
    """Batch VQE optimization (for batch parallelism)."""
    if not HAS_QUANTRS2 or not hasattr(quantrs2, 'ml'):
        # Mock implementation
        return [{'index': i, 'energy': np.random.random()} for i in circuit_indices]
    
    from quantrs2.ml import VQE, RealAmplitudes
    
    results = []
    
    for idx in circuit_indices:
        # Create simple Hamiltonian
        n_qubits = 4
        hamiltonian = [
            ('ZZ', [0, 1], 1.0),
            ('ZZ', [1, 2], 1.0),
            ('ZZ', [2, 3], 1.0),
            ('X', [0], 0.5),
            ('X', [3], 0.5)
        ]
        
        # Random initial parameters
        np.random.seed(idx)
        
        ansatz = RealAmplitudes(n_qubits, 2)
        vqe = VQE(ansatz, hamiltonian, optimizer='L-BFGS-B')
        
        # Quick optimization
        result = vqe.minimize(max_iter=5)
        
        results.append({
            'index': idx,
            'energy': result.fun if hasattr(result, 'fun') else 0,
            'converged': result.success if hasattr(result, 'success') else False
        })
    
    return results


def run_parallel_benchmarks():
    """Run complete parallel benchmark suite."""
    import platform
    
    benchmark = ParallelBenchmark()
    
    print(f"System: {benchmark.cpu_count} physical cores, {benchmark.thread_count} logical cores")
    
    # Test 1: Thread scaling for circuit simulation
    thread_counts = [1, 2, 4, 8, benchmark.thread_count]
    thread_counts = [t for t in thread_counts if t <= benchmark.thread_count]
    
    benchmark.benchmark_thread_scaling(
        parallel_circuit_simulation,
        "Circuit Simulation",
        thread_counts=thread_counts,
        workload_size=50
    )
    
    # Test 2: GPU vs CPU comparison
    if HAS_QUANTRS2:
        problem_sizes = [5, 10, 15, 20]
        benchmark.benchmark_gpu_vs_cpu(
            create_benchmark_circuit,
            "State Vector Simulation",
            problem_sizes=problem_sizes
        )
    
    # Test 3: Batch processing efficiency
    batch_sizes = [1, 5, 10, 20, 50, 100]
    benchmark.benchmark_batch_parallelism(
        batch_vqe_optimization,
        "VQE Optimization",
        batch_sizes=batch_sizes,
        n_items=200
    )
    
    # Save results
    benchmark.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("Parallel Benchmark Summary")
    print("="*60)
    print(f"Total benchmarks: {len(benchmark.results)}")
    
    # Find best configurations
    for result in benchmark.results:
        if result['type'] == 'thread_scaling':
            data = result['data']
            best = max(data, key=lambda x: x['speedup'])
            print(f"\n{result['name']} - Best speedup: {best['speedup']:.2f}x with {best['threads']} threads")
        
        elif result['type'] == 'gpu_comparison':
            data = result['data']
            gpu_data = [d for d in data if d['gpu_available'] and d['speedup'] > 0]
            if gpu_data:
                best = max(gpu_data, key=lambda x: x['speedup'])
                print(f"\n{result['name']} - Best GPU speedup: {best['speedup']:.2f}x at size {best['size']}")
        
        elif result['type'] == 'batch_parallelism':
            data = result['data']
            best = max(data, key=lambda x: x['throughput'])
            print(f"\n{result['name']} - Best throughput: {best['throughput']:.0f} items/s with batch size {best['batch_size']}")


if __name__ == "__main__":
    run_parallel_benchmarks()