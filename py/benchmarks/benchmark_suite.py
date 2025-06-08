#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for QuantRS2

This module provides a complete benchmarking framework for testing performance
of various QuantRS2 components including simulators, ML algorithms, and hardware backends.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import json
import platform
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import concurrent.futures
import traceback

try:
    import quantrs2
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False
    print("Warning: quantrs2 not available, using mock implementations")


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run."""
    name: str
    category: str
    parameters: Dict[str, Any]
    execution_time: float
    memory_usage: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())


@dataclass
class SystemInfo:
    """Store system information for reproducibility."""
    platform: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())
    cpu_count: int = field(default_factory=lambda: psutil.cpu_count())
    memory_total_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / 1e9)
    gpu_available: bool = field(default_factory=lambda: check_gpu_availability())
    quantrs2_version: str = field(default_factory=lambda: get_quantrs2_version())


def check_gpu_availability() -> bool:
    """Check if GPU is available for computation."""
    try:
        import quantrs2
        # Try to create a simple circuit with GPU
        circuit = quantrs2.Circuit(1)
        circuit.h(0)
        result = circuit.run(backend="gpu")
        return True
    except:
        return False


def get_quantrs2_version() -> str:
    """Get QuantRS2 version."""
    try:
        import quantrs2
        return getattr(quantrs2, '__version__', 'unknown')
    except:
        return 'not_installed'


class BenchmarkSuite:
    """Comprehensive benchmarking suite for QuantRS2."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.system_info = SystemInfo()
        
    def run_benchmark(
        self,
        func: Callable,
        name: str,
        category: str,
        parameters: Dict[str, Any],
        warmup_runs: int = 2,
        test_runs: int = 10
    ) -> BenchmarkResult:
        """Run a single benchmark with timing and memory profiling."""
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                func(**parameters)
            except Exception:
                pass
        
        # Actual benchmark runs
        times = []
        memory_usages = []
        additional_metrics = {}
        
        for _ in range(test_runs):
            # Memory before
            memory_before = psutil.Process().memory_info().rss / 1e6  # MB
            
            # Time execution
            start_time = time.perf_counter()
            try:
                result = func(**parameters)
                end_time = time.perf_counter()
                
                # Extract additional metrics if available
                if isinstance(result, dict) and 'metrics' in result:
                    for key, value in result['metrics'].items():
                        if key not in additional_metrics:
                            additional_metrics[key] = []
                        additional_metrics[key].append(value)
                
                times.append(end_time - start_time)
                
                # Memory after
                memory_after = psutil.Process().memory_info().rss / 1e6  # MB
                memory_usages.append(memory_after - memory_before)
                
            except Exception as e:
                return BenchmarkResult(
                    name=name,
                    category=category,
                    parameters=parameters,
                    execution_time=-1,
                    memory_usage=-1,
                    error=str(e) + "\n" + traceback.format_exc()
                )
        
        # Calculate statistics
        avg_time = np.mean(times)
        avg_memory = np.mean(memory_usages)
        
        # Average additional metrics
        avg_additional_metrics = {
            key: np.mean(values) for key, values in additional_metrics.items()
        }
        avg_additional_metrics['time_std'] = np.std(times)
        avg_additional_metrics['memory_std'] = np.std(memory_usages)
        
        result = BenchmarkResult(
            name=name,
            category=category,
            parameters=parameters,
            execution_time=avg_time,
            memory_usage=avg_memory,
            additional_metrics=avg_additional_metrics
        )
        
        self.results.append(result)
        return result
    
    def run_category(self, category: str, benchmarks: List[Tuple[str, Callable, Dict]]):
        """Run all benchmarks in a category."""
        print(f"\n{'='*60}")
        print(f"Running {category} benchmarks")
        print('='*60)
        
        for name, func, params_list in benchmarks:
            print(f"\n{name}:")
            for params in params_list:
                param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                print(f"  Parameters: {param_str}")
                
                result = self.run_benchmark(
                    func=func,
                    name=name,
                    category=category,
                    parameters=params
                )
                
                if result.error:
                    print(f"    ERROR: {result.error.split(chr(10))[0]}")
                else:
                    print(f"    Time: {result.execution_time*1000:.2f} ms")
                    print(f"    Memory: {result.memory_usage:.2f} MB")
                    if result.additional_metrics:
                        for key, value in result.additional_metrics.items():
                            if key not in ['time_std', 'memory_std']:
                                print(f"    {key}: {value}")
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        results_dict = {
            'system_info': self.system_info.__dict__,
            'results': [r.__dict__ for r in self.results]
        }
        
        with open(self.output_dir / f"results_{timestamp}.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Create DataFrame for analysis
        df_data = []
        for r in self.results:
            row = {
                'name': r.name,
                'category': r.category,
                'execution_time_ms': r.execution_time * 1000,
                'memory_usage_mb': r.memory_usage,
                'error': r.error is not None
            }
            row.update(r.parameters)
            row.update(r.additional_metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(self.output_dir / f"results_{timestamp}.csv", index=False)
        
        # Generate plots
        self._generate_plots(df, timestamp)
        
        print(f"\nResults saved to {self.output_dir}")
        
        return df
    
    def _generate_plots(self, df: pd.DataFrame, timestamp: str):
        """Generate visualization plots."""
        # Filter out error results
        df_clean = df[~df['error']]
        
        if df_clean.empty:
            print("No successful benchmarks to plot")
            return
        
        # Create figure directory
        fig_dir = self.output_dir / f"figures_{timestamp}"
        fig_dir.mkdir(exist_ok=True)
        
        # Plot 1: Execution time by category
        if 'category' in df_clean.columns:
            plt.figure(figsize=(10, 6))
            df_clean.boxplot(column='execution_time_ms', by='category', rot=45)
            plt.title('Execution Time by Category')
            plt.ylabel('Time (ms)')
            plt.tight_layout()
            plt.savefig(fig_dir / 'time_by_category.png')
            plt.close()
        
        # Plot 2: Memory usage by category
        if 'category' in df_clean.columns:
            plt.figure(figsize=(10, 6))
            df_clean.boxplot(column='memory_usage_mb', by='category', rot=45)
            plt.title('Memory Usage by Category')
            plt.ylabel('Memory (MB)')
            plt.tight_layout()
            plt.savefig(fig_dir / 'memory_by_category.png')
            plt.close()
        
        # Plot 3: Scaling analysis (if n_qubits parameter exists)
        if 'n_qubits' in df_clean.columns:
            plt.figure(figsize=(10, 6))
            for category in df_clean['category'].unique():
                cat_data = df_clean[df_clean['category'] == category]
                if not cat_data.empty:
                    avg_times = cat_data.groupby('n_qubits')['execution_time_ms'].mean()
                    plt.plot(avg_times.index, avg_times.values, 'o-', label=category)
            
            plt.xlabel('Number of Qubits')
            plt.ylabel('Execution Time (ms)')
            plt.title('Scaling Analysis')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / 'scaling_analysis.png')
            plt.close()
    
    def compare_backends(self):
        """Generate backend comparison report."""
        df = pd.DataFrame([r.__dict__ for r in self.results])
        
        if 'backend' not in df.columns:
            print("No backend comparison data available")
            return
        
        # Group by backend and calculate statistics
        backend_stats = df.groupby('backend').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'memory_usage': ['mean', 'std', 'min', 'max']
        })
        
        print("\n" + "="*60)
        print("Backend Comparison")
        print("="*60)
        print(backend_stats)
        
        # Save comparison
        backend_stats.to_csv(self.output_dir / 'backend_comparison.csv')


# Benchmark implementations
def benchmark_circuit_simulation(**kwargs):
    """Benchmark basic circuit simulation."""
    if not HAS_QUANTRS2:
        return {'metrics': {'gates_per_second': 0}}
    
    n_qubits = kwargs.get('n_qubits', 5)
    depth = kwargs.get('depth', 10)
    backend = kwargs.get('backend', 'auto')
    
    circuit = quantrs2.Circuit(n_qubits)
    
    # Create random circuit
    gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
    np.random.seed(42)
    
    gate_count = 0
    for _ in range(depth):
        for q in range(n_qubits):
            gate = np.random.choice(gates)
            if gate == 'h':
                circuit.h(q)
            elif gate == 'x':
                circuit.x(q)
            elif gate == 'y':
                circuit.y(q)
            elif gate == 'z':
                circuit.z(q)
            elif gate in ['rx', 'ry', 'rz']:
                angle = np.random.uniform(0, 2*np.pi)
                getattr(circuit, gate)(q, angle)
            gate_count += 1
            
            # Add CNOT gates
            if q < n_qubits - 1 and np.random.random() < 0.3:
                circuit.cx(q, q + 1)
                gate_count += 1
    
    # Run simulation
    start = time.perf_counter()
    result = circuit.run(backend=backend)
    sim_time = time.perf_counter() - start
    
    return {
        'metrics': {
            'gates_per_second': gate_count / sim_time,
            'total_gates': gate_count,
            'simulation_time': sim_time
        }
    }


def benchmark_vqe_optimization(**kwargs):
    """Benchmark VQE optimization."""
    if not HAS_QUANTRS2 or not hasattr(quantrs2, 'ml'):
        return {'metrics': {'iterations_per_second': 0}}
    
    n_qubits = kwargs.get('n_qubits', 4)
    n_layers = kwargs.get('n_layers', 2)
    n_iterations = kwargs.get('n_iterations', 10)
    
    # Create simple Hamiltonian (Ising chain)
    hamiltonian = []
    for i in range(n_qubits - 1):
        hamiltonian.append(('ZZ', [i, i+1], 1.0))
    for i in range(n_qubits):
        hamiltonian.append(('X', [i], 0.5))
    
    # Initialize VQE
    from quantrs2.ml import VQE, RealAmplitudes
    
    ansatz = RealAmplitudes(n_qubits, n_layers)
    vqe = VQE(ansatz, hamiltonian, optimizer='L-BFGS-B')
    
    # Run optimization
    start = time.perf_counter()
    result = vqe.minimize(max_iter=n_iterations)
    opt_time = time.perf_counter() - start
    
    return {
        'metrics': {
            'iterations_per_second': n_iterations / opt_time,
            'final_energy': result.fun if hasattr(result, 'fun') else 0,
            'optimization_time': opt_time
        }
    }


def benchmark_quantum_annealing(**kwargs):
    """Benchmark quantum annealing simulation."""
    if not HAS_QUANTRS2 or not hasattr(quantrs2, 'anneal'):
        return {'metrics': {'samples_per_second': 0}}
    
    n_vars = kwargs.get('n_vars', 10)
    n_samples = kwargs.get('n_samples', 100)
    density = kwargs.get('density', 0.3)
    
    from quantrs2.anneal import QuboModel, SimulatedAnnealingSampler
    
    # Create random QUBO
    qubo = QuboModel(n_vars)
    np.random.seed(42)
    
    # Add linear terms
    for i in range(n_vars):
        qubo.add_linear(i, np.random.uniform(-1, 1))
    
    # Add quadratic terms
    n_quadratic = int(n_vars * (n_vars - 1) / 2 * density)
    pairs = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
    selected_pairs = np.random.choice(len(pairs), n_quadratic, replace=False)
    
    for idx in selected_pairs:
        i, j = pairs[idx]
        qubo.add_quadratic(i, j, np.random.uniform(-1, 1))
    
    # Run annealing
    sampler = SimulatedAnnealingSampler()
    
    start = time.perf_counter()
    results = sampler.sample(qubo, num_reads=n_samples)
    sample_time = time.perf_counter() - start
    
    return {
        'metrics': {
            'samples_per_second': n_samples / sample_time,
            'best_energy': min(r.energy for r in results),
            'sampling_time': sample_time
        }
    }


def benchmark_transfer_learning(**kwargs):
    """Benchmark quantum transfer learning."""
    if not HAS_QUANTRS2 or not hasattr(quantrs2, 'transfer_learning'):
        return {'metrics': {'adaptation_time': 0}}
    
    n_qubits = kwargs.get('n_qubits', 4)
    n_classes = kwargs.get('n_classes', 3)
    strategy = kwargs.get('strategy', 'fine_tuning')
    
    from quantrs2.transfer_learning import QuantumModelZoo, TransferLearningHelper
    
    # Load pretrained model
    start = time.perf_counter()
    model = QuantumModelZoo.vqe_feature_extractor(n_qubits)
    load_time = time.perf_counter() - start
    
    # Setup transfer learning
    helper = TransferLearningHelper(model, strategy)
    
    # Adapt model
    start = time.perf_counter()
    helper.adapt_for_classification(n_classes)
    adapt_time = time.perf_counter() - start
    
    # Get model info
    info = helper.get_model_info()
    
    return {
        'metrics': {
            'model_load_time': load_time,
            'adaptation_time': adapt_time,
            'trainable_parameters': info.get('trainable_parameters', 0)
        }
    }


def benchmark_visualization(**kwargs):
    """Benchmark visualization data preparation."""
    if not HAS_QUANTRS2 or not hasattr(quantrs2, 'tytan_viz'):
        return {'metrics': {'analysis_time': 0}}
    
    n_samples = kwargs.get('n_samples', 100)
    n_vars = kwargs.get('n_vars', 10)
    
    from quantrs2.tytan_viz import SampleResult, VisualizationHelper
    
    # Generate sample results
    np.random.seed(42)
    results = []
    
    for i in range(n_samples):
        assignments = {f'x{j}': bool(np.random.randint(2)) for j in range(n_vars)}
        energy = np.random.normal(0, 1)
        results.append(SampleResult(assignments, energy, 1))
    
    # Create visualization helper
    viz = VisualizationHelper(results)
    
    # Benchmark different analyses
    start = time.perf_counter()
    
    # Energy landscape
    energy_data = viz.prepare_energy_landscape(num_bins=30, compute_kde=True)
    landscape_time = time.perf_counter() - start
    
    # Solution distribution
    start = time.perf_counter()
    solution_data = viz.analyze_solutions(
        compute_correlations=True,
        compute_pca=True,
        n_components=2
    )
    solution_time = time.perf_counter() - start
    
    # Variable statistics
    start = time.perf_counter()
    stats = viz.get_variable_statistics()
    stats_time = time.perf_counter() - start
    
    return {
        'metrics': {
            'landscape_analysis_time': landscape_time,
            'solution_analysis_time': solution_time,
            'statistics_time': stats_time,
            'total_analysis_time': landscape_time + solution_time + stats_time
        }
    }


def main():
    """Run complete benchmark suite."""
    suite = BenchmarkSuite()
    
    # Define benchmark configurations
    circuit_benchmarks = [
        ("Basic Circuit Simulation", benchmark_circuit_simulation, [
            {'n_qubits': n, 'depth': 20, 'backend': 'cpu'}
            for n in [5, 10, 15, 20]
        ] + [
            {'n_qubits': n, 'depth': 20, 'backend': 'gpu'}
            for n in [5, 10, 15, 20]
        ] if suite.system_info.gpu_available else [])
    ]
    
    ml_benchmarks = [
        ("VQE Optimization", benchmark_vqe_optimization, [
            {'n_qubits': n, 'n_layers': 3, 'n_iterations': 20}
            for n in [4, 6, 8, 10]
        ]),
        ("Transfer Learning", benchmark_transfer_learning, [
            {'n_qubits': 4, 'n_classes': n, 'strategy': 'fine_tuning'}
            for n in [2, 3, 4, 5]
        ])
    ]
    
    anneal_benchmarks = [
        ("Quantum Annealing", benchmark_quantum_annealing, [
            {'n_vars': n, 'n_samples': 100, 'density': 0.3}
            for n in [10, 20, 30, 40, 50]
        ])
    ]
    
    viz_benchmarks = [
        ("Visualization Analysis", benchmark_visualization, [
            {'n_samples': n, 'n_vars': 20}
            for n in [100, 500, 1000, 2000]
        ])
    ]
    
    # Run all benchmarks
    suite.run_category("Circuit Simulation", circuit_benchmarks)
    suite.run_category("Machine Learning", ml_benchmarks)
    suite.run_category("Quantum Annealing", anneal_benchmarks)
    suite.run_category("Visualization", viz_benchmarks)
    
    # Save results and generate reports
    df = suite.save_results()
    suite.compare_backends()
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"Total benchmarks run: {len(suite.results)}")
    print(f"Successful: {sum(1 for r in suite.results if r.error is None)}")
    print(f"Failed: {sum(1 for r in suite.results if r.error is not None)}")
    
    # Print top performers
    if not df.empty:
        print("\nTop performers by category:")
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            if not cat_df.empty and 'execution_time_ms' in cat_df.columns:
                fastest = cat_df.nsmallest(1, 'execution_time_ms').iloc[0]
                print(f"  {category}: {fastest['name']} ({fastest['execution_time_ms']:.2f} ms)")


if __name__ == "__main__":
    main()