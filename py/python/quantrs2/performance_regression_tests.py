"""
Performance Regression Testing Framework for QuantRS2

This module provides comprehensive performance regression testing capabilities to ensure
that new changes don't introduce performance degradations.
"""

import time
import os
import json
import pickle
import hashlib
import statistics
import warnings
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from quantrs2 import Circuit
    from quantrs2.profiler import CircuitProfiler
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    warnings.warn("QuantRS2 core not available. Using mock implementations.")
    
    class Circuit:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits
            self.gates = []
        def h(self, qubit): self.gates.append(('h', qubit))
        def x(self, qubit): self.gates.append(('x', qubit))
        def cnot(self, control, target): self.gates.append(('cnot', control, target))
        def run(self): return MockResult()
    
    class MockResult:
        def __init__(self):
            self.state_vector = np.array([0.7071, 0, 0, 0.7071])
    
    class CircuitProfiler:
        def profile_circuit(self, circuit):
            return MockProfileResult()
    
    class MockProfileResult:
        def __init__(self):
            self.total_time = 0.001
            self.gate_times = {'h': 0.0005, 'cnot': 0.0005}
            self.memory_usage = 1024


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    benchmark_name: str
    execution_time: float
    memory_usage: int
    additional_metrics: Dict[str, float]
    environment_info: Dict[str, str]
    timestamp: datetime
    commit_hash: Optional[str] = None


@dataclass
class RegressionThreshold:
    """Threshold configuration for regression detection."""
    metric_name: str
    max_degradation_percent: float
    max_absolute_change: Optional[float] = None
    min_samples: int = 3


class PerformanceDatabase:
    """Database for storing performance benchmarks."""
    
    def __init__(self, db_path: str = "performance_results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.load_results()
    
    def save_results(self):
        """Save results to disk."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.results, f)
        except Exception as e:
            warnings.warn(f"Failed to save performance results: {e}")
    
    def load_results(self):
        """Load results from disk."""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'rb') as f:
                    self.results = pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load performance results: {e}")
            self.results = []
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
        self.save_results()
    
    def get_results_for_benchmark(self, benchmark_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific benchmark."""
        return [r for r in self.results if r.benchmark_name == benchmark_name]
    
    def get_recent_results(self, benchmark_name: str, count: int = 10) -> List[BenchmarkResult]:
        """Get recent results for a benchmark."""
        results = self.get_results_for_benchmark(benchmark_name)
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:count]


class QuantumBenchmarkSuite:
    """Suite of quantum computing benchmarks."""
    
    def __init__(self):
        self.benchmarks: Dict[str, Callable] = {}
        self.setup_default_benchmarks()
    
    def setup_default_benchmarks(self):
        """Setup default quantum computing benchmarks."""
        self.register_benchmark("bell_state_creation", self.benchmark_bell_state)
        self.register_benchmark("grover_3_qubits", self.benchmark_grover_3q)
        self.register_benchmark("qft_4_qubits", self.benchmark_qft_4q)
        self.register_benchmark("random_circuit_5q_depth20", self.benchmark_random_circuit_5q)
        self.register_benchmark("vqe_h2_molecule", self.benchmark_vqe_h2)
        self.register_benchmark("circuit_compilation", self.benchmark_circuit_compilation)
        self.register_benchmark("state_vector_simulation", self.benchmark_state_vector_sim)
        self.register_benchmark("measurement_sampling", self.benchmark_measurement_sampling)
    
    def register_benchmark(self, name: str, func: Callable):
        """Register a benchmark function."""
        self.benchmarks[name] = func
    
    def benchmark_bell_state(self) -> Dict[str, float]:
        """Benchmark Bell state creation and simulation."""
        start_time = time.time()
        
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Simulate multiple times
        for _ in range(100):
            result = circuit.run()
        
        execution_time = time.time() - start_time
        
        return {
            'execution_time': execution_time,
            'operations_per_second': 100 / execution_time,
            'circuit_depth': 2,
            'qubit_count': 2
        }
    
    def benchmark_grover_3q(self) -> Dict[str, float]:
        """Benchmark 3-qubit Grover's algorithm."""
        start_time = time.time()
        
        circuit = Circuit(3)
        
        # Initial superposition
        for i in range(3):
            circuit.h(i)
        
        # Grover iterations (simplified)
        for _ in range(2):
            # Oracle (mark state |101>)
            circuit.x(0)
            circuit.x(2)
            # Multi-controlled Z would go here
            circuit.x(0)
            circuit.x(2)
            
            # Diffusion operator
            for i in range(3):
                circuit.h(i)
                circuit.x(i)
            # Multi-controlled Z would go here
            for i in range(3):
                circuit.x(i)
                circuit.h(i)
        
        result = circuit.run()
        execution_time = time.time() - start_time
        
        return {
            'execution_time': execution_time,
            'circuit_depth': 14,  # Approximate
            'qubit_count': 3,
            'algorithm_complexity': 'quadratic_speedup'
        }
    
    def benchmark_qft_4q(self) -> Dict[str, float]:
        """Benchmark 4-qubit Quantum Fourier Transform."""
        start_time = time.time()
        
        circuit = Circuit(4)
        
        # QFT implementation (simplified)
        for i in range(4):
            circuit.h(i)
            # Controlled rotations would go here
            # Using CNOTs as approximation
            for j in range(i + 1, 4):
                circuit.cnot(i, j)
        
        result = circuit.run()
        execution_time = time.time() - start_time
        
        return {
            'execution_time': execution_time,
            'circuit_depth': 10,  # Approximate
            'qubit_count': 4,
            'gate_count': 10
        }
    
    def benchmark_random_circuit_5q(self) -> Dict[str, float]:
        """Benchmark random circuit with 5 qubits and depth 20."""
        start_time = time.time()
        
        circuit = Circuit(5)
        
        # Random circuit (deterministic for consistency)
        np.random.seed(42)
        for layer in range(20):
            for qubit in range(5):
                if np.random.random() < 0.3:
                    circuit.h(qubit)
                elif np.random.random() < 0.6:
                    circuit.x(qubit)
            
            # Entangling layer
            for qubit in range(4):
                if np.random.random() < 0.5:
                    circuit.cnot(qubit, qubit + 1)
        
        result = circuit.run()
        execution_time = time.time() - start_time
        
        return {
            'execution_time': execution_time,
            'circuit_depth': 20,
            'qubit_count': 5,
            'gate_count': 60  # Approximate
        }
    
    def benchmark_vqe_h2(self) -> Dict[str, float]:
        """Benchmark VQE for H2 molecule (simplified)."""
        start_time = time.time()
        
        # Simplified VQE ansatz
        def create_ansatz(params):
            circuit = Circuit(2)
            circuit.ry(0, params[0])
            circuit.ry(1, params[1])
            circuit.cnot(0, 1)
            circuit.ry(0, params[2])
            circuit.ry(1, params[3])
            return circuit
        
        # Mock optimization loop
        best_energy = float('inf')
        params = [0.5, 1.0, 1.5, 2.0]
        
        for iteration in range(10):
            circuit = create_ansatz(params)
            result = circuit.run()
            
            # Mock energy calculation
            energy = sum(p**2 for p in params) - 1.0
            if energy < best_energy:
                best_energy = energy
            
            # Update parameters (mock optimization)
            params = [p + 0.1 * np.random.randn() for p in params]
        
        execution_time = time.time() - start_time
        
        return {
            'execution_time': execution_time,
            'optimization_iterations': 10,
            'final_energy': best_energy,
            'convergence_time': execution_time
        }
    
    def benchmark_circuit_compilation(self) -> Dict[str, float]:
        """Benchmark circuit compilation and optimization."""
        start_time = time.time()
        
        # Create complex circuit
        circuit = Circuit(4)
        for i in range(4):
            circuit.h(i)
        for i in range(3):
            circuit.cnot(i, i + 1)
        circuit.cnot(3, 0)  # Ring connectivity
        
        # Simulate compilation process
        time.sleep(0.001)  # Mock compilation time
        
        compilation_time = time.time() - start_time
        
        return {
            'compilation_time': compilation_time,
            'original_gate_count': 7,
            'optimized_gate_count': 7,  # No optimization in mock
            'circuit_depth': 3
        }
    
    def benchmark_state_vector_sim(self) -> Dict[str, float]:
        """Benchmark state vector simulation."""
        start_time = time.time()
        
        # Large-ish circuit for simulation
        circuit = Circuit(6)
        for i in range(6):
            circuit.h(i)
        for i in range(5):
            circuit.cnot(i, i + 1)
        
        result = circuit.run()
        execution_time = time.time() - start_time
        
        return {
            'simulation_time': execution_time,
            'state_vector_size': 2**6,
            'qubit_count': 6,
            'simulation_method': 'state_vector'
        }
    
    def benchmark_measurement_sampling(self) -> Dict[str, float]:
        """Benchmark measurement sampling."""
        start_time = time.time()
        
        circuit = Circuit(3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)
        
        # Simulate measurement sampling
        measurements = []
        for _ in range(1000):
            result = circuit.run()
            # Mock measurement
            measurements.append(np.random.choice(['000', '111']))
        
        sampling_time = time.time() - start_time
        
        return {
            'sampling_time': sampling_time,
            'shots': 1000,
            'samples_per_second': 1000 / sampling_time,
            'circuit_qubits': 3
        }


class RegressionDetector:
    """Detect performance regressions in benchmarks."""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
        self.thresholds = self.setup_default_thresholds()
    
    def setup_default_thresholds(self) -> Dict[str, RegressionThreshold]:
        """Setup default regression thresholds."""
        return {
            'execution_time': RegressionThreshold('execution_time', 15.0, min_samples=3),
            'memory_usage': RegressionThreshold('memory_usage', 20.0, min_samples=3),
            'operations_per_second': RegressionThreshold('operations_per_second', -10.0, min_samples=3),
            'compilation_time': RegressionThreshold('compilation_time', 25.0, min_samples=3),
            'simulation_time': RegressionThreshold('simulation_time', 15.0, min_samples=3),
        }
    
    def detect_regressions(self, benchmark_name: str, 
                          latest_result: BenchmarkResult) -> List[Dict[str, Any]]:
        """Detect regressions in a benchmark result."""
        regressions = []
        historical_results = self.db.get_recent_results(benchmark_name, count=10)
        
        if len(historical_results) < 2:
            return regressions  # Need at least 2 results for comparison
        
        # Exclude the latest result from historical data
        historical_results = [r for r in historical_results if r.timestamp != latest_result.timestamp]
        
        if len(historical_results) < 1:
            return regressions
        
        # Check primary metrics
        primary_metrics = ['execution_time', 'memory_usage']
        for metric in primary_metrics:
            regression = self._check_metric_regression(
                metric, latest_result, historical_results
            )
            if regression:
                regressions.append(regression)
        
        # Check additional metrics
        for metric_name in latest_result.additional_metrics:
            if metric_name in self.thresholds:
                regression = self._check_additional_metric_regression(
                    metric_name, latest_result, historical_results
                )
                if regression:
                    regressions.append(regression)
        
        return regressions
    
    def _check_metric_regression(self, metric_name: str, 
                                latest_result: BenchmarkResult,
                                historical_results: List[BenchmarkResult]) -> Optional[Dict[str, Any]]:
        """Check if a primary metric has regressed."""
        if metric_name not in self.thresholds:
            return None
        
        threshold = self.thresholds[metric_name]
        
        # Get latest value
        if metric_name == 'execution_time':
            latest_value = latest_result.execution_time
        elif metric_name == 'memory_usage':
            latest_value = latest_result.memory_usage
        else:
            return None
        
        # Get historical values
        historical_values = []
        for result in historical_results:
            if metric_name == 'execution_time':
                historical_values.append(result.execution_time)
            elif metric_name == 'memory_usage':
                historical_values.append(result.memory_usage)
        
        if len(historical_values) < threshold.min_samples:
            return None
        
        # Calculate baseline (median of historical values)
        baseline = statistics.median(historical_values)
        
        # Check for regression
        percent_change = ((latest_value - baseline) / baseline) * 100
        
        is_regression = False
        if metric_name in ['execution_time', 'memory_usage']:
            # Higher is worse for these metrics
            is_regression = percent_change > threshold.max_degradation_percent
        else:
            # Lower is worse for performance metrics
            is_regression = percent_change < threshold.max_degradation_percent
        
        if is_regression:
            return {
                'metric': metric_name,
                'latest_value': latest_value,
                'baseline_value': baseline,
                'percent_change': percent_change,
                'threshold': threshold.max_degradation_percent,
                'severity': self._calculate_severity(abs(percent_change), threshold.max_degradation_percent),
                'historical_samples': len(historical_values)
            }
        
        return None
    
    def _check_additional_metric_regression(self, metric_name: str,
                                          latest_result: BenchmarkResult,
                                          historical_results: List[BenchmarkResult]) -> Optional[Dict[str, Any]]:
        """Check if an additional metric has regressed."""
        if metric_name not in latest_result.additional_metrics:
            return None
        
        latest_value = latest_result.additional_metrics[metric_name]
        
        # Get historical values
        historical_values = []
        for result in historical_results:
            if metric_name in result.additional_metrics:
                historical_values.append(result.additional_metrics[metric_name])
        
        threshold = self.thresholds.get(metric_name)
        if not threshold or len(historical_values) < threshold.min_samples:
            return None
        
        baseline = statistics.median(historical_values)
        percent_change = ((latest_value - baseline) / baseline) * 100
        
        # Determine if this is a regression based on metric type
        is_regression = False
        if 'time' in metric_name.lower() or 'memory' in metric_name.lower():
            is_regression = percent_change > threshold.max_degradation_percent
        else:
            # Performance metrics (higher is better)
            is_regression = percent_change < threshold.max_degradation_percent
        
        if is_regression:
            return {
                'metric': metric_name,
                'latest_value': latest_value,
                'baseline_value': baseline,
                'percent_change': percent_change,
                'threshold': threshold.max_degradation_percent,
                'severity': self._calculate_severity(abs(percent_change), threshold.max_degradation_percent),
                'historical_samples': len(historical_values)
            }
        
        return None
    
    def _calculate_severity(self, percent_change: float, threshold: float) -> str:
        """Calculate severity of regression."""
        if percent_change > threshold * 2:
            return "critical"
        elif percent_change > threshold * 1.5:
            return "high"
        elif percent_change > threshold:
            return "medium"
        else:
            return "low"


class PerformanceRegressionRunner:
    """Main class for running performance regression tests."""
    
    def __init__(self, db_path: str = "performance_results.db"):
        self.db = PerformanceDatabase(db_path)
        self.benchmark_suite = QuantumBenchmarkSuite()
        self.regression_detector = RegressionDetector(self.db)
        self.environment_info = self._collect_environment_info()
    
    def _collect_environment_info(self) -> Dict[str, str]:
        """Collect environment information."""
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'quantrs2_version': getattr(__import__('quantrs2'), '__version__', 'unknown'),
            'numpy_version': np.__version__,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_all_benchmarks(self, detect_regressions: bool = True) -> Dict[str, Any]:
        """Run all benchmarks and detect regressions."""
        results = {}
        all_regressions = []
        
        for benchmark_name, benchmark_func in self.benchmark_suite.benchmarks.items():
            print(f"Running benchmark: {benchmark_name}")
            result = self.run_single_benchmark(benchmark_name, benchmark_func)
            results[benchmark_name] = result
            
            if detect_regressions:
                regressions = self.regression_detector.detect_regressions(
                    benchmark_name, result
                )
                if regressions:
                    all_regressions.extend([
                        {**reg, 'benchmark': benchmark_name} for reg in regressions
                    ])
        
        return {
            'results': results,
            'regressions': all_regressions,
            'environment': self.environment_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_single_benchmark(self, name: str, benchmark_func: Callable) -> BenchmarkResult:
        """Run a single benchmark."""
        # Memory usage before
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Run benchmark
        start_time = time.time()
        metrics = benchmark_func()
        execution_time = time.time() - start_time
        
        # Memory usage after
        memory_after = process.memory_info().rss
        memory_usage = memory_after - memory_before
        
        # Create result
        result = BenchmarkResult(
            benchmark_name=name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            additional_metrics=metrics,
            environment_info=self.environment_info,
            timestamp=datetime.now(),
            commit_hash=self._get_git_commit_hash()
        )
        
        # Store in database
        self.db.add_result(result)
        
        return result
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def generate_regression_report(self, regressions: List[Dict[str, Any]]) -> str:
        """Generate a human-readable regression report."""
        if not regressions:
            return "✅ No performance regressions detected!"
        
        report = ["🚨 Performance Regressions Detected!", "=" * 50, ""]
        
        # Group by severity
        by_severity = {}
        for regression in regressions:
            severity = regression['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(regression)
        
        # Report by severity
        severity_order = ['critical', 'high', 'medium', 'low']
        for severity in severity_order:
            if severity in by_severity:
                report.append(f"{severity.upper()} SEVERITY:")
                report.append("-" * 20)
                
                for reg in by_severity[severity]:
                    report.append(f"Benchmark: {reg['benchmark']}")
                    report.append(f"Metric: {reg['metric']}")
                    report.append(f"Change: {reg['percent_change']:.1f}% "
                                f"(threshold: {reg['threshold']:.1f}%)")
                    report.append(f"Latest: {reg['latest_value']:.6f}")
                    report.append(f"Baseline: {reg['baseline_value']:.6f}")
                    report.append("")
        
        report.append("Recommendations:")
        report.append("- Review recent changes for performance impact")
        report.append("- Profile affected benchmarks for optimization opportunities")
        report.append("- Consider if regression is acceptable for new features")
        
        return "\n".join(report)
    
    def export_results(self, filename: str = "performance_report.json"):
        """Export all results to JSON file."""
        export_data = {
            'results': [asdict(result) for result in self.db.results],
            'environment': self.environment_info,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Performance results exported to {filename}")


# Convenience functions
def run_performance_regression_tests(db_path: str = "performance_results.db") -> Dict[str, Any]:
    """Run performance regression tests."""
    runner = PerformanceRegressionRunner(db_path)
    return runner.run_all_benchmarks()


def detect_performance_regressions(benchmark_name: str, 
                                 db_path: str = "performance_results.db") -> List[Dict[str, Any]]:
    """Detect regressions for a specific benchmark."""
    db = PerformanceDatabase(db_path)
    detector = RegressionDetector(db)
    
    recent_results = db.get_recent_results(benchmark_name, count=1)
    if not recent_results:
        return []
    
    return detector.detect_regressions(benchmark_name, recent_results[0])


def benchmark_quantum_operations(operations: List[str] = None) -> Dict[str, float]:
    """Benchmark specific quantum operations."""
    if operations is None:
        operations = ['bell_state', 'grover', 'qft']
    
    suite = QuantumBenchmarkSuite()
    results = {}
    
    for op in operations:
        if op == 'bell_state':
            results['bell_state'] = suite.benchmark_bell_state()
        elif op == 'grover':
            results['grover'] = suite.benchmark_grover_3q()
        elif op == 'qft':
            results['qft'] = suite.benchmark_qft_4q()
    
    return results


def setup_ci_performance_tests(threshold_degradation: float = 10.0):
    """Setup performance tests for CI/CD pipeline."""
    runner = PerformanceRegressionRunner()
    
    # Run benchmarks
    results = runner.run_all_benchmarks(detect_regressions=True)
    
    # Check for regressions
    regressions = results['regressions']
    if regressions:
        # Check if any critical regressions
        critical_regressions = [r for r in regressions if r['severity'] == 'critical']
        if critical_regressions:
            report = runner.generate_regression_report(regressions)
            print(report)
            raise Exception("Critical performance regressions detected!")
        else:
            print("⚠️ Performance regressions detected but not critical")
            print(runner.generate_regression_report(regressions))
    
    return results


# Export main classes and functions
__all__ = [
    'PerformanceMetric',
    'BenchmarkResult', 
    'RegressionThreshold',
    'PerformanceDatabase',
    'QuantumBenchmarkSuite',
    'RegressionDetector',
    'PerformanceRegressionRunner',
    'run_performance_regression_tests',
    'detect_performance_regressions',
    'benchmark_quantum_operations',
    'setup_ci_performance_tests'
]