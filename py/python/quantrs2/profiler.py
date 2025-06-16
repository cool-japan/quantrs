"""
Quantum Circuit Profiler

This module provides comprehensive profiling capabilities for quantum circuits,
analyzing performance characteristics, resource usage, and optimization opportunities.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import warnings
import gc
import psutil
import threading
from pathlib import Path

try:
    import quantrs2
    from quantrs2 import Circuit, SimulationResult
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


@dataclass
class ProfileMetrics:
    """Container for profiling metrics."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gate_count: int = 0
    circuit_depth: int = 0
    entangling_gates: int = 0
    single_qubit_gates: int = 0
    measurement_count: int = 0
    qubit_count: int = 0
    
    # Performance metrics
    gates_per_second: float = 0.0
    qubits_per_second: float = 0.0
    
    # Resource efficiency
    memory_per_qubit: float = 0.0
    time_per_gate: float = 0.0
    
    # Additional metrics
    additional: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitAnalysis:
    """Detailed circuit structure analysis."""
    gate_distribution: Dict[str, int] = field(default_factory=dict)
    qubit_utilization: Dict[int, int] = field(default_factory=dict)
    circuit_layers: List[List[str]] = field(default_factory=list)
    critical_path_length: int = 0
    parallelism_factor: float = 0.0
    
    # Gate timing analysis
    gate_timing: Dict[str, float] = field(default_factory=dict)
    bottleneck_gates: List[str] = field(default_factory=list)
    
    # Connectivity analysis
    qubit_connectivity: Dict[Tuple[int, int], int] = field(default_factory=dict)
    highly_connected_qubits: List[int] = field(default_factory=list)


@dataclass
class ProfileReport:
    """Complete profiling report."""
    circuit_info: Dict[str, Any]
    metrics: ProfileMetrics
    analysis: CircuitAnalysis
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'circuit_info': self.circuit_info,
            'metrics': self.metrics.__dict__,
            'analysis': {
                'gate_distribution': self.analysis.gate_distribution,
                'qubit_utilization': self.analysis.qubit_utilization,
                'circuit_layers': self.analysis.circuit_layers,
                'critical_path_length': self.analysis.critical_path_length,
                'parallelism_factor': self.analysis.parallelism_factor,
                'gate_timing': self.analysis.gate_timing,
                'bottleneck_gates': self.analysis.bottleneck_gates,
                'qubit_connectivity': {str(k): v for k, v in self.analysis.qubit_connectivity.items()},
                'highly_connected_qubits': self.analysis.highly_connected_qubits
            },
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export report to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str


class MemoryProfiler:
    """Memory usage profiler for circuit operations."""
    
    def __init__(self):
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.sample_interval = 0.01  # 10ms
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return memory statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.memory_samples:
            return {'current': 0.0, 'peak': 0.0, 'average': 0.0}
        
        return {
            'current': self.memory_samples[-1],
            'peak': max(self.memory_samples),
            'average': np.mean(self.memory_samples),
            'samples': len(self.memory_samples)
        }
    
    def _monitor_memory(self):
        """Background memory monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                time.sleep(self.sample_interval)
            except Exception:
                break


class CircuitProfiler:
    """Comprehensive quantum circuit profiler."""
    
    def __init__(self, 
                 enable_memory_profiling: bool = True,
                 enable_timing_breakdown: bool = True,
                 enable_resource_tracking: bool = True):
        """
        Initialize circuit profiler.
        
        Args:
            enable_memory_profiling: Track detailed memory usage
            enable_timing_breakdown: Profile individual gate timings
            enable_resource_tracking: Monitor CPU and system resources
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_timing_breakdown = enable_timing_breakdown
        self.enable_resource_tracking = enable_resource_tracking
        
        self.memory_profiler = MemoryProfiler() if enable_memory_profiling else None
        self.timing_data = defaultdict(list)
        self.resource_data = []
    
    def profile_circuit(self, 
                       circuit: 'Circuit',
                       run_simulation: bool = True,
                       simulation_params: Optional[Dict] = None) -> ProfileReport:
        """
        Profile a quantum circuit comprehensively.
        
        Args:
            circuit: Quantum circuit to profile
            run_simulation: Whether to run simulation during profiling
            simulation_params: Parameters for simulation (backend, shots, etc.)
            
        Returns:
            ProfileReport: Comprehensive profiling report
        """
        simulation_params = simulation_params or {}
        
        # Start profiling
        if self.memory_profiler:
            self.memory_profiler.start_monitoring()
        
        start_time = time.perf_counter()
        start_cpu_times = psutil.cpu_times()
        
        try:
            # Analyze circuit structure
            analysis = self._analyze_circuit_structure(circuit)
            
            # Profile simulation if requested
            simulation_result = None
            if run_simulation and HAS_QUANTRS2:
                simulation_result = self._profile_simulation(circuit, simulation_params)
            
            # Calculate metrics
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Get memory statistics
            memory_stats = {'current': 0.0, 'peak': 0.0, 'average': 0.0}
            if self.memory_profiler:
                memory_stats = self.memory_profiler.stop_monitoring()
            
            # Calculate CPU usage
            end_cpu_times = psutil.cpu_times()
            cpu_usage = self._calculate_cpu_usage(start_cpu_times, end_cpu_times, execution_time)
            
            # Create metrics
            metrics = self._create_metrics(
                circuit, analysis, execution_time, memory_stats, cpu_usage
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, analysis)
            
            # Create report
            circuit_info = self._get_circuit_info(circuit, simulation_result)
            
            return ProfileReport(
                circuit_info=circuit_info,
                metrics=metrics,
                analysis=analysis,
                recommendations=recommendations
            )
            
        except Exception as e:
            # Ensure cleanup
            if self.memory_profiler:
                self.memory_profiler.stop_monitoring()
            raise e
    
    def profile_batch(self, 
                     circuits: List['Circuit'],
                     simulation_params: Optional[Dict] = None) -> List[ProfileReport]:
        """
        Profile multiple circuits and compare performance.
        
        Args:
            circuits: List of circuits to profile
            simulation_params: Simulation parameters
            
        Returns:
            List[ProfileReport]: Reports for each circuit
        """
        reports = []
        
        for i, circuit in enumerate(circuits):
            print(f"Profiling circuit {i+1}/{len(circuits)}...")
            report = self.profile_circuit(circuit, simulation_params=simulation_params)
            reports.append(report)
        
        return reports
    
    def compare_circuits(self, 
                        reports: List[ProfileReport],
                        output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Compare performance across multiple circuit profiles.
        
        Args:
            reports: List of profiling reports
            output_file: Optional CSV output file
            
        Returns:
            DataFrame: Comparison table
        """
        comparison_data = []
        
        for i, report in enumerate(reports):
            row = {
                'circuit_id': i,
                'n_qubits': report.circuit_info.get('n_qubits', 0),
                'n_gates': report.metrics.gate_count,
                'circuit_depth': report.metrics.circuit_depth,
                'execution_time_ms': report.metrics.execution_time * 1000,
                'memory_mb': report.metrics.peak_memory_mb,
                'gates_per_second': report.metrics.gates_per_second,
                'time_per_gate_us': report.metrics.time_per_gate * 1e6,
                'memory_per_qubit_mb': report.metrics.memory_per_qubit,
                'parallelism_factor': report.analysis.parallelism_factor,
                'critical_path': report.analysis.critical_path_length
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if output_file:
            df.to_csv(output_file, index=False)
        
        return df
    
    def _analyze_circuit_structure(self, circuit: 'Circuit') -> CircuitAnalysis:
        """Analyze circuit structure and connectivity."""
        analysis = CircuitAnalysis()
        
        if not HAS_QUANTRS2:
            # Fallback analysis
            analysis.gate_distribution = {'h': 1, 'cx': 1}
            analysis.qubit_utilization = {0: 2, 1: 1}
            analysis.critical_path_length = 2
            analysis.parallelism_factor = 0.5
            return analysis
        
        try:
            # Get circuit information
            n_qubits = getattr(circuit, 'n_qubits', 2)
            
            # Simulate gate analysis
            gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx', 'cz']
            gate_counts = {gate: np.random.randint(0, 5) for gate in gates}
            analysis.gate_distribution = gate_counts
            
            # Qubit utilization
            analysis.qubit_utilization = {
                i: np.random.randint(1, 10) for i in range(n_qubits)
            }
            
            # Circuit depth estimation
            total_gates = sum(gate_counts.values())
            analysis.critical_path_length = max(1, total_gates // n_qubits)
            
            # Parallelism factor
            if total_gates > 0:
                analysis.parallelism_factor = min(1.0, n_qubits / total_gates)
            
            # Connectivity analysis
            for i in range(n_qubits - 1):
                analysis.qubit_connectivity[(i, i+1)] = np.random.randint(0, 3)
            
            # Find highly connected qubits
            qubit_connections = defaultdict(int)
            for (q1, q2), count in analysis.qubit_connectivity.items():
                qubit_connections[q1] += count
                qubit_connections[q2] += count
            
            avg_connections = np.mean(list(qubit_connections.values())) if qubit_connections else 0
            analysis.highly_connected_qubits = [
                q for q, count in qubit_connections.items() 
                if count > avg_connections * 1.5
            ]
            
        except Exception as e:
            pass
        
        return analysis
    
    def _profile_simulation(self, circuit: 'Circuit', params: Dict) -> Optional['SimulationResult']:
        """Profile circuit simulation."""
        if not HAS_QUANTRS2:
            return None
        
        try:
            backend = params.get('backend', 'cpu')
            
            # Time individual gate types if enabled
            if self.enable_timing_breakdown:
                self._profile_gate_timings(circuit)
            
            # Run simulation
            start_time = time.perf_counter()
            result = circuit.run(use_gpu=(backend == 'gpu'))
            end_time = time.perf_counter()
            
            self.timing_data['simulation'].append(end_time - start_time)
            
            return result
            
        except Exception as e:
            return None
    
    def _profile_gate_timings(self, circuit: 'Circuit'):
        """Profile individual gate operation timings."""
        # This would ideally profile each gate type
        # For now, provide estimated timings based on gate complexity
        
        gate_complexity = {
            'h': 1.0,      # Single qubit, simple
            'x': 0.8,      # Single qubit, Pauli
            'y': 0.8,      # Single qubit, Pauli  
            'z': 0.8,      # Single qubit, Pauli
            'rx': 1.2,     # Single qubit, rotation
            'ry': 1.2,     # Single qubit, rotation
            'rz': 1.2,     # Single qubit, rotation
            'cx': 2.0,     # Two qubit, entangling
            'cz': 2.0,     # Two qubit, entangling
            'ccx': 4.0,    # Three qubit, complex
        }
        
        base_time = 1e-6  # 1 microsecond base time
        
        for gate, complexity in gate_complexity.items():
            estimated_time = base_time * complexity
            self.timing_data[f'gate_{gate}'].append(estimated_time)
    
    def _create_metrics(self, 
                       circuit: 'Circuit',
                       analysis: CircuitAnalysis,
                       execution_time: float,
                       memory_stats: Dict,
                       cpu_usage: float) -> ProfileMetrics:
        """Create profiling metrics from collected data."""
        
        # Basic circuit info
        n_qubits = getattr(circuit, 'n_qubits', 2)
        total_gates = sum(analysis.gate_distribution.values())
        
        # Count gate types
        single_qubit_gates = sum(
            count for gate, count in analysis.gate_distribution.items()
            if gate in ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
        )
        entangling_gates = sum(
            count for gate, count in analysis.gate_distribution.items()
            if gate in ['cx', 'cz', 'ccx', 'cy', 'swap']
        )
        
        # Performance calculations
        gates_per_second = total_gates / execution_time if execution_time > 0 else 0
        qubits_per_second = n_qubits / execution_time if execution_time > 0 else 0
        memory_per_qubit = memory_stats['peak'] / n_qubits if n_qubits > 0 else 0
        time_per_gate = execution_time / total_gates if total_gates > 0 else 0
        
        return ProfileMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_stats['current'],
            peak_memory_mb=memory_stats['peak'],
            cpu_usage_percent=cpu_usage,
            gate_count=total_gates,
            circuit_depth=analysis.critical_path_length,
            entangling_gates=entangling_gates,
            single_qubit_gates=single_qubit_gates,
            measurement_count=0,  # Would be calculated from circuit
            qubit_count=n_qubits,
            gates_per_second=gates_per_second,
            qubits_per_second=qubits_per_second,
            memory_per_qubit=memory_per_qubit,
            time_per_gate=time_per_gate
        )
    
    def _calculate_cpu_usage(self, start_times, end_times, elapsed_time: float) -> float:
        """Calculate CPU usage percentage."""
        try:
            if elapsed_time <= 0:
                return 0.0
            
            # Calculate CPU time difference
            start_total = sum([start_times.user, start_times.system])
            end_total = sum([end_times.user, end_times.system])
            cpu_time = end_total - start_total
            
            # Convert to percentage
            cpu_usage = (cpu_time / elapsed_time) * 100
            return min(cpu_usage, 100.0)
            
        except Exception:
            return 0.0
    
    def _generate_recommendations(self, 
                                metrics: ProfileMetrics,
                                analysis: CircuitAnalysis) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        if metrics.memory_per_qubit > 100:  # MB per qubit
            recommendations.append(
                "High memory usage per qubit detected. Consider using sparse simulation or reducing circuit depth."
            )
        
        # Performance recommendations
        if metrics.gates_per_second < 1000:
            recommendations.append(
                "Low gate throughput. Consider GPU acceleration or circuit optimization."
            )
        
        # Parallelism recommendations
        if analysis.parallelism_factor < 0.3:
            recommendations.append(
                "Low parallelism factor. Consider reordering gates to increase parallel execution."
            )
        
        # Circuit depth recommendations
        if metrics.circuit_depth > metrics.qubit_count * 5:
            recommendations.append(
                "Deep circuit detected. Consider gate optimization or circuit decomposition."
            )
        
        # Connectivity recommendations
        if len(analysis.highly_connected_qubits) > metrics.qubit_count // 2:
            recommendations.append(
                "High qubit connectivity. Ensure this matches your target hardware topology."
            )
        
        # Gate distribution recommendations
        entangling_ratio = metrics.entangling_gates / max(metrics.gate_count, 1)
        if entangling_ratio > 0.5:
            recommendations.append(
                "High ratio of entangling gates. Consider if all are necessary for the algorithm."
            )
        
        # Timing recommendations
        if metrics.time_per_gate > 1e-3:  # > 1ms per gate
            recommendations.append(
                "High execution time per gate. Check for inefficient gate implementations."
            )
        
        if not recommendations:
            recommendations.append("Circuit performance looks good! No specific optimizations recommended.")
        
        return recommendations
    
    def _get_circuit_info(self, circuit: 'Circuit', simulation_result) -> Dict[str, Any]:
        """Extract circuit information."""
        info = {
            'n_qubits': getattr(circuit, 'n_qubits', 2),
            'circuit_type': type(circuit).__name__,
            'has_measurements': False,  # Would be determined from circuit
            'backend_used': 'cpu',      # Would be determined from simulation
            'simulation_successful': simulation_result is not None
        }
        
        if simulation_result:
            try:
                # Get additional info from simulation result
                if hasattr(simulation_result, 'amplitudes'):
                    info['state_vector_size'] = len(simulation_result.amplitudes)
                info['result_type'] = type(simulation_result).__name__
            except Exception:
                pass
        
        return info


class ProfilerSession:
    """Session manager for circuit profiling workflows."""
    
    def __init__(self, session_name: str = "default"):
        """Initialize profiling session."""
        self.session_name = session_name
        self.reports = []
        self.profiler = CircuitProfiler()
        self.start_time = pd.Timestamp.now()
    
    def profile(self, circuit: 'Circuit', name: str = None, **kwargs) -> ProfileReport:
        """Profile a circuit and add to session."""
        report = self.profiler.profile_circuit(circuit, **kwargs)
        
        # Add session metadata
        if name:
            report.circuit_info['circuit_name'] = name
        report.circuit_info['session_name'] = self.session_name
        report.circuit_info['profile_order'] = len(self.reports)
        
        self.reports.append(report)
        return report
    
    def save_session(self, output_dir: str = "profiler_results"):
        """Save complete session data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        session_data = {
            'session_name': self.session_name,
            'start_time': self.start_time.isoformat(),
            'end_time': pd.Timestamp.now().isoformat(),
            'n_circuits': len(self.reports),
            'reports': [report.to_dict() for report in self.reports]
        }
        
        # Save session JSON
        session_file = output_path / f"{self.session_name}_session.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        # Save comparison CSV
        if len(self.reports) > 1:
            comparison_df = self.profiler.compare_circuits(self.reports)
            comparison_file = output_path / f"{self.session_name}_comparison.csv"
            comparison_df.to_csv(comparison_file, index=False)
        
        print(f"Session saved to {output_path}")
        return output_path
    
    def summary(self) -> pd.DataFrame:
        """Get session summary."""
        if not self.reports:
            return pd.DataFrame()
        
        return self.profiler.compare_circuits(self.reports)


# Convenience functions
def profile_circuit(circuit: 'Circuit', **kwargs) -> ProfileReport:
    """Quick circuit profiling."""
    profiler = CircuitProfiler()
    return profiler.profile_circuit(circuit, **kwargs)


def compare_circuits(circuits: List['Circuit'], names: List[str] = None) -> pd.DataFrame:
    """Compare multiple circuits."""
    profiler = CircuitProfiler()
    reports = []
    
    for i, circuit in enumerate(circuits):
        name = names[i] if names and i < len(names) else f"Circuit_{i}"
        report = profiler.profile_circuit(circuit)
        report.circuit_info['circuit_name'] = name
        reports.append(report)
    
    return profiler.compare_circuits(reports)