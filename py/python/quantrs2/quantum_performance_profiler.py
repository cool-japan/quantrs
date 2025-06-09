"""
QuantRS2 Quantum Performance Profiling System

A comprehensive performance profiling and analysis framework for quantum circuits,
gates, and operations in QuantRS2. This module provides detailed performance metrics,
bottleneck identification, optimization recommendations, and comparative analysis.

Author: QuantRS2 Team
License: MIT
"""

import time
import psutil
import threading
import asyncio
import statistics
import json
import warnings
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
import numpy as np
from pathlib import Path
import logging

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for quantum operations."""
    
    # Timing metrics
    execution_time: float = 0.0
    setup_time: float = 0.0
    teardown_time: float = 0.0
    gate_times: Dict[str, List[float]] = field(default_factory=dict)
    
    # Memory metrics
    memory_usage: float = 0.0
    peak_memory: float = 0.0
    memory_delta: float = 0.0
    memory_efficiency: float = 0.0
    
    # Circuit metrics
    circuit_depth: int = 0
    gate_count: int = 0
    qubit_count: int = 0
    entanglement_degree: float = 0.0
    
    # Resource metrics
    cpu_usage: float = 0.0
    threads_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Quality metrics
    fidelity: float = 1.0
    error_rate: float = 0.0
    coherence_loss: float = 0.0
    
    # Scalability metrics
    scaling_factor: float = 1.0
    parallelization_efficiency: float = 1.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    operation_type: str = ""
    backend: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return asdict(self)
    
    def summary(self) -> str:
        """Generate a human-readable summary of metrics."""
        return f"""
Performance Summary:
  Execution Time: {self.execution_time:.4f}s
  Memory Usage: {self.memory_usage:.2f}MB
  Peak Memory: {self.peak_memory:.2f}MB
  Circuit Depth: {self.circuit_depth}
  Gate Count: {self.gate_count}
  Qubit Count: {self.qubit_count}
  CPU Usage: {self.cpu_usage:.1f}%
  Fidelity: {self.fidelity:.6f}
  Error Rate: {self.error_rate:.6f}
        """.strip()


class PerformanceProfiler:
    """Base profiler class with common functionality."""
    
    def __init__(self, name: str = "ProfilerBase"):
        self.name = name
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_measurements: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.process = psutil.Process()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except Exception:
            return 0.0
    
    @contextmanager
    def profile_operation(self, operation_name: str, **kwargs):
        """Context manager for profiling operations."""
        metrics = PerformanceMetrics(operation_type=operation_name, **kwargs)
        
        # Setup measurements
        start_memory = self._get_memory_usage()
        start_time = time.perf_counter()
        start_cpu = self._get_cpu_usage()
        
        try:
            yield metrics
        finally:
            # Collect final measurements
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            metrics.execution_time = end_time - start_time
            metrics.memory_usage = end_memory
            metrics.memory_delta = end_memory - start_memory
            metrics.cpu_usage = (start_cpu + end_cpu) / 2
            
            self.metrics_history.append(metrics)


class CircuitProfiler(PerformanceProfiler):
    """Specialized profiler for quantum circuit analysis."""
    
    def __init__(self):
        super().__init__("CircuitProfiler")
        self.gate_timings: Dict[str, List[float]] = defaultdict(list)
        self.circuit_cache: Dict[str, Any] = {}
        
    def profile_circuit(self, circuit, backend: str = "default") -> PerformanceMetrics:
        """Profile a complete quantum circuit execution."""
        with self.profile_operation("circuit_execution", backend=backend) as metrics:
            try:
                # Extract circuit properties
                metrics.qubit_count = getattr(circuit, 'qubit_count', 0)
                metrics.gate_count = getattr(circuit, 'gate_count', 0)
                metrics.circuit_depth = getattr(circuit, 'depth', 0)
                
                # Analyze circuit structure
                self._analyze_circuit_structure(circuit, metrics)
                
                # Execute circuit with timing
                start_time = time.perf_counter()
                result = self._execute_circuit(circuit, backend)
                execution_time = time.perf_counter() - start_time
                
                metrics.execution_time = execution_time
                
                # Calculate quality metrics if result available
                if result:
                    self._calculate_quality_metrics(result, metrics)
                
            except Exception as e:
                logger.error(f"Circuit profiling failed: {e}")
                metrics.error_rate = 1.0
                
        return metrics
    
    def _analyze_circuit_structure(self, circuit, metrics: PerformanceMetrics):
        """Analyze circuit structure for performance insights."""
        try:
            # Calculate entanglement degree (simplified)
            entangling_gates = 0
            total_gates = metrics.gate_count
            
            if hasattr(circuit, 'gates'):
                for gate in circuit.gates:
                    if hasattr(gate, 'qubit_count') and gate.qubit_count > 1:
                        entangling_gates += 1
            
            metrics.entanglement_degree = entangling_gates / max(total_gates, 1)
            
        except Exception as e:
            logger.warning(f"Circuit structure analysis failed: {e}")
    
    def _execute_circuit(self, circuit, backend: str):
        """Execute circuit and measure performance."""
        try:
            # This would integrate with actual QuantRS2 execution
            if hasattr(circuit, 'execute'):
                return circuit.execute(backend=backend)
            elif hasattr(circuit, 'run'):
                return circuit.run()
            else:
                # Mock execution for demonstration
                time.sleep(0.001 * getattr(circuit, 'gate_count', 10))
                return {"success": True}
        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            return None
    
    def _calculate_quality_metrics(self, result, metrics: PerformanceMetrics):
        """Calculate quality metrics from execution results."""
        try:
            if isinstance(result, dict):
                metrics.fidelity = result.get('fidelity', 1.0)
                metrics.error_rate = result.get('error_rate', 0.0)
                metrics.coherence_loss = result.get('coherence_loss', 0.0)
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
    
    def analyze_bottlenecks(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance bottlenecks from multiple circuit runs."""
        if not metrics_list:
            return {}
        
        analysis = {
            'timing_bottlenecks': self._identify_timing_bottlenecks(metrics_list),
            'memory_bottlenecks': self._identify_memory_bottlenecks(metrics_list),
            'scalability_issues': self._identify_scalability_issues(metrics_list),
            'optimization_opportunities': self._identify_optimization_opportunities(metrics_list)
        }
        
        return analysis
    
    def _identify_timing_bottlenecks(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Identify timing-related bottlenecks."""
        execution_times = [m.execution_time for m in metrics_list]
        
        return {
            'mean_time': statistics.mean(execution_times),
            'std_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'max_time': max(execution_times),
            'min_time': min(execution_times),
            'outliers': [t for t in execution_times if t > statistics.mean(execution_times) + 2 * statistics.stdev(execution_times)] if len(execution_times) > 1 else []
        }
    
    def _identify_memory_bottlenecks(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Identify memory-related bottlenecks."""
        memory_usage = [m.memory_usage for m in metrics_list]
        memory_deltas = [m.memory_delta for m in metrics_list]
        
        return {
            'mean_memory': statistics.mean(memory_usage),
            'peak_memory': max(memory_usage),
            'memory_leaks': sum(1 for delta in memory_deltas if delta > 10),  # MB threshold
            'memory_efficiency': statistics.mean([m.memory_efficiency for m in metrics_list if m.memory_efficiency > 0])
        }
    
    def _identify_scalability_issues(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Identify scalability issues."""
        qubit_counts = [m.qubit_count for m in metrics_list]
        execution_times = [m.execution_time for m in metrics_list]
        
        if len(set(qubit_counts)) < 2:
            return {'insufficient_data': True}
        
        # Simple scaling analysis
        scaling_ratios = []
        for i in range(1, len(metrics_list)):
            if qubit_counts[i] != qubit_counts[i-1] and qubit_counts[i-1] > 0:
                time_ratio = execution_times[i] / execution_times[i-1]
                qubit_ratio = qubit_counts[i] / qubit_counts[i-1]
                scaling_ratios.append(time_ratio / qubit_ratio)
        
        return {
            'scaling_efficiency': statistics.mean(scaling_ratios) if scaling_ratios else 1.0,
            'sublinear_scaling': sum(1 for r in scaling_ratios if r < 1),
            'superlinear_scaling': sum(1 for r in scaling_ratios if r > 2)
        }
    
    def _identify_optimization_opportunities(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        avg_error_rate = statistics.mean([m.error_rate for m in metrics_list])
        avg_fidelity = statistics.mean([m.fidelity for m in metrics_list])
        avg_memory_delta = statistics.mean([m.memory_delta for m in metrics_list])
        
        if avg_error_rate > 0.1:
            opportunities.append("High error rate detected - consider error mitigation")
        
        if avg_fidelity < 0.9:
            opportunities.append("Low fidelity - review gate implementation or add error correction")
        
        if avg_memory_delta > 100:  # MB
            opportunities.append("High memory consumption - consider memory optimization")
        
        # Check for circuit optimization opportunities
        avg_depth = statistics.mean([m.circuit_depth for m in metrics_list])
        avg_gates = statistics.mean([m.gate_count for m in metrics_list])
        
        if avg_depth > avg_gates * 0.5:
            opportunities.append("High circuit depth - consider gate parallelization")
        
        return opportunities


class GateProfiler(PerformanceProfiler):
    """Profiler for individual quantum gate operations."""
    
    def __init__(self):
        super().__init__("GateProfiler")
        self.gate_statistics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
    def profile_gate(self, gate_name: str, gate_operation: Callable, *args, **kwargs) -> PerformanceMetrics:
        """Profile a single gate operation."""
        with self.profile_operation(f"gate_{gate_name}", gate=gate_name) as metrics:
            try:
                start_time = time.perf_counter()
                result = gate_operation(*args, **kwargs)
                end_time = time.perf_counter()
                
                gate_time = end_time - start_time
                metrics.gate_times[gate_name] = [gate_time]
                
                # Store in statistics
                self.gate_statistics[gate_name]['execution_times'].append(gate_time)
                
                # Calculate gate-specific metrics
                metrics.gate_count = 1
                metrics.execution_time = gate_time
                
            except Exception as e:
                logger.error(f"Gate profiling failed for {gate_name}: {e}")
                metrics.error_rate = 1.0
                
        return metrics
    
    def get_gate_statistics(self, gate_name: str) -> Dict[str, float]:
        """Get statistical summary for a specific gate."""
        if gate_name not in self.gate_statistics:
            return {}
        
        times = self.gate_statistics[gate_name]['execution_times']
        if not times:
            return {}
        
        return {
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'min_time': min(times),
            'max_time': max(times),
            'sample_count': len(times)
        }
    
    def compare_gates(self, gate_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare performance across multiple gate types."""
        comparison = {}
        for gate_name in gate_names:
            comparison[gate_name] = self.get_gate_statistics(gate_name)
        return comparison


class MemoryProfiler(PerformanceProfiler):
    """Specialized profiler for memory usage analysis."""
    
    def __init__(self):
        super().__init__("MemoryProfiler")
        self.memory_snapshots: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self, interval: float = 0.1):
        """Start continuous memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_memory(self, interval: float):
        """Background memory monitoring."""
        while self.monitoring:
            timestamp = time.time()
            memory_mb = self._get_memory_usage()
            self.memory_snapshots.append((timestamp, memory_mb))
            time.sleep(interval)
    
    def analyze_memory_pattern(self, operation_start: float, operation_end: float) -> Dict[str, Any]:
        """Analyze memory usage pattern during a specific operation."""
        relevant_snapshots = [
            (ts, mem) for ts, mem in self.memory_snapshots 
            if operation_start <= ts <= operation_end
        ]
        
        if not relevant_snapshots:
            return {}
        
        memories = [mem for _, mem in relevant_snapshots]
        timestamps = [ts for ts, _ in relevant_snapshots]
        
        return {
            'initial_memory': memories[0],
            'final_memory': memories[-1],
            'peak_memory': max(memories),
            'min_memory': min(memories),
            'memory_delta': memories[-1] - memories[0],
            'memory_variance': statistics.variance(memories) if len(memories) > 1 else 0,
            'duration': timestamps[-1] - timestamps[0],
            'memory_growth_rate': (memories[-1] - memories[0]) / max(timestamps[-1] - timestamps[0], 1e-6)
        }
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        if len(self.memory_snapshots) < 10:
            return leaks
        
        # Analyze memory trend over time
        window_size = 10
        for i in range(window_size, len(self.memory_snapshots)):
            window = self.memory_snapshots[i-window_size:i]
            memories = [mem for _, mem in window]
            
            # Check for consistent growth
            if all(memories[j] <= memories[j+1] for j in range(len(memories)-1)):
                growth = memories[-1] - memories[0]
                if growth > threshold_mb:
                    leaks.append({
                        'start_time': window[0][0],
                        'end_time': window[-1][0],
                        'memory_growth': growth,
                        'growth_rate': growth / max(window[-1][0] - window[0][0], 1e-6)
                    })
        
        return leaks


class PerformanceComparator:
    """Compare performance between different implementations or configurations."""
    
    def __init__(self):
        self.comparison_data: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
    def add_measurement(self, implementation_name: str, metrics: PerformanceMetrics):
        """Add a performance measurement for comparison."""
        self.comparison_data[implementation_name].append(metrics)
    
    def compare_implementations(self, metric_name: str = "execution_time") -> Dict[str, Any]:
        """Compare implementations across a specific metric."""
        if not self.comparison_data:
            return {}
        
        comparison = {}
        for impl_name, metrics_list in self.comparison_data.items():
            values = [getattr(m, metric_name, 0) for m in metrics_list]
            if values:
                comparison[impl_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        # Add relative performance
        if len(comparison) > 1:
            baseline = min(comparison.values(), key=lambda x: x['mean'])['mean']
            for impl_name in comparison:
                comparison[impl_name]['relative_performance'] = comparison[impl_name]['mean'] / baseline
        
        return comparison
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if not self.comparison_data:
            return "No comparison data available."
        
        report_lines = ["Performance Comparison Report", "=" * 30, ""]
        
        for metric in ['execution_time', 'memory_usage', 'error_rate', 'fidelity']:
            comparison = self.compare_implementations(metric)
            if comparison:
                report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
                report_lines.append("-" * 20)
                
                for impl_name, stats in comparison.items():
                    report_lines.append(f"{impl_name}:")
                    report_lines.append(f"  Mean: {stats['mean']:.6f}")
                    report_lines.append(f"  Std:  {stats['std']:.6f}")
                    if 'relative_performance' in stats:
                        report_lines.append(f"  Relative: {stats['relative_performance']:.2f}x")
                    report_lines.append("")
        
        return "\n".join(report_lines)


class PerformanceOptimizer:
    """Provide optimization recommendations based on performance analysis."""
    
    def __init__(self):
        self.optimization_rules: List[Callable] = [
            self._check_memory_optimization,
            self._check_timing_optimization,
            self._check_circuit_optimization,
            self._check_gate_optimization,
            self._check_scalability_optimization
        ]
    
    def analyze_and_recommend(self, metrics: Union[PerformanceMetrics, List[PerformanceMetrics]]) -> List[str]:
        """Analyze metrics and provide optimization recommendations."""
        if isinstance(metrics, PerformanceMetrics):
            metrics = [metrics]
        
        recommendations = []
        for rule in self.optimization_rules:
            try:
                rule_recommendations = rule(metrics)
                recommendations.extend(rule_recommendations)
            except Exception as e:
                logger.warning(f"Optimization rule failed: {e}")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _check_memory_optimization(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Check for memory optimization opportunities."""
        recommendations = []
        
        avg_memory = statistics.mean([m.memory_usage for m in metrics_list])
        avg_delta = statistics.mean([m.memory_delta for m in metrics_list])
        
        if avg_memory > 1000:  # MB
            recommendations.append("Consider memory optimization - high memory usage detected")
        
        if avg_delta > 100:  # MB
            recommendations.append("Memory growth detected - check for memory leaks")
        
        memory_efficiency = [m.memory_efficiency for m in metrics_list if m.memory_efficiency > 0]
        if memory_efficiency and statistics.mean(memory_efficiency) < 0.5:
            recommendations.append("Low memory efficiency - consider state vector optimization")
        
        return recommendations
    
    def _check_timing_optimization(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Check for timing optimization opportunities."""
        recommendations = []
        
        execution_times = [m.execution_time for m in metrics_list]
        if not execution_times:
            return recommendations
        
        avg_time = statistics.mean(execution_times)
        if len(execution_times) > 1:
            time_variance = statistics.variance(execution_times)
            if time_variance > avg_time * 0.5:
                recommendations.append("High execution time variance - investigate timing inconsistencies")
        
        cpu_usage = [m.cpu_usage for m in metrics_list if m.cpu_usage > 0]
        if cpu_usage and statistics.mean(cpu_usage) < 50:
            recommendations.append("Low CPU utilization - consider parallelization")
        
        return recommendations
    
    def _check_circuit_optimization(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Check for circuit-level optimization opportunities."""
        recommendations = []
        
        depths = [m.circuit_depth for m in metrics_list if m.circuit_depth > 0]
        gate_counts = [m.gate_count for m in metrics_list if m.gate_count > 0]
        
        if depths and gate_counts:
            avg_depth = statistics.mean(depths)
            avg_gates = statistics.mean(gate_counts)
            
            if avg_depth > avg_gates * 0.7:
                recommendations.append("High circuit depth relative to gate count - consider gate reordering")
        
        entanglement_degrees = [m.entanglement_degree for m in metrics_list if m.entanglement_degree > 0]
        if entanglement_degrees and statistics.mean(entanglement_degrees) > 0.8:
            recommendations.append("High entanglement degree - consider circuit decomposition")
        
        return recommendations
    
    def _check_gate_optimization(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Check for gate-level optimization opportunities."""
        recommendations = []
        
        # Analyze gate timing patterns
        all_gate_times = {}
        for metrics in metrics_list:
            for gate_name, times in metrics.gate_times.items():
                if gate_name not in all_gate_times:
                    all_gate_times[gate_name] = []
                all_gate_times[gate_name].extend(times)
        
        for gate_name, times in all_gate_times.items():
            if len(times) > 1:
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times)
                if std_time > avg_time * 0.5:
                    recommendations.append(f"High timing variance for {gate_name} gate - investigate implementation")
        
        return recommendations
    
    def _check_scalability_optimization(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Check for scalability optimization opportunities."""
        recommendations = []
        
        scaling_factors = [m.scaling_factor for m in metrics_list if m.scaling_factor != 1.0]
        if scaling_factors and statistics.mean(scaling_factors) > 2.0:
            recommendations.append("Poor scaling detected - consider algorithmic improvements")
        
        parallel_efficiency = [m.parallelization_efficiency for m in metrics_list if m.parallelization_efficiency != 1.0]
        if parallel_efficiency and statistics.mean(parallel_efficiency) < 0.7:
            recommendations.append("Low parallelization efficiency - review parallel implementation")
        
        return recommendations


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        self.alert_thresholds = alert_thresholds or {
            'execution_time': 10.0,  # seconds
            'memory_usage': 1000.0,  # MB
            'error_rate': 0.1,  # 10%
            'cpu_usage': 90.0  # %
        }
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.metrics_buffer = deque(maxlen=1000)
        
    async def start_monitoring(self, check_interval: float = 1.0):
        """Start asynchronous performance monitoring."""
        self.monitoring_active = True
        while self.monitoring_active:
            try:
                current_metrics = await self._collect_current_metrics()
                self.metrics_buffer.append(current_metrics)
                self._check_alerts(current_metrics)
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(check_interval)
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
    
    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        metrics = PerformanceMetrics()
        
        # System metrics
        metrics.memory_usage = psutil.virtual_memory().used / 1024 / 1024
        metrics.cpu_usage = psutil.cpu_percent()
        
        # Add any quantum-specific metrics here
        return metrics
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check if any metrics exceed alert thresholds."""
        for metric_name, threshold in self.alert_thresholds.items():
            value = getattr(metrics, metric_name, 0)
            if value > threshold:
                alert = {
                    'timestamp': time.time(),
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'severity': 'high' if value > threshold * 1.5 else 'medium'
                }
                self.alerts.append(alert)
                logger.warning(f"Performance alert: {metric_name} = {value} exceeds threshold {threshold}")
    
    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get alerts from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring data."""
        if not self.metrics_buffer:
            return {}
        
        recent_metrics = list(self.metrics_buffer)
        
        return {
            'total_samples': len(recent_metrics),
            'time_range': {
                'start': recent_metrics[0].timestamp,
                'end': recent_metrics[-1].timestamp
            },
            'averages': {
                'execution_time': statistics.mean([m.execution_time for m in recent_metrics if m.execution_time > 0]),
                'memory_usage': statistics.mean([m.memory_usage for m in recent_metrics]),
                'cpu_usage': statistics.mean([m.cpu_usage for m in recent_metrics if m.cpu_usage > 0])
            },
            'alert_count': len(self.get_recent_alerts()),
            'monitoring_active': self.monitoring_active
        }


class PerformanceReporter:
    """Generate comprehensive performance reports."""
    
    def __init__(self):
        self.report_templates = {
            'summary': self._generate_summary_report,
            'detailed': self._generate_detailed_report,
            'comparison': self._generate_comparison_report,
            'optimization': self._generate_optimization_report
        }
    
    def generate_report(self, metrics_data: List[PerformanceMetrics], 
                       report_type: str = "summary", 
                       output_format: str = "text") -> str:
        """Generate a performance report."""
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        report_content = self.report_templates[report_type](metrics_data)
        
        if output_format == "json":
            return self._format_as_json(report_content)
        elif output_format == "html":
            return self._format_as_html(report_content)
        else:
            return report_content
    
    def _generate_summary_report(self, metrics_data: List[PerformanceMetrics]) -> str:
        """Generate a summary report."""
        if not metrics_data:
            return "No performance data available."
        
        # Calculate summary statistics
        execution_times = [m.execution_time for m in metrics_data if m.execution_time > 0]
        memory_usage = [m.memory_usage for m in metrics_data if m.memory_usage > 0]
        error_rates = [m.error_rate for m in metrics_data]
        
        report = f"""
Performance Summary Report
=========================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Measurements: {len(metrics_data)}

Execution Time Statistics:
  Mean: {statistics.mean(execution_times):.4f}s
  Median: {statistics.median(execution_times):.4f}s
  Std Dev: {statistics.stdev(execution_times) if len(execution_times) > 1 else 0:.4f}s
  Min: {min(execution_times):.4f}s
  Max: {max(execution_times):.4f}s

Memory Usage Statistics:
  Mean: {statistics.mean(memory_usage):.2f}MB
  Peak: {max(memory_usage):.2f}MB
  Min: {min(memory_usage):.2f}MB

Quality Metrics:
  Mean Error Rate: {statistics.mean(error_rates):.6f}
  Mean Fidelity: {statistics.mean([m.fidelity for m in metrics_data]):.6f}

Circuit Characteristics:
  Mean Qubit Count: {statistics.mean([m.qubit_count for m in metrics_data if m.qubit_count > 0]):.1f}
  Mean Gate Count: {statistics.mean([m.gate_count for m in metrics_data if m.gate_count > 0]):.1f}
  Mean Circuit Depth: {statistics.mean([m.circuit_depth for m in metrics_data if m.circuit_depth > 0]):.1f}
        """.strip()
        
        return report
    
    def _generate_detailed_report(self, metrics_data: List[PerformanceMetrics]) -> str:
        """Generate a detailed report with individual measurements."""
        report_lines = [
            "Detailed Performance Report",
            "=" * 30,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Measurements: {len(metrics_data)}",
            ""
        ]
        
        for i, metrics in enumerate(metrics_data):
            report_lines.append(f"Measurement {i + 1}:")
            report_lines.append(f"  Timestamp: {time.strftime('%H:%M:%S', time.localtime(metrics.timestamp))}")
            report_lines.append(f"  Operation: {metrics.operation_type}")
            report_lines.append(f"  Execution Time: {metrics.execution_time:.4f}s")
            report_lines.append(f"  Memory Usage: {metrics.memory_usage:.2f}MB")
            report_lines.append(f"  CPU Usage: {metrics.cpu_usage:.1f}%")
            report_lines.append(f"  Error Rate: {metrics.error_rate:.6f}")
            report_lines.append(f"  Fidelity: {metrics.fidelity:.6f}")
            if metrics.qubit_count > 0:
                report_lines.append(f"  Qubits: {metrics.qubit_count}")
            if metrics.gate_count > 0:
                report_lines.append(f"  Gates: {metrics.gate_count}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _generate_comparison_report(self, metrics_data: List[PerformanceMetrics]) -> str:
        """Generate a comparison report between different backends or configurations."""
        # Group by backend or operation type
        grouped_data = defaultdict(list)
        for metrics in metrics_data:
            key = metrics.backend or metrics.operation_type or "default"
            grouped_data[key].append(metrics)
        
        if len(grouped_data) < 2:
            return "Insufficient data for comparison (need at least 2 different backends/operations)."
        
        report_lines = [
            "Performance Comparison Report",
            "=" * 30,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Compare execution times
        report_lines.append("Execution Time Comparison:")
        report_lines.append("-" * 25)
        for group_name, group_metrics in grouped_data.items():
            times = [m.execution_time for m in group_metrics if m.execution_time > 0]
            if times:
                report_lines.append(f"{group_name}:")
                report_lines.append(f"  Mean: {statistics.mean(times):.4f}s")
                report_lines.append(f"  Samples: {len(times)}")
        
        report_lines.append("")
        
        # Compare memory usage
        report_lines.append("Memory Usage Comparison:")
        report_lines.append("-" * 23)
        for group_name, group_metrics in grouped_data.items():
            memory = [m.memory_usage for m in group_metrics if m.memory_usage > 0]
            if memory:
                report_lines.append(f"{group_name}:")
                report_lines.append(f"  Mean: {statistics.mean(memory):.2f}MB")
                report_lines.append(f"  Peak: {max(memory):.2f}MB")
        
        return "\n".join(report_lines)
    
    def _generate_optimization_report(self, metrics_data: List[PerformanceMetrics]) -> str:
        """Generate an optimization recommendations report."""
        optimizer = PerformanceOptimizer()
        recommendations = optimizer.analyze_and_recommend(metrics_data)
        
        # Analyze bottlenecks
        circuit_profiler = CircuitProfiler()
        bottlenecks = circuit_profiler.analyze_bottlenecks(metrics_data)
        
        report_lines = [
            "Performance Optimization Report",
            "=" * 32,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Optimization Recommendations:",
            "-" * 28
        ]
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        else:
            report_lines.append("No specific optimization recommendations at this time.")
        
        report_lines.extend(["", "Bottleneck Analysis:", "-" * 18])
        
        if bottlenecks:
            for category, analysis in bottlenecks.items():
                if analysis:
                    report_lines.append(f"\n{category.replace('_', ' ').title()}:")
                    if isinstance(analysis, dict):
                        for key, value in analysis.items():
                            report_lines.append(f"  {key}: {value}")
                    elif isinstance(analysis, list):
                        for item in analysis:
                            report_lines.append(f"  - {item}")
        
        return "\n".join(report_lines)
    
    def _format_as_json(self, report_content: str) -> str:
        """Format report as JSON."""
        return json.dumps({
            'report': report_content,
            'generated_at': time.time(),
            'format': 'json'
        }, indent=2)
    
    def _format_as_html(self, report_content: str) -> str:
        """Format report as HTML."""
        html_content = report_content.replace('\n', '<br>\n')
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>QuantRS2 Performance Report</title>
    <style>
        body {{ font-family: monospace; margin: 20px; }}
        .report {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="report">
        {html_content}
    </div>
</body>
</html>
        """.strip()
    
    def save_report(self, report_content: str, filename: str):
        """Save report to file."""
        try:
            with open(filename, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


class QuantumPerformanceProfiler:
    """Main profiler class that coordinates all profiling activities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize sub-profilers
        self.circuit_profiler = CircuitProfiler()
        self.gate_profiler = GateProfiler()
        self.memory_profiler = MemoryProfiler()
        self.comparator = PerformanceComparator()
        self.optimizer = PerformanceOptimizer()
        self.monitor = PerformanceMonitor(
            alert_thresholds=self.config.get('alert_thresholds')
        )
        self.reporter = PerformanceReporter()
        
        # Storage for all metrics
        self.all_metrics: List[PerformanceMetrics] = []
        
        # Performance regression detection
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.regression_threshold = self.config.get('regression_threshold', 0.1)  # 10%
    
    def profile_circuit_execution(self, circuit, backend: str = "default", 
                                 label: Optional[str] = None) -> PerformanceMetrics:
        """Profile a complete circuit execution."""
        logger.info(f"Profiling circuit execution on backend: {backend}")
        
        # Start memory monitoring
        self.memory_profiler.start_monitoring()
        
        try:
            # Profile the circuit
            metrics = self.circuit_profiler.profile_circuit(circuit, backend)
            
            # Add label if provided
            if label:
                metrics.configuration['label'] = label
            
            # Store metrics
            self.all_metrics.append(metrics)
            self.comparator.add_measurement(backend, metrics)
            
            # Check for regressions
            self._check_performance_regression(metrics, backend)
            
            return metrics
            
        finally:
            # Stop memory monitoring
            self.memory_profiler.stop_monitoring()
    
    def profile_gate_sequence(self, gates: List[Tuple[str, Callable]], 
                             label: Optional[str] = None) -> List[PerformanceMetrics]:
        """Profile a sequence of gate operations."""
        logger.info(f"Profiling {len(gates)} gate operations")
        
        gate_metrics = []
        for gate_name, gate_operation in gates:
            metrics = self.gate_profiler.profile_gate(gate_name, gate_operation)
            if label:
                metrics.configuration['label'] = label
            gate_metrics.append(metrics)
            self.all_metrics.append(metrics)
        
        return gate_metrics
    
    def benchmark_scalability(self, circuit_generator: Callable, 
                             qubit_counts: List[int], 
                             backend: str = "default") -> Dict[int, PerformanceMetrics]:
        """Benchmark circuit scalability across different qubit counts."""
        logger.info(f"Running scalability benchmark for qubit counts: {qubit_counts}")
        
        scalability_results = {}
        
        for qubit_count in qubit_counts:
            try:
                # Generate circuit for this qubit count
                circuit = circuit_generator(qubit_count)
                
                # Profile execution
                metrics = self.profile_circuit_execution(circuit, backend, 
                                                       label=f"scalability_{qubit_count}")
                
                # Calculate scaling factor relative to smallest circuit
                if len(scalability_results) > 0:
                    base_metrics = scalability_results[min(scalability_results.keys())]
                    metrics.scaling_factor = metrics.execution_time / base_metrics.execution_time
                
                scalability_results[qubit_count] = metrics
                
            except Exception as e:
                logger.error(f"Scalability test failed for {qubit_count} qubits: {e}")
        
        return scalability_results
    
    def compare_backends(self, circuit, backends: List[str], 
                        runs_per_backend: int = 5) -> Dict[str, Any]:
        """Compare performance across multiple backends."""
        logger.info(f"Comparing backends: {backends}")
        
        comparison_results = {}
        
        for backend in backends:
            backend_metrics = []
            for run in range(runs_per_backend):
                try:
                    metrics = self.profile_circuit_execution(circuit, backend, 
                                                           label=f"backend_comparison_{run}")
                    backend_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Backend comparison failed for {backend}, run {run}: {e}")
            
            comparison_results[backend] = backend_metrics
        
        # Generate comparison analysis
        return self.comparator.compare_implementations("execution_time")
    
    def detect_performance_regressions(self, baseline_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to baseline."""
        if baseline_file:
            self._load_baseline_metrics(baseline_file)
        
        if not self.baseline_metrics:
            logger.warning("No baseline metrics available for regression detection")
            return []
        
        regressions = []
        recent_metrics = self.all_metrics[-10:]  # Check last 10 measurements
        
        for metrics in recent_metrics:
            backend = metrics.backend or "default"
            baseline_key = f"{backend}_execution_time"
            
            if baseline_key in self.baseline_metrics:
                baseline_time = self.baseline_metrics[baseline_key]
                current_time = metrics.execution_time
                
                if current_time > baseline_time * (1 + self.regression_threshold):
                    regression = {
                        'timestamp': metrics.timestamp,
                        'backend': backend,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'regression_factor': current_time / baseline_time,
                        'operation_type': metrics.operation_type
                    }
                    regressions.append(regression)
        
        return regressions
    
    def start_real_time_monitoring(self, check_interval: float = 1.0):
        """Start real-time performance monitoring."""
        logger.info("Starting real-time performance monitoring")
        asyncio.create_task(self.monitor.start_monitoring(check_interval))
    
    def stop_real_time_monitoring(self):
        """Stop real-time performance monitoring."""
        logger.info("Stopping real-time performance monitoring")
        self.monitor.stop_monitoring()
    
    def generate_comprehensive_report(self, report_type: str = "summary", 
                                    output_file: Optional[str] = None) -> str:
        """Generate a comprehensive performance report."""
        logger.info(f"Generating {report_type} performance report")
        
        if not self.all_metrics:
            return "No performance data available for reporting."
        
        report = self.reporter.generate_report(self.all_metrics, report_type)
        
        if output_file:
            self.reporter.save_report(report, output_file)
        
        return report
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on all collected metrics."""
        return self.optimizer.analyze_and_recommend(self.all_metrics)
    
    def export_metrics(self, filename: str, format: str = "json"):
        """Export all collected metrics to file."""
        try:
            if format == "json":
                data = [metrics.to_dict() for metrics in self.all_metrics]
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == "csv" and HAS_PANDAS:
                df = pd.DataFrame([metrics.to_dict() for metrics in self.all_metrics])
                df.to_csv(filename, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Metrics exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def create_performance_visualization(self, metric_name: str = "execution_time", 
                                       output_file: str = "performance_plot.png"):
        """Create performance visualization plots."""
        if not HAS_PLOTTING:
            logger.warning("Matplotlib not available - cannot create visualizations")
            return
        
        if not self.all_metrics:
            logger.warning("No metrics available for visualization")
            return
        
        try:
            # Extract data for plotting
            timestamps = [m.timestamp for m in self.all_metrics]
            values = [getattr(m, metric_name, 0) for m in self.all_metrics]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, values, marker='o', linestyle='-', alpha=0.7)
            plt.title(f'Performance Trend: {metric_name.replace("_", " ").title()}')
            plt.xlabel('Time')
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance visualization saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
    
    def _check_performance_regression(self, metrics: PerformanceMetrics, backend: str):
        """Check if current metrics indicate a performance regression."""
        if not self.baseline_metrics:
            return
        
        baseline_key = f"{backend}_execution_time"
        if baseline_key in self.baseline_metrics:
            baseline_time = self.baseline_metrics[baseline_key]
            if metrics.execution_time > baseline_time * (1 + self.regression_threshold):
                logger.warning(f"Performance regression detected for {backend}: "
                             f"{metrics.execution_time:.4f}s vs baseline {baseline_time:.4f}s")
    
    def _load_baseline_metrics(self, filename: str):
        """Load baseline metrics from file."""
        try:
            with open(filename, 'r') as f:
                self.baseline_metrics = json.load(f)
            logger.info(f"Baseline metrics loaded from {filename}")
        except Exception as e:
            logger.error(f"Failed to load baseline metrics: {e}")
    
    def save_baseline_metrics(self, filename: str):
        """Save current metrics as baseline for future regression detection."""
        try:
            # Calculate baseline metrics from recent data
            baseline = {}
            
            # Group by backend
            backend_metrics = defaultdict(list)
            for metrics in self.all_metrics[-50:]:  # Use last 50 measurements
                backend = metrics.backend or "default"
                backend_metrics[backend].append(metrics)
            
            # Calculate baseline values
            for backend, metrics_list in backend_metrics.items():
                execution_times = [m.execution_time for m in metrics_list if m.execution_time > 0]
                if execution_times:
                    baseline[f"{backend}_execution_time"] = statistics.median(execution_times)
            
            # Save baseline
            with open(filename, 'w') as f:
                json.dump(baseline, f, indent=2)
            
            self.baseline_metrics = baseline
            logger.info(f"Baseline metrics saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save baseline metrics: {e}")


# Convenience functions for easy usage
def profile_quantum_circuit(circuit, backend: str = "default", 
                           profiler: Optional[QuantumPerformanceProfiler] = None) -> PerformanceMetrics:
    """Convenience function to profile a quantum circuit."""
    if profiler is None:
        profiler = QuantumPerformanceProfiler()
    return profiler.profile_circuit_execution(circuit, backend)


def benchmark_circuit_scalability(circuit_generator: Callable, 
                                 qubit_range: Tuple[int, int, int] = (2, 10, 2),
                                 backend: str = "default") -> Dict[int, PerformanceMetrics]:
    """Convenience function to benchmark circuit scalability."""
    profiler = QuantumPerformanceProfiler()
    start, stop, step = qubit_range
    qubit_counts = list(range(start, stop + 1, step))
    return profiler.benchmark_scalability(circuit_generator, qubit_counts, backend)


def compare_quantum_backends(circuit, backends: List[str], 
                           runs_per_backend: int = 5) -> Dict[str, Any]:
    """Convenience function to compare quantum backends."""
    profiler = QuantumPerformanceProfiler()
    return profiler.compare_backends(circuit, backends, runs_per_backend)


def generate_performance_report(metrics_data: List[PerformanceMetrics], 
                              report_type: str = "summary") -> str:
    """Convenience function to generate performance reports."""
    reporter = PerformanceReporter()
    return reporter.generate_report(metrics_data, report_type)


# CLI interface for profiling operations
def main():
    """CLI interface for quantum performance profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Performance Profiler")
    parser.add_argument("--mode", choices=["profile", "monitor", "report", "export"], 
                       default="profile", help="Operation mode")
    parser.add_argument("--circuit", help="Path to circuit file or circuit specification")
    parser.add_argument("--backend", default="default", help="Backend to use")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["text", "json", "html"], default="text")
    parser.add_argument("--baseline", help="Baseline metrics file for regression detection")
    parser.add_argument("--threshold", type=float, default=0.1, 
                       help="Regression threshold (default: 0.1)")
    
    args = parser.parse_args()
    
    # Create profiler
    config = {
        'regression_threshold': args.threshold
    }
    profiler = QuantumPerformanceProfiler(config)
    
    if args.mode == "profile":
        # Profile mode - would need actual circuit loading logic
        print("Profile mode - circuit profiling functionality")
        
    elif args.mode == "monitor":
        # Monitor mode
        print("Starting real-time monitoring...")
        profiler.start_real_time_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping monitoring...")
            profiler.stop_real_time_monitoring()
            
    elif args.mode == "report":
        # Report generation
        report = profiler.generate_comprehensive_report("summary", args.output)
        if not args.output:
            print(report)
            
    elif args.mode == "export":
        # Export metrics
        if args.output:
            format_type = args.format if args.format in ["json", "csv"] else "json"
            profiler.export_metrics(args.output, format_type)
        else:
            print("Output file required for export mode")


if __name__ == "__main__":
    main()