#!/usr/bin/env python3
"""
Comprehensive test suite for the QuantRS2 Quantum Performance Profiler module.

This test suite provides complete coverage of all profiling functionality including:
- PerformanceMetrics dataclass
- CircuitProfiler with circuit analysis and bottleneck detection
- GateProfiler with gate-level timing and statistics
- MemoryProfiler with memory monitoring and leak detection
- PerformanceComparator with backend comparisons
- PerformanceOptimizer with optimization recommendations
- PerformanceMonitor with real-time monitoring and alerts
- PerformanceReporter with report generation in multiple formats
- QuantumPerformanceProfiler main class with comprehensive profiling
- Convenience functions and integration tests
"""

import pytest
import tempfile
import os
import json
import time
import asyncio
import threading
import statistics
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import quantrs2
    from quantrs2.quantum_performance_profiler import (
        PerformanceMetrics, PerformanceProfiler, CircuitProfiler, GateProfiler,
        MemoryProfiler, PerformanceComparator, PerformanceOptimizer,
        PerformanceMonitor, PerformanceReporter, QuantumPerformanceProfiler,
        profile_quantum_circuit, benchmark_circuit_scalability,
        compare_quantum_backends, generate_performance_report
    )
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Test fixtures and mock objects
@pytest.fixture
def sample_metrics():
    """Create sample performance metrics for testing."""
    return PerformanceMetrics(
        execution_time=1.5,
        setup_time=0.1,
        teardown_time=0.05,
        memory_usage=128.0,
        peak_memory=150.0,
        memory_delta=22.0,
        circuit_depth=10,
        gate_count=25,
        qubit_count=4,
        cpu_usage=65.0,
        fidelity=0.95,
        error_rate=0.05,
        entanglement_degree=0.6,
        operation_type="circuit_execution",
        backend="test_backend"
    )

@pytest.fixture
def mock_circuit():
    """Create a mock quantum circuit for testing."""
    circuit = Mock()
    circuit.qubit_count = 2
    circuit.gate_count = 5
    circuit.depth = 3
    circuit.gates = [Mock(qubit_count=1), Mock(qubit_count=2), Mock(qubit_count=1)]
    circuit.run = Mock(return_value={"success": True, "fidelity": 0.98})
    circuit.execute = Mock(return_value={"success": True, "fidelity": 0.98})
    return circuit

@pytest.fixture
def metrics_list():
    """Create a list of sample metrics for testing."""
    metrics = []
    for i in range(5):
        m = PerformanceMetrics(
            execution_time=1.0 + i * 0.1,
            memory_usage=100.0 + i * 10,
            gate_count=10 + i * 2,
            circuit_depth=5 + i,
            qubit_count=2 + i,
            error_rate=0.01 * i,
            fidelity=1.0 - 0.01 * i,
            operation_type=f"test_operation_{i}",
            backend="test_backend"
        )
        metrics.append(m)
    return metrics


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass functionality."""
    
    def test_default_initialization(self):
        """Test default PerformanceMetrics initialization."""
        metrics = PerformanceMetrics()
        assert metrics.execution_time == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.circuit_depth == 0
        assert metrics.gate_count == 0
        assert metrics.qubit_count == 0
        assert metrics.fidelity == 1.0
        assert metrics.error_rate == 0.0
        assert metrics.operation_type == ""
        assert metrics.backend == ""
        assert isinstance(metrics.configuration, dict)
        assert isinstance(metrics.gate_times, dict)
    
    def test_custom_initialization(self, sample_metrics):
        """Test PerformanceMetrics with custom values."""
        assert sample_metrics.execution_time == 1.5
        assert sample_metrics.memory_usage == 128.0
        assert sample_metrics.circuit_depth == 10
        assert sample_metrics.gate_count == 25
        assert sample_metrics.qubit_count == 4
        assert sample_metrics.fidelity == 0.95
        assert sample_metrics.error_rate == 0.05
    
    def test_to_dict_conversion(self, sample_metrics):
        """Test conversion to dictionary format."""
        data_dict = sample_metrics.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['execution_time'] == 1.5
        assert data_dict['memory_usage'] == 128.0
        assert data_dict['operation_type'] == "circuit_execution"
        assert 'timestamp' in data_dict
        assert 'configuration' in data_dict
    
    def test_summary_generation(self, sample_metrics):
        """Test human-readable summary generation."""
        summary = sample_metrics.summary()
        assert isinstance(summary, str)
        assert "Performance Summary:" in summary
        assert "1.5000s" in summary  # Execution time
        assert "128.00MB" in summary  # Memory usage
        assert "25" in summary  # Gate count
        assert "4" in summary  # Qubit count
        assert "0.950000" in summary  # Fidelity
    
    def test_gate_times_handling(self):
        """Test gate timing information handling."""
        metrics = PerformanceMetrics()
        metrics.gate_times['h'] = [0.001, 0.002, 0.0015]
        metrics.gate_times['cnot'] = [0.005, 0.0055]
        
        assert len(metrics.gate_times['h']) == 3
        assert len(metrics.gate_times['cnot']) == 2
        assert metrics.gate_times['h'][0] == 0.001


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceProfiler:
    """Test base PerformanceProfiler functionality."""
    
    def test_profiler_initialization(self):
        """Test base profiler initialization."""
        profiler = PerformanceProfiler("TestProfiler")
        assert profiler.name == "TestProfiler"
        assert profiler.metrics_history == []
        assert profiler.active_measurements == {}
        assert profiler.start_time is None
        assert profiler.process is not None
    
    @patch('psutil.Process')
    def test_memory_usage_collection(self, mock_process):
        """Test memory usage collection."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 128 * 1024 * 1024  # 128 MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        profiler = PerformanceProfiler()
        memory_mb = profiler._get_memory_usage()
        assert memory_mb == 128.0
    
    @patch('psutil.Process')
    def test_cpu_usage_collection(self, mock_process):
        """Test CPU usage collection."""
        mock_process.return_value.cpu_percent.return_value = 75.5
        
        profiler = PerformanceProfiler()
        cpu_usage = profiler._get_cpu_usage()
        assert cpu_usage == 75.5
    
    @patch('psutil.Process')
    def test_profile_operation_context_manager(self, mock_process):
        """Test the profile_operation context manager."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        profiler = PerformanceProfiler()
        
        with profiler.profile_operation("test_operation", test_param="value") as metrics:
            time.sleep(0.01)  # Small delay to ensure execution time > 0
            assert metrics.operation_type == "test_operation"
        
        assert len(profiler.metrics_history) == 1
        recorded_metrics = profiler.metrics_history[0]
        assert recorded_metrics.execution_time > 0
        assert recorded_metrics.memory_usage == 100.0
        assert recorded_metrics.operation_type == "test_operation"


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestCircuitProfiler:
    """Test CircuitProfiler with circuit analysis and bottleneck detection."""
    
    def test_circuit_profiler_initialization(self):
        """Test CircuitProfiler initialization."""
        profiler = CircuitProfiler()
        assert profiler.name == "CircuitProfiler"
        assert isinstance(profiler.gate_timings, dict)
        assert isinstance(profiler.circuit_cache, dict)
    
    @patch('psutil.Process')
    def test_profile_circuit_basic(self, mock_process, mock_circuit):
        """Test basic circuit profiling."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        profiler = CircuitProfiler()
        metrics = profiler.profile_circuit(mock_circuit, backend="test_backend")
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.qubit_count == 2
        assert metrics.gate_count == 5
        assert metrics.circuit_depth == 3
        assert metrics.execution_time > 0
        assert metrics.backend == "test_backend"
        mock_circuit.run.assert_called_once()
    
    @patch('psutil.Process')
    def test_circuit_structure_analysis(self, mock_process, mock_circuit):
        """Test circuit structure analysis."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        profiler = CircuitProfiler()
        metrics = profiler.profile_circuit(mock_circuit)
        
        # Should calculate entanglement degree based on multi-qubit gates
        assert 0 <= metrics.entanglement_degree <= 1
        # With 1 two-qubit gate out of 3 total gates, entanglement degree should be ~0.33
        assert abs(metrics.entanglement_degree - (1/3)) < 0.1
    
    def test_circuit_execution_with_execute_method(self, mock_circuit):
        """Test circuit execution using execute method."""
        mock_circuit.execute.return_value = {"success": True, "fidelity": 0.95}
        del mock_circuit.run  # Remove run method to force use of execute
        
        profiler = CircuitProfiler()
        result = profiler._execute_circuit(mock_circuit, "test_backend")
        
        assert result["success"] is True
        assert result["fidelity"] == 0.95
        mock_circuit.execute.assert_called_once_with(backend="test_backend")
    
    def test_circuit_execution_fallback(self):
        """Test circuit execution fallback for circuits without run/execute methods."""
        mock_circuit = Mock()
        mock_circuit.gate_count = 10
        del mock_circuit.run
        del mock_circuit.execute
        
        profiler = CircuitProfiler()
        result = profiler._execute_circuit(mock_circuit, "test_backend")
        
        assert result == {"success": True}
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation from results."""
        profiler = CircuitProfiler()
        metrics = PerformanceMetrics()
        
        result = {
            "fidelity": 0.98,
            "error_rate": 0.02,
            "coherence_loss": 0.05
        }
        
        profiler._calculate_quality_metrics(result, metrics)
        
        assert metrics.fidelity == 0.98
        assert metrics.error_rate == 0.02
        assert metrics.coherence_loss == 0.05
    
    def test_bottleneck_analysis(self, metrics_list):
        """Test bottleneck analysis functionality."""
        profiler = CircuitProfiler()
        analysis = profiler.analyze_bottlenecks(metrics_list)
        
        assert isinstance(analysis, dict)
        assert 'timing_bottlenecks' in analysis
        assert 'memory_bottlenecks' in analysis
        assert 'scalability_issues' in analysis
        assert 'optimization_opportunities' in analysis
        
        timing_analysis = analysis['timing_bottlenecks']
        assert 'mean_time' in timing_analysis
        assert 'std_time' in timing_analysis
        assert 'max_time' in timing_analysis
        assert 'min_time' in timing_analysis
    
    def test_timing_bottleneck_identification(self, metrics_list):
        """Test timing bottleneck identification."""
        profiler = CircuitProfiler()
        timing_bottlenecks = profiler._identify_timing_bottlenecks(metrics_list)
        
        execution_times = [m.execution_time for m in metrics_list]
        expected_mean = statistics.mean(execution_times)
        
        assert timing_bottlenecks['mean_time'] == expected_mean
        assert timing_bottlenecks['max_time'] == max(execution_times)
        assert timing_bottlenecks['min_time'] == min(execution_times)
        assert isinstance(timing_bottlenecks['outliers'], list)
    
    def test_memory_bottleneck_identification(self, metrics_list):
        """Test memory bottleneck identification."""
        profiler = CircuitProfiler()
        memory_bottlenecks = profiler._identify_memory_bottlenecks(metrics_list)
        
        memory_usage = [m.memory_usage for m in metrics_list]
        expected_mean = statistics.mean(memory_usage)
        
        assert memory_bottlenecks['mean_memory'] == expected_mean
        assert memory_bottlenecks['peak_memory'] == max(memory_usage)
        assert 'memory_leaks' in memory_bottlenecks
        assert 'memory_efficiency' in memory_bottlenecks
    
    def test_scalability_analysis(self, metrics_list):
        """Test scalability issue identification."""
        profiler = CircuitProfiler()
        scalability_issues = profiler._identify_scalability_issues(metrics_list)
        
        # With varying qubit counts in metrics_list, should provide scaling analysis
        assert 'scaling_efficiency' in scalability_issues
        assert 'sublinear_scaling' in scalability_issues
        assert 'superlinear_scaling' in scalability_issues
    
    def test_optimization_opportunities_identification(self, metrics_list):
        """Test optimization opportunity identification."""
        # Create metrics with issues that should trigger recommendations
        high_error_metrics = [
            PerformanceMetrics(error_rate=0.2, fidelity=0.7, memory_delta=150, 
                             circuit_depth=50, gate_count=100)
        ]
        
        profiler = CircuitProfiler()
        opportunities = profiler._identify_optimization_opportunities(high_error_metrics)
        
        assert isinstance(opportunities, list)
        assert any("High error rate" in opp for opp in opportunities)
        assert any("Low fidelity" in opp for opp in opportunities)
        assert any("High memory consumption" in opp for opp in opportunities)
        assert any("High circuit depth" in opp for opp in opportunities)
    
    def test_empty_metrics_bottleneck_analysis(self):
        """Test bottleneck analysis with empty metrics list."""
        profiler = CircuitProfiler()
        analysis = profiler.analyze_bottlenecks([])
        
        assert analysis == {}


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestGateProfiler:
    """Test GateProfiler with gate-level timing and statistics."""
    
    def test_gate_profiler_initialization(self):
        """Test GateProfiler initialization."""
        profiler = GateProfiler()
        assert profiler.name == "GateProfiler"
        assert isinstance(profiler.gate_statistics, dict)
    
    @patch('psutil.Process')
    def test_profile_gate_operation(self, mock_process):
        """Test profiling individual gate operations."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        def mock_gate_operation(qubit):
            time.sleep(0.001)  # Simulate gate execution time
            return f"Applied to qubit {qubit}"
        
        profiler = GateProfiler()
        metrics = profiler.profile_gate("H", mock_gate_operation, 0)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.gate_count == 1
        assert metrics.execution_time > 0
        assert 'H' in metrics.gate_times
        assert len(metrics.gate_times['H']) == 1
        assert metrics.gate_times['H'][0] > 0
        
        # Check that statistics were recorded
        assert 'H' in profiler.gate_statistics
        assert len(profiler.gate_statistics['H']['execution_times']) == 1
    
    def test_gate_statistics_collection(self):
        """Test gate statistics collection and retrieval."""
        profiler = GateProfiler()
        
        # Manually add some gate timing data
        profiler.gate_statistics['H']['execution_times'] = [0.001, 0.0015, 0.0012, 0.0008, 0.0014]
        
        stats = profiler.get_gate_statistics('H')
        
        assert 'mean_time' in stats
        assert 'median_time' in stats
        assert 'std_time' in stats
        assert 'min_time' in stats
        assert 'max_time' in stats
        assert 'sample_count' in stats
        
        assert stats['sample_count'] == 5
        assert stats['min_time'] == 0.0008
        assert stats['max_time'] == 0.0015
        assert abs(stats['mean_time'] - 0.00118) < 0.0001
    
    def test_gate_statistics_empty_data(self):
        """Test gate statistics with no data."""
        profiler = GateProfiler()
        
        # Test non-existent gate
        stats = profiler.get_gate_statistics('NonExistentGate')
        assert stats == {}
        
        # Test gate with empty statistics
        profiler.gate_statistics['EmptyGate'] = {'execution_times': []}
        stats = profiler.get_gate_statistics('EmptyGate')
        assert stats == {}
    
    def test_compare_gates(self):
        """Test gate performance comparison."""
        profiler = GateProfiler()
        
        # Add statistics for multiple gates
        profiler.gate_statistics['H']['execution_times'] = [0.001, 0.0015, 0.0012]
        profiler.gate_statistics['CNOT']['execution_times'] = [0.005, 0.0055, 0.0048]
        profiler.gate_statistics['X']['execution_times'] = [0.0008, 0.0009, 0.0007]
        
        comparison = profiler.compare_gates(['H', 'CNOT', 'X'])
        
        assert 'H' in comparison
        assert 'CNOT' in comparison
        assert 'X' in comparison
        
        # CNOT should have higher average time than single-qubit gates
        assert comparison['CNOT']['mean_time'] > comparison['H']['mean_time']
        assert comparison['CNOT']['mean_time'] > comparison['X']['mean_time']
    
    @patch('psutil.Process')
    def test_gate_profiling_error_handling(self, mock_process):
        """Test error handling in gate profiling."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        def failing_gate_operation():
            raise Exception("Gate operation failed")
        
        profiler = GateProfiler()
        metrics = profiler.profile_gate("FailingGate", failing_gate_operation)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.error_rate == 1.0


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestMemoryProfiler:
    """Test MemoryProfiler with memory monitoring and leak detection."""
    
    def test_memory_profiler_initialization(self):
        """Test MemoryProfiler initialization."""
        profiler = MemoryProfiler()
        assert profiler.name == "MemoryProfiler"
        assert profiler.memory_snapshots == []
        assert profiler.monitoring is False
        assert profiler.monitor_thread is None
    
    @patch('psutil.Process')
    def test_start_stop_monitoring(self, mock_process):
        """Test starting and stopping memory monitoring."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        profiler = MemoryProfiler()
        
        # Start monitoring
        profiler.start_monitoring(interval=0.01)
        assert profiler.monitoring is True
        assert profiler.monitor_thread is not None
        assert profiler.monitor_thread.is_alive()
        
        # Let it collect some samples
        time.sleep(0.05)
        
        # Stop monitoring
        profiler.stop_monitoring()
        assert profiler.monitoring is False
        
        # Check that samples were collected
        assert len(profiler.memory_snapshots) > 0
    
    @patch('psutil.Process')
    def test_memory_pattern_analysis(self, mock_process):
        """Test memory usage pattern analysis."""
        profiler = MemoryProfiler()
        
        # Add mock memory snapshots
        start_time = time.time()
        profiler.memory_snapshots = [
            (start_time, 100.0),
            (start_time + 0.1, 110.0),
            (start_time + 0.2, 120.0),
            (start_time + 0.3, 115.0),
            (start_time + 0.4, 125.0)
        ]
        
        analysis = profiler.analyze_memory_pattern(start_time, start_time + 0.4)
        
        assert analysis['initial_memory'] == 100.0
        assert analysis['final_memory'] == 125.0
        assert analysis['peak_memory'] == 125.0
        assert analysis['min_memory'] == 100.0
        assert analysis['memory_delta'] == 25.0
        assert analysis['duration'] == 0.4
        assert analysis['memory_growth_rate'] > 0
    
    def test_memory_pattern_analysis_no_data(self):
        """Test memory pattern analysis with no relevant data."""
        profiler = MemoryProfiler()
        
        start_time = time.time()
        analysis = profiler.analyze_memory_pattern(start_time, start_time + 1.0)
        
        assert analysis == {}
    
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        profiler = MemoryProfiler()
        
        # Create snapshots showing consistent memory growth
        base_time = time.time()
        for i in range(20):
            timestamp = base_time + i * 0.1
            memory = 100.0 + i * 2.0  # 2MB growth per sample
            profiler.memory_snapshots.append((timestamp, memory))
        
        leaks = profiler.detect_memory_leaks(threshold_mb=15.0)
        
        assert len(leaks) > 0
        leak = leaks[0]
        assert 'start_time' in leak
        assert 'end_time' in leak
        assert 'memory_growth' in leak
        assert 'growth_rate' in leak
        assert leak['memory_growth'] >= 15.0
    
    def test_memory_leak_detection_insufficient_data(self):
        """Test memory leak detection with insufficient data."""
        profiler = MemoryProfiler()
        
        # Add only a few snapshots
        base_time = time.time()
        for i in range(5):
            profiler.memory_snapshots.append((base_time + i * 0.1, 100.0 + i))
        
        leaks = profiler.detect_memory_leaks()
        
        assert leaks == []
    
    def test_memory_leak_detection_no_growth(self):
        """Test memory leak detection with stable memory usage."""
        profiler = MemoryProfiler()
        
        # Create snapshots with stable memory
        base_time = time.time()
        for i in range(20):
            timestamp = base_time + i * 0.1
            memory = 100.0 + (i % 3)  # Oscillating memory pattern
            profiler.memory_snapshots.append((timestamp, memory))
        
        leaks = profiler.detect_memory_leaks(threshold_mb=10.0)
        
        assert len(leaks) == 0


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceComparator:
    """Test PerformanceComparator with backend comparisons."""
    
    def test_comparator_initialization(self):
        """Test PerformanceComparator initialization."""
        comparator = PerformanceComparator()
        assert isinstance(comparator.comparison_data, dict)
        assert len(comparator.comparison_data) == 0
    
    def test_add_measurement(self, sample_metrics):
        """Test adding measurements for comparison."""
        comparator = PerformanceComparator()
        
        comparator.add_measurement("backend_a", sample_metrics)
        comparator.add_measurement("backend_a", sample_metrics)
        
        assert "backend_a" in comparator.comparison_data
        assert len(comparator.comparison_data["backend_a"]) == 2
    
    def test_compare_implementations_single_metric(self):
        """Test comparing implementations on a single metric."""
        comparator = PerformanceComparator()
        
        # Add measurements for two backends
        backend_a_metrics = [
            PerformanceMetrics(execution_time=1.0),
            PerformanceMetrics(execution_time=1.2),
            PerformanceMetrics(execution_time=0.9)
        ]
        
        backend_b_metrics = [
            PerformanceMetrics(execution_time=1.5),
            PerformanceMetrics(execution_time=1.4),
            PerformanceMetrics(execution_time=1.6)
        ]
        
        for metrics in backend_a_metrics:
            comparator.add_measurement("backend_a", metrics)
        for metrics in backend_b_metrics:
            comparator.add_measurement("backend_b", metrics)
        
        comparison = comparator.compare_implementations("execution_time")
        
        assert "backend_a" in comparison
        assert "backend_b" in comparison
        
        # Backend A should have better (lower) average execution time
        assert comparison["backend_a"]["mean"] < comparison["backend_b"]["mean"]
        
        # Check relative performance calculation
        assert "relative_performance" in comparison["backend_a"]
        assert "relative_performance" in comparison["backend_b"]
        assert comparison["backend_a"]["relative_performance"] == 1.0  # Best performance baseline
        assert comparison["backend_b"]["relative_performance"] > 1.0
    
    def test_compare_implementations_empty_data(self):
        """Test comparison with no data."""
        comparator = PerformanceComparator()
        comparison = comparator.compare_implementations("execution_time")
        
        assert comparison == {}
    
    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        comparator = PerformanceComparator()
        
        # Add some test data
        metrics_a = PerformanceMetrics(execution_time=1.0, memory_usage=100.0, error_rate=0.01, fidelity=0.99)
        metrics_b = PerformanceMetrics(execution_time=1.5, memory_usage=150.0, error_rate=0.02, fidelity=0.98)
        
        comparator.add_measurement("backend_a", metrics_a)
        comparator.add_measurement("backend_b", metrics_b)
        
        report = comparator.generate_comparison_report()
        
        assert isinstance(report, str)
        assert "Performance Comparison Report" in report
        assert "Execution Time:" in report
        assert "Memory Usage:" in report
        assert "Error Rate:" in report
        assert "Fidelity:" in report
        assert "backend_a" in report
        assert "backend_b" in report
    
    def test_generate_comparison_report_no_data(self):
        """Test comparison report with no data."""
        comparator = PerformanceComparator()
        report = comparator.generate_comparison_report()
        
        assert report == "No comparison data available."


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceOptimizer:
    """Test PerformanceOptimizer with optimization recommendations."""
    
    def test_optimizer_initialization(self):
        """Test PerformanceOptimizer initialization."""
        optimizer = PerformanceOptimizer()
        assert len(optimizer.optimization_rules) == 5
        assert callable(optimizer.optimization_rules[0])
    
    def test_analyze_and_recommend_single_metrics(self):
        """Test optimization analysis with single metrics object."""
        optimizer = PerformanceOptimizer()
        
        metrics = PerformanceMetrics(
            memory_usage=1500.0,  # High memory
            memory_delta=150.0,   # High memory growth
            memory_efficiency=0.3, # Low efficiency
            execution_time=5.0,
            cpu_usage=30.0,       # Low CPU usage
            circuit_depth=50,
            gate_count=100,
            entanglement_degree=0.9  # High entanglement
        )
        
        recommendations = optimizer.analyze_and_recommend(metrics)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("memory optimization" in rec.lower() for rec in recommendations)
        assert any("parallelization" in rec.lower() for rec in recommendations)
    
    def test_analyze_and_recommend_metrics_list(self, metrics_list):
        """Test optimization analysis with list of metrics."""
        optimizer = PerformanceOptimizer()
        
        recommendations = optimizer.analyze_and_recommend(metrics_list)
        
        assert isinstance(recommendations, list)
        # Should get some recommendations based on the metrics
    
    def test_memory_optimization_recommendations(self):
        """Test memory optimization rule."""
        optimizer = PerformanceOptimizer()
        
        high_memory_metrics = [
            PerformanceMetrics(memory_usage=1200.0, memory_delta=120.0, memory_efficiency=0.4),
            PerformanceMetrics(memory_usage=1100.0, memory_delta=110.0, memory_efficiency=0.3)
        ]
        
        recommendations = optimizer._check_memory_optimization(high_memory_metrics)
        
        assert len(recommendations) >= 2
        assert any("memory optimization" in rec for rec in recommendations)
        assert any("Memory growth detected" in rec for rec in recommendations)
        assert any("memory efficiency" in rec for rec in recommendations)
    
    def test_timing_optimization_recommendations(self):
        """Test timing optimization rule."""
        optimizer = PerformanceOptimizer()
        
        timing_metrics = [
            PerformanceMetrics(execution_time=1.0, cpu_usage=30.0),
            PerformanceMetrics(execution_time=3.0, cpu_usage=25.0),  # High variance
            PerformanceMetrics(execution_time=1.5, cpu_usage=35.0)
        ]
        
        recommendations = optimizer._check_timing_optimization(timing_metrics)
        
        assert len(recommendations) >= 1
        assert any("timing inconsistencies" in rec or "parallelization" in rec for rec in recommendations)
    
    def test_circuit_optimization_recommendations(self):
        """Test circuit optimization rule."""
        optimizer = PerformanceOptimizer()
        
        circuit_metrics = [
            PerformanceMetrics(circuit_depth=70, gate_count=100, entanglement_degree=0.85),
            PerformanceMetrics(circuit_depth=75, gate_count=105, entanglement_degree=0.90)
        ]
        
        recommendations = optimizer._check_circuit_optimization(circuit_metrics)
        
        assert len(recommendations) >= 1
        assert any("gate reordering" in rec or "circuit decomposition" in rec for rec in recommendations)
    
    def test_gate_optimization_recommendations(self):
        """Test gate optimization rule."""
        optimizer = PerformanceOptimizer()
        
        gate_metrics = [
            PerformanceMetrics(gate_times={'H': [0.001, 0.005, 0.002]}),  # High variance
            PerformanceMetrics(gate_times={'CNOT': [0.01, 0.03, 0.015]})  # High variance
        ]
        
        recommendations = optimizer._check_gate_optimization(gate_metrics)
        
        assert len(recommendations) >= 1
        assert any("timing variance" in rec for rec in recommendations)
    
    def test_scalability_optimization_recommendations(self):
        """Test scalability optimization rule."""
        optimizer = PerformanceOptimizer()
        
        scalability_metrics = [
            PerformanceMetrics(scaling_factor=2.5, parallelization_efficiency=0.6),
            PerformanceMetrics(scaling_factor=3.0, parallelization_efficiency=0.5)
        ]
        
        recommendations = optimizer._check_scalability_optimization(scalability_metrics)
        
        assert len(recommendations) >= 1
        assert any("scaling" in rec or "parallel" in rec for rec in recommendations)
    
    def test_no_optimization_needed(self):
        """Test case where no optimization recommendations are needed."""
        optimizer = PerformanceOptimizer()
        
        good_metrics = [
            PerformanceMetrics(
                memory_usage=100.0, memory_delta=5.0, memory_efficiency=0.8,
                execution_time=1.0, cpu_usage=75.0,
                circuit_depth=10, gate_count=20, entanglement_degree=0.5,
                scaling_factor=1.0, parallelization_efficiency=0.9,
                error_rate=0.01, fidelity=0.99
            )
        ]
        
        recommendations = optimizer.analyze_and_recommend(good_metrics)
        
        # Should have minimal recommendations for good metrics
        assert len(recommendations) <= 2


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceMonitor:
    """Test PerformanceMonitor with real-time monitoring and alerts."""
    
    def test_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.alert_thresholds['execution_time'] == 10.0
        assert monitor.alert_thresholds['memory_usage'] == 1000.0
        assert monitor.alert_thresholds['error_rate'] == 0.1
        assert monitor.alert_thresholds['cpu_usage'] == 90.0
        assert monitor.alerts == []
        assert monitor.monitoring_active is False
        assert len(monitor.metrics_buffer) == 0
    
    def test_custom_alert_thresholds(self):
        """Test monitor with custom alert thresholds."""
        custom_thresholds = {
            'execution_time': 5.0,
            'memory_usage': 500.0,
            'error_rate': 0.05
        }
        
        monitor = PerformanceMonitor(custom_thresholds)
        assert monitor.alert_thresholds['execution_time'] == 5.0
        assert monitor.alert_thresholds['memory_usage'] == 500.0
        assert monitor.alert_thresholds['error_rate'] == 0.05
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    async def test_collect_current_metrics(self, mock_cpu_percent, mock_virtual_memory):
        """Test current metrics collection."""
        mock_virtual_memory.return_value.used = 512 * 1024 * 1024  # 512 MB
        mock_cpu_percent.return_value = 65.0
        
        monitor = PerformanceMonitor()
        metrics = await monitor._collect_current_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.memory_usage == 512.0
        assert metrics.cpu_usage == 65.0
    
    def test_alert_checking(self):
        """Test alert threshold checking."""
        monitor = PerformanceMonitor()
        
        # Create metrics that exceed thresholds
        high_metrics = PerformanceMetrics(
            execution_time=15.0,  # Exceeds 10.0 threshold
            memory_usage=1200.0,  # Exceeds 1000.0 threshold
            error_rate=0.15       # Exceeds 0.1 threshold
        )
        
        monitor._check_alerts(high_metrics)
        
        assert len(monitor.alerts) == 3  # Should have 3 alerts
        
        # Check alert details
        execution_alert = next(a for a in monitor.alerts if a['metric'] == 'execution_time')
        assert execution_alert['value'] == 15.0
        assert execution_alert['threshold'] == 10.0
        assert execution_alert['severity'] == 'high'  # 15.0 > 10.0 * 1.5
        
        memory_alert = next(a for a in monitor.alerts if a['metric'] == 'memory_usage')
        assert memory_alert['value'] == 1200.0
        assert memory_alert['threshold'] == 1000.0
        assert memory_alert['severity'] == 'medium'  # 1200.0 < 1000.0 * 1.5
    
    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        monitor = PerformanceMonitor()
        
        # Add some alerts with different timestamps
        current_time = time.time()
        old_alert = {
            'timestamp': current_time - 3700,  # More than 1 hour ago
            'metric': 'execution_time',
            'value': 15.0,
            'threshold': 10.0,
            'severity': 'high'
        }
        recent_alert = {
            'timestamp': current_time - 1800,  # 30 minutes ago
            'metric': 'memory_usage',
            'value': 1200.0,
            'threshold': 1000.0,
            'severity': 'medium'
        }
        
        monitor.alerts = [old_alert, recent_alert]
        
        # Get alerts from last 60 minutes
        recent_alerts = monitor.get_recent_alerts(60)
        
        assert len(recent_alerts) == 1
        assert recent_alerts[0]['metric'] == 'memory_usage'
    
    def test_monitoring_summary(self):
        """Test monitoring summary generation."""
        monitor = PerformanceMonitor()
        
        # Add some metrics to buffer
        for i in range(5):
            metrics = PerformanceMetrics(
                execution_time=1.0 + i * 0.1,
                memory_usage=100.0 + i * 10,
                cpu_usage=50.0 + i * 5,
                timestamp=time.time() + i
            )
            monitor.metrics_buffer.append(metrics)
        
        # Add an alert
        monitor.alerts.append({
            'timestamp': time.time(),
            'metric': 'test',
            'value': 100,
            'threshold': 50,
            'severity': 'high'
        })
        
        summary = monitor.get_monitoring_summary()
        
        assert summary['total_samples'] == 5
        assert 'time_range' in summary
        assert 'averages' in summary
        assert summary['alert_count'] == 1
        assert summary['monitoring_active'] is False
        
        # Check averages
        assert summary['averages']['execution_time'] == 1.2  # Mean of 1.0, 1.1, 1.2, 1.3, 1.4
        assert summary['averages']['memory_usage'] == 140.0  # Mean of 100, 110, 120, 130, 140
    
    def test_monitoring_summary_empty(self):
        """Test monitoring summary with no data."""
        monitor = PerformanceMonitor()
        summary = monitor.get_monitoring_summary()
        
        assert summary == {}
    
    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        monitor = PerformanceMonitor()
        monitor.monitoring_active = True
        
        monitor.stop_monitoring()
        
        assert monitor.monitoring_active is False


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceReporter:
    """Test PerformanceReporter with report generation in multiple formats."""
    
    def test_reporter_initialization(self):
        """Test PerformanceReporter initialization."""
        reporter = PerformanceReporter()
        assert len(reporter.report_templates) == 4
        assert 'summary' in reporter.report_templates
        assert 'detailed' in reporter.report_templates
        assert 'comparison' in reporter.report_templates
        assert 'optimization' in reporter.report_templates
    
    def test_generate_summary_report(self, metrics_list):
        """Test summary report generation."""
        reporter = PerformanceReporter()
        report = reporter.generate_report(metrics_list, "summary")
        
        assert isinstance(report, str)
        assert "Performance Summary Report" in report
        assert "Total Measurements: 5" in report
        assert "Execution Time Statistics:" in report
        assert "Memory Usage Statistics:" in report
        assert "Quality Metrics:" in report
        assert "Circuit Characteristics:" in report
    
    def test_generate_detailed_report(self, metrics_list):
        """Test detailed report generation."""
        reporter = PerformanceReporter()
        report = reporter.generate_report(metrics_list, "detailed")
        
        assert isinstance(report, str)
        assert "Detailed Performance Report" in report
        assert "Total Measurements: 5" in report
        assert "Measurement 1:" in report
        assert "Measurement 5:" in report
        assert "Operation:" in report
        assert "Execution Time:" in report
    
    def test_generate_comparison_report(self, metrics_list):
        """Test comparison report generation."""
        # Modify metrics to have different backends
        for i, metrics in enumerate(metrics_list):
            metrics.backend = f"backend_{i % 2}"  # Two different backends
        
        reporter = PerformanceReporter()
        report = reporter.generate_report(metrics_list, "comparison")
        
        assert isinstance(report, str)
        assert "Performance Comparison Report" in report
        assert "Execution Time Comparison:" in report
        assert "Memory Usage Comparison:" in report
        assert "backend_0" in report
        assert "backend_1" in report
    
    def test_generate_comparison_report_insufficient_data(self):
        """Test comparison report with insufficient data."""
        metrics_list = [PerformanceMetrics(backend="single_backend")]
        
        reporter = PerformanceReporter()
        report = reporter.generate_report(metrics_list, "comparison")
        
        assert "Insufficient data for comparison" in report
    
    def test_generate_optimization_report(self, metrics_list):
        """Test optimization report generation."""
        reporter = PerformanceReporter()
        report = reporter.generate_report(metrics_list, "optimization")
        
        assert isinstance(report, str)
        assert "Performance Optimization Report" in report
        assert "Optimization Recommendations:" in report
        assert "Bottleneck Analysis:" in report
    
    def test_json_format_output(self, metrics_list):
        """Test JSON format report output."""
        reporter = PerformanceReporter()
        json_report = reporter.generate_report(metrics_list, "summary", "json")
        
        # Should be valid JSON
        data = json.loads(json_report)
        assert 'report' in data
        assert 'generated_at' in data
        assert 'format' in data
        assert data['format'] == 'json'
    
    def test_html_format_output(self, metrics_list):
        """Test HTML format report output."""
        reporter = PerformanceReporter()
        html_report = reporter.generate_report(metrics_list, "summary", "html")
        
        assert isinstance(html_report, str)
        assert "<!DOCTYPE html>" in html_report
        assert "<html>" in html_report
        assert "<body>" in html_report
        assert "QuantRS2 Performance Report" in html_report
        assert "Performance Summary Report" in html_report
    
    def test_unknown_report_type(self, metrics_list):
        """Test error handling for unknown report type."""
        reporter = PerformanceReporter()
        
        with pytest.raises(ValueError, match="Unknown report type"):
            reporter.generate_report(metrics_list, "unknown_type")
    
    def test_save_report(self, metrics_list):
        """Test saving report to file."""
        reporter = PerformanceReporter()
        report = reporter.generate_report(metrics_list, "summary")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            reporter.save_report(report, temp_path)
            
            # Verify file was created and contains expected content
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "Performance Summary Report" in content
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_empty_metrics_report(self):
        """Test report generation with empty metrics."""
        reporter = PerformanceReporter()
        report = reporter.generate_report([], "summary")
        
        assert report == "No performance data available."


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQuantumPerformanceProfiler:
    """Test QuantumPerformanceProfiler main class with comprehensive profiling."""
    
    def test_main_profiler_initialization(self):
        """Test QuantumPerformanceProfiler initialization."""
        profiler = QuantumPerformanceProfiler()
        
        assert profiler.config == {}
        assert isinstance(profiler.circuit_profiler, CircuitProfiler)
        assert isinstance(profiler.gate_profiler, GateProfiler)
        assert isinstance(profiler.memory_profiler, MemoryProfiler)
        assert isinstance(profiler.comparator, PerformanceComparator)
        assert isinstance(profiler.optimizer, PerformanceOptimizer)
        assert isinstance(profiler.monitor, PerformanceMonitor)
        assert isinstance(profiler.reporter, PerformanceReporter)
        assert profiler.all_metrics == []
        assert profiler.baseline_metrics is None
        assert profiler.regression_threshold == 0.1
    
    def test_custom_configuration(self):
        """Test profiler with custom configuration."""
        custom_config = {
            'alert_thresholds': {'memory_usage': 500.0},
            'regression_threshold': 0.05
        }
        
        profiler = QuantumPerformanceProfiler(custom_config)
        
        assert profiler.config == custom_config
        assert profiler.regression_threshold == 0.05
        assert profiler.monitor.alert_thresholds['memory_usage'] == 500.0
    
    @patch('psutil.Process')
    def test_profile_circuit_execution(self, mock_process, mock_circuit):
        """Test comprehensive circuit execution profiling."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        profiler = QuantumPerformanceProfiler()
        metrics = profiler.profile_circuit_execution(mock_circuit, "test_backend", "test_label")
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.backend == "test_backend"
        assert metrics.configuration['label'] == "test_label"
        assert len(profiler.all_metrics) == 1
        assert "test_backend" in profiler.comparator.comparison_data
    
    @patch('psutil.Process')
    def test_profile_gate_sequence(self, mock_process):
        """Test gate sequence profiling."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        def mock_h_gate():
            time.sleep(0.001)
            return "H applied"
        
        def mock_cnot_gate():
            time.sleep(0.002)
            return "CNOT applied"
        
        gates = [("H", mock_h_gate), ("CNOT", mock_cnot_gate)]
        
        profiler = QuantumPerformanceProfiler()
        gate_metrics = profiler.profile_gate_sequence(gates, "gate_test")
        
        assert len(gate_metrics) == 2
        assert all(isinstance(m, PerformanceMetrics) for m in gate_metrics)
        assert all(m.configuration['label'] == "gate_test" for m in gate_metrics)
        assert len(profiler.all_metrics) == 2
    
    @patch('psutil.Process')
    def test_benchmark_scalability(self, mock_process):
        """Test scalability benchmarking."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        def circuit_generator(qubit_count):
            circuit = Mock()
            circuit.qubit_count = qubit_count
            circuit.gate_count = qubit_count * 2
            circuit.depth = qubit_count
            circuit.run = Mock(return_value={"success": True})
            return circuit
        
        profiler = QuantumPerformanceProfiler()
        results = profiler.benchmark_scalability(circuit_generator, [2, 4, 6], "test_backend")
        
        assert len(results) == 3
        assert 2 in results and 4 in results and 6 in results
        assert all(isinstance(m, PerformanceMetrics) for m in results.values())
        
        # Check scaling factors (should be relative to 2-qubit circuit)
        assert results[2].scaling_factor == 1.0  # First circuit, no scaling factor
        assert results[4].scaling_factor > 0  # Should have some scaling factor
        assert results[6].scaling_factor > 0
    
    @patch('psutil.Process')
    def test_compare_backends(self, mock_process, mock_circuit):
        """Test backend comparison."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        profiler = QuantumPerformanceProfiler()
        comparison = profiler.compare_backends(mock_circuit, ["backend_a", "backend_b"], runs_per_backend=2)
        
        # Should return comparison data
        assert isinstance(comparison, dict)
        assert len(profiler.all_metrics) == 4  # 2 backends * 2 runs each
    
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        profiler = QuantumPerformanceProfiler()
        
        # Set baseline metrics
        profiler.baseline_metrics = {
            "test_backend_execution_time": 1.0
        }
        
        # Add metrics with regression
        regression_metrics = PerformanceMetrics(
            execution_time=1.2,  # 20% slower than baseline
            backend="test_backend",
            operation_type="test_operation"
        )
        profiler.all_metrics.append(regression_metrics)
        
        regressions = profiler.detect_performance_regressions()
        
        assert len(regressions) == 1
        regression = regressions[0]
        assert regression['backend'] == "test_backend"
        assert regression['baseline_time'] == 1.0
        assert regression['current_time'] == 1.2
        assert regression['regression_factor'] == 1.2
    
    def test_generate_comprehensive_report(self, metrics_list):
        """Test comprehensive report generation."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        report = profiler.generate_comprehensive_report("summary")
        
        assert isinstance(report, str)
        assert "Performance Summary Report" in report
    
    def test_generate_report_no_data(self):
        """Test report generation with no data."""
        profiler = QuantumPerformanceProfiler()
        
        report = profiler.generate_comprehensive_report()
        
        assert report == "No performance data available for reporting."
    
    def test_get_optimization_recommendations(self, metrics_list):
        """Test optimization recommendations."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        recommendations = profiler.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
    
    def test_export_metrics_json(self, metrics_list):
        """Test exporting metrics to JSON."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            profiler.export_metrics(temp_path, "json")
            
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert len(data) == 5  # 5 metrics in metrics_list
                assert all('execution_time' in item for item in data)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_export_metrics_csv(self, metrics_list):
        """Test exporting metrics to CSV."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            profiler.export_metrics(temp_path, "csv")
            
            assert os.path.exists(temp_path)
            # Verify it's a valid CSV with pandas
            df = pd.read_csv(temp_path)
            assert len(df) == 5
            assert 'execution_time' in df.columns
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_metrics_unsupported_format(self, metrics_list):
        """Test error handling for unsupported export format."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Should not raise exception but log error
            profiler.export_metrics(temp_path, "unsupported")
            # File should not be created or should be empty
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.skipif(not HAS_PLOTTING, reason="matplotlib not available")
    def test_create_performance_visualization(self, metrics_list):
        """Test performance visualization creation."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            profiler.create_performance_visualization("execution_time", temp_path)
            
            # Should create a file (though we can't easily verify it's a valid image)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_and_load_baseline_metrics(self, metrics_list):
        """Test saving and loading baseline metrics."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save baseline
            profiler.save_baseline_metrics(temp_path)
            
            assert os.path.exists(temp_path)
            assert profiler.baseline_metrics is not None
            
            # Load baseline in new profiler
            new_profiler = QuantumPerformanceProfiler()
            new_profiler._load_baseline_metrics(temp_path)
            
            assert new_profiler.baseline_metrics is not None
            assert len(new_profiler.baseline_metrics) > 0
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestConvenienceFunctions:
    """Test convenience functions for easy profiling."""
    
    @patch('psutil.Process')
    def test_profile_quantum_circuit_function(self, mock_process, mock_circuit):
        """Test profile_quantum_circuit convenience function."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        metrics = profile_quantum_circuit(mock_circuit, "test_backend")
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.backend == "test_backend"
    
    @patch('psutil.Process')
    def test_profile_quantum_circuit_with_profiler(self, mock_process, mock_circuit):
        """Test profile_quantum_circuit with custom profiler."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        custom_profiler = QuantumPerformanceProfiler()
        metrics = profile_quantum_circuit(mock_circuit, profiler=custom_profiler)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert len(custom_profiler.all_metrics) == 1
    
    @patch('psutil.Process')
    def test_benchmark_circuit_scalability_function(self, mock_process):
        """Test benchmark_circuit_scalability convenience function."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        def circuit_generator(qubit_count):
            circuit = Mock()
            circuit.qubit_count = qubit_count
            circuit.gate_count = qubit_count * 2
            circuit.depth = qubit_count
            circuit.run = Mock(return_value={"success": True})
            return circuit
        
        results = benchmark_circuit_scalability(circuit_generator, (2, 6, 2), "test_backend")
        
        assert isinstance(results, dict)
        assert len(results) == 3  # qubits 2, 4, 6
        assert all(isinstance(m, PerformanceMetrics) for m in results.values())
    
    @patch('psutil.Process')
    def test_compare_quantum_backends_function(self, mock_process, mock_circuit):
        """Test compare_quantum_backends convenience function."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        comparison = compare_quantum_backends(mock_circuit, ["backend_a", "backend_b"], 2)
        
        assert isinstance(comparison, dict)
    
    def test_generate_performance_report_function(self, metrics_list):
        """Test generate_performance_report convenience function."""
        report = generate_performance_report(metrics_list, "summary")
        
        assert isinstance(report, str)
        assert "Performance Summary Report" in report


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestIntegrationScenarios:
    """Integration tests for complete profiling workflows."""
    
    @patch('psutil.Process')
    def test_complete_profiling_workflow(self, mock_process, mock_circuit):
        """Test complete end-to-end profiling workflow."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        # Initialize profiler
        profiler = QuantumPerformanceProfiler()
        
        # Profile circuit execution
        metrics = profiler.profile_circuit_execution(mock_circuit, "test_backend")
        assert isinstance(metrics, PerformanceMetrics)
        
        # Generate recommendations
        recommendations = profiler.get_optimization_recommendations()
        assert isinstance(recommendations, list)
        
        # Generate report
        report = profiler.generate_comprehensive_report("summary")
        assert isinstance(report, str)
        assert "Performance Summary Report" in report
        
        # Export metrics
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            profiler.export_metrics(temp_path, "json")
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('psutil.Process')
    def test_multi_backend_comparison_workflow(self, mock_process, mock_circuit):
        """Test workflow for comparing multiple backends."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        profiler = QuantumPerformanceProfiler()
        
        # Compare multiple backends
        backends = ["backend_a", "backend_b", "backend_c"]
        for backend in backends:
            for _ in range(3):  # Multiple runs per backend
                profiler.profile_circuit_execution(mock_circuit, backend)
        
        # Generate comparison report
        report = profiler.generate_comprehensive_report("comparison")
        assert isinstance(report, str)
        assert "Performance Comparison Report" in report
        
        # Check that all backends appear in the report
        for backend in backends:
            assert backend in report
    
    @patch('psutil.Process')
    def test_scalability_analysis_workflow(self, mock_process):
        """Test workflow for scalability analysis."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        def circuit_generator(qubit_count):
            circuit = Mock()
            circuit.qubit_count = qubit_count
            circuit.gate_count = qubit_count * 3
            circuit.depth = qubit_count * 2
            # Simulate increasing execution time with circuit size
            circuit.run = Mock(return_value={"success": True})
            return circuit
        
        profiler = QuantumPerformanceProfiler()
        
        # Run scalability benchmark
        qubit_counts = [2, 4, 6, 8]
        results = profiler.benchmark_scalability(circuit_generator, qubit_counts)
        
        assert len(results) == len(qubit_counts)
        
        # Analyze scaling behavior
        execution_times = [results[qc].execution_time for qc in qubit_counts]
        # Should generally increase with circuit size (though mock doesn't simulate this)
        
        # Generate optimization report focusing on scalability
        report = profiler.generate_comprehensive_report("optimization")
        assert "Optimization Report" in report


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceTests:
    """Tests to verify the profiler itself performs well."""
    
    @patch('psutil.Process')
    def test_profiler_overhead(self, mock_process, mock_circuit):
        """Test that profiler itself has minimal overhead."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        mock_process.return_value.cpu_percent.return_value = 50.0
        
        # Time profiling operation
        start_time = time.perf_counter()
        
        profiler = QuantumPerformanceProfiler()
        for _ in range(10):
            profiler.profile_circuit_execution(mock_circuit)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Profiling 10 circuits should complete quickly (under 1 second)
        assert total_time < 1.0
        
        # Check memory usage is reasonable
        assert len(profiler.all_metrics) == 10
    
    def test_large_metrics_collection_performance(self):
        """Test performance with large numbers of metrics."""
        profiler = QuantumPerformanceProfiler()
        
        # Add large number of metrics
        start_time = time.perf_counter()
        
        for i in range(1000):
            metrics = PerformanceMetrics(
                execution_time=1.0 + i * 0.001,
                memory_usage=100.0 + i,
                operation_type=f"operation_{i % 10}"
            )
            profiler.all_metrics.append(metrics)
        
        # Generate report with many metrics
        report = profiler.generate_comprehensive_report("summary")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should handle 1000 metrics efficiently
        assert total_time < 5.0
        assert "1000" in report  # Should show correct count
    
    def test_memory_profiler_efficiency(self):
        """Test memory profiler efficiency during monitoring."""
        profiler = MemoryProfiler()
        
        # Start monitoring and let it run briefly
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 100 * 1024 * 1024
            mock_process.return_value.memory_info.return_value = mock_memory_info
            
            profiler.start_monitoring(interval=0.01)
            time.sleep(0.1)  # Let it collect samples
            profiler.stop_monitoring()
        
        # Should have collected samples without excessive overhead
        assert len(profiler.memory_snapshots) > 0
        assert len(profiler.memory_snapshots) < 50  # Shouldn't be excessive


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_profiling_with_null_circuit(self):
        """Test profiling with None circuit."""
        profiler = CircuitProfiler()
        
        # Should handle None gracefully
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 100 * 1024 * 1024
            mock_process.return_value.memory_info.return_value = mock_memory_info
            mock_process.return_value.cpu_percent.return_value = 50.0
            
            metrics = profiler.profile_circuit(None)
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.qubit_count == 0
            assert metrics.gate_count == 0
    
    def test_profiling_circuit_with_execution_error(self):
        """Test profiling circuit that fails during execution."""
        failing_circuit = Mock()
        failing_circuit.qubit_count = 2
        failing_circuit.gate_count = 5
        failing_circuit.depth = 3
        failing_circuit.run = Mock(side_effect=Exception("Execution failed"))
        
        profiler = CircuitProfiler()
        
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 100 * 1024 * 1024
            mock_process.return_value.memory_info.return_value = mock_memory_info
            mock_process.return_value.cpu_percent.return_value = 50.0
            
            metrics = profiler.profile_circuit(failing_circuit)
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.error_rate == 1.0
    
    def test_gate_profiling_with_no_gate_function(self):
        """Test gate profiling with invalid gate function."""
        profiler = GateProfiler()
        
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 100 * 1024 * 1024
            mock_process.return_value.memory_info.return_value = mock_memory_info
            mock_process.return_value.cpu_percent.return_value = 50.0
            
            # Test with None function
            metrics = profiler.profile_gate("invalid_gate", None)
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.error_rate == 1.0
    
    def test_memory_profiler_system_error(self):
        """Test memory profiler handling system errors."""
        profiler = MemoryProfiler()
        
        # Mock psutil to raise exception
        with patch.object(profiler, '_get_memory_usage', side_effect=Exception("System error")):
            profiler.start_monitoring(interval=0.01)
            time.sleep(0.05)
            profiler.stop_monitoring()
        
        # Should handle errors gracefully and not crash
        assert profiler.monitoring is False
    
    def test_optimizer_with_empty_metrics(self):
        """Test optimizer with empty or invalid metrics."""
        optimizer = PerformanceOptimizer()
        
        # Empty list
        recommendations = optimizer.analyze_and_recommend([])
        assert isinstance(recommendations, list)
        
        # List with None
        recommendations = optimizer.analyze_and_recommend([None])
        assert isinstance(recommendations, list)
    
    def test_reporter_with_invalid_metrics(self):
        """Test reporter with invalid metrics data."""
        reporter = PerformanceReporter()
        
        # Empty metrics
        report = reporter.generate_report([], "summary")
        assert "No performance data available" in report
        
        # Metrics with missing attributes
        invalid_metrics = [Mock()]
        report = reporter.generate_report(invalid_metrics, "summary")
        assert isinstance(report, str)  # Should not crash
    
    def test_comparator_with_mismatched_data(self):
        """Test comparator with mismatched or invalid data."""
        comparator = PerformanceComparator()
        
        # Add metrics with missing attributes
        invalid_metrics = Mock()
        del invalid_metrics.execution_time  # Remove expected attribute
        
        comparator.add_measurement("test_backend", invalid_metrics)
        
        # Should handle gracefully
        comparison = comparator.compare_implementations("execution_time")
        assert isinstance(comparison, dict)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestMockDependencies:
    """Test behavior when optional dependencies are not available."""
    
    def test_profiler_without_pandas(self, metrics_list):
        """Test profiler functionality when pandas is not available."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        # Mock pandas as unavailable
        with patch('quantrs2.quantum_performance_profiler.HAS_PANDAS', False):
            # Should not be able to export to CSV
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                temp_path = f.name
            
            try:
                # Should handle gracefully (might log error but not crash)
                profiler.export_metrics(temp_path, "csv")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_profiler_without_matplotlib(self, metrics_list):
        """Test profiler functionality when matplotlib is not available."""
        profiler = QuantumPerformanceProfiler()
        profiler.all_metrics = metrics_list
        
        # Mock matplotlib as unavailable
        with patch('quantrs2.quantum_performance_profiler.HAS_PLOTTING', False):
            # Should handle visualization request gracefully
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            try:
                # Should not crash, just log warning
                profiler.create_performance_visualization("execution_time", temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_monitor_without_asyncio_event_loop(self):
        """Test monitor behavior without running event loop."""
        monitor = PerformanceMonitor()
        
        # Try to start monitoring without event loop
        # Should handle gracefully (might not start but shouldn't crash)
        try:
            monitor.start_real_time_monitoring()
        except RuntimeError:
            # Expected when no event loop is running
            pass
        
        # Stop should work regardless
        monitor.stop_real_time_monitoring()


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestAsyncFunctionality:
    """Test asynchronous functionality in the profiler."""
    
    @pytest.mark.asyncio
    async def test_async_monitoring(self):
        """Test asynchronous monitoring functionality."""
        monitor = PerformanceMonitor()
        
        # Mock system metrics
        with patch('psutil.virtual_memory') as mock_vm, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_vm.return_value.used = 512 * 1024 * 1024  # 512 MB
            mock_cpu.return_value = 75.0
            
            # Start monitoring for a short time
            monitoring_task = asyncio.create_task(monitor.start_monitoring(0.01))
            
            # Let it run briefly
            await asyncio.sleep(0.05)
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Wait for task to complete
            try:
                await asyncio.wait_for(monitoring_task, timeout=1.0)
            except asyncio.TimeoutError:
                monitoring_task.cancel()
            
            # Should have collected some metrics
            assert len(monitor.metrics_buffer) > 0
    
    @pytest.mark.asyncio
    async def test_async_metrics_collection(self):
        """Test asynchronous metrics collection."""
        monitor = PerformanceMonitor()
        
        with patch('psutil.virtual_memory') as mock_vm, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_vm.return_value.used = 256 * 1024 * 1024  # 256 MB
            mock_cpu.return_value = 60.0
            
            metrics = await monitor._collect_current_metrics()
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.memory_usage == 256.0
            assert metrics.cpu_usage == 60.0


if __name__ == "__main__":
    pytest.main([__file__])