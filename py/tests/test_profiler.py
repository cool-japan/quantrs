#!/usr/bin/env python3
"""
Test suite for quantum circuit profiler.
"""

import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path

try:
    import quantrs2
    from quantrs2.profiler import (
        CircuitProfiler, ProfilerSession, MemoryProfiler,
        ProfileMetrics, CircuitAnalysis, ProfileReport,
        profile_circuit, compare_circuits
    )
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestProfileMetrics:
    """Test ProfileMetrics dataclass."""
    
    def test_default_metrics(self):
        """Test default ProfileMetrics initialization."""
        metrics = ProfileMetrics()
        assert metrics.execution_time == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.gate_count == 0
        assert metrics.circuit_depth == 0
        assert isinstance(metrics.additional, dict)
    
    def test_custom_metrics(self):
        """Test ProfileMetrics with custom values."""
        metrics = ProfileMetrics(
            execution_time=1.5,
            memory_usage_mb=128.0,
            gate_count=10,
            circuit_depth=5
        )
        assert metrics.execution_time == 1.5
        assert metrics.memory_usage_mb == 128.0
        assert metrics.gate_count == 10
        assert metrics.circuit_depth == 5


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestCircuitAnalysis:
    """Test CircuitAnalysis dataclass."""
    
    def test_default_analysis(self):
        """Test default CircuitAnalysis initialization."""
        analysis = CircuitAnalysis()
        assert isinstance(analysis.gate_distribution, dict)
        assert isinstance(analysis.qubit_utilization, dict)
        assert isinstance(analysis.circuit_layers, list)
        assert analysis.critical_path_length == 0
        assert analysis.parallelism_factor == 0.0
    
    def test_custom_analysis(self):
        """Test CircuitAnalysis with custom values."""
        analysis = CircuitAnalysis(
            gate_distribution={'h': 2, 'cx': 1},
            qubit_utilization={0: 3, 1: 2},
            critical_path_length=3,
            parallelism_factor=0.75
        )
        assert analysis.gate_distribution == {'h': 2, 'cx': 1}
        assert analysis.qubit_utilization == {0: 3, 1: 2}
        assert analysis.critical_path_length == 3
        assert analysis.parallelism_factor == 0.75


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestProfileReport:
    """Test ProfileReport functionality."""
    
    def test_report_creation(self):
        """Test ProfileReport creation."""
        metrics = ProfileMetrics(execution_time=1.0, gate_count=5)
        analysis = CircuitAnalysis(gate_distribution={'h': 1, 'cx': 1})
        circuit_info = {'n_qubits': 2, 'circuit_type': 'Circuit'}
        
        report = ProfileReport(
            circuit_info=circuit_info,
            metrics=metrics,
            analysis=analysis,
            recommendations=['Test recommendation']
        )
        
        assert report.circuit_info == circuit_info
        assert report.metrics.execution_time == 1.0
        assert report.analysis.gate_distribution == {'h': 1, 'cx': 1}
        assert len(report.recommendations) == 1
    
    def test_report_to_dict(self):
        """Test ProfileReport to_dict conversion."""
        metrics = ProfileMetrics(execution_time=1.0)
        analysis = CircuitAnalysis()
        report = ProfileReport(
            circuit_info={'n_qubits': 2},
            metrics=metrics,
            analysis=analysis
        )
        
        data = report.to_dict()
        assert isinstance(data, dict)
        assert 'circuit_info' in data
        assert 'metrics' in data
        assert 'analysis' in data
        assert 'recommendations' in data
        assert 'timestamp' in data
    
    def test_report_to_json(self):
        """Test ProfileReport JSON export."""
        metrics = ProfileMetrics(execution_time=1.0)
        analysis = CircuitAnalysis()
        report = ProfileReport(
            circuit_info={'n_qubits': 2},
            metrics=metrics,
            analysis=analysis
        )
        
        json_str = report.to_json()
        assert isinstance(json_str, str)
        assert '"circuit_info"' in json_str
        assert '"metrics"' in json_str
    
    def test_report_to_json_file(self):
        """Test ProfileReport JSON file export."""
        metrics = ProfileMetrics(execution_time=1.0)
        analysis = CircuitAnalysis()
        report = ProfileReport(
            circuit_info={'n_qubits': 2},
            metrics=metrics,
            analysis=analysis
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            json_str = report.to_json(temp_path)
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert '"circuit_info"' in content
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestMemoryProfiler:
    """Test MemoryProfiler functionality."""
    
    def test_memory_profiler_creation(self):
        """Test MemoryProfiler initialization."""
        profiler = MemoryProfiler()
        assert profiler.memory_samples == []
        assert profiler.monitoring is False
        assert profiler.monitor_thread is None
    
    def test_memory_monitoring(self):
        """Test memory monitoring start/stop."""
        profiler = MemoryProfiler()
        
        # Start monitoring
        profiler.start_monitoring()
        assert profiler.monitoring is True
        
        # Let it run briefly
        import time
        time.sleep(0.1)
        
        # Stop monitoring
        stats = profiler.stop_monitoring()
        assert profiler.monitoring is False
        assert isinstance(stats, dict)
        assert 'current' in stats
        assert 'peak' in stats
        assert 'average' in stats
    
    def test_memory_stats_empty(self):
        """Test memory statistics with no samples."""
        profiler = MemoryProfiler()
        stats = profiler.stop_monitoring()
        
        assert stats['current'] == 0.0
        assert stats['peak'] == 0.0
        assert stats['average'] == 0.0


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestCircuitProfiler:
    """Test CircuitProfiler functionality."""
    
    def test_profiler_creation(self):
        """Test CircuitProfiler initialization."""
        profiler = CircuitProfiler()
        assert profiler.enable_memory_profiling is True
        assert profiler.enable_timing_breakdown is True
        assert profiler.enable_resource_tracking is True
        assert profiler.memory_profiler is not None
    
    def test_profiler_custom_options(self):
        """Test CircuitProfiler with custom options."""
        profiler = CircuitProfiler(
            enable_memory_profiling=False,
            enable_timing_breakdown=False,
            enable_resource_tracking=False
        )
        assert profiler.enable_memory_profiling is False
        assert profiler.enable_timing_breakdown is False
        assert profiler.enable_resource_tracking is False
        assert profiler.memory_profiler is None
    
    def test_profile_basic_circuit(self):
        """Test profiling a basic circuit."""
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        profiler = CircuitProfiler()
        report = profiler.profile_circuit(circuit, run_simulation=True)
        
        assert isinstance(report, ProfileReport)
        assert report.circuit_info['n_qubits'] == 2
        assert report.metrics.execution_time >= 0
        assert report.metrics.gate_count >= 0
        assert len(report.recommendations) > 0
    
    def test_profile_without_simulation(self):
        """Test profiling without running simulation."""
        circuit = quantrs2.Circuit(1)
        circuit.h(0)
        
        profiler = CircuitProfiler()
        report = profiler.profile_circuit(circuit, run_simulation=False)
        
        assert isinstance(report, ProfileReport)
        assert report.circuit_info['simulation_successful'] is False
    
    def test_profile_batch(self):
        """Test batch profiling of multiple circuits."""
        circuits = []
        for i in range(3):
            circuit = quantrs2.Circuit(2)
            circuit.h(0)
            if i > 0:
                circuit.cx(0, 1)
            circuits.append(circuit)
        
        profiler = CircuitProfiler()
        reports = profiler.profile_batch(circuits)
        
        assert len(reports) == 3
        assert all(isinstance(r, ProfileReport) for r in reports)
    
    def test_compare_circuits(self):
        """Test circuit comparison functionality."""
        # Create different circuits
        circuit1 = quantrs2.Circuit(2)
        circuit1.h(0)
        
        circuit2 = quantrs2.Circuit(2)
        circuit2.h(0)
        circuit2.cx(0, 1)
        
        profiler = CircuitProfiler()
        report1 = profiler.profile_circuit(circuit1)
        report2 = profiler.profile_circuit(circuit2)
        
        comparison_df = profiler.compare_circuits([report1, report2])
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'circuit_id' in comparison_df.columns
        assert 'n_qubits' in comparison_df.columns
        assert 'execution_time_ms' in comparison_df.columns


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestProfilerSession:
    """Test ProfilerSession functionality."""
    
    def test_session_creation(self):
        """Test ProfilerSession initialization."""
        session = ProfilerSession("test_session")
        assert session.session_name == "test_session"
        assert session.reports == []
        assert isinstance(session.profiler, CircuitProfiler)
    
    def test_session_profiling(self):
        """Test profiling within a session."""
        session = ProfilerSession("test_session")
        
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        report = session.profile(circuit, name="bell_circuit")
        
        assert len(session.reports) == 1
        assert report.circuit_info['circuit_name'] == "bell_circuit"
        assert report.circuit_info['session_name'] == "test_session"
        assert report.circuit_info['profile_order'] == 0
    
    def test_session_summary(self):
        """Test session summary generation."""
        session = ProfilerSession("test_session")
        
        # Add multiple circuits
        for i in range(2):
            circuit = quantrs2.Circuit(2)
            circuit.h(0)
            if i > 0:
                circuit.cx(0, 1)
            session.profile(circuit, name=f"circuit_{i}")
        
        summary = session.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
    
    def test_session_save(self):
        """Test session saving functionality."""
        session = ProfilerSession("test_session")
        
        circuit = quantrs2.Circuit(1)
        circuit.h(0)
        session.profile(circuit, name="test_circuit")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = session.save_session(temp_dir)
            
            assert output_path.exists()
            session_file = output_path / "test_session_session.json"
            assert session_file.exists()


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_profile_circuit_function(self):
        """Test profile_circuit convenience function."""
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        report = profile_circuit(circuit)
        
        assert isinstance(report, ProfileReport)
        assert report.circuit_info['n_qubits'] == 2
    
    def test_compare_circuits_function(self):
        """Test compare_circuits convenience function."""
        circuit1 = quantrs2.Circuit(1)
        circuit1.h(0)
        
        circuit2 = quantrs2.Circuit(2)
        circuit2.h(0)
        circuit2.cx(0, 1)
        
        comparison_df = compare_circuits(
            [circuit1, circuit2], 
            names=["simple", "bell"]
        )
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestProfilerIntegration:
    """Test profiler integration with main module."""
    
    def test_profiler_functions_available(self):
        """Test that profiler functions are available from main module."""
        try:
            from quantrs2 import profile_circuit, compare_circuits, CircuitProfiler
            assert callable(profile_circuit)
            assert callable(compare_circuits)
            assert CircuitProfiler is not None
        except ImportError:
            # This is acceptable if profiler not available
            pass
    
    def test_profiler_with_various_circuits(self):
        """Test profiler with different circuit types."""
        circuits = []
        
        # Simple single-qubit circuit
        circuit1 = quantrs2.Circuit(1)
        circuit1.h(0)
        circuits.append(circuit1)
        
        # Bell state circuit
        circuit2 = quantrs2.Circuit(2)
        circuit2.h(0)
        circuit2.cx(0, 1)
        circuits.append(circuit2)
        
        # More complex circuit
        circuit3 = quantrs2.Circuit(3)
        circuit3.h(0)
        circuit3.h(1)
        circuit3.h(2)
        circuit3.cx(0, 1)
        circuit3.cx(1, 2)
        circuits.append(circuit3)
        
        # Profile all circuits
        profiler = CircuitProfiler()
        reports = []
        
        for i, circuit in enumerate(circuits):
            report = profiler.profile_circuit(circuit)
            assert isinstance(report, ProfileReport)
            reports.append(report)
        
        # Compare circuits
        comparison = profiler.compare_circuits(reports)
        assert len(comparison) == 3


if __name__ == "__main__":
    pytest.main([__file__])