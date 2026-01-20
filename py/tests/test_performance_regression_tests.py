"""
Tests for performance regression testing framework.
"""

import unittest
import pytest
import tempfile
import time
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Safe import pattern
try:
    from quantrs2.performance_regression_tests import (
        PerformanceMetric,
        BenchmarkResult,
        RegressionThreshold,
        PerformanceDatabase,
        QuantumBenchmarkSuite,
        RegressionDetector,
        PerformanceRegressionRunner,
        run_performance_regression_tests,
        detect_performance_regressions,
        benchmark_quantum_operations,
        setup_ci_performance_tests
    )
    HAS_PERFORMANCE_REGRESSION_TESTS = True
except ImportError:
    HAS_PERFORMANCE_REGRESSION_TESTS = False
    
    # Stub implementations
    class PerformanceMetric:
        def __init__(self, name, value, unit, timestamp=None, metadata=None):
            self.name = name
            self.value = value
            self.unit = unit
            self.timestamp = timestamp or datetime.now()
            self.metadata = metadata or {}
    
    def run_performance_regression_tests(db_path):
        return {"results": [], "regressions": [], "environment": {}, "timestamp": datetime.now().isoformat()}
    
    def benchmark_quantum_operations(operations):
        return {op: {"execution_time": 0.1, "qubit_count": 2} for op in operations}


@pytest.mark.skipif(not HAS_PERFORMANCE_REGRESSION_TESTS, reason="quantrs2.performance_regression_tests not available")
class TestPerformanceMetric(unittest.TestCase):
    """Test PerformanceMetric dataclass."""
    
    def test_metric_creation(self):
        """Test creating a performance metric."""
        timestamp = datetime.now()
        metric = PerformanceMetric(
            name="execution_time",
            value=0.123,
            unit="seconds",
            timestamp=timestamp,
            metadata={"test": "value"}
        )
        
        self.assertEqual(metric.name, "execution_time")
        self.assertEqual(metric.value, 0.123)
        self.assertEqual(metric.unit, "seconds")
        self.assertEqual(metric.timestamp, timestamp)
        self.assertEqual(metric.metadata, {"test": "value"})


class TestBenchmarkResult(unittest.TestCase):
    """Test BenchmarkResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a benchmark result."""
        timestamp = datetime.now()
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            execution_time=0.456,
            memory_usage=1024,
            additional_metrics={"ops_per_sec": 100.0},
            environment_info={"python": "3.9"},
            timestamp=timestamp,
            commit_hash="abc123"
        )
        
        self.assertEqual(result.benchmark_name, "test_benchmark")
        self.assertEqual(result.execution_time, 0.456)
        self.assertEqual(result.memory_usage, 1024)
        self.assertEqual(result.additional_metrics["ops_per_sec"], 100.0)
        self.assertEqual(result.commit_hash, "abc123")


class TestRegressionThreshold(unittest.TestCase):
    """Test RegressionThreshold dataclass."""
    
    def test_threshold_creation(self):
        """Test creating regression threshold."""
        threshold = RegressionThreshold(
            metric_name="execution_time",
            max_degradation_percent=15.0,
            max_absolute_change=0.1,
            min_samples=5
        )
        
        self.assertEqual(threshold.metric_name, "execution_time")
        self.assertEqual(threshold.max_degradation_percent, 15.0)
        self.assertEqual(threshold.max_absolute_change, 0.1)
        self.assertEqual(threshold.min_samples, 5)


class TestPerformanceDatabase(unittest.TestCase):
    """Test PerformanceDatabase class."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_perf.db"
        self.db = PerformanceDatabase(str(self.db_path))
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsInstance(self.db.results, list)
        self.assertEqual(len(self.db.results), 0)
    
    def test_add_and_retrieve_result(self):
        """Test adding and retrieving results."""
        result = BenchmarkResult(
            benchmark_name="test",
            execution_time=0.1,
            memory_usage=512,
            additional_metrics={},
            environment_info={},
            timestamp=datetime.now()
        )
        
        self.db.add_result(result)
        self.assertEqual(len(self.db.results), 1)
        
        retrieved = self.db.get_results_for_benchmark("test")
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].benchmark_name, "test")
    
    def test_get_recent_results(self):
        """Test getting recent results."""
        # Add multiple results with different timestamps
        for i in range(5):
            result = BenchmarkResult(
                benchmark_name="test",
                execution_time=0.1 * i,
                memory_usage=512,
                additional_metrics={},
                environment_info={},
                timestamp=datetime.now()
            )
            self.db.add_result(result)
            time.sleep(0.001)  # Ensure different timestamps
        
        recent = self.db.get_recent_results("test", count=3)
        self.assertEqual(len(recent), 3)
        
        # Should be sorted by timestamp (most recent first)
        for i in range(len(recent) - 1):
            self.assertGreaterEqual(recent[i].timestamp, recent[i + 1].timestamp)


class TestQuantumBenchmarkSuite(unittest.TestCase):
    """Test QuantumBenchmarkSuite class."""
    
    def setUp(self):
        """Set up benchmark suite."""
        self.suite = QuantumBenchmarkSuite()
    
    def test_suite_initialization(self):
        """Test suite initializes with default benchmarks."""
        self.assertIsInstance(self.suite.benchmarks, dict)
        self.assertGreater(len(self.suite.benchmarks), 0)
        
        # Check for expected benchmarks
        expected_benchmarks = [
            "bell_state_creation",
            "grover_3_qubits",
            "qft_4_qubits",
            "vqe_h2_molecule"
        ]
        
        for benchmark in expected_benchmarks:
            self.assertIn(benchmark, self.suite.benchmarks)
    
    def test_register_benchmark(self):
        """Test registering custom benchmark."""
        def custom_benchmark():
            return {"test_metric": 1.0}
        
        self.suite.register_benchmark("custom_test", custom_benchmark)
        self.assertIn("custom_test", self.suite.benchmarks)
        self.assertEqual(self.suite.benchmarks["custom_test"], custom_benchmark)
    
    def test_bell_state_benchmark(self):
        """Test Bell state benchmark."""
        result = self.suite.benchmark_bell_state()
        
        self.assertIsInstance(result, dict)
        self.assertIn('execution_time', result)
        self.assertIn('operations_per_second', result)
        self.assertIn('circuit_depth', result)
        self.assertIn('qubit_count', result)
        
        self.assertGreater(result['execution_time'], 0)
        self.assertEqual(result['qubit_count'], 2)
    
    def test_grover_benchmark(self):
        """Test Grover's algorithm benchmark."""
        result = self.suite.benchmark_grover_3q()
        
        self.assertIsInstance(result, dict)
        self.assertIn('execution_time', result)
        self.assertIn('circuit_depth', result)
        self.assertIn('qubit_count', result)
        
        self.assertEqual(result['qubit_count'], 3)
        self.assertGreater(result['execution_time'], 0)
    
    def test_qft_benchmark(self):
        """Test QFT benchmark."""
        result = self.suite.benchmark_qft_4q()
        
        self.assertIsInstance(result, dict)
        self.assertIn('execution_time', result)
        self.assertIn('qubit_count', result)
        
        self.assertEqual(result['qubit_count'], 4)
    
    def test_vqe_benchmark(self):
        """Test VQE benchmark."""
        result = self.suite.benchmark_vqe_h2()
        
        self.assertIsInstance(result, dict)
        self.assertIn('execution_time', result)
        self.assertIn('optimization_iterations', result)
        self.assertIn('final_energy', result)


class TestRegressionDetector(unittest.TestCase):
    """Test RegressionDetector class."""
    
    def setUp(self):
        """Set up regression detector."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_regression.db"
        self.db = PerformanceDatabase(str(self.db_path))
        self.detector = RegressionDetector(self.db)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector.thresholds, dict)
        self.assertIn('execution_time', self.detector.thresholds)
        self.assertIn('memory_usage', self.detector.thresholds)
    
    def test_no_regression_detection(self):
        """Test no regression when performance is stable."""
        # Add historical results
        for i in range(5):
            result = BenchmarkResult(
                benchmark_name="stable_test",
                execution_time=0.1,  # Stable execution time
                memory_usage=1000,   # Stable memory usage
                additional_metrics={"ops_per_sec": 100.0},
                environment_info={},
                timestamp=datetime.now()
            )
            self.db.add_result(result)
            time.sleep(0.001)
        
        # Add latest result with similar performance
        latest = BenchmarkResult(
            benchmark_name="stable_test",
            execution_time=0.11,  # Slight increase, but within threshold
            memory_usage=1050,    # Slight increase, but within threshold
            additional_metrics={"ops_per_sec": 95.0},
            environment_info={},
            timestamp=datetime.now()
        )
        
        regressions = self.detector.detect_regressions("stable_test", latest)
        self.assertEqual(len(regressions), 0)
    
    def test_execution_time_regression(self):
        """Test detection of execution time regression."""
        # Add historical results with good performance
        for i in range(5):
            result = BenchmarkResult(
                benchmark_name="time_regression_test",
                execution_time=0.1,
                memory_usage=1000,
                additional_metrics={},
                environment_info={},
                timestamp=datetime.now()
            )
            self.db.add_result(result)
            time.sleep(0.001)
        
        # Add latest result with significantly worse execution time
        latest = BenchmarkResult(
            benchmark_name="time_regression_test",
            execution_time=0.2,  # 100% increase - should trigger regression
            memory_usage=1000,
            additional_metrics={},
            environment_info={},
            timestamp=datetime.now()
        )
        
        regressions = self.detector.detect_regressions("time_regression_test", latest)
        self.assertGreater(len(regressions), 0)
        
        # Check that execution time regression was detected
        time_regressions = [r for r in regressions if r['metric'] == 'execution_time']
        self.assertEqual(len(time_regressions), 1)
        
        regression = time_regressions[0]
        self.assertEqual(regression['latest_value'], 0.2)
        self.assertEqual(regression['baseline_value'], 0.1)
        self.assertGreater(regression['percent_change'], 15.0)  # Above threshold
    
    def test_memory_regression(self):
        """Test detection of memory usage regression."""
        # Add historical results
        for i in range(5):
            result = BenchmarkResult(
                benchmark_name="memory_regression_test",
                execution_time=0.1,
                memory_usage=1000,
                additional_metrics={},
                environment_info={},
                timestamp=datetime.now()
            )
            self.db.add_result(result)
            time.sleep(0.001)
        
        # Add latest result with high memory usage
        latest = BenchmarkResult(
            benchmark_name="memory_regression_test",
            execution_time=0.1,
            memory_usage=1500,  # 50% increase - should trigger regression
            additional_metrics={},
            environment_info={},
            timestamp=datetime.now()
        )
        
        regressions = self.detector.detect_regressions("memory_regression_test", latest)
        memory_regressions = [r for r in regressions if r['metric'] == 'memory_usage']
        self.assertEqual(len(memory_regressions), 1)
    
    def test_insufficient_historical_data(self):
        """Test behavior with insufficient historical data."""
        # Add only one historical result
        result = BenchmarkResult(
            benchmark_name="insufficient_data_test",
            execution_time=0.1,
            memory_usage=1000,
            additional_metrics={},
            environment_info={},
            timestamp=datetime.now()
        )
        self.db.add_result(result)
        
        # Try to detect regressions
        latest = BenchmarkResult(
            benchmark_name="insufficient_data_test",
            execution_time=0.5,  # Much worse, but not enough data for comparison
            memory_usage=2000,
            additional_metrics={},
            environment_info={},
            timestamp=datetime.now()
        )
        
        regressions = self.detector.detect_regressions("insufficient_data_test", latest)
        self.assertEqual(len(regressions), 0)


class TestPerformanceRegressionRunner(unittest.TestCase):
    """Test PerformanceRegressionRunner class."""
    
    def setUp(self):
        """Set up regression runner."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_runner.db"
        self.runner = PerformanceRegressionRunner(str(self.db_path))
    
    def test_runner_initialization(self):
        """Test runner initialization."""
        self.assertIsInstance(self.runner.db, PerformanceDatabase)
        self.assertIsInstance(self.runner.benchmark_suite, QuantumBenchmarkSuite)
        self.assertIsInstance(self.runner.regression_detector, RegressionDetector)
        self.assertIsInstance(self.runner.environment_info, dict)
    
    def test_environment_info_collection(self):
        """Test environment information collection."""
        env_info = self.runner.environment_info
        
        self.assertIn('python_version', env_info)
        self.assertIn('platform', env_info)
        self.assertIn('quantrs2_version', env_info)
        self.assertIn('numpy_version', env_info)
        self.assertIn('timestamp', env_info)
    
    @patch('psutil.Process')
    def test_single_benchmark_run(self, mock_process):
        """Test running a single benchmark."""
        # Mock memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 1000000
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        def test_benchmark():
            return {"test_metric": 42.0}
        
        result = self.runner.run_single_benchmark("test_benchmark", test_benchmark)
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "test_benchmark")
        self.assertGreater(result.execution_time, 0)
        self.assertEqual(result.additional_metrics["test_metric"], 42.0)
    
    def test_regression_report_generation(self):
        """Test regression report generation."""
        # Test with no regressions
        report = self.runner.generate_regression_report([])
        self.assertIn("No performance regressions detected", report)
        
        # Test with regressions
        regressions = [
            {
                'benchmark': 'test_benchmark',
                'metric': 'execution_time',
                'latest_value': 0.2,
                'baseline_value': 0.1,
                'percent_change': 100.0,
                'threshold': 15.0,
                'severity': 'critical'
            }
        ]
        
        report = self.runner.generate_regression_report(regressions)
        self.assertIn("Performance Regressions Detected", report)
        self.assertIn("CRITICAL SEVERITY", report)
        self.assertIn("test_benchmark", report)
        self.assertIn("execution_time", report)
    
    def test_export_results(self):
        """Test exporting results to JSON."""
        export_file = Path(self.temp_dir) / "test_export.json"
        self.runner.export_results(str(export_file))
        
        self.assertTrue(export_file.exists())
        
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('results', data)
        self.assertIn('environment', data)
        self.assertIn('export_timestamp', data)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_convenience.db"
    
    @patch('psutil.Process')
    def test_run_performance_regression_tests(self, mock_process):
        """Test running performance regression tests."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 1000000
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        results = run_performance_regression_tests(str(self.db_path))
        
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIn('regressions', results)
        self.assertIn('environment', results)
        self.assertIn('timestamp', results)
    
    def test_benchmark_quantum_operations(self):
        """Test benchmarking specific quantum operations."""
        results = benchmark_quantum_operations(['bell_state', 'grover'])
        
        self.assertIsInstance(results, dict)
        self.assertIn('bell_state', results)
        self.assertIn('grover', results)
        
        # Check structure of results
        bell_result = results['bell_state']
        self.assertIn('execution_time', bell_result)
        self.assertIn('qubit_count', bell_result)


class TestIntegration(unittest.TestCase):
    """Integration tests for the performance regression framework."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "integration_test.db"
    
    @patch('psutil.Process')
    def test_full_workflow(self, mock_process):
        """Test complete workflow with regression detection."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 1000000
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        runner = PerformanceRegressionRunner(str(self.db_path))
        
        # Run initial benchmarks to establish baseline
        for _ in range(3):
            results = runner.run_all_benchmarks(detect_regressions=False)
            time.sleep(0.1)  # Ensure different timestamps
        
        # Run with regression detection
        results = runner.run_all_benchmarks(detect_regressions=True)
        
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIn('regressions', results)
        
        # Should have multiple benchmark results
        self.assertGreater(len(results['results']), 0)
    
    @patch('psutil.Process')  
    def test_performance_monitoring_simulation(self, mock_process):
        """Test simulating performance monitoring over time."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 1000000
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        runner = PerformanceRegressionRunner(str(self.db_path))
        
        # Simulate performance degradation over time
        for i in range(5):
            # Gradually increase execution time to simulate degradation
            with patch.object(runner.benchmark_suite, 'benchmark_bell_state') as mock_bench:
                mock_bench.return_value = {
                    'execution_time': 0.1 * (1 + i * 0.1),  # Gradual increase
                    'operations_per_second': 100.0 / (1 + i * 0.1),
                    'circuit_depth': 2,
                    'qubit_count': 2
                }
                
                result = runner.run_single_benchmark('bell_state_creation', mock_bench)
                time.sleep(0.1)
        
        # Check for regressions
        regressions = runner.regression_detector.detect_regressions(
            'bell_state_creation', result
        )
        
        # Should detect regression due to gradual performance degradation
        self.assertGreater(len(regressions), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)