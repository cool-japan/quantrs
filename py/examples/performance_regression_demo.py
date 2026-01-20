"""
Performance Regression Testing Framework Demonstration

This example shows how to use the QuantRS2 performance regression testing framework to:
1. Set up comprehensive performance benchmarks
2. Detect performance regressions over time
3. Integrate with CI/CD pipelines
4. Monitor quantum algorithm performance
"""

import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import performance regression testing framework
from quantrs2.performance_regression_tests import (
    PerformanceRegressionRunner,
    QuantumBenchmarkSuite,
    RegressionDetector,
    PerformanceDatabase,
    BenchmarkResult,
    RegressionThreshold,
    run_performance_regression_tests,
    benchmark_quantum_operations,
    setup_ci_performance_tests
)


def demo_basic_benchmarking():
    """Demonstrate basic performance benchmarking."""
    print("=" * 60)
    print("BASIC PERFORMANCE BENCHMARKING DEMO")
    print("=" * 60)
    
    # Create benchmark suite
    print("1. Creating quantum benchmark suite...")
    suite = QuantumBenchmarkSuite()
    
    print(f"   Available benchmarks: {list(suite.benchmarks.keys())}")
    
    # Run individual benchmarks
    print("\n2. Running individual benchmarks...")
    
    benchmark_names = ['bell_state_creation', 'grover_3_qubits', 'qft_4_qubits']
    results = {}
    
    for name in benchmark_names:
        print(f"   Running {name}...")
        start_time = time.time()
        result = suite.benchmarks[name]()
        end_time = time.time()
        
        results[name] = result
        print(f"     Execution time: {end_time - start_time:.6f}s")
        print(f"     Metrics: {list(result.keys())}")
    
    # Display detailed results
    print("\n3. Detailed benchmark results:")
    for name, result in results.items():
        print(f"\n   {name}:")
        for metric, value in result.items():
            if isinstance(value, float):
                print(f"     {metric}: {value:.6f}")
            else:
                print(f"     {metric}: {value}")
    
    return results


def demo_regression_detection():
    """Demonstrate regression detection capabilities."""
    print("\n" + "=" * 60)
    print("REGRESSION DETECTION DEMO")
    print("=" * 60)
    
    # Set up temporary database
    db_path = "demo_performance.db"
    runner = PerformanceRegressionRunner(db_path)
    
    print("1. Setting up performance baseline...")
    
    # Create baseline performance data (simulate historical good performance)
    for i in range(5):
        # Simulate consistent good performance
        mock_result = BenchmarkResult(
            benchmark_name="demo_benchmark",
            execution_time=0.1 + np.random.normal(0, 0.01),  # Stable ~0.1s
            memory_usage=1000 + int(np.random.normal(0, 50)), # Stable ~1000 bytes
            additional_metrics={
                "operations_per_second": 100 + np.random.normal(0, 5),
                "circuit_depth": 5,
                "gate_count": 10
            },
            environment_info=runner.environment_info,
            timestamp=datetime.now() - timedelta(days=i),
            commit_hash=f"baseline_{i}"
        )
        runner.db.add_result(mock_result)
    
    print(f"   Added {len(runner.db.get_results_for_benchmark('demo_benchmark'))} baseline results")
    
    # Test case 1: No regression (good performance)
    print("\n2. Testing with good performance (no regression expected)...")
    good_result = BenchmarkResult(
        benchmark_name="demo_benchmark",
        execution_time=0.105,  # Slight increase, but within threshold
        memory_usage=1020,     # Slight increase, but within threshold
        additional_metrics={
            "operations_per_second": 98,
            "circuit_depth": 5,
            "gate_count": 10
        },
        environment_info=runner.environment_info,
        timestamp=datetime.now(),
        commit_hash="good_commit"
    )
    
    regressions = runner.regression_detector.detect_regressions("demo_benchmark", good_result)
    print(f"   Regressions detected: {len(regressions)}")
    
    if regressions:
        for reg in regressions:
            print(f"     - {reg['metric']}: {reg['percent_change']:.1f}% change")
    else:
        print("   ✅ No regressions detected - performance is stable!")
    
    # Test case 2: Performance regression
    print("\n3. Testing with poor performance (regression expected)...")
    bad_result = BenchmarkResult(
        benchmark_name="demo_benchmark",
        execution_time=0.25,   # 150% increase - should trigger regression
        memory_usage=1800,     # 80% increase - should trigger regression
        additional_metrics={
            "operations_per_second": 40,  # 60% decrease - should trigger regression
            "circuit_depth": 5,
            "gate_count": 10
        },
        environment_info=runner.environment_info,
        timestamp=datetime.now(),
        commit_hash="bad_commit"
    )
    
    regressions = runner.regression_detector.detect_regressions("demo_benchmark", bad_result)
    print(f"   Regressions detected: {len(regressions)}")
    
    if regressions:
        print("   🚨 Performance regressions found:")
        for reg in regressions:
            print(f"     - {reg['metric']}: {reg['percent_change']:.1f}% change "
                  f"(severity: {reg['severity']})")
            print(f"       Latest: {reg['latest_value']:.6f}, "
                  f"Baseline: {reg['baseline_value']:.6f}")
    
    # Generate regression report
    print("\n4. Generating regression report...")
    report = runner.generate_regression_report(regressions)
    print(report)
    
    return regressions


def demo_custom_benchmarks():
    """Demonstrate creating custom benchmarks."""
    print("\n" + "=" * 60)
    print("CUSTOM BENCHMARKS DEMO")
    print("=" * 60)
    
    # Create custom benchmark suite
    suite = QuantumBenchmarkSuite()
    
    print("1. Creating custom benchmarks...")
    
    # Custom benchmark 1: Quantum Machine Learning
    def benchmark_qml_training():
        """Benchmark quantum machine learning training."""
        start_time = time.time()
        
        # Simulate QML training process
        training_epochs = 10
        data_points = 100
        
        for epoch in range(training_epochs):
            # Simulate training epoch
            for _ in range(data_points // 10):
                # Simulate quantum circuit evaluation
                time.sleep(0.001)  # Mock computation time
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'epochs': training_epochs,
            'data_points': data_points,
            'convergence_rate': np.random.uniform(0.8, 0.95),
            'final_accuracy': np.random.uniform(0.85, 0.98)
        }
    
    # Custom benchmark 2: Quantum Error Correction
    def benchmark_error_correction():
        """Benchmark quantum error correction overhead."""
        start_time = time.time()
        
        # Simulate error correction encoding/decoding
        logical_qubits = 5
        physical_qubits = logical_qubits * 9  # Surface code overhead
        
        # Simulate syndrome extraction
        for _ in range(100):
            for _ in range(physical_qubits):
                time.sleep(0.0001)  # Mock stabilizer measurement
        
        correction_time = time.time() - start_time
        
        return {
            'correction_time': correction_time,
            'logical_qubits': logical_qubits,
            'physical_qubits': physical_qubits,
            'overhead_ratio': physical_qubits / logical_qubits,
            'error_rate': np.random.uniform(0.001, 0.01)
        }
    
    # Register custom benchmarks
    suite.register_benchmark("qml_training", benchmark_qml_training)
    suite.register_benchmark("error_correction", benchmark_error_correction)
    
    print(f"   Registered benchmarks: qml_training, error_correction")
    
    # Run custom benchmarks
    print("\n2. Running custom benchmarks...")
    
    custom_results = {}
    for name in ["qml_training", "error_correction"]:
        print(f"   Running {name}...")
        result = suite.benchmarks[name]()
        custom_results[name] = result
        
        print(f"     Results: {list(result.keys())}")
        for metric, value in result.items():
            if isinstance(value, float):
                print(f"       {metric}: {value:.6f}")
            else:
                print(f"       {metric}: {value}")
    
    return custom_results


def demo_performance_monitoring():
    """Demonstrate continuous performance monitoring."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MONITORING DEMO")
    print("=" * 60)
    
    db_path = "monitoring_demo.db"
    runner = PerformanceRegressionRunner(db_path)
    
    print("1. Simulating performance monitoring over time...")
    
    # Simulate performance data over several "releases"
    releases = [
        ("v1.0.0", {"performance_factor": 1.0, "memory_factor": 1.0}),
        ("v1.1.0", {"performance_factor": 0.95, "memory_factor": 1.05}),  # Slight improvement
        ("v1.2.0", {"performance_factor": 0.90, "memory_factor": 1.10}),  # More improvement
        ("v1.3.0", {"performance_factor": 1.20, "memory_factor": 1.30}),  # Regression!
        ("v1.3.1", {"performance_factor": 0.92, "memory_factor": 1.08}),  # Fixed
    ]
    
    performance_history = []
    
    for i, (version, factors) in enumerate(releases):
        print(f"\n   Simulating release {version}...")
        
        # Create mock benchmark result
        base_time = 0.1
        base_memory = 1000
        
        result = BenchmarkResult(
            benchmark_name="monitoring_test",
            execution_time=base_time * factors["performance_factor"],
            memory_usage=int(base_memory * factors["memory_factor"]),
            additional_metrics={
                "version": version,
                "operations_per_second": 100 / factors["performance_factor"],
                "efficiency_score": 1 / (factors["performance_factor"] * factors["memory_factor"])
            },
            environment_info={**runner.environment_info, "version": version},
            timestamp=datetime.now() - timedelta(days=len(releases) - i),
            commit_hash=f"commit_{version}"
        )
        
        runner.db.add_result(result)
        performance_history.append((version, result))
        
        # Check for regressions (if we have enough historical data)
        if i >= 2:  # Need at least 3 results for regression detection
            regressions = runner.regression_detector.detect_regressions(
                "monitoring_test", result
            )
            
            if regressions:
                print(f"     🚨 Regressions detected in {version}:")
                for reg in regressions:
                    print(f"       - {reg['metric']}: {reg['percent_change']:.1f}% "
                          f"({reg['severity']} severity)")
            else:
                print(f"     ✅ No regressions in {version}")
    
    # Analyze performance trends
    print("\n2. Performance trend analysis:")
    print("   Version | Exec Time | Memory | Regressions")
    print("   --------|-----------|--------|------------")
    
    for version, result in performance_history:
        regressions = runner.regression_detector.detect_regressions(
            "monitoring_test", result
        )
        regression_count = len(regressions)
        status = "🚨" if regression_count > 0 else "✅"
        
        print(f"   {version:7} | {result.execution_time:8.3f}s | "
              f"{result.memory_usage:5d}B | {status} ({regression_count})")
    
    return performance_history


def demo_ci_integration():
    """Demonstrate CI/CD integration."""
    print("\n" + "=" * 60)
    print("CI/CD INTEGRATION DEMO")
    print("=" * 60)
    
    print("1. Simulating CI pipeline performance check...")
    
    # Simulate a CI environment
    try:
        # This would normally run in CI
        print("   Running performance regression tests...")
        
        # Use convenience function for CI
        results = run_performance_regression_tests("ci_demo.db")
        
        print(f"   ✅ Performance tests completed successfully")
        print(f"   Benchmarks run: {len(results['results'])}")
        print(f"   Regressions detected: {len(results['regressions'])}")
        
        if results['regressions']:
            print("   ⚠️ Performance regressions found:")
            for reg in results['regressions']:
                print(f"     - {reg['benchmark']}.{reg['metric']}: "
                      f"{reg['percent_change']:.1f}% ({reg['severity']})")
        
        # Simulate CI decision making
        critical_regressions = [r for r in results['regressions'] 
                               if r['severity'] == 'critical']
        
        if critical_regressions:
            print("\n   🚨 CI PIPELINE WOULD FAIL - Critical regressions detected!")
            return False
        else:
            print("\n   ✅ CI PIPELINE WOULD PASS - No critical regressions")
            return True
            
    except Exception as e:
        print(f"   ❌ CI performance check failed: {e}")
        return False


def demo_benchmark_comparison():
    """Demonstrate comparing different quantum algorithms."""
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON DEMO")
    print("=" * 60)
    
    print("1. Benchmarking quantum algorithms...")
    
    # Benchmark different quantum operations
    algorithms = ['bell_state', 'grover', 'qft']
    results = benchmark_quantum_operations(algorithms)
    
    print("\n2. Algorithm performance comparison:")
    print("   Algorithm   | Exec Time | Qubits | Depth | Special Metrics")
    print("   ------------|-----------|--------|-------|----------------")
    
    for alg_name, metrics in results.items():
        exec_time = metrics.get('execution_time', 0)
        qubits = metrics.get('qubit_count', 'N/A')
        depth = metrics.get('circuit_depth', 'N/A')
        
        # Get a special metric for each algorithm
        special = ""
        if 'operations_per_second' in metrics:
            special = f"OPS: {metrics['operations_per_second']:.1f}"
        elif 'gate_count' in metrics:
            special = f"Gates: {metrics['gate_count']}"
        elif 'algorithm_complexity' in metrics:
            special = f"Type: {metrics['algorithm_complexity']}"
        
        print(f"   {alg_name:11} | {exec_time:8.3f}s | {qubits:6} | {depth:5} | {special}")
    
    # Identify best performing algorithm
    best_alg = min(results.keys(), key=lambda k: results[k].get('execution_time', float('inf')))
    worst_alg = max(results.keys(), key=lambda k: results[k].get('execution_time', 0))
    
    print(f"\n3. Performance analysis:")
    print(f"   Fastest algorithm: {best_alg} ({results[best_alg]['execution_time']:.6f}s)")
    print(f"   Slowest algorithm: {worst_alg} ({results[worst_alg]['execution_time']:.6f}s)")
    
    if results[worst_alg]['execution_time'] > 0:
        speedup = results[worst_alg]['execution_time'] / results[best_alg]['execution_time']
        print(f"   Speedup ratio: {speedup:.2f}x")
    
    return results


def demo_threshold_configuration():
    """Demonstrate configuring regression thresholds."""
    print("\n" + "=" * 60)
    print("THRESHOLD CONFIGURATION DEMO")
    print("=" * 60)
    
    # Create database and detector
    db = PerformanceDatabase("threshold_demo.db")
    detector = RegressionDetector(db)
    
    print("1. Default regression thresholds:")
    for metric, threshold in detector.thresholds.items():
        print(f"   {metric}: {threshold.max_degradation_percent}% degradation limit")
        print(f"     Min samples: {threshold.min_samples}")
        if threshold.max_absolute_change:
            print(f"     Max absolute change: {threshold.max_absolute_change}")
    
    print("\n2. Configuring custom thresholds...")
    
    # Add custom threshold for a specific use case
    from quantrs2.performance_regression_tests import RegressionThreshold
    
    # Stricter threshold for critical operations
    detector.thresholds['critical_execution_time'] = RegressionThreshold(
        metric_name='critical_execution_time',
        max_degradation_percent=5.0,  # Very strict - only 5% degradation allowed
        min_samples=5
    )
    
    # More lenient threshold for experimental features
    detector.thresholds['experimental_feature_time'] = RegressionThreshold(
        metric_name='experimental_feature_time',
        max_degradation_percent=50.0,  # More lenient for experimental features
        min_samples=2
    )
    
    print("   Added custom thresholds:")
    print("   - critical_execution_time: 5% (strict)")
    print("   - experimental_feature_time: 50% (lenient)")
    
    print("\n3. Threshold recommendations:")
    print("   - Production code: 10-15% degradation threshold")
    print("   - Development code: 20-25% degradation threshold")  
    print("   - Experimental features: 30-50% degradation threshold")
    print("   - Memory usage: 15-20% increase threshold")
    print("   - Critical path operations: 5-10% degradation threshold")


def main():
    """Run all performance regression testing demonstrations."""
    print("QuantRS2 Performance Regression Testing Framework")
    print("=" * 60)
    
    print("This demonstration shows how to:")
    print("1. Set up comprehensive quantum algorithm benchmarks")
    print("2. Detect performance regressions automatically")
    print("3. Monitor performance trends over time")
    print("4. Integrate with CI/CD pipelines")
    print("5. Compare quantum algorithm performance")
    print("6. Configure regression detection thresholds")
    
    # Run demonstrations
    demos = [
        ("Basic Benchmarking", demo_basic_benchmarking),
        ("Regression Detection", demo_regression_detection),
        ("Custom Benchmarks", demo_custom_benchmarks),
        ("Performance Monitoring", demo_performance_monitoring),
        ("CI/CD Integration", demo_ci_integration),
        ("Algorithm Comparison", demo_benchmark_comparison),
        ("Threshold Configuration", demo_threshold_configuration)
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            demo_func()
        except Exception as e:
            print(f"\n{demo_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    print("\nKey takeaways:")
    print("1. Automated performance regression detection prevents degradations")
    print("2. Historical performance data enables trend analysis")
    print("3. Custom benchmarks can target specific use cases")
    print("4. CI/CD integration catches regressions early")
    print("5. Configurable thresholds adapt to different requirements")
    print("6. Comprehensive reporting aids in debugging performance issues")
    
    print("\nNext steps:")
    print("- Set up performance benchmarks for your quantum algorithms")
    print("- Integrate with your CI/CD pipeline")
    print("- Configure appropriate regression thresholds")
    print("- Monitor performance trends over development cycles")
    print("- Use benchmarks to guide optimization efforts")
    
    # Clean up demo files
    import os
    demo_files = [
        "demo_performance.db", "monitoring_demo.db", 
        "ci_demo.db", "threshold_demo.db"
    ]
    
    print("\nCleaning up demo files...")
    for file in demo_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"  Removed {file}")
        except Exception:
            pass


if __name__ == "__main__":
    main()