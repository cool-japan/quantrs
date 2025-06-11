# Testing Tools API Reference

The QuantRS2 testing framework provides comprehensive testing capabilities specifically designed for quantum software development.

## Quantum Testing Tools

::: quantrs2.quantum_testing_tools
    options:
      members:
        - QuantumTestSuite
        - PropertyBasedTester
        - QuantumMockBackend
        - TestCase
        - TestResult
        - run_quantum_tests
        - create_test_suite
        - mock_quantum_backend

### QuantumTestSuite

Main test suite management class with comprehensive testing capabilities.

#### Methods

- `add_test(test_case)`: Add individual test case
- `run_all_tests()`: Execute all tests in the suite
- `run_filtered_tests(filter_func)`: Run tests matching criteria
- `generate_test_report()`: Create detailed test report
- `setup_test_environment()`: Initialize testing environment
- `teardown_test_environment()`: Clean up after tests

#### Usage Example

```python
from quantrs2.quantum_testing_tools import (
    QuantumTestSuite, TestCase, PropertyBasedTester
)
from quantrs2 import Circuit

# Create test suite
test_suite = QuantumTestSuite("VQE Algorithm Tests")

# Add functional tests
def test_vqe_convergence():
    """Test VQE algorithm convergence."""
    vqe = VQEAlgorithm(hamiltonian="H2")
    result = vqe.optimize(max_iterations=100)
    assert result.converged
    assert result.energy < -1.0  # Expected ground state energy

test_suite.add_test(TestCase(test_vqe_convergence, "VQE Convergence"))

# Add property-based tests
property_tester = PropertyBasedTester()
test_suite.add_property_test(
    property_tester.test_circuit_unitarity,
    "Circuit Unitarity"
)

# Run all tests
results = test_suite.run_all_tests()
print(f"Passed: {results.passed_count}")
print(f"Failed: {results.failed_count}")
```

### PropertyBasedTester

Advanced property-based testing framework for quantum operations.

#### Test Properties

- **Unitarity**: Verify quantum operations are unitary
- **Normalization**: Ensure quantum states remain normalized
- **Hermiticity**: Check observables are Hermitian
- **Commutativity**: Test gate commutation relations
- **Entanglement**: Verify entanglement generation/preservation

#### Methods

- `test_circuit_unitarity(circuit)`: Verify circuit preserves unitarity
- `test_state_normalization(state)`: Check state vector normalization
- `test_gate_commutativity(gate1, gate2)`: Test if gates commute
- `test_observable_hermiticity(observable)`: Verify observable properties
- `generate_random_circuits(n_qubits, depth)`: Create test circuits

#### Usage Example

```python
from quantrs2.quantum_testing_tools import PropertyBasedTester
import numpy as np

tester = PropertyBasedTester()

# Test circuit unitarity
circuit = Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)

result = tester.test_circuit_unitarity(circuit)
assert result.passed, f"Unitarity test failed: {result.message}"

# Test state normalization
state = np.array([0.6, 0.8j, 0.0, 0.0])
norm_result = tester.test_state_normalization(state)
assert norm_result.passed

# Generate and test random circuits
for _ in range(100):
    random_circuit = tester.generate_random_circuit(3, depth=10)
    unitarity_result = tester.test_circuit_unitarity(random_circuit)
    assert unitarity_result.passed
```

### QuantumMockBackend

Mock quantum backend for testing with configurable noise and latency.

#### Features

- **Configurable Noise**: Add realistic noise models
- **Latency Simulation**: Simulate hardware execution delays
- **Error Injection**: Introduce specific error types
- **Resource Limits**: Test resource constraints
- **Concurrent Testing**: Support parallel test execution

#### Methods

- `set_noise_model(noise_params)`: Configure noise characteristics
- `set_latency(mean, variance)`: Set execution timing
- `inject_errors(error_types)`: Add specific error patterns
- `limit_resources(max_qubits, max_shots)`: Set resource constraints
- `execute_circuit(circuit, shots)`: Execute with mock backend

#### Usage Example

```python
from quantrs2.quantum_testing_tools import QuantumMockBackend
from quantrs2.noise import NoiseModel

# Create mock backend with noise
mock_backend = QuantumMockBackend()

# Configure noise model
noise_model = NoiseModel()
noise_model.add_depolarizing_noise(0.01, ['h', 'x', 'y', 'z'])
noise_model.add_readout_error(0.05)
mock_backend.set_noise_model(noise_model)

# Set realistic latency
mock_backend.set_latency(mean=2.0, variance=0.5)  # seconds

# Test algorithm with noise
circuit = create_grover_circuit(3)
results = []
for _ in range(10):
    result = mock_backend.execute_circuit(circuit, shots=1024)
    results.append(result)

# Analyze noise impact
success_rates = [analyze_grover_success(r) for r in results]
avg_success_rate = np.mean(success_rates)
print(f"Average success rate with noise: {avg_success_rate:.2%}")
```

## Test Types and Categories

### Functional Testing

Test basic functionality and correctness:

```python
class FunctionalTests:
    def test_bell_state_preparation(self):
        """Test Bell state preparation circuit."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        result = simulate_circuit(circuit)
        expected_state = np.array([1/√2, 0, 0, 1/√2])
        
        assert np.allclose(result.state_vector, expected_state)
    
    def test_quantum_fourier_transform(self):
        """Test QFT implementation."""
        for n_qubits in range(2, 6):
            qft_circuit = create_qft_circuit(n_qubits)
            # Test against classical DFT
            assert verify_qft_correctness(qft_circuit, n_qubits)
```

### Performance Testing

Validate performance characteristics:

```python
class PerformanceTests:
    def test_circuit_simulation_scaling(self):
        """Test simulation performance scaling."""
        for n_qubits in range(4, 12):
            circuit = create_random_circuit(n_qubits, depth=20)
            
            start_time = time.time()
            result = simulate_circuit(circuit)
            execution_time = time.time() - start_time
            
            # Check scaling behavior
            expected_max_time = 2**n_qubits * 1e-6  # 1μs per amplitude
            assert execution_time < expected_max_time
    
    def test_memory_usage(self):
        """Test memory usage patterns."""
        memory_tracker = MemoryTracker()
        
        with memory_tracker.track():
            circuit = Circuit(10)
            for i in range(9):
                circuit.cnot(i, i+1)
            result = simulate_circuit(circuit)
        
        assert memory_tracker.peak_usage < 1024 * 1024 * 100  # 100MB limit
```

### Integration Testing

Test module interactions:

```python
class IntegrationTests:
    def test_vqe_with_noise_mitigation(self):
        """Test VQE with error mitigation."""
        # Create noisy backend
        noisy_backend = create_noisy_backend()
        
        # Setup VQE with mitigation
        vqe = VQE(backend=noisy_backend)
        vqe.enable_error_mitigation("zero_noise_extrapolation")
        
        # Run optimization
        result = vqe.optimize(hamiltonian="H2")
        
        # Verify mitigation effectiveness
        assert result.mitigated_energy < result.raw_energy
        assert abs(result.mitigated_energy - (-1.137)) < 0.1
```

### Regression Testing

Detect performance regressions:

```python
class RegressionTests:
    def test_benchmark_performance(self):
        """Ensure performance hasn't regressed."""
        benchmark_results = load_benchmark_baselines()
        
        for algorithm, baseline in benchmark_results.items():
            current_time = benchmark_algorithm(algorithm)
            regression_threshold = baseline * 1.1  # 10% tolerance
            
            assert current_time < regression_threshold, (
                f"Performance regression in {algorithm}: "
                f"{current_time:.3f}s > {regression_threshold:.3f}s"
            )
```

### Fuzz Testing

Test with random inputs:

```python
class FuzzTests:
    def test_random_circuit_simulation(self):
        """Fuzz test circuit simulation."""
        fuzz_tester = FuzzTester()
        
        for _ in range(1000):
            # Generate random valid circuit
            circuit = fuzz_tester.generate_random_circuit(
                n_qubits=random.randint(2, 8),
                depth=random.randint(5, 50),
                gate_set=['h', 'x', 'y', 'z', 'cnot', 'cz']
            )
            
            # Should not crash or produce invalid results
            result = simulate_circuit(circuit)
            assert is_valid_quantum_state(result.state_vector)
            assert abs(np.linalg.norm(result.state_vector) - 1.0) < 1e-10
```

## Test Configuration and Setup

### Test Configuration

```python
from quantrs2.quantum_testing_tools import TestConfig

config = TestConfig(
    # Test execution settings
    parallel_execution=True,
    max_workers=4,
    timeout_per_test=30.0,
    
    # Noise and simulation settings
    default_shots=1024,
    noise_tolerance=0.01,
    fidelity_threshold=0.95,
    
    # Reporting options
    verbose_output=True,
    generate_html_report=True,
    save_test_data=True,
    
    # Performance testing
    enable_profiling=True,
    memory_limit_mb=1024,
    
    # Property testing
    hypothesis_max_examples=100,
    hypothesis_deadline=5000  # milliseconds
)
```

### Test Environment Setup

```python
def setup_test_environment():
    """Initialize testing environment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Configure test backends
    setup_test_backends()
    
    # Initialize test data
    generate_test_datasets()
    
    # Setup temporary directories
    create_test_workspace()
```

## Test Reporting and Analysis

### Test Results

```python
class TestResult:
    """Comprehensive test result information."""
    
    def __init__(self):
        self.test_name: str = ""
        self.status: str = "pending"  # passed, failed, skipped, error
        self.execution_time: float = 0.0
        self.memory_usage: int = 0
        self.error_message: Optional[str] = None
        self.performance_metrics: Dict[str, float] = {}
        self.coverage_data: Dict[str, float] = {}
```

### Report Generation

```python
def generate_comprehensive_report(test_results):
    """Generate detailed test report."""
    report = TestReport()
    
    # Summary statistics
    report.add_summary(test_results)
    
    # Performance analysis
    report.add_performance_analysis(test_results)
    
    # Coverage report
    report.add_coverage_analysis(test_results)
    
    # Failure analysis
    report.add_failure_analysis(test_results)
    
    # Trend analysis (compare with historical data)
    report.add_trend_analysis(test_results)
    
    return report
```

## Advanced Testing Features

### Concurrent Testing

```python
def run_concurrent_tests(test_suite, max_workers=4):
    """Run tests in parallel for better performance."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for test in test_suite.tests:
            future = executor.submit(run_single_test, test)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return aggregate_test_results(results)
```

### Test Data Management

```python
class TestDataManager:
    """Manage test datasets and fixtures."""
    
    def __init__(self):
        self.datasets = {}
        self.fixtures = {}
    
    def load_quantum_dataset(self, name):
        """Load quantum test dataset."""
        if name not in self.datasets:
            self.datasets[name] = self._generate_dataset(name)
        return self.datasets[name]
    
    def create_fixture(self, name, generator_func):
        """Create reusable test fixture."""
        self.fixtures[name] = generator_func
    
    def get_fixture(self, name):
        """Get test fixture."""
        return self.fixtures[name]()
```

## Error Handling and Debugging

### Test-specific Exceptions

```python
class TestingError(Exception):
    """Base exception for testing framework."""
    pass

class PropertyTestFailure(TestingError):
    """Property-based test failure."""
    pass

class MockBackendError(TestingError):
    """Mock backend operation error."""
    pass

class TestTimeoutError(TestingError):
    """Test execution timeout."""
    pass
```

### Debugging Support

```python
def debug_failed_test(test_result):
    """Debug a failed test with detailed analysis."""
    if test_result.status == "failed":
        # Analyze failure context
        context = extract_failure_context(test_result)
        
        # Generate debug report
        debug_report = create_debug_report(context)
        
        # Suggest fixes
        suggestions = generate_fix_suggestions(test_result)
        
        return {
            "context": context,
            "debug_report": debug_report,
            "suggestions": suggestions
        }
```

## Integration with CI/CD

### Continuous Integration

```python
def run_ci_tests():
    """Run tests suitable for CI environment."""
    # Quick smoke tests
    smoke_results = run_smoke_tests()
    if not smoke_results.all_passed:
        return smoke_results
    
    # Core functionality tests
    core_results = run_core_tests()
    if not core_results.all_passed:
        return core_results
    
    # Performance regression tests
    perf_results = run_performance_tests()
    
    return aggregate_results([smoke_results, core_results, perf_results])
```

### Test Metrics Collection

```python
def collect_test_metrics(test_results):
    """Collect metrics for monitoring test health."""
    metrics = {
        "test_count": len(test_results),
        "pass_rate": calculate_pass_rate(test_results),
        "average_execution_time": calculate_avg_time(test_results),
        "flaky_test_count": count_flaky_tests(test_results),
        "coverage_percentage": calculate_coverage(test_results)
    }
    
    # Send to monitoring system
    send_metrics_to_dashboard(metrics)
    
    return metrics
```

## See Also

- [Core Module](core.md) for basic circuit operations
- [Performance Profiling](../user-guide/performance.md) for optimization testing
- [Development Tools](dev-tools.md) for IDE integration
- [Debugging Tools](../advanced/debugging.md) for troubleshooting