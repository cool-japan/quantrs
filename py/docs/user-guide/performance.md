# Performance Optimization Guide

Learn how to optimize your quantum circuits and applications for maximum performance with QuantRS2.

## Understanding Quantum Performance

Quantum circuit performance differs significantly from classical computing. This guide covers optimization strategies specific to quantum computing.

### Performance Metrics

#### Circuit-Level Metrics
- **Gate count**: Total number of quantum gates
- **Circuit depth**: Maximum gates in any qubit's timeline
- **Two-qubit gate count**: Most expensive operations
- **Connectivity requirements**: Qubit interaction patterns

#### Simulation Metrics
- **Memory usage**: Scales as 2ⁿ for n qubits
- **Execution time**: Depends on gates and qubits
- **Parallelization efficiency**: Multi-core utilization
- **Cache performance**: Data access patterns

#### Hardware Metrics (Real Devices)
- **Fidelity**: Quality of gate operations
- **Coherence time**: How long qubits maintain quantum properties
- **Error rates**: Probability of gate failures
- **Calibration drift**: Time-dependent performance changes

## Circuit Optimization

### 1. Gate Count Reduction

Minimize the total number of gates in your circuit:

```python
import quantrs2
import numpy as np

# ❌ Inefficient: Redundant gates
def inefficient_circuit():
    circuit = quantrs2.Circuit(2)
    
    # Redundant operations
    circuit.h(0)
    circuit.h(0)  # H·H = I (identity)
    
    circuit.x(1)
    circuit.x(1)  # X·X = I
    
    # Multiple single-qubit rotations
    circuit.rx(0, np.pi/4)
    circuit.ry(0, np.pi/6)
    circuit.rz(0, np.pi/8)
    
    return circuit

# ✅ Efficient: Combined operations
def efficient_circuit():
    circuit = quantrs2.Circuit(2)
    
    # Single combined rotation (equivalent to above)
    # Use U3 gate or matrix decomposition
    circuit.u3(0, theta=0.123, phi=0.456, lam=0.789)
    
    return circuit

# Compare gate counts
inefficient = inefficient_circuit()
efficient = efficient_circuit()

print(f"Inefficient gates: {inefficient.gate_count}")  # 6 gates
print(f"Efficient gates: {efficient.gate_count}")      # 1 gate
```

### 2. Circuit Depth Optimization

Reduce circuit depth by parallelizing operations:

```python
# ❌ Sequential operations (high depth)
def high_depth_circuit(n_qubits=4):
    circuit = quantrs2.Circuit(n_qubits)
    
    # Sequential single-qubit gates
    for i in range(n_qubits):
        circuit.h(i)
    
    # Sequential two-qubit gates
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    
    return circuit

# ✅ Parallel operations (low depth)
def low_depth_circuit(n_qubits=4):
    circuit = quantrs2.Circuit(n_qubits)
    
    # All single-qubit gates can be parallel
    for i in range(n_qubits):
        circuit.h(i)
    
    # Parallelize two-qubit gates where possible
    # Even-odd pairing allows parallelization
    for i in range(0, n_qubits - 1, 2):
        circuit.cx(i, i + 1)
    
    for i in range(1, n_qubits - 1, 2):
        circuit.cx(i, i + 1)
    
    return circuit

# Compare depths
high_depth = high_depth_circuit(8)
low_depth = low_depth_circuit(8)

print(f"High depth: {high_depth.depth}")    # 11
print(f"Low depth: {low_depth.depth}")      # 3
```

### 3. Gate Fusion and Decomposition

Use QuantRS2's optimization tools:

```python
from quantrs2.optimization import (
    optimize_circuit, 
    fuse_single_qubit_gates,
    decompose_multi_controlled_gates
)

def demonstrate_optimization():
    """Show circuit optimization techniques."""
    
    # Create unoptimized circuit
    circuit = quantrs2.Circuit(3)
    
    # Add multiple rotation gates (can be fused)
    circuit.rx(0, np.pi/4)
    circuit.ry(0, np.pi/6)
    circuit.rz(0, np.pi/8)
    
    # Add redundant operations
    circuit.h(1)
    circuit.z(1)
    circuit.h(1)  # Equivalent to X gate
    
    # Complex multi-controlled gate
    circuit.ccx(0, 1, 2)
    
    print(f"Original circuit:")
    print(f"  Gates: {circuit.gate_count}")
    print(f"  Depth: {circuit.depth}")
    
    # Apply optimizations
    optimized = optimize_circuit(circuit)
    
    print(f"\nOptimized circuit:")
    print(f"  Gates: {optimized.gate_count}")
    print(f"  Depth: {optimized.depth}")
    
    # Verify equivalence
    original_result = circuit.run()
    optimized_result = optimized.run()
    
    print(f"\nResults equivalent: {original_result.state_probabilities() == optimized_result.state_probabilities()}")

demonstrate_optimization()
```

## Memory Optimization

### Understanding Memory Scaling

Quantum simulation memory requirements:

| Qubits | States | Memory (Complex128) |
|--------|--------|-------------------|
| 10     | 1,024  | 16 KB            |
| 20     | 1M     | 16 MB            |
| 30     | 1B     | 16 GB            |
| 40     | 1T     | 16 TB            |

### Memory-Efficient Patterns

```python
import quantrs2
import numpy as np

# ❌ Memory inefficient: Large intermediate states
def memory_heavy_approach():
    """Avoid storing large intermediate states."""
    
    # Don't create large arrays unnecessarily
    large_array = np.zeros((2**20, 2**20), dtype=complex)  # 16 TB!
    
    circuit = quantrs2.Circuit(20)
    # ... circuit operations
    
    return circuit

# ✅ Memory efficient: Stream processing
def memory_efficient_approach():
    """Process data in chunks, avoid large allocations."""
    
    # Use generators and streaming
    def quantum_data_processor():
        for batch in range(100):
            circuit = quantrs2.Circuit(10)  # Smaller circuits
            # Process batch
            yield circuit.run()
    
    # Process results without storing all at once
    results = []
    for result in quantum_data_processor():
        # Process immediately, don't accumulate
        processed = analyze_result(result)
        results.append(processed)
    
    return results

def analyze_result(result):
    """Analyze result without storing large data."""
    probs = result.state_probabilities()
    return {
        'max_prob': max(probs.values()),
        'entropy': calculate_entropy(probs),
        'num_nonzero': len(probs)
    }

def calculate_entropy(probs):
    """Calculate Shannon entropy."""
    return -sum(p * np.log2(p) for p in probs.values() if p > 0)
```

### Sparse Representations

For circuits with sparse state vectors:

```python
# Use sparse representations when appropriate
from quantrs2.sparse import SparseSimulator

def sparse_simulation_example():
    """Use sparse simulation for appropriate circuits."""
    
    # Circuits with many |0⟩ states benefit from sparse simulation
    circuit = quantrs2.Circuit(15)
    
    # Only affect a few qubits
    circuit.h(0)
    circuit.cx(0, 1)
    # Most qubits remain in |0⟩
    
    # Use sparse simulator
    sparse_sim = SparseSimulator()
    result = sparse_sim.run(circuit)
    
    print(f"Sparse simulation memory: {sparse_sim.memory_usage} MB")
    return result
```

## GPU Acceleration

### When to Use GPU

GPU acceleration is beneficial for:
- **Large circuits** (15+ qubits)
- **Many shots** (repeated executions)
- **Parameter sweeps** (variational algorithms)

```python
import quantrs2

def gpu_optimization_example():
    """Demonstrate GPU acceleration best practices."""
    
    # Check GPU availability
    if not quantrs2.gpu_available():
        print("GPU not available, using CPU")
        return
    
    # Create large circuit that benefits from GPU
    circuit = quantrs2.Circuit(20)
    
    # Add many gates (GPU overhead is amortized)
    for layer in range(10):
        for qubit in range(20):
            circuit.ry(qubit, np.random.random() * 2 * np.pi)
        
        for qubit in range(19):
            circuit.cx(qubit, qubit + 1)
    
    # Benchmark CPU vs GPU
    import time
    
    # CPU execution
    start_time = time.time()
    cpu_result = circuit.run(use_gpu=False)
    cpu_time = time.time() - start_time
    
    # GPU execution
    start_time = time.time()
    gpu_result = circuit.run(use_gpu=True)
    gpu_time = time.time() - start_time
    
    print(f"CPU time: {cpu_time:.3f}s")
    print(f"GPU time: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    return gpu_result

# Run GPU optimization example
gpu_optimization_example()
```

### GPU Memory Management

```python
# Manage GPU memory efficiently
import quantrs2

def gpu_memory_management():
    """Best practices for GPU memory management."""
    
    # Batch similar circuits together
    circuits = []
    for i in range(10):
        circuit = quantrs2.Circuit(15)
        # ... build circuit
        circuits.append(circuit)
    
    # Run batch on GPU (more efficient than individual runs)
    results = quantrs2.run_circuits_batch(circuits, use_gpu=True)
    
    # Clear GPU memory when done
    quantrs2.gpu_clear_cache()
    
    return results
```

## Algorithmic Optimizations

### Variational Algorithm Optimization

```python
import quantrs2
import numpy as np
from scipy.optimize import minimize

class OptimizedVQE:
    """Optimized Variational Quantum Eigensolver."""
    
    def __init__(self, num_qubits, depth):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit_cache = {}  # Cache compiled circuits
        
    def create_ansatz(self, parameters):
        """Create ansatz with caching."""
        param_key = tuple(np.round(parameters, 6))  # Round for cache key
        
        if param_key in self.circuit_cache:
            return self.circuit_cache[param_key]
        
        circuit = quantrs2.Circuit(self.num_qubits)
        
        param_idx = 0
        for layer in range(self.depth):
            # Efficient layer construction
            self._add_rotation_layer(circuit, parameters[param_idx:param_idx+self.num_qubits])
            param_idx += self.num_qubits
            
            self._add_entangling_layer(circuit)
        
        # Cache the circuit
        self.circuit_cache[param_key] = circuit
        return circuit
    
    def _add_rotation_layer(self, circuit, layer_params):
        """Add rotation layer efficiently."""
        for qubit, param in enumerate(layer_params):
            circuit.ry(qubit, param)
    
    def _add_entangling_layer(self, circuit):
        """Add entangling layer with optimal connectivity."""
        # Use linear connectivity for NISQ devices
        for qubit in range(self.num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
    
    def cost_function(self, parameters):
        """Optimized cost function."""
        circuit = self.create_ansatz(parameters)
        
        # Use GPU for large circuits
        use_gpu = self.num_qubits >= 15
        result = circuit.run(use_gpu=use_gpu)
        
        # Calculate expectation value efficiently
        return self._calculate_energy(result)
    
    def _calculate_energy(self, result):
        """Efficient energy calculation."""
        # Implement specific Hamiltonian efficiently
        # This is problem-specific
        return 0.0  # Placeholder
    
    def optimize(self, initial_params):
        """Run optimization with performance monitoring."""
        
        # Track optimization performance
        self.function_calls = 0
        self.best_energy = float('inf')
        
        def wrapped_cost_function(params):
            self.function_calls += 1
            energy = self.cost_function(params)
            
            if energy < self.best_energy:
                self.best_energy = energy
                print(f"Call {self.function_calls}: Energy = {energy:.6f}")
            
            return energy
        
        # Use efficient optimizer
        result = minimize(
            wrapped_cost_function,
            initial_params,
            method='COBYLA',  # Good for noisy functions
            options={'maxiter': 100}
        )
        
        print(f"Optimization completed in {self.function_calls} function calls")
        print(f"Cache size: {len(self.circuit_cache)} circuits")
        
        return result

# Example usage
vqe = OptimizedVQE(num_qubits=8, depth=3)
initial_params = np.random.random(24) * 2 * np.pi
result = vqe.optimize(initial_params)
```

### Parameter Sweep Optimization

```python
def efficient_parameter_sweep():
    """Efficiently explore parameter space."""
    
    # Use parallel processing for parameter sweeps
    from concurrent.futures import ThreadPoolExecutor
    import itertools
    
    def evaluate_parameters(params):
        """Evaluate single parameter set."""
        circuit = quantrs2.Circuit(4)
        
        for i, param in enumerate(params):
            circuit.ry(i, param)
        
        for i in range(3):
            circuit.cx(i, i + 1)
        
        result = circuit.run()
        return calculate_cost(result)
    
    def calculate_cost(result):
        """Calculate cost from result."""
        probs = result.state_probabilities()
        # Example cost function
        return sum(prob * int(state, 2) for state, prob in probs.items())
    
    # Generate parameter grid
    param_ranges = [np.linspace(0, 2*np.pi, 10) for _ in range(4)]
    param_grid = list(itertools.product(*param_ranges))
    
    print(f"Evaluating {len(param_grid)} parameter combinations...")
    
    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=4) as executor:
        costs = list(executor.map(evaluate_parameters, param_grid))
    
    # Find best parameters
    best_idx = np.argmin(costs)
    best_params = param_grid[best_idx]
    best_cost = costs[best_idx]
    
    print(f"Best parameters: {best_params}")
    print(f"Best cost: {best_cost:.6f}")
    
    return best_params, best_cost

# Run parameter sweep
best_params, best_cost = efficient_parameter_sweep()
```

## Performance Monitoring

### Built-in Profiling Tools

```python
from quantrs2.profiler import CircuitProfiler, profile_circuit

def performance_monitoring_example():
    """Demonstrate performance monitoring tools."""
    
    # Create test circuit
    circuit = quantrs2.Circuit(10)
    
    for layer in range(5):
        for qubit in range(10):
            circuit.ry(qubit, np.random.random())
        
        for qubit in range(9):
            circuit.cx(qubit, qubit + 1)
    
    # Profile circuit execution
    profiler = CircuitProfiler()
    
    with profiler:
        result = circuit.run()
    
    # Get performance metrics
    report = profiler.get_report()
    
    print("Performance Report:")
    print(f"  Execution time: {report.execution_time:.4f}s")
    print(f"  Memory usage: {report.memory_usage:.2f} MB")
    print(f"  Gate operations: {report.gate_operations}")
    print(f"  Cache hits: {report.cache_hits}")
    print(f"  Cache misses: {report.cache_misses}")
    
    return report

# Alternative: One-line profiling
circuit = quantrs2.Circuit(8)
# ... build circuit ...

profile_result = profile_circuit(circuit)
print(f"Quick profile: {profile_result.execution_time:.3f}s, {profile_result.memory_usage:.1f}MB")
```

### Custom Performance Metrics

```python
import time
import psutil
import os

class QuantumPerformanceMonitor:
    """Custom performance monitoring for quantum applications."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        self.metrics = {
            'execution_time': end_time - self.start_time,
            'memory_delta': end_memory - self.start_memory,
            'peak_memory': end_memory
        }
    
    def add_metric(self, name, value):
        """Add custom metric."""
        self.metrics[name] = value
    
    def get_metrics(self):
        """Get all collected metrics."""
        return self.metrics

# Usage example
def monitored_quantum_computation():
    """Example of monitored quantum computation."""
    
    with QuantumPerformanceMonitor() as monitor:
        # Create large circuit
        circuit = quantrs2.Circuit(15)
        
        # Add complexity
        for i in range(10):
            for qubit in range(15):
                circuit.ry(qubit, np.random.random())
            
            for qubit in range(14):
                circuit.cx(qubit, qubit + 1)
        
        # Track circuit metrics
        monitor.add_metric('gate_count', circuit.gate_count)
        monitor.add_metric('circuit_depth', circuit.depth)
        
        # Run simulation
        result = circuit.run()
        
        # Track result metrics
        probs = result.state_probabilities()
        monitor.add_metric('num_nonzero_states', len(probs))
        monitor.add_metric('max_probability', max(probs.values()))
    
    # Print performance report
    metrics = monitor.get_metrics()
    print("Performance Metrics:")
    for name, value in metrics.items():
        if 'time' in name:
            print(f"  {name}: {value:.4f}s")
        elif 'memory' in name:
            print(f"  {name}: {value:.2f}MB")
        else:
            print(f"  {name}: {value}")

monitored_quantum_computation()
```

## Hardware-Specific Optimizations

### NISQ Device Optimization

```python
from quantrs2.hardware import get_device_topology, optimize_for_device

def nisq_optimization_example():
    """Optimize circuits for NISQ devices."""
    
    # Get device specifications
    device_info = get_device_topology('ibm_perth')  # Example device
    
    print(f"Device: {device_info.name}")
    print(f"Qubits: {device_info.num_qubits}")
    print(f"Connectivity: {device_info.coupling_map}")
    print(f"Gate errors: {device_info.gate_errors}")
    
    # Create circuit
    circuit = quantrs2.Circuit(5)
    
    # Original circuit (may not match device topology)
    circuit.h(0)
    circuit.cx(0, 2)  # May require SWAP gates on some devices
    circuit.cx(1, 3)
    circuit.cx(2, 4)
    
    print(f"Original circuit depth: {circuit.depth}")
    
    # Optimize for device
    optimized_circuit = optimize_for_device(circuit, device_info)
    
    print(f"Optimized circuit depth: {optimized_circuit.depth}")
    print(f"Added SWAP gates: {optimized_circuit.gate_count - circuit.gate_count}")
    
    return optimized_circuit

# Device-aware circuit construction
def device_aware_circuit_construction():
    """Build circuits that match device topology."""
    
    # Use linear connectivity for most NISQ devices
    circuit = quantrs2.Circuit(5)
    
    # Create entanglement using device-native gates
    for i in range(4):
        circuit.cx(i, i + 1)  # Linear chain
    
    # Avoid long-range interactions that require many SWAPs
    # Instead of circuit.cx(0, 4), use:
    for i in range(4):
        circuit.cx(i, i + 1)  # Creates connectivity
    
    return circuit
```

### Error Mitigation

```python
from quantrs2.mitigation import (
    zero_noise_extrapolation,
    readout_error_mitigation,
    symmetry_verification
)

def error_mitigation_example():
    """Demonstrate error mitigation techniques."""
    
    # Original circuit
    circuit = quantrs2.Circuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()
    
    # Method 1: Zero-noise extrapolation
    noise_levels = [0.0, 0.01, 0.02]  # Different noise strengths
    results = []
    
    for noise in noise_levels:
        noisy_circuit = add_noise(circuit, noise)
        result = noisy_circuit.run()
        results.append(result)
    
    # Extrapolate to zero noise
    mitigated_result = zero_noise_extrapolation(results, noise_levels)
    
    # Method 2: Readout error mitigation
    calibration_circuits = create_readout_calibration_circuits(3)
    calibration_results = [c.run() for c in calibration_circuits]
    
    readout_matrix = calculate_readout_matrix(calibration_results)
    corrected_result = readout_error_mitigation(result, readout_matrix)
    
    # Method 3: Symmetry verification
    # Verify results using known symmetries
    verification_passed = symmetry_verification(
        circuit, result, symmetry_operators=['Z0*Z1', 'Z1*Z2']
    )
    
    print(f"Symmetry verification: {'PASSED' if verification_passed else 'FAILED'}")
    
    return mitigated_result

def add_noise(circuit, noise_level):
    """Add noise to circuit (placeholder)."""
    # Implementation depends on noise model
    return circuit

def create_readout_calibration_circuits(num_qubits):
    """Create circuits for readout calibration."""
    circuits = []
    
    # Create all computational basis states
    for i in range(2**num_qubits):
        circuit = quantrs2.Circuit(num_qubits)
        
        # Prepare basis state
        binary = format(i, f'0{num_qubits}b')
        for qubit, bit in enumerate(binary):
            if bit == '1':
                circuit.x(qubit)
        
        circuit.measure_all()
        circuits.append(circuit)
    
    return circuits

def calculate_readout_matrix(results):
    """Calculate readout error matrix."""
    # Implementation of readout matrix calculation
    # This is a simplified placeholder
    return np.eye(len(results))
```

## Best Practices Summary

### Do's ✅

1. **Profile before optimizing**: Measure to find bottlenecks
2. **Minimize gate count**: Fewer gates = faster execution
3. **Reduce circuit depth**: Enables parallelization
4. **Use GPU for large circuits**: 15+ qubits benefit from GPU
5. **Cache compiled circuits**: Avoid recompilation
6. **Batch similar operations**: Amortize overhead costs
7. **Use sparse simulation**: When state vectors are sparse
8. **Monitor memory usage**: Prevent out-of-memory errors
9. **Optimize for target hardware**: Match device topology
10. **Apply error mitigation**: Improve result quality

### Don'ts ❌

1. **Don't optimize prematurely**: Profile first
2. **Don't ignore memory scaling**: Plan for exponential growth
3. **Don't use GPU for small circuits**: Overhead outweighs benefit
4. **Don't forget to clear caches**: Prevent memory leaks
5. **Don't ignore device constraints**: Real hardware has limitations
6. **Don't skip error mitigation**: NISQ devices need error correction
7. **Don't use synchronous operations**: Prefer asynchronous when available
8. **Don't hardcode parameters**: Make algorithms configurable
9. **Don't ignore algorithmic complexity**: Some problems are inherently hard
10. **Don't skip validation**: Verify optimizations preserve correctness

### Performance Checklist

Before deploying quantum applications:

- [ ] **Profiled circuit execution** times and memory usage
- [ ] **Optimized gate count** using circuit optimization tools
- [ ] **Minimized circuit depth** through parallelization
- [ ] **Tested GPU acceleration** for large circuits
- [ ] **Implemented caching** for repeated computations
- [ ] **Added performance monitoring** to track degradation
- [ ] **Optimized for target hardware** topology and constraints
- [ ] **Applied error mitigation** techniques for NISQ devices
- [ ] **Validated correctness** after optimizations
- [ ] **Documented performance** characteristics for users

---

**Ready to optimize?** Start with [circuit profiling](../dev-tools/profiling.md) to identify bottlenecks in your quantum applications.

Remember: "Premature optimization is the root of all evil" applies to quantum computing too. Always measure first, then optimize based on actual performance data! 🚀