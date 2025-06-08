# QuantRS2 Benchmarking Suite

Comprehensive performance benchmarking tools for the QuantRS2 quantum computing framework.

## Overview

This benchmarking suite provides tools to measure and analyze the performance of QuantRS2 across various dimensions:

- **Performance Benchmarks**: Execution time and throughput for different quantum operations
- **Memory Benchmarks**: Memory usage patterns and scaling behavior
- **Parallel Benchmarks**: Thread scaling, GPU acceleration, and batch processing efficiency
- **System Benchmarks**: Hardware-specific optimizations and backend comparisons

## Quick Start

Run all benchmarks with a single command:

```bash
python benchmarks/run_benchmarks.py
```

For quick testing (excludes memory and parallel benchmarks):

```bash
python benchmarks/run_benchmarks.py --quick
```

## Benchmark Modules

### 1. Comprehensive Benchmark Suite (`benchmark_suite.py`)

Tests overall performance across different QuantRS2 components:

- Circuit simulation with various backends
- VQE optimization performance
- Quantum annealing simulation
- Transfer learning adaptation
- Visualization data preparation

**Usage:**
```bash
python benchmarks/benchmark_suite.py
```

**Key Metrics:**
- Gates per second
- Optimization iterations per second
- Samples per second (annealing)
- Time to prepare visualization data

### 2. Memory Benchmarks (`memory_benchmark.py`)

Analyzes memory usage patterns:

- State vector memory scaling
- Tensor network memory efficiency
- Batch processing memory overhead
- ML model memory footprint
- Sparse operations memory usage

**Usage:**
```bash
python benchmarks/memory_benchmark.py
```

**Key Metrics:**
- Peak memory usage
- Memory scaling with problem size
- Memory efficiency comparisons

### 3. Parallel Benchmarks (`parallel_benchmark.py`)

Tests parallel execution and GPU acceleration:

- Thread scaling efficiency
- CPU vs GPU performance comparison
- Batch processing throughput
- Parallel algorithm speedup

**Usage:**
```bash
python benchmarks/parallel_benchmark.py
```

**Key Metrics:**
- Parallel speedup and efficiency
- GPU acceleration factors
- Optimal batch sizes
- Thread scaling behavior

## Running Specific Benchmarks

### Skip certain modules:

```bash
python benchmarks/run_benchmarks.py --skip memory parallel
```

### Custom output directory:

```bash
python benchmarks/run_benchmarks.py --output-dir my_results
```

## Understanding Results

After running benchmarks, you'll find:

1. **HTML Report**: `benchmark_results/report_<timestamp>/benchmark_report.html`
   - System information
   - Performance summaries
   - Visualizations
   - Recommendations

2. **Raw Data**: JSON and CSV files in `benchmark_results/`
   - Detailed timing data
   - Memory usage traces
   - Scaling analysis

3. **Visualizations**: PNG plots showing:
   - Scaling behavior
   - Performance comparisons
   - Memory usage patterns
   - Parallel efficiency

## Benchmark Configuration

Each benchmark module can be configured:

### Circuit Simulation
- `n_qubits`: Number of qubits (5-20)
- `depth`: Circuit depth
- `backend`: cpu, gpu, or auto

### VQE Optimization
- `n_qubits`: System size
- `n_layers`: Ansatz depth
- `n_iterations`: Optimization steps

### Memory Tests
- Size ranges for scaling analysis
- Comparison configurations

### Parallel Tests
- Thread counts
- Batch sizes
- Problem sizes for GPU comparison

## Interpreting Results

### Performance Metrics

1. **Execution Time**: Lower is better
   - Look for scaling behavior with problem size
   - Compare backends for optimal choice

2. **Throughput**: Higher is better
   - Gates/second for circuits
   - Samples/second for annealing
   - Items/second for batch processing

3. **Memory Usage**: Lower is better
   - Check for unexpected scaling
   - Identify memory bottlenecks

4. **Parallel Efficiency**: Closer to 100% is better
   - Identifies parallelization bottlenecks
   - Helps choose optimal thread count

### Common Issues

1. **GPU Not Available**: 
   - Install with GPU support: `pip install quantrs2[gpu]`
   - Check CUDA/GPU drivers

2. **Memory Errors**:
   - Reduce problem sizes
   - Use tensor network backend for large systems

3. **Poor Parallel Scaling**:
   - May indicate memory bandwidth limits
   - Try different batch sizes

## Customizing Benchmarks

Add new benchmarks by:

1. Create a function that performs the operation:
```python
def my_benchmark(**kwargs):
    # Perform operation
    return {'metrics': {'my_metric': value}}
```

2. Add to benchmark list:
```python
my_benchmarks = [
    ("My Benchmark", my_benchmark, [
        {'param1': value1, 'param2': value2},
        # More parameter sets
    ])
]
```

3. Run with the suite:
```python
suite.run_category("My Category", my_benchmarks)
```

## Best Practices

1. **Warm-up Runs**: Each benchmark includes warm-up runs to ensure stable measurements

2. **Multiple Runs**: Results are averaged over multiple runs to reduce variance

3. **System State**: Close other applications for consistent results

4. **Reproducibility**: System information is recorded with each run

## Continuous Benchmarking

For tracking performance over time:

1. Run benchmarks regularly:
```bash
# Add to CI/CD pipeline
python benchmarks/run_benchmarks.py --quick
```

2. Compare results:
```python
# Compare two benchmark runs
python benchmarks/compare_results.py results1.json results2.json
```

3. Track regressions:
- Set performance thresholds
- Alert on significant slowdowns

## Contributing

When adding new features to QuantRS2:

1. Add corresponding benchmarks
2. Run before and after changes
3. Document any performance impacts
4. Optimize if regressions found

## Troubleshooting

### "Module not found" errors
- Ensure you're in the correct directory
- Install required dependencies: `pip install -r requirements-benchmark.txt`

### Benchmarks hang or crash
- Reduce problem sizes
- Check available memory
- Disable GPU benchmarks if GPU issues

### Inconsistent results
- Ensure exclusive system access
- Disable CPU frequency scaling
- Run multiple times and average

## Future Enhancements

Planned improvements:
- Network-distributed benchmarks
- Hardware profiling integration
- Automated regression detection
- Cloud backend benchmarks
- Real quantum hardware benchmarks