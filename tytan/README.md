# QuantRS2-Tytan

[![Crates.io](https://img.shields.io/crates/v/quantrs2-tytan.svg)](https://crates.io/crates/quantrs2-tytan)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](docs/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Tytan is a comprehensive, high-performance quantum annealing library for the QuantRS2 framework. Inspired by the Python [Tytan](https://github.com/tytansdk/tytan) library, it provides powerful tools for formulating and solving quantum optimization problems with state-of-the-art performance.

## Version 0.1.0

This release features refined integration with [SciRS2](https://github.com/cool-japan/scirs2) v0.1.0:
- High-performance sparse matrix operations via SciRS2
- Parallel optimization using `scirs2_core::parallel_ops`
- SIMD-accelerated energy calculations
- Memory-efficient large problem handling

## Key Features

### Core Capabilities
- **Symbolic Problem Construction**: Define optimization problems using intuitive symbolic expressions
- **Higher-Order Binary Optimization (HOBO)**: Support for problems beyond quadratic (3rd order and higher)
- **Advanced Samplers**: 
  - Simulated Annealing (SA) with SIMD optimization
  - Genetic Algorithm (GA) with adaptive operators
  - GPU-accelerated samplers (Armin, MIKAS)
  - Parallel Tempering with adaptive scheduling
  - Machine Learning guided sampling
  - D-Wave quantum hardware integration
- **Auto Result Processing**: Intelligent conversion of solutions to multi-dimensional arrays

### Performance Features
- **GPU Acceleration**: Up to 50x speedup for large problems
- **SIMD Optimizations**: 2-5x faster energy calculations
- **Sparse Matrix Support**: 80-97% memory reduction
- **Problem Decomposition**: Handle problems with 10,000+ variables
- **Multi-GPU Support**: Scale across multiple GPUs

### Advanced Capabilities
- **Constraint Handling**: Equality, inequality, and soft constraints
- **Variable Encodings**: One-hot, binary, unary, and custom encodings
- **Hybrid Algorithms**: Combine quantum and classical approaches
- **Solution Analysis**: Clustering, diversity metrics, correlation analysis
- **Visualization Tools**: Energy landscapes, convergence plots, solution analysis
- **ML Integration**: Neural networks, reinforcement learning, quantum ML

### 🆕 Cutting-Edge Quantum Computing Features
- **Quantum Neural Networks**: Hybrid quantum-classical architectures with advanced training
- **Quantum State Tomography**: State reconstruction with shadow tomography and ML methods
- **Quantum Error Correction**: Advanced QEC codes with ML-based decoding algorithms
- **Tensor Network Algorithms**: MPS, PEPS, MERA algorithms for quantum optimization
- **Advanced Performance Analysis**: Real-time monitoring with ML-based predictions

### Enterprise Features
- **Cloud Integration**: AWS, Azure, and Google Cloud support
- **Benchmarking Framework**: Comprehensive performance analysis
- **Problem Templates**: Pre-built solutions for finance, logistics, drug discovery
- **Testing Framework**: Property-based testing and performance regression
- **Production Ready**: Error handling, logging, monitoring

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
quantrs2-tytan = "0.1.0"

# Optional features
# quantrs2-tytan = { version = "0.1.0", features = ["gpu", "dwave", "scirs"] }
```

### Basic Example

```rust
use quantrs2_tytan::sampler::{SASampler, Sampler};
use ndarray::Array2;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a QUBO matrix
    let mut qubo = Array2::zeros((3, 3));
    qubo[[0, 0]] = -1.0;
    qubo[[1, 1]] = -1.0;
    qubo[[2, 2]] = -1.0;
    qubo[[0, 1]] = 2.0;
    qubo[[1, 0]] = 2.0;

    // Create variable map
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);
    var_map.insert("z".to_string(), 2);

    // Solve with simulated annealing
    let solver = SASampler::new(None);
    let results = solver.run_qubo(&(qubo, var_map), 100)?;

    // Print best solution
    let best = results.iter().min_by_key(|r| r.energy as i32).unwrap();
    println!("Best energy: {}, solution: {:?}", best.energy, best.solution);

    Ok(())
}
```

### Symbolic Example (requires 'dwave' feature)

```rust
#[cfg(feature = "dwave")]
use quantrs2_tytan::{symbols, Compile};
use quantrs2_tytan::sampler::{SASampler, Sampler};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define symbolic variables
    let x = symbols("x");
    let y = symbols("y");
    let z = symbols("z");

    // Create constraint: exactly one variable should be 1
    let expr = (x + y + z - 1).pow(2);

    // Compile to QUBO
    let (qubo, offset) = Compile::new(&expr).get_qubo()?;

    // Solve
    let solver = SASampler::new(None);
    let results = solver.run_qubo(&qubo, 100)?;

    Ok(())
}
```

## Performance

QuantRS2-Tytan delivers exceptional performance across all problem types:

- **Small problems (< 50 variables)**: 2-5x faster with SIMD
- **Medium problems (50-500 variables)**: 10-50x faster with GPU
- **Large problems (> 1000 variables)**: 40-45x faster with GPU
- **HOBO problems**: 50-100x faster with tensor decomposition

See [BENCHMARKS.md](docs/BENCHMARKS.md) for detailed performance analysis.

## Advanced Examples

### 🆕 Quantum Neural Networks

```rust
use quantrs2_tytan::quantum_neural_networks::{QuantumNeuralNetwork, QNNConfig, create_qnn_for_optimization};

fn qnn_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create QNN for optimization
    let mut qnn = create_qnn_for_optimization(4)?; // 4 qubits
    
    // Train on quantum data
    qnn.train_quantum_model()?;
    
    // Use for quantum-enhanced optimization
    let optimized_params = qnn.optimize_parameters()?;
    
    Ok(())
}
```

### 🆕 Tensor Network Sampler

```rust
use quantrs2_tytan::tensor_network_sampler::{create_mps_sampler, create_peps_sampler};
use quantrs2_tytan::sampler::Sampler;

fn tensor_network_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create MPS sampler for 1D problems
    let mps_sampler = create_mps_sampler(64); // bond dimension
    
    // Create PEPS sampler for 2D problems
    let peps_sampler = create_peps_sampler(16, (5, 5)); // 5x5 lattice
    
    // Use with existing QUBO/HOBO interface
    let results = mps_sampler.run_qubo(&qubo, 100)?;
    
    Ok(())
}
```

### 🆕 Advanced Performance Analysis

```rust
use quantrs2_tytan::advanced_performance_analysis::{create_comprehensive_analyzer, AnalysisConfig};

fn performance_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create performance analyzer
    let mut analyzer = create_comprehensive_analyzer();
    
    // Start analysis
    analyzer.start_analysis()?;
    
    // Run your quantum optimization
    // ... optimization code ...
    
    // Perform comprehensive analysis
    analyzer.perform_comprehensive_analysis()?;
    
    // Get bottleneck recommendations
    for recommendation in &analyzer.analysis_results.optimization_recommendations {
        println!("Optimization: {}", recommendation.title);
        println!("Expected benefit: {:.2}%", recommendation.expected_benefit * 100.0);
    }
    
    Ok(())
}
```

### GPU-Accelerated Solving

```rust
#[cfg(feature = "gpu")]
use quantrs2_tytan::sampler::{ArminSampler, Sampler};

fn gpu_example() -> Result<(), Box<dyn std::error::Error>> {
    // Check GPU availability
    if !quantrs2_tytan::is_gpu_available() {
        println!("GPU not available, falling back to CPU");
        return Ok(());
    }

    // Use GPU-accelerated sampler for large problems
    let solver = ArminSampler::new(None);
    let results = solver.run_qubo(&large_qubo, 1000)?;
    
    Ok(())
}
```

### Parallel Tempering for Complex Problems

```rust
use quantrs2_tytan::parallel_tempering::{ParallelTemperingSampler, PTConfig};

fn parallel_tempering_example() -> Result<(), Box<dyn std::error::Error>> {
    let solver = ParallelTemperingSampler::new(PTConfig {
        num_replicas: 20,
        temperature_range: (0.01, 50.0),
        swap_interval: 5,
        num_sweeps: 50000,
    });

    let results = solver.run_qubo(&qubo, 10)?;
    
    Ok(())
}
```

### Problem with Constraints

```rust
use quantrs2_tytan::constraints::add_equality_constraint;

fn constrained_problem() -> Result<(), Box<dyn std::error::Error>> {
    let mut qubo = Array2::zeros((5, 5));
    
    // Add objective function terms
    // ...
    
    // Add constraint: x1 + x2 + x3 = 2
    add_equality_constraint(&mut qubo, &[0, 1, 2], 2.0, 10.0);
    
    // Solve
    let solver = SASampler::new(None);
    let results = solver.run_qubo(&(qubo, var_map), 100)?;
    
    Ok(())
}
```

## Available Features

### Core Features
- `parallel`: Multi-threading support (enabled by default)
- `gpu`: GPU-accelerated samplers using OpenCL/CUDA
- `dwave`: Symbolic math and D-Wave quantum hardware support

### Performance Features
- `scirs`: High-performance computing with SciRS2 libraries (leverages `scirs2_core::parallel_ops` and sparse matrix operations)
- `advanced_optimization`: State-of-the-art optimization algorithms
- `gpu_accelerated`: Full GPU acceleration pipeline
- `simd`: SIMD optimizations for CPU operations

### Analysis Features
- `clustering`: Solution clustering and pattern analysis
- `plotters`: Visualization tools for results and convergence
- `ml`: Machine learning integration
- `benchmark`: Performance benchmarking tools

## Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Feature Summary](docs/FEATURE_SUMMARY.md) - Detailed feature overview
- [Benchmarks](docs/BENCHMARKS.md) - Performance analysis and comparisons
- [Migration Guide](docs/MIGRATION_GUIDE.md) - Migrate from other frameworks
- [Examples](examples/) - Working code examples

## Integration with QuantRS2 Ecosystem

QuantRS2-Tytan seamlessly integrates with the entire QuantRS2 quantum computing framework:

- **quantrs2-core**: Quantum circuit operations and gates
- **quantrs2-sim**: State vector and tensor network simulation
- **quantrs2-anneal**: Core annealing algorithms
- **quantrs2-device**: Hardware backend integration
- **quantrs2-circuit**: Circuit optimization and compilation
- **quantrs2-ml**: Quantum machine learning algorithms

## Building from Source

### Standard Build
```bash
cargo build --release
```

### With All Features
```bash
cargo build --release --all-features
```

### With GPU Support
```bash
cargo build --release --features gpu,gpu_accelerated
```

### Building with SymEngine (for symbolic math)

On macOS:
```bash
brew install symengine gmp mpfr
export SYMENGINE_DIR=$(brew --prefix symengine)
export GMP_DIR=$(brew --prefix gmp)
export MPFR_DIR=$(brew --prefix mpfr)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix symengine)/include -I$(brew --prefix gmp)/include -I$(brew --prefix mpfr)/include"
cargo build --features dwave
```

On Linux:
```bash
sudo apt-get install libsymengine-dev libgmp-dev libmpfr-dev
cargo build --features dwave
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/cool-japan/quantrs.git
cd quantrs/tytan

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check code style
cargo fmt -- --check
cargo clippy -- -D warnings
```

## License

This project is licensed under either:

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.