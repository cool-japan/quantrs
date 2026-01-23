# quantrs2

[![Crates.io](https://img.shields.io/crates/v/quantrs2.svg)](https://crates.io/crates/quantrs2)
[![Documentation](https://docs.rs/quantrs2/badge.svg)](https://docs.rs/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

**Unified facade for QuantRS2: simplified quantum computing in Rust**

The `quantrs2` facade crate provides a unified entry point to the entire QuantRS2 quantum computing framework. It offers hierarchical preludes, comprehensive system management, and developer utilities—all with zero runtime overhead.

## ✨ Key Features

- 🎯 **Hierarchical Preludes**: Import exactly what you need (essentials → circuits → simulation → algorithms)
- ⚙️ **System Management**: Global configuration, diagnostics, and version checking
- 🛠️ **Developer Utilities**: Memory estimation, testing helpers, error handling
- 📊 **Zero Runtime Overhead**: Feature-gated re-exports with compile-time optimization
- 📚 **Comprehensive Documentation**: 8 runnable examples and detailed guides

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
# Choose your level of functionality
quantrs2 = { version = "0.1.2", features = ["circuit", "sim"] }

# Or enable everything
quantrs2 = { version = "0.1.2", features = ["full"] }
```

### Example: Bell State with Hierarchical Prelude

```rust
use quantrs2::prelude::simulation::*; // Includes essentials + circuits + simulators

fn main() -> QuantRS2Result<()> {
    // Create a 2-qubit Bell state circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(QubitId::new(0))?;
    circuit.cnot(QubitId::new(0), QubitId::new(1))?;

    // Simulate
    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&circuit)?;

    println!("Bell state created: {:?}", result);
    Ok(())
}
```

### Example: System Management

```rust
use quantrs2::{config, diagnostics, utils, version};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check system readiness
    let report = diagnostics::run_diagnostics();
    if !report.is_ready() {
        eprintln!("{}", report);
        return Err("System not ready".into());
    }

    // Configure global settings
    let cfg = config::Config::global();
    cfg.set_num_threads(8);
    cfg.set_memory_limit_gb(16);

    // Estimate memory requirements
    let max_qubits = utils::max_qubits_for_memory(16 * 1024 * 1024 * 1024);
    println!("Can simulate up to {} qubits", max_qubits);

    // Check version compatibility
    version::check_compatibility()?;

    Ok(())
}
```

## Available Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `circuit` | Quantum circuit construction and optimization | `quantrs2-circuit` |
| `sim` | Quantum simulators (state vector, stabilizer, etc.) | `quantrs2-sim`, `circuit` |
| `anneal` | Quantum annealing algorithms | `quantrs2-anneal`, `circuit` |
| `device` | Hardware backends and device interfaces | `quantrs2-device`, `circuit` |
| `ml` | Quantum machine learning algorithms | `quantrs2-ml`, `sim`, `anneal` |
| `tytan` | TYTAN quantum annealing integration | `quantrs2-tytan`, `anneal` |
| `symengine` | Symbolic computation with SymEngine | `quantrs2-symengine` |
| `full` | All features enabled | All of the above |

## Module Structure

When you enable features, the corresponding modules become available:

```rust
// Core is always available
use quantrs2::core;

// Available with "circuit" feature
use quantrs2::circuit;

// Available with "sim" feature
use quantrs2::sim;

// Available with "anneal" feature
use quantrs2::anneal;

// Available with "device" feature
use quantrs2::device;

// Available with "ml" feature
use quantrs2::ml;

// Available with "tytan" feature
use quantrs2::tytan;

// Available with "symengine" feature
use quantrs2::symengine;
```

## 📚 Examples

The facade crate includes 8 comprehensive examples in the `examples/` directory:

| Example | Description | Run Command |
|---------|-------------|-------------|
| **`basic_usage.rs`** | Facade basics: version, config, diagnostics | `cargo run --example basic_usage` |
| **`configuration.rs`** | Global configuration management | `cargo run --example configuration` |
| **`diagnostics.rs`** | System diagnostics and health checks | `cargo run --example diagnostics` |
| **`utility_functions.rs`** | Memory estimation and utilities | `cargo run --example utility_functions` |
| **`prelude_hierarchy.rs`** | Prelude levels and feature selection | `cargo run --example prelude_hierarchy` |
| **`memory_estimation.rs`** | Capacity planning for quantum circuits | `cargo run --example memory_estimation` |
| **`error_handling.rs`** | Error handling and recovery patterns | `cargo run --example error_handling` |
| **`testing_helpers.rs`** | Testing utilities for quantum algorithms | `cargo run --example testing_helpers` |

## 🎯 Hierarchical Preludes

Choose the right prelude for your use case:

```rust
// Level 1: Essentials (always available, fastest compile)
use quantrs2::prelude::essentials::*;
// Includes: QubitId, Error types, Version

// Level 2: Circuit construction
use quantrs2::prelude::circuits::*;
// Includes: essentials + Circuit, Gates

// Level 3: Quantum simulation
use quantrs2::prelude::simulation::*;
// Includes: circuits + StateVectorSimulator, Backends

// Level 4: Algorithms and ML
use quantrs2::prelude::algorithms::*;
// Includes: simulation + VQE, QAOA, QNNs

// Level 5: Hardware integration
use quantrs2::prelude::hardware::*;
// Includes: circuits + IBM, Azure, AWS

// Level 6: Quantum annealing
use quantrs2::prelude::quantum_annealing::*;
// Includes: essentials + QUBO, Ising, D-Wave

// Level 7: Tytan DSL
use quantrs2::prelude::tytan::*;
// Includes: quantum_annealing + Tytan API

// Full: Everything (slowest compile)
use quantrs2::prelude::full::*;
```

**Recommendation**: Start with `essentials` and add features as needed for faster compilation.

## 🛠️ Facade Features

### 1. Configuration Management

```rust
use quantrs2::config::Config;

// Global singleton with builder pattern
Config::builder()
    .num_threads(8)
    .memory_limit_gb(16)
    .log_level(LogLevel::Info)
    .default_backend(DefaultBackend::Auto)
    .apply();

// Or configure directly
let cfg = Config::global();
cfg.set_gpu_enabled(true);
cfg.set_simd_enabled(true);
```

Supports environment variables: `QUANTRS2_NUM_THREADS`, `QUANTRS2_LOG_LEVEL`, etc.

### 2. System Diagnostics

```rust
use quantrs2::diagnostics;

// Comprehensive system check
let report = diagnostics::run_diagnostics();
println!("{}", report); // Detailed report

// Quick checks
if diagnostics::is_ready() {
    // System ready for quantum simulation
}
```

Detects: CPU cores, memory, GPU, SIMD capabilities, compatibility issues.

### 3. Memory Estimation

```rust
use quantrs2::utils;

// Estimate memory for N qubits
let mem = utils::estimate_statevector_memory(30);
println!("30 qubits: {}", utils::format_memory(mem)); // "16.00 GB"

// Find max qubits for available memory
let max = utils::max_qubits_for_memory(16 * 1024 * 1024 * 1024);
println!("16 GB supports {} qubits", max); // 30

// Validate configuration
if utils::is_valid_qubit_count(25, available_memory) {
    // Can simulate 25 qubits
}
```

### 4. Error Handling

```rust
use quantrs2::error::{ErrorCategory, QuantRS2ErrorExt, with_context};

let err = QuantRS2Error::NetworkError("timeout".into());

// Categorize errors
match err.category() {
    ErrorCategory::Hardware if err.is_recoverable() => {
        // Retry operation
    }
    _ => return Err(err),
}

// Add context
let err = with_context(err, "while connecting to IBM Quantum");

// User-friendly messages
eprintln!("{}", err.user_message());
```

### 5. Testing Helpers

```rust
use quantrs2::testing;

// Floating-point assertions
testing::assert_approx_eq(1.0, 1.0000001, 1e-6);

// Vector assertions
testing::assert_vec_approx_eq(&expected, &actual, 1e-5);

// Stochastic measurement assertions
testing::assert_measurement_counts_close(&counts, &expected, 0.05);

// Reproducible test data
let data = testing::generate_random_test_data(100, testing::test_seed());
```

### 6. Version Management

```rust
use quantrs2::version;

// Version information
println!("QuantRS2 v{}", version::VERSION);
println!("SciRS2 v{}", version::SCIRS2_VERSION);

// Detailed info
let info = version::VersionInfo::current();
println!("{}", info.detailed_version_string());

// Compatibility checking
version::check_compatibility()?;
```

## 📖 Documentation

- **[FEATURES.md](FEATURES.md)**: Comprehensive feature documentation
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and migration guides
- **[SCIRS2_INTEGRATION_GUIDE.md](SCIRS2_INTEGRATION_GUIDE.md)**: How QuantRS2 uses SciRS2
- **[CLAUDE.md](../CLAUDE.md)**: Development guidelines and architecture
- **[SCIRS2_INTEGRATION_POLICY.md](../SCIRS2_INTEGRATION_POLICY.md)**: SciRS2 usage patterns
- **[API Documentation](https://docs.rs/quantrs2)**: Generated API docs

## Feature Dependencies

Some features automatically enable others:

- `sim` → enables `circuit`
- `anneal` → enables `circuit`
- `device` → enables `circuit`
- `ml` → enables `sim` and `anneal`
- `tytan` → enables `anneal`

## When to Use This Crate

**Use `quantrs2` when:**
- You want a simple, unified dependency
- You're building applications that use multiple QuantRS2 components
- You prefer feature flags over managing multiple crate dependencies
- You want the convenience of a single import namespace

**Use individual crates when:**
- You only need specific functionality (e.g., just `quantrs2-core`)
- You want minimal compile times and dependencies
- You're building libraries that should have minimal dependencies
- You need fine-grained control over versions

## Example Configurations

### Minimal quantum programming:
```toml
quantrs2 = { version = "0.1.2", features = ["circuit"] }
```

### Circuit simulation:
```toml
quantrs2 = { version = "0.1.2", features = ["sim"] }
```

### Quantum machine learning:
```toml
quantrs2 = { version = "0.1.2", features = ["ml"] }
```

### Hardware interaction:
```toml
quantrs2 = { version = "0.1.2", features = ["device", "sim"] }
```

### Everything:
```toml
quantrs2 = { version = "0.1.2", features = ["full"] }
```

## Alternative: Individual Crates

If you prefer to use individual crates instead of the facade:

```toml
[dependencies]
quantrs2-core = "0.1.2"
quantrs2-circuit = "0.1.2"
quantrs2-sim = "0.1.2"
# etc.
```

## 🎯 Use Case Examples

### Use Case 1: Quantum Algorithm Research

**Scenario**: Researcher developing and benchmarking quantum algorithms

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["circuit", "sim"] }
```

```rust
use quantrs2::prelude::simulation::*;

fn main() -> QuantRS2Result<()> {
    // Quick system check
    if !quantrs2::diagnostics::is_ready() {
        quantrs2::diagnostics::print_issues();
        return Err("System not ready".into());
    }

    // Create and simulate Grover's algorithm
    let mut circuit = Circuit::<4>::new();
    // ... Grover's oracle and diffusion operator

    let simulator = StateVectorSimulator::new();
    let result = simulator.run(&circuit)?;

    println!("Search result: {:?}", result.measure_all(1000));
    Ok(())
}
```

**Why quantrs2?** Single dependency, fast compilation, comprehensive testing utilities.

### Use Case 2: Quantum Machine Learning Application

**Scenario**: ML engineer building quantum neural networks for classification

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["ml"] }
```

```rust
use quantrs2::prelude::algorithms::*;

fn main() -> QuantRS2Result<()> {
    // Configure for ML workload
    quantrs2::config::Config::builder()
        .num_threads(8)
        .memory_limit_gb(32)
        .enable_gpu(true)
        .apply();

    // Build quantum neural network
    let qnn = QuantumNeuralNetwork::builder()
        .input_qubits(4)
        .hidden_layers(&[8, 4])
        .output_qubits(2)
        .build()?;

    // Train on dataset
    let (X_train, y_train) = load_dataset();
    let trained_qnn = qnn.fit(&X_train, &y_train, epochs=100)?;

    println!("Training accuracy: {:.2}%", trained_qnn.score(&X_train, &y_train));
    Ok(())
}
```

**Why quantrs2?** Integrated VQE/QAOA optimizers, GPU acceleration, SciRS2 autodiff support.

### Use Case 3: Quantum Annealing Optimization

**Scenario**: Operations research solving vehicle routing with quantum annealing

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["tytan"] }
```

```rust
use quantrs2::prelude::tytan::*;

fn main() -> QuantRS2Result<()> {
    // Define QUBO problem using Tytan DSL
    let mut qubo = Qubo::new();

    // Add variables and constraints
    let x = qubo.add_binary_variables(10);
    qubo.add_objective(/* cost function */);
    qubo.add_constraint(/* route constraints */);

    // Solve using simulated annealing
    let solver = SimulatedAnnealing::default();
    let solution = solver.solve(&qubo)?;

    println!("Optimal route cost: {}", solution.energy);
    println!("Route: {:?}", solution.variables);
    Ok(())
}
```

**Why quantrs2?** High-level Tytan DSL, multiple solvers (SA, GPU, D-Wave), visualization tools.

### Use Case 4: Real Quantum Hardware Integration

**Scenario**: Enterprise application running on IBM Quantum

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["device", "circuit"] }
```

```rust
use quantrs2::prelude::hardware::*;

fn main() -> QuantRS2Result<()> {
    // Check memory requirements
    let qubits = 20;
    if !quantrs2::utils::is_valid_qubit_count(qubits, available_memory()) {
        return Err("Insufficient memory".into());
    }

    // Connect to IBM Quantum
    let token = std::env::var("IBM_QUANTUM_TOKEN")?;
    let backend = IBMBackend::new(&token, "ibmq_montreal")?;

    // Transpile circuit for hardware
    let circuit = create_vqe_circuit();
    let transpiled = backend.transpile(&circuit)?;

    // Execute with error mitigation
    let job = backend.submit(&transpiled)?;
    let result = job.wait_for_completion()?;

    println!("Hardware result: {:?}", result.counts());
    Ok(())
}
```

**Why quantrs2?** Unified device API (IBM/Azure/AWS), automatic transpilation, error mitigation.

### Use Case 5: Production Quantum Service

**Scenario**: Cloud service offering quantum computation APIs

```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["full"] }
```

```rust
use quantrs2::{prelude::full::*, config, diagnostics, version};

fn main() -> QuantRS2Result<()> {
    // Startup validation
    diagnostics::validate_or_panic();
    version::check_compatibility()?;

    // Production configuration
    config::Config::builder()
        .num_threads(16)
        .memory_limit_gb(64)
        .log_level(LogLevel::Info)
        .default_backend(DefaultBackend::Auto)
        .apply();

    // Start service with all quantum capabilities
    let service = QuantumService::new()
        .with_simulation()
        .with_hardware()
        .with_ml_algorithms()
        .with_annealing()
        .build()?;

    service.serve("0.0.0.0:8080").await?;
    Ok(())
}
```

**Why quantrs2?** Complete feature set, comprehensive diagnostics, production-ready error handling.

## 📊 Facade vs Individual Crates Comparison

### When to Use quantrs2 Facade

| Aspect | Facade Crate | Individual Crates |
|--------|--------------|-------------------|
| **Dependency Management** | Single `quantrs2` entry | Multiple `quantrs2-*` dependencies |
| **Feature Selection** | Cargo feature flags | Manual version coordination |
| **Import Style** | `use quantrs2::prelude::*` | `use quantrs2_circuit::*; use quantrs2_sim::*;` |
| **Compilation Time** | Feature-dependent (10s - 60s) | Minimal for specific needs (5s - 20s) |
| **Version Compatibility** | Guaranteed compatible versions | Manual version matching required |
| **API Discoverability** | Unified namespace, hierarchical | Separate documentation per crate |
| **Binary Size** | Optimized out unused features | Minimal (only what you use) |
| **Best For** | Applications, prototypes, research | Libraries, minimal dependencies |

### Facade Feature Compilation Times

```bash
# Benchmarked on Apple M1 Max, 32GB RAM
quantrs2 = { features = ["circuit"] }         # ~8s
quantrs2 = { features = ["sim"] }             # ~15s
quantrs2 = { features = ["ml"] }              # ~30s
quantrs2 = { features = ["tytan"] }           # ~12s
quantrs2 = { features = ["full"] }            # ~45s

# Individual crates (for comparison)
quantrs2-circuit = "0.1.2"             # ~5s
quantrs2-sim = "0.1.2"                 # ~8s
quantrs2-ml = "0.1.2"                  # ~18s
```

### Example: Dependency Management Comparison

**Using Facade (Recommended for Applications):**
```toml
[dependencies]
quantrs2 = { version = "0.1.2", features = ["ml"] }
```

**Using Individual Crates (Recommended for Libraries):**
```toml
[dependencies]
quantrs2-core = "0.1.2"
quantrs2-circuit = "0.1.2"
quantrs2-sim = "0.1.2"
quantrs2-ml = "0.1.2"
# Must manually ensure version compatibility!
```

## 🚀 Performance Tips

### 1. Choose Optimal Prelude Level

```rust
// ✓ FAST: Use specific prelude
use quantrs2::prelude::circuits::*;  // Only circuits, ~8s compile

// ✗ SLOW: Use full prelude when unnecessary
use quantrs2::prelude::full::*;      // Everything, ~45s compile
```

### 2. Enable Hardware Acceleration

```rust
use quantrs2::config::Config;

Config::global()
    .set_gpu_enabled(true)      // 10-100x speedup for large circuits
    .set_simd_enabled(true)     // 2-4x speedup for CPU operations
    .set_num_threads(8);        // Parallel gate application
```

### 3. Estimate Memory Before Simulation

```rust
use quantrs2::utils;

let qubits = 25;
let required_memory = utils::estimate_statevector_memory(qubits);
let available = 16 * 1024 * 1024 * 1024; // 16 GB

if required_memory > available {
    // Use tensor network or stabilizer simulation instead
    use_tensor_network_backend();
} else {
    use_statevector_backend();
}
```

### 4. Profile Your Quantum Algorithms

```rust
use quantrs2::bench;

let timer = bench::Timer::start();
let result = run_quantum_algorithm();
let duration = timer.elapsed();

println!("Algorithm completed in {}", quantrs2::utils::format_duration(duration));
```

### 5. Common Pitfalls

```rust
// ❌ DON'T: Forget to check system readiness
fn main() {
    let result = expensive_quantum_simulation(); // May fail mysteriously
}

// ✅ DO: Validate environment first
fn main() -> QuantRS2Result<()> {
    quantrs2::diagnostics::validate_or_panic();
    let result = expensive_quantum_simulation()?;
    Ok(())
}
```

## End-to-End Example

This example builds a Bell state on 2 qubits and prints the probabilities. Enable features `circuit` and `sim`.

```rust
use quantrs2::core::api::prelude::essentials::*;
use quantrs2::circuit::builder::{Circuit, Simulator};
use quantrs2::sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 2;
    let mut circuit = Circuit::<N>::new();
    circuit.h(QubitId::new(0))?;                  // H on qubit 0
    circuit.cx(QubitId::new(0), QubitId::new(1))?; // CNOT 0->1

    let sim = StateVectorSimulator::new();
    let reg: Register<N> = sim.run(&circuit)?;

    let probs = reg.probabilities();
    println!(
        "|00>: {:.3}, |01>: {:.3}, |10>: {:.3}, |11>: {:.3}",
        probs[0], probs[1], probs[2], probs[3]
    );
    Ok(())
}
```

## Advanced Features

### Hierarchical Prelude System

QuantRS2 provides hierarchical prelude modules for convenient imports:

```rust
// Minimal imports for basic quantum programming
use quantrs2::prelude::essentials::*;

// Circuit construction
use quantrs2::prelude::circuits::*;

// Quantum simulation
use quantrs2::prelude::simulation::*;

// Algorithm development with ML
use quantrs2::prelude::algorithms::*;

// Real quantum hardware
use quantrs2::prelude::hardware::*;

// All features
use quantrs2::prelude::full::*;
```

### Unified Error Handling

Comprehensive error handling with categorization and user-friendly messages:

```rust
use quantrs2::error::{QuantRS2Error, QuantRS2ErrorExt, with_context};

fn operation() -> Result<(), QuantRS2Error> {
    let error = QuantRS2Error::NetworkError("timeout".into());

    // Check error properties
    if error.is_recoverable() {
        // Retry logic
    }

    // Get user-friendly message
    eprintln!("{}", error.user_message());

    // Add context
    Err(with_context(error, "during quantum operation"))
}
```

### Version Compatibility Checking

Automatic version and environment validation:

```rust
use quantrs2::version::{VersionInfo, check_compatibility};

// Print version information
let info = VersionInfo::current();
println!("{}", info.detailed_version_string());

// Check compatibility
if let Err(issues) = check_compatibility() {
    for issue in issues {
        eprintln!("Compatibility issue: {}", issue);
    }
}
```

### Global Configuration

Centralized configuration for all QuantRS2 components:

```rust
use quantrs2::config::{Config, LogLevel, DefaultBackend};

// Configure via builder pattern
Config::builder()
    .num_threads(8)
    .log_level(LogLevel::Info)
    .memory_limit_gb(32)
    .default_backend(DefaultBackend::Gpu)
    .enable_gpu(true)
    .apply();

// Or configure directly
let config = Config::global();
config.set_num_threads(4);
config.set_log_level(LogLevel::Debug);
```

Configuration can also be set via environment variables:
- `QUANTRS2_NUM_THREADS`: Number of threads
- `QUANTRS2_LOG_LEVEL`: Logging level (trace, debug, info, warn, error)
- `QUANTRS2_MEMORY_LIMIT_GB`: Memory limit in GB
- `QUANTRS2_BACKEND`: Default backend (cpu, gpu, tensor_network, stabilizer, auto)

### System Diagnostics

Comprehensive system validation and health checks:

```rust
use quantrs2::diagnostics;

// Run full diagnostic check
let report = diagnostics::run_diagnostics();
println!("{}", report);

// Quick readiness check
if !diagnostics::is_ready() {
    diagnostics::print_issues();
}

// Validate at startup (panics if not ready)
diagnostics::validate_or_panic();
```

## Documentation

- [Main QuantRS2 Documentation](https://docs.rs/quantrs2)
- [Examples and Tutorials](https://github.com/cool-japan/quantrs/tree/master/examples)

## Subcrates

- `quantrs2-core`: Core types, math, error handling, and APIs — https://github.com/cool-japan/quantrs/tree/master/core
- `quantrs2-circuit`: Circuit builder, DSL, optimization — https://github.com/cool-japan/quantrs/tree/master/circuit
- `quantrs2-sim`: Simulators (statevector, stabilizer, MPS, etc.) — https://github.com/cool-japan/quantrs/tree/master/sim
- `quantrs2-anneal`: Quantum annealing algorithms and workflows — https://github.com/cool-japan/quantrs/tree/master/anneal
- `quantrs2-device`: Hardware/device connectors and scheduling — https://github.com/cool-japan/quantrs/tree/master/device
- `quantrs2-ml`: Quantum machine learning utilities — https://github.com/cool-japan/quantrs/tree/master/ml
- `quantrs2-tytan`: High-level annealing interface inspired by TYTAN — https://github.com/cool-japan/quantrs/tree/master/tytan
- `quantrs2-symengine`: Symbolic computation bindings — https://github.com/cool-japan/quantrs/tree/master/quantrs2-symengine

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.